"""The trading core: one bar-close core for replay, paper and live.

``TradingCore.on_bar_close(tf, ts)`` runs everything that happens when the bars of one
timeframe close at ``ts``. Replay drives it from recorded bars with a replay clock; the
live runtime drives it from the bar-close scheduler with the wall clock and a bar store
the feed appends to. Brokers are injected per book: in paper and live the ensemble book
gets the venue adapter (behind the order guard), while every virtual book stays a
SimBroker fed the same live bars. Same strategies and same bars give the same decision
rows in both (tested).

For each symbol with a bar closing at ``ts``:

1. On the base (finest) timeframe, simulated brokers process the bar: orders placed at
   the last close fill at this open, stops and targets are checked inside the bar. Every
   book's new fills (polled with ``fills(since)``, so venues work the same) become Fill
   and Close rows.
2. Open positions of strategies on this timeframe are reviewed with the same
   PositionReviewer the backtester uses (time stop, stale, unreachable target, rollover,
   events, break-even and trailing).
3. Strategies subscribed to this timeframe vote from incremental signals (a rolling
   window of at least three lookbacks). Qualified strategies vote in the ensemble book;
   every active strategy also trades its own virtual book.
4. Gate 2 finalises a plan, gate 3 applies context vetoes (event calendar, short-side
   checks), gate 4 (the risk gate) sets the size, within the free margin of the venue
   the trade goes to. The order cites the verdict ID.
5. Every step is a ledger row under one decision ID. Rejected and vetoed plans are
   followed by the counterfactual tracker to the exit they would have had.

Marks come from MarketData and FX rates from a rate source; in paper and live a missing
mark or rate blocks the trade (Health row) instead of being estimated. Once per New York
day the core writes an equity snapshot per book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradex.backtest.engine import _policy_from_spec
from tradex.core.counterfactual import CounterfactualTracker
from tradex.core.interfaces import Broker, BrokerFill, Clock, MarketData, OrderRequest
from tradex.core.ledger import Ledger
from tradex.core.records import (Close, DecisionIds, EquitySnapshot, ExitChange, Fill, Health, Order, TradePlan,
                                 Veto, Vote)
from tradex.costs.models import model_for, next_rollover, split_pair
from tradex.decision.ensemble import DEFAULT_HIT_RATE, PlanRules, finalise
from tradex.events import EventCalendar
from tradex.execution.checks import SanityPolicy, ShortInfo, ShortPolicy, sanity_check, short_check
from tradex.execution.guard import OrderRefused
from tradex.execution.sim import MARGIN_RATES, SimBroker
from tradex.positions.review import ActionKind, MarketSnapshot, OpenPosition, PositionReviewer
from tradex.risk.exposure import Leg, net_open_position
from tradex.risk.gate import BookState, RiskGate
from tradex.runtime.fx import LIVE_MODES, MissingRate, RateSource, SeriesRates, rate_snapshot
from tradex.runtime.signals import SignalCache
from tradex.runtime.venues import margin_max_qty
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

NY = "America/New_York"
QUALIFIED = {"validated", "paper", "live"}
INACTIVE = {"retired", "rejected"}


class MissingData(LookupError):
    """A mark or bar the core needs in paper/live is not there."""


@dataclass
class CoreConfig:
    initial_equity: float = 10_000.0
    virtual_equity: float = 10_000.0
    virtual_books: bool = True
    qualified_statuses: set[str] = field(default_factory=lambda: set(QUALIFIED))
    simulation: bool = True                       # unknown borrow data is allowed (and labelled) in simulation
    family_weights: dict[str, float] | None = None
    mode: str = "replay"                          # backtest | replay | paper | live
    signal_window_mult: int = 3                   # rolling signal window, in strategy lookbacks
    margin_rates: dict[str, float] = field(default_factory=lambda: dict(MARGIN_RATES))  # conservative: 20:1 fx, cash stocks

    @property
    def live(self) -> bool:
        return self.mode in LIVE_MODES


@dataclass
class _Meta:
    strategy_id: str
    initial_risk: float
    max_bars: int
    target: float
    targets: list[float]
    stop: float                                   # stop at entry: the R unit
    bars_held: int = 0
    best_price: float | None = None
    qty: float = 0.0                              # filled and still open
    risk_usd: float = 0.0                         # loss to the entry stop on what is still open


class TradingCore:
    def __init__(self, strategies: list[StrategySpec], data: MarketData, clock: Clock, ledger: Ledger,
                 gate: RiskGate, brokers: dict[str, Broker] | None = None, rates: RateSource | None = None,
                 calendar: EventCalendar | None = None, short_info: dict[str, ShortInfo] | None = None,
                 cfg: CoreConfig | None = None, symbols: list[str] | None = None, base_tf: str | None = None):
        self.cfg = cfg or CoreConfig()
        self.strategies = [s for s in strategies if s.status not in INACTIVE]
        self.data, self.clock, self.ledger, self.gate = data, clock, ledger, gate
        self.rates = rates or SeriesRates(None, self.cfg.mode)
        self.calendar = calendar or EventCalendar()
        self.short_info = short_info or {}
        pol = gate.policy
        self.rules = PlanRules.from_policy(pol)
        self.short_pol = ShortPolicy(**{k: v for k, v in pol.get("shorts", {}).items()},
                                     require_known_borrow=not self.cfg.simulation or self.cfg.live)
        self.sanity_pol = SanityPolicy(**pol.get("sanity", {}))
        self.tier = int(pol["book"].get("start_tier", 2))
        self.costs = {"stocks": model_for("stocks"), "forex": model_for("forex")}
        self.ids = DecisionIds()
        self.paused = False
        self.tfs = sorted({s.signal_tf for s in self.strategies}, key=duration)
        self.base_tf = base_tf or (self.tfs[0] if self.tfs else "H1")
        self.bar = duration(self.base_tf)
        if self.tfs and duration(self.tfs[0]) < self.bar:
            raise ValueError(f"base timeframe {self.base_tf} is coarser than strategy timeframe {self.tfs[0]}")
        self.symbols = symbols or list(dict.fromkeys(u for s in self.strategies for u in s.universe
                                                     if not u.startswith("$")))
        self.brokers: dict[str, Broker] = dict(brokers or {})
        self.brokers.setdefault("ensemble", self._sim("ensemble", self.cfg.initial_equity))
        if self.cfg.virtual_books:
            for s in self.strategies:
                self.brokers.setdefault(f"virtual:{s.id}", self._sim(f"virtual:{s.id}", self.cfg.virtual_equity))
        self.meta: dict[tuple[str, str], _Meta] = {}
        self.reviewers = {s.id: PositionReviewer(_policy_from_spec(s)) for s in self.strategies}
        self.spec_by_id = {s.id: s for s in self.strategies}
        self.signals = SignalCache(data, self.cfg.signal_window_mult)
        self.cf = CounterfactualTracker()
        self._orders: dict[str, dict[str, str]] = {b: {} for b in self.brokers}      # open client IDs -> symbol
        self._cursor: dict[tuple[str, str], tuple[pd.Timestamp, set[str]]] = {}
        self._stats: dict[str, list[float]] = {b: [0, 0.0] for b in self.brokers}    # closed trades, net P&L
        self._last_snap_day = None
        self._last_begin: pd.Timestamp | None = None
        self._config_set = False

    def _sim(self, book: str, cash: float) -> SimBroker:
        return SimBroker(cash, self.costs, bar=self.bar, book=book, rate_fn=self.rates.usd_per_unit)

    # --- the bar-close entry point --------------------------------------------------

    def on_bar_close(self, tf: str, ts: pd.Timestamp) -> None:
        """Run everything for the ``tf`` bars that closed at ``ts``. Idempotence across
        restarts is the scheduler's job (jobs table); this call does the work once."""
        d = duration(tf)
        subscribed = {u for s in self.strategies if s.signal_tf == tf for u in s.universe}
        todo = []
        for sym in self.symbols:
            if tf != self.base_tf and sym not in subscribed:
                continue
            b = self.data.bars(sym, ts, tf)
            if len(b) and b.index[-1] == ts - d:
                todo.append((sym, b.iloc[-1]))
        if not todo:
            return
        self._begin(ts)
        for sym, bar in todo:
            self._step(sym, tf, ts, bar)

    def finish(self, t: pd.Timestamp) -> dict[str, Any]:
        """End of a replay: close out the counterfactuals and write final snapshots."""
        for rec in self.cf.finish(t):
            self.ledger.append(rec)
        self._snapshot_all(t)
        return self.summary()

    def _begin(self, ts: pd.Timestamp) -> None:
        if ts == self._last_begin:
            return
        self._last_begin = ts
        if not self._config_set:
            self.ledger.set_config(ts.isoformat(), "config/risk/policy.yaml", self.gate.policy)
            self._config_set = True
        self._apply_commands(ts)
        self._maybe_snapshot(ts)

    def _step(self, sym: str, tf: str, close_t: pd.Timestamp, bar: pd.Series) -> None:
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        if tf == self.base_tf:
            for br in self.brokers.values():
                if getattr(br, "simulated", False):
                    br.on_bar(sym, close_t - self.bar, o, h, l, c)
        for book, br in self.brokers.items():
            self._record_fills(book, br, sym)
        if tf == self.base_tf:
            for rec in self.cf.on_bar(sym, close_t, h, l, c):
                self.ledger.append(rec)
        for book, br in self.brokers.items():
            self._review_positions(book, br, sym, tf, close_t, c)
        votes = self._votes(sym, tf, close_t)
        if not votes:
            return
        qualified = [v for v in votes if self.spec_by_id[v.strategy_id].status in self.cfg.qualified_statuses]
        if qualified and not self.paused and not self._busy("ensemble", sym):
            self._decide("ensemble", qualified, self.rules, close_t, tf)
        if self.cfg.virtual_books:
            single = PlanRules(1, 0.0, 1.0, self.rules.min_reward_risk)
            for v in votes:
                book = f"virtual:{v.strategy_id}"
                if book in self.brokers and not self._busy(book, sym):
                    self._decide(book, [v], single, close_t, tf)

    # --- fills and closes, from any venue ----------------------------------------------

    def _record_fills(self, book: str, br: Broker, sym: str) -> None:
        since, seen = self._cursor.get((book, sym), (None, set()))
        new = [f for f in br.fills(since) if f.symbol == sym and f.fill_id not in seen]
        if not new:
            return
        last = max(f.time for f in new)
        keep = {i for i in seen if since == last} | {f.fill_id for f in new if f.time == last}
        self._cursor[(book, sym)] = (last, keep)
        for f in new:
            self.ledger.append(Fill(f.decision_id, f.client_order_id, f.time.isoformat(), f.symbol, f.side, f.qty,
                                    f.price, round(f.fees_usd, 4), round(f.spread_slippage_usd, 4), f.book))
        for f in new:
            if f.reason == "entry":
                self._on_entry(book, f)
            else:
                self._on_close(book, br, f)

    def _on_entry(self, book: str, f: BrokerFill) -> None:
        m = self.meta.get((book, f.decision_id))
        if m is None:
            return
        try:
            bu = self._bu(f.symbol, self._asset_class(m), f.time)
        except MissingRate as exc:
            self._health("fx_rate", False, f"{f.decision_id}: {exc}; R multiple unknown", f.time)
            bu = 0.0
        m.qty += f.qty
        m.risk_usd += abs(f.price - m.stop) * f.qty * bu

    def _on_close(self, book: str, br: Broker, f: BrokerFill) -> None:
        m = self.meta.get((book, f.decision_id))
        net = f.net_pnl_usd if f.net_pnl_usd is not None else 0.0
        risk = 0.0
        if m is not None and m.qty > 0:
            share = min(1.0, f.qty / m.qty)
            risk = m.risk_usd * share
            m.risk_usd -= risk
            m.qty -= f.qty
        closed = f.position_closed if f.position_closed is not None else \
            not any(p.decision_id == f.decision_id for p in br.positions())
        self.ledger.append(Close(f.decision_id, f.time.isoformat(), f.symbol, f.price, f.qty, round(net, 4),
                                 round(net / risk if risk else 0.0, 4), f.reason, f.book))
        st = self._stats.setdefault(book, [0, 0.0])
        st[1] += net
        if closed:
            st[0] += 1
            self.meta.pop((book, f.decision_id), None)

    def _asset_class(self, m: _Meta) -> str:
        spec = self.spec_by_id.get(m.strategy_id)
        return spec.asset_class if spec else "stocks"

    # --- the four gates ----------------------------------------------------------------

    def _votes(self, sym: str, tf: str, close_t: pd.Timestamp) -> list[Vote]:
        ct = close_t.isoformat()
        out = []
        for s in self.strategies:
            if s.signal_tf != tf or sym not in s.universe:
                continue
            sp = self.signals.at(s, sym, close_t)
            if sp is None or sp.bars < 2 or sp.long_entry == sp.short_entry:
                continue
            if not np.isfinite(sp.atr) or sp.atr <= 0:
                continue
            d = 1 if sp.long_entry else -1
            c = sp.close
            stop_dist = s.exit.stop_atr * sp.atr
            out.append(Vote(time=ct, strategy_id=s.id, strategy_version=s.version, family=s.family, symbol=sym,
                            asset_class=s.asset_class, direction=d,
                            strength=float(s.stats.get("hit_rate", DEFAULT_HIT_RATE)), entry_ref=c,
                            stop=c - d * stop_dist, targets=[c + d * s.exit.target_r * stop_dist],
                            max_bars=s.exit.max_bars, knowable_at=ct))
        return out

    def _decide(self, book: str, votes: list[Vote], rules: PlanRules, close_t: pd.Timestamp, tf: str) -> None:
        br = self.brokers[book]
        did = self.ids.next(close_t)
        plan, why = finalise(votes, did, rules, self.costs[votes[0].asset_class], self.cfg.family_weights, book)
        if plan is None:
            return
        self.ledger.append(plan)
        for v in votes:
            v.decision_id, v.book = did, book
            self.ledger.append(v)
        if why:
            self._block(plan, "plan", why, close_t)
            return
        # gate 3: context, subtract only
        hold_end = close_t + duration(tf) * plan.max_bars
        veto, factor, _ = self.calendar.check(plan.symbol, plan.asset_class, close_t, hold_end)
        if veto:
            self._block(plan, "calendar", veto, close_t)
            return
        sv = short_check(plan.asset_class, plan.direction, self.short_info.get(plan.symbol), self.short_pol)
        if sv:
            self._block(plan, "short_check", sv, close_t)
            return
        # gate 4: size, from live equity, marks and rates; anything missing blocks
        try:
            bu = self._bu(plan.symbol, plan.asset_class, close_t)
            legs = self._legs(br)
            eq = self._equity(book, close_t)
            fx = rate_snapshot(self.rates, _currencies(legs) | _currencies([plan]), close_t)
            last = self._mark(plan.symbol)
            venue = br.venue_for(plan.asset_class) if hasattr(br, "venue_for") else br
            q_margin, margin = margin_max_qty(venue, self.rates, close_t, self.cfg.margin_rates[plan.asset_class],
                                              plan.entry_price * bu)
        except (MissingRate, MissingData) as exc:
            self._health("data", False, f"{did}: {exc}", close_t)
            self._block(plan, "data", str(exc), close_t)
            return
        cm = self.costs[plan.asset_class]
        fee_fn = lambda q: (sum(cm.order_fees(plan.symbol, plan.direction, q, plan.entry_price, close_t).values())  # noqa: E731
                            + sum(cm.order_fees(plan.symbol, -plan.direction, q, plan.targets[0], close_t).values()))
        state = BookState(eq, legs, self.tier, fx, q_margin, margin)
        promoted = book == "ensemble" and all(self.spec_by_id[s].status == "live" for s in plan.strategies)
        verdict = self.gate.review(plan, state, bu, fee_fn, factor, promoted)
        verdict.verdict_id = f"{did}-v"
        self.ledger.append(verdict)
        if verdict.outcome != "accepted":
            self._block(plan, "risk", "; ".join(verdict.reasons), close_t)
            return
        insane = sanity_check(verdict.qty, plan.entry_price, bu, eq, last, self.sanity_pol)
        if insane:
            self._block(plan, "sanity", insane, close_t)
            return
        side = plan.direction
        req = OrderRequest(f"{did}-entry", did, plan.symbol, plan.asset_class, side, verdict.qty, "market", None,
                           plan.stop, plan.targets[0], "entry", book, verdict_id=verdict.verdict_id)
        if not self._place(book, br, req, close_t):
            self._block(plan, "order_guard", "order refused", close_t)
            return
        self.ledger.append(Order(did, req.client_order_id, close_t.isoformat(), plan.symbol, side, verdict.qty,
                                 "market", None, "entry", book))
        self.meta[(book, did)] = _Meta(plan.strategies[0], abs(plan.entry_price - plan.stop), plan.max_bars,
                                       plan.targets[0], list(plan.targets), plan.stop)

    def _place(self, book: str, br: Broker, req: OrderRequest, t: pd.Timestamp) -> bool:
        try:
            br.place(req)
        except OrderRefused as exc:
            self._health("order_guard", False, str(exc), t)
            return False
        self._orders.setdefault(book, {})[req.client_order_id] = req.symbol
        return True

    def _block(self, plan: TradePlan, source: str, reason: str, close_t: pd.Timestamp) -> None:
        self.ledger.append(Veto(plan.decision_id, close_t.isoformat(), source, reason))
        if plan.book == "ensemble":
            self.cf.track(plan, source)

    # --- position management --------------------------------------------------------------

    def _review_positions(self, book: str, br: Broker, sym: str, tf: str, close_t: pd.Timestamp, c: float) -> None:
        for p in br.positions():
            if p.symbol != sym:
                continue
            m = self.meta.get((book, p.decision_id))
            if m is None:
                continue
            spec = self.spec_by_id.get(m.strategy_id)
            if spec is None or spec.signal_tf != tf:
                continue
            m.bars_held += 1
            m.best_price = c if m.best_price is None else (max(m.best_price, c) if p.direction > 0 else min(m.best_price, c))
            sp = self.signals.at(spec, sym, close_t)
            atr = sp.atr if sp is not None and np.isfinite(sp.atr) else 0.0
            try:
                bu = self._bu(sym, p.asset_class, close_t)
            except MissingRate as exc:
                self._health("fx_rate", False, f"review {p.decision_id}: {exc}", close_t)
                continue
            op = OpenPosition(sym, p.asset_class, m.strategy_id, p.direction, p.qty, p.entry_time, p.entry_price,
                              p.stop if p.stop is not None else p.entry_price, m.target, m.initial_risk, m.max_bars,
                              m.bars_held, m.best_price, bu)
            snap = MarketSnapshot(close_t, c, atr, duration(tf).total_seconds() / 3600)
            if p.asset_class == "forex":
                cut, days = next_rollover(close_t)
                rate = self.costs["forex"].annual_rate(sym, p.direction, close_t)
                snap.next_rollover_time = cut
                snap.next_rollover_cost_usd = -p.qty * c * op.base_to_usd * rate * days / 365.0
            reviewer = self.reviewers.get(m.strategy_id) or PositionReviewer()
            for act in reviewer.review(op, snap):
                if act.kind == ActionKind.MOVE_STOP and act.price is not None:
                    self.ledger.append(ExitChange(p.decision_id, close_t.isoformat(), "stop", p.stop, act.price, act.reason))
                    br.amend_stop(p.decision_id, act.price)
                elif act.kind in (ActionKind.CLOSE, ActionKind.REDUCE) and not self._has_open_order(book, sym):
                    qty = p.qty if act.kind == ActionKind.CLOSE else max(1.0, np.floor(p.qty * (act.fraction or 0.5)))
                    req = OrderRequest(f"{p.decision_id}-exit-{m.bars_held}", p.decision_id, sym, p.asset_class,
                                       -p.direction, qty, "market", purpose="exit", book=book)
                    if self._place(book, br, req, close_t):
                        self.ledger.append(Order(p.decision_id, req.client_order_id, close_t.isoformat(), sym,
                                                 -p.direction, qty, "market", None, f"exit: {act.reason}", book))

    def _has_open_order(self, book: str, sym: str) -> bool:
        br, orders = self.brokers[book], self._orders.setdefault(book, {})
        for coid, s in list(orders.items()):
            if s != sym:
                continue
            if br.order_status(coid).open:
                return True
            del orders[coid]
        return False

    def _busy(self, book: str, sym: str) -> bool:
        return self._has_open_order(book, sym) or any(p.symbol == sym for p in self.brokers[book].positions())

    # --- marks, rates and equity -------------------------------------------------------------

    def _mark(self, sym: str, fallback: float | None = None) -> float | None:
        try:
            px = self.data.last_price(sym)
        except KeyError:
            px = float("nan")
        if np.isfinite(px):
            return float(px)
        if self.cfg.live:
            raise MissingData(f"no mark for {sym}")
        return fallback

    def _bu(self, symbol: str, asset_class: str, ts: pd.Timestamp) -> float:
        if asset_class != "forex":
            return 1.0
        return self.rates.usd_per_unit(split_pair(symbol)[1], ts)

    def _equity(self, book: str, ts: pd.Timestamp) -> float:
        """Book equity in USD: the venue account's equity converted at the live rate."""
        acct = self.brokers[book].account()
        return acct.equity * self.rates.usd_per_unit(acct.currency, ts)

    def _legs(self, br: Broker) -> list[Leg]:
        return [Leg(p.symbol, p.asset_class, p.direction, p.qty, self._mark(p.symbol, p.entry_price), p.stop,
                    p.account) for p in br.positions(account=None)]

    def _open_risk_usd(self, br: Broker, ts: pd.Timestamp) -> float:
        out = 0.0
        for p in br.positions():
            if p.stop is not None:
                px = self._mark(p.symbol, p.entry_price)
                out += max(0.0, p.direction * (px - p.stop)) * p.qty * self._bu(p.symbol, p.asset_class, ts)
        return out

    def _health(self, check: str, ok: bool, detail: str, t: pd.Timestamp) -> None:
        self.ledger.append(_health(t, check, ok, detail))

    # --- commands and snapshots -------------------------------------------------------------

    def _apply_commands(self, t: pd.Timestamp) -> None:
        for cmd in self.ledger.pending_commands():
            name = cmd["command"]
            if name == "pause":
                self.paused, res = True, "no new entries"
            elif name == "resume":
                self.paused, res = False, "entries allowed"
            elif name == "flatten":
                br = self.brokers["ensemble"]
                for p in br.positions():
                    self._place("ensemble", br, OrderRequest(f"{p.decision_id}-flatten", p.decision_id, p.symbol,
                                                             p.asset_class, -p.direction, p.qty, "market",
                                                             purpose="exit"), t)
                self.paused, res = True, f"closing {len(br.positions())} positions at the next open; paused"
            else:
                res = "unknown command"
            self.ledger.mark_command(cmd["id"], t.isoformat(), res)

    def _maybe_snapshot(self, close_t: pd.Timestamp) -> None:
        day = close_t.tz_convert(NY).date()
        if self._last_snap_day is None:
            self._last_snap_day = day
        elif day != self._last_snap_day:
            self._snapshot_all(close_t)
            self._last_snap_day = day

    def _snapshot_all(self, t: pd.Timestamp) -> None:
        bk = self.gate.policy["book"]
        for book, br in self.brokers.items():
            try:
                legs = self._legs(br)
                acct = br.account()
                rate = self.rates.usd_per_unit(acct.currency, t)
                eq = acct.equity * rate
                fx = rate_snapshot(self.rates, _currencies(legs), t)
                risk = self._open_risk_usd(br, t)
            except (MissingRate, MissingData) as exc:
                self._health("snapshot", False, f"{book}: {exc}", t)
                continue
            self.ledger.append(EquitySnapshot(
                time=t.isoformat(), book=book, equity_usd=round(eq, 2), cash_usd=round(acct.cash * rate, 2),
                open_risk_usd=round(risk, 2),
                positions=[{"decision_id": p.decision_id, "symbol": p.symbol, "direction": p.direction, "qty": p.qty,
                            "entry": p.entry_price, "stop": p.stop, "target": p.take_profit, "account": p.account,
                            "mark": self._mark(p.symbol)} for p in br.positions(account=None)],
                exposure_by_currency={k: round(v, 2) for k, v in net_open_position(legs, fx).items()},
                limits={"heat_cap_usd": round(eq * bk["heat_cap_pct"] / 100, 2), "tier": self.tier,
                        "paused": self.paused},
                config_hash=self.ledger.config_hash))

    def summary(self) -> dict[str, Any]:
        out = {}
        for book, br in self.brokers.items():
            n, net = self._stats.get(book, [0, 0.0])
            out[book] = {"equity": round(br.account().equity, 2), "closed_trades": int(n),
                         "net_pnl": round(net, 2), "open_positions": len(br.positions())}
        return out


def _currencies(items) -> set[str]:
    out: set[str] = set()
    for x in items:
        if x.asset_class == "forex":
            out |= set(split_pair(x.symbol))
    return out


def _health(t: pd.Timestamp, check: str, ok: bool, detail: str) -> Health:
    return Health(time=t.isoformat(), check=check, ok=ok, detail=detail)
