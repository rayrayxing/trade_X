"""The trading core: one loop for replay, paper and live.

At each closed bar, for each symbol:

1. The broker processes the bar: orders placed at the last close fill at this open,
   stops and targets are checked inside the bar.
2. Open positions are reviewed with the same PositionReviewer the backtester uses
   (time stop, stale, unreachable target, rollover, events, break-even and trailing).
3. Strategies vote. Qualified strategies vote in the ensemble book; every active
   strategy also trades its own virtual book, so all of them build a forward record.
4. Gate 2 finalises a plan, gate 3 applies context vetoes (event calendar, short-side
   checks, agent vetoes), gate 4 (the risk gate) sets the size.
5. Every step is a ledger row under one decision ID. Rejected and vetoed plans are
   followed by the counterfactual tracker to the exit they would have had.

Once per New York day the loop writes an equity snapshot per book (positions,
exposure per currency, limits, config hash) so the dashboard can look back in time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradex.backtest.engine import _policy_from_spec
from tradex.core.counterfactual import CounterfactualTracker
from tradex.core.interfaces import OrderRequest, ReplayClock, ReplayData
from tradex.core.ledger import Ledger
from tradex.core.records import (Close, DecisionIds, EquitySnapshot, ExitChange, Fill, Order, TradePlan, Veto,
                                 Vote)
from tradex.costs.models import model_for, next_rollover
from tradex.decision.ensemble import DEFAULT_HIT_RATE, PlanRules, finalise
from tradex.events import EventCalendar
from tradex.execution.checks import SanityPolicy, ShortInfo, ShortPolicy, sanity_check, short_check
from tradex.execution.sim import SimBroker
from tradex.positions.review import ActionKind, MarketSnapshot, OpenPosition, PositionReviewer
from tradex.risk.exposure import Leg, net_open_position
from tradex.risk.gate import BookState, RiskGate
from tradex.strategy.spec import SignalFrame, StrategySpec, compute_signals

NY = "America/New_York"
QUALIFIED = {"validated", "paper", "live"}
INACTIVE = {"retired", "rejected"}


@dataclass
class CoreConfig:
    initial_equity: float = 10_000.0
    virtual_equity: float = 10_000.0
    virtual_books: bool = True
    qualified_statuses: set[str] = field(default_factory=lambda: set(QUALIFIED))
    simulation: bool = True                       # unknown borrow data is allowed (and labelled) in simulation
    family_weights: dict[str, float] | None = None


@dataclass
class _Meta:
    strategy_id: str
    initial_risk: float
    max_bars: int
    target: float
    targets: list[float]
    bars_held: int = 0
    best_price: float | None = None


class TradingCore:
    def __init__(self, strategies: list[StrategySpec], data: ReplayData, clock: ReplayClock, ledger: Ledger,
                 gate: RiskGate, calendar: EventCalendar | None = None,
                 short_info: dict[str, ShortInfo] | None = None, fx: dict[str, pd.Series] | None = None,
                 cfg: CoreConfig | None = None):
        self.cfg = cfg or CoreConfig()
        self.strategies = [s for s in strategies if s.status not in INACTIVE]
        self.data, self.clock, self.ledger, self.gate = data, clock, ledger, gate
        self.calendar = calendar or EventCalendar()
        self.short_info = short_info or {}
        self.fx = fx
        pol = gate.policy
        self.rules = PlanRules.from_policy(pol)
        self.short_pol = ShortPolicy(**{k: v for k, v in pol.get("shorts", {}).items()},
                                     require_known_borrow=not self.cfg.simulation)
        self.sanity_pol = SanityPolicy(**pol.get("sanity", {}))
        self.tier = int(pol["book"].get("start_tier", 2))
        self.costs = {"stocks": model_for("stocks"), "forex": model_for("forex")}
        self.ids = DecisionIds()
        self.paused = False
        bar = data.bar
        self.brokers: dict[str, SimBroker] = {
            "ensemble": SimBroker(self.cfg.initial_equity, self.costs, fx, bar, book="ensemble")}
        if self.cfg.virtual_books:
            for s in self.strategies:
                self.brokers[f"virtual:{s.id}"] = SimBroker(self.cfg.virtual_equity, self.costs, fx, bar,
                                                            book=f"virtual:{s.id}")
        self.meta: dict[tuple[str, str], _Meta] = {}
        self.reviewers = {s.id: PositionReviewer(_policy_from_spec(s)) for s in self.strategies}
        self.spec_by_id = {s.id: s for s in self.strategies}
        self.signals: dict[tuple[str, str], SignalFrame] = {}
        for s in self.strategies:
            for sym in s.universe:
                if sym in data.frames:
                    self.signals[(s.id, sym)] = compute_signals(s, data.frames[sym])
        self.cf = CounterfactualTracker()
        self._closed_seen: dict[str, int] = {b: 0 for b in self.brokers}
        self._last_snap_day = None

    # --- running ---------------------------------------------------------------------

    def run(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict[str, Any]:
        t0 = start.isoformat() if start is not None else "start"
        self.ledger.set_config(t0, "config/risk/policy.yaml", self.gate.policy)
        last_close = None
        for ts, items in self.data.timeline(start, end):
            close_t = ts + self.data.bar
            self.clock.set(close_t)
            self._apply_commands(close_t)
            self._maybe_snapshot(close_t)
            for sym, i in items:
                self.on_bar(sym, ts, i)
            last_close = close_t
        if last_close is not None:
            for rec in self.cf.finish(last_close):
                self.ledger.append(rec)
            self._snapshot_all(last_close)
        return self.summary()

    def on_bar(self, sym: str, ts: pd.Timestamp, i: int) -> None:
        df = self.data.frames[sym]
        o, h, l, c = (float(df[k].iloc[i]) for k in ("open", "high", "low", "close"))
        close_t = ts + self.data.bar
        ct = close_t.isoformat()
        for book, br in self.brokers.items():
            for f in br.on_bar(sym, ts, o, h, l, c):
                self.ledger.append(Fill(f.decision_id, f.client_order_id, f.time.isoformat(), f.symbol, f.side, f.qty,
                                        f.price, round(f.fees_usd, 4), round(f.spread_slippage_usd, 4), f.book))
            self._record_closes(book, br)
        for rec in self.cf.on_bar(sym, close_t, h, l, c):
            self.ledger.append(rec)
        for book, br in self.brokers.items():
            self._review_positions(book, br, sym, close_t, c, i)
        votes = self._votes(sym, i, ct)
        if not votes:
            return
        qualified = [v for v in votes if self.spec_by_id[v.strategy_id].status in self.cfg.qualified_statuses]
        if qualified and not self.paused and not self._busy("ensemble", sym):
            self._decide("ensemble", qualified, self.rules, close_t)
        if self.cfg.virtual_books:
            single = PlanRules(1, 0.0, 1.0, self.rules.min_reward_risk)
            for v in votes:
                book = f"virtual:{v.strategy_id}"
                if not self._busy(book, sym):
                    self._decide(book, [v], single, close_t)

    # --- the four gates ----------------------------------------------------------------

    def _votes(self, sym: str, i: int, ct: str) -> list[Vote]:
        out = []
        for s in self.strategies:
            sig = self.signals.get((s.id, sym))
            if sig is None or i < 1:
                continue
            le, se = bool(sig.long_entry.iloc[i]), bool(sig.short_entry.iloc[i])
            if le == se:
                continue
            atr = float(sig.atr.iloc[i])
            if not np.isfinite(atr) or atr <= 0:
                continue
            d = 1 if le else -1
            c = float(self.data.frames[sym]["close"].iloc[i])
            stop_dist = s.exit.stop_atr * atr
            out.append(Vote(time=ct, strategy_id=s.id, strategy_version=s.version, family=s.family, symbol=sym,
                            asset_class=s.asset_class, direction=d,
                            strength=float(s.stats.get("hit_rate", DEFAULT_HIT_RATE)), entry_ref=c,
                            stop=c - d * stop_dist, targets=[c + d * s.exit.target_r * stop_dist],
                            max_bars=s.exit.max_bars, knowable_at=ct))
        return out

    def _decide(self, book: str, votes: list[Vote], rules: PlanRules, close_t: pd.Timestamp) -> None:
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
        hold_end = close_t + self.data.bar * plan.max_bars
        veto, factor, _ = self.calendar.check(plan.symbol, plan.asset_class, close_t, hold_end)
        if veto:
            self._block(plan, "calendar", veto, close_t)
            return
        sv = short_check(plan.asset_class, plan.direction, self.short_info.get(plan.symbol), self.short_pol)
        if sv:
            self._block(plan, "short_check", sv, close_t)
            return
        # gate 4: size
        bu = br.base_to_usd(plan.symbol, plan.asset_class, close_t)
        cm = self.costs[plan.asset_class]
        fee_fn = lambda q: (sum(cm.order_fees(plan.symbol, plan.direction, q, plan.entry_price, close_t).values())  # noqa: E731
                            + sum(cm.order_fees(plan.symbol, -plan.direction, q, plan.targets[0], close_t).values()))
        state = BookState(br.equity(), self._legs(br), self.tier, self.fx)
        promoted = book == "ensemble" and all(self.spec_by_id[s].status == "live" for s in plan.strategies)
        verdict = self.gate.review(plan, state, bu, fee_fn, factor, promoted)
        self.ledger.append(verdict)
        if verdict.outcome != "accepted":
            self._block(plan, "risk", "; ".join(verdict.reasons), close_t)
            return
        insane = sanity_check(verdict.qty, plan.entry_price, bu, br.equity(), br.mark(plan.symbol), self.sanity_pol)
        if insane:
            self._block(plan, "sanity", insane, close_t)
            return
        side = plan.direction
        req = OrderRequest(f"{did}-entry", did, plan.symbol, plan.asset_class, side, verdict.qty, "market", None,
                           plan.stop, plan.targets[0], "entry", book)
        br.place(req)
        self.ledger.append(Order(did, req.client_order_id, close_t.isoformat(), plan.symbol, side, verdict.qty,
                                 "market", None, "entry", book))
        self.meta[(book, did)] = _Meta(plan.strategies[0], abs(plan.entry_price - plan.stop), plan.max_bars,
                                       plan.targets[0], list(plan.targets))

    def _block(self, plan: TradePlan, source: str, reason: str, close_t: pd.Timestamp) -> None:
        self.ledger.append(Veto(plan.decision_id, close_t.isoformat(), source, reason))
        if plan.book == "ensemble":
            self.cf.track(plan, source)

    # --- position management --------------------------------------------------------------

    def _review_positions(self, book: str, br: SimBroker, sym: str, close_t: pd.Timestamp, c: float, i: int) -> None:
        for p in br.positions():
            if p.symbol != sym:
                continue
            m = self.meta.get((book, p.decision_id))
            if m is None:
                continue
            m.bars_held += 1
            m.best_price = c if m.best_price is None else (max(m.best_price, c) if p.direction > 0 else min(m.best_price, c))
            spec = self.spec_by_id.get(m.strategy_id)
            atr = float(self.signals[(m.strategy_id, sym)].atr.iloc[i]) if spec and (m.strategy_id, sym) in self.signals else 0.0
            op = OpenPosition(sym, p.asset_class, m.strategy_id, p.direction, p.qty, p.entry_time, p.entry_price,
                              p.stop if p.stop is not None else p.entry_price, m.target, m.initial_risk, m.max_bars,
                              m.bars_held, m.best_price, br.base_to_usd(sym, p.asset_class, close_t))
            snap = MarketSnapshot(close_t, c, atr if np.isfinite(atr) else 0.0, self.data.bar.total_seconds() / 3600)
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
                elif act.kind in (ActionKind.CLOSE, ActionKind.REDUCE) and not br.has_pending(sym):
                    qty = p.qty if act.kind == ActionKind.CLOSE else max(1.0, np.floor(p.qty * (act.fraction or 0.5)))
                    req = OrderRequest(f"{p.decision_id}-exit-{m.bars_held}", p.decision_id, sym, p.asset_class,
                                       -p.direction, qty, "market", purpose="exit", book=book)
                    br.place(req)
                    self.ledger.append(Order(p.decision_id, req.client_order_id, close_t.isoformat(), sym, -p.direction,
                                             qty, "market", None, f"exit: {act.reason}", book))

    def _record_closes(self, book: str, br: SimBroker) -> None:
        new = br.closed[self._closed_seen[book]:]
        self._closed_seen[book] = len(br.closed)
        for lot in new:
            self.ledger.append(Close(lot.decision_id, lot.time.isoformat(), lot.symbol, lot.exit_price, lot.qty,
                                     round(lot.net_pnl_usd, 4), round(lot.r_multiple, 4), lot.reason, lot.book))
            if lot.fully_closed:
                self.meta.pop((book, lot.decision_id), None)

    def _busy(self, book: str, sym: str) -> bool:
        br = self.brokers[book]
        return br.has_pending(sym) or any(p.symbol == sym for p in br.positions())

    def _legs(self, br: SimBroker) -> list[Leg]:
        out = []
        for p in br.positions(account=None):
            px = br.mark(p.symbol) or p.entry_price
            out.append(Leg(p.symbol, p.asset_class, p.direction, p.qty, px, p.stop, p.account))
        return out

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
                    br.place(OrderRequest(f"{p.decision_id}-flatten", p.decision_id, p.symbol, p.asset_class,
                                          -p.direction, p.qty, "market", purpose="exit"))
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
            legs = self._legs(br)
            eq = br.equity()
            self.ledger.append(EquitySnapshot(
                time=t.isoformat(), book=book, equity_usd=round(eq, 2), cash_usd=round(br.cash(), 2),
                open_risk_usd=round(br.open_risk_usd(), 2),
                positions=[{"decision_id": p.decision_id, "symbol": p.symbol, "direction": p.direction, "qty": p.qty,
                            "entry": p.entry_price, "stop": p.stop, "target": p.take_profit, "account": p.account,
                            "mark": br.mark(p.symbol)} for p in br.positions(account=None)],
                exposure_by_currency={k: round(v, 2) for k, v in net_open_position(legs, self.fx).items()},
                limits={"heat_cap_usd": round(eq * bk["heat_cap_pct"] / 100, 2), "tier": self.tier,
                        "paused": self.paused},
                config_hash=self.ledger.config_hash))

    def summary(self) -> dict[str, Any]:
        out = {}
        for book, br in self.brokers.items():
            lots = [x for x in br.closed if x.fully_closed]
            out[book] = {"equity": round(br.equity(), 2), "closed_trades": len(lots),
                         "net_pnl": round(sum(x.net_pnl_usd for x in br.closed), 2),
                         "open_positions": len(br.positions())}
        return out
