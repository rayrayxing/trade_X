"""Bar-by-bar portfolio backtester with full trading costs.

Rules that keep results honest:
- Signals use closed bars only; orders fill at the NEXT bar's open (or, for a spec with
  ``fill: next_close``, at the next bar's close: a market-on-close order placed a bar ahead).
- Stops and targets are checked inside each bar; if both are touched in one bar the
  stop is assumed to hit first. A gap through the stop fills at the open.
- Every fill pays half the spread plus slippage; every order pays broker and
  regulatory fees; open positions pay forex rollover, short borrow and margin interest.
- Open positions are reviewed every bar by the same PositionReviewer the live loop
  uses, so time limits, stale-trade exits and rollover exits are in the backtest.
- An optional day-trade cap (US pattern-day-trader style) can be switched on per account.
- ``exit.session_close`` (stocks): a position still open on the last bar of a US regular
  session is closed at that bar's close (16:00 New York, or the early close on half days),
  and no entry is taken from that bar, so the strategy never holds overnight.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tradex.costs.models import CostModel, model_for, next_rollover, split_pair, usd_per_unit
from tradex.positions.review import (ActionKind, MarketSnapshot, OpenPosition, PositionReviewer,
                                     ReviewPolicy)
from tradex.strategy.spec import SignalFrame, StrategySpec, compute_signals
from tradex.timeframes import duration

NY = "America/New_York"


@dataclass
class EngineConfig:
    initial_equity: float = 10_000.0
    risk_pct: float = 1.0               # equity % lost if a stop is hit, before strategy cap
    max_leverage: float = 1.0           # gross notional / equity
    max_positions: int = 10
    max_heat_pct: float = 12.0          # total open risk as % of equity
    allow_short: bool = True
    # Day-trade cap. Off by default: Ray reports moomoo SG does not apply the US 3-in-5 rule
    # (3 Oct 2026). The live loop should set this from the account's own flags.
    pdt: bool = False
    pdt_threshold: float = 25_000.0
    pdt_max_day_trades: int = 3
    pdt_window_bdays: int = 5
    review: ReviewPolicy | None = None
    start: pd.Timestamp | str | None = None
    end: pd.Timestamp | str | None = None
    fx_rates: dict[str, pd.Series] | None = None   # USD per unit of each currency
    events: dict[str, list] | None = None          # symbol -> event timestamps (earnings etc.)
    size_fn: Callable[[dict], float] | None = None # optional override: returns risk % for a trade


@dataclass
class Trade:
    strategy_id: str
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    qty: float
    notional_usd: float
    risk_usd: float
    gross_pnl: float          # from fill prices, so it already includes spread and slippage
    spread_slippage: float    # spread and slippage paid, for reporting
    fees: float
    financing: float
    borrow: float
    margin_interest: float
    net_pnl: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    pdt_violation: bool = False


@dataclass
class BacktestResult:
    strategy_id: str
    params: dict
    trades: pd.DataFrame
    equity: pd.Series
    warnings: list[str]
    config: EngineConfig

    @property
    def daily_returns(self) -> pd.Series:
        if self.equity.empty:
            return pd.Series(dtype=float)
        d = self.equity.resample("1D").last().dropna()
        return d.pct_change().dropna()


@dataclass
class _Pos:
    p: OpenPosition
    entry_fill: float
    risk_usd: float
    spread_slip: float
    fees: float
    hold: dict = field(default_factory=lambda: defaultdict(float))
    last_accrual: pd.Timestamp | None = None
    entry_date: object = None


def run_backtest(
    spec: StrategySpec,
    data: dict[str, pd.DataFrame],
    costs: CostModel | None = None,
    cfg: EngineConfig | None = None,
    tradable: dict[str, pd.Series] | None = None,
    filter_ctx: dict | None = None,
    params: dict | None = None,
    precomputed: dict[str, SignalFrame] | None = None,
) -> BacktestResult:
    cfg = cfg or EngineConfig()
    if params:
        spec = spec.with_params(params)
    costs = costs or model_for(spec.asset_class)
    reviewer = PositionReviewer(cfg.review or _policy_from_spec(spec))
    bar_td = duration(spec.signal_tf)
    bar_hours = bar_td.total_seconds() / 3600
    warnings: list[str] = []
    start = pd.Timestamp(cfg.start, tz="UTC") if isinstance(cfg.start, str) else cfg.start
    end = pd.Timestamp(cfg.end, tz="UTC") if isinstance(cfg.end, str) else cfg.end

    # Precompute per-symbol arrays. Features see full history so indicators are warm at `start`.
    S = {}
    for sym, bars in data.items():
        if bars.empty:
            continue
        sig = precomputed[sym] if precomputed and sym in precomputed else compute_signals(spec, bars, filter_ctx, sym)
        warnings += [f"{sym}: {w}" for w in sig.warnings]
        trad = np.ones(len(bars), dtype=bool)
        if tradable is not None:
            m = tradable.get(sym)
            trad = np.zeros(len(bars), dtype=bool) if m is None else \
                m.reindex(bars.index, method="ffill").fillna(False).astype(bool).to_numpy()
        S[sym] = dict(
            idx=bars.index, o=bars["open"].to_numpy(float), h=bars["high"].to_numpy(float),
            l=bars["low"].to_numpy(float), c=bars["close"].to_numpy(float),
            le=sig.long_entry.to_numpy(bool), se=sig.short_entry.to_numpy(bool) & cfg.allow_short,
            lx=sig.long_exit.to_numpy(bool), sx=sig.short_exit.to_numpy(bool),
            atr=sig.atr.to_numpy(float), trad=trad,
            events=sorted(pd.DatetimeIndex(cfg.events.get(sym, []))) if cfg.events else [],
            sess_end=session_end_mask(bars.index, bar_td) if spec.exit.session_close and spec.asset_class == "stocks"
            else np.zeros(len(bars), dtype=bool),
        )

    timeline: dict[pd.Timestamp, list[tuple[str, int]]] = defaultdict(list)
    for sym, s in S.items():
        for i, ts in enumerate(s["idx"]):
            if (start is None or ts >= start) and (end is None or ts < end):
                timeline[ts].append((sym, i))

    cash = cfg.initial_equity
    peak_equity = cash
    open_pos: dict[str, _Pos] = {}
    pending_entry: dict[str, int] = {}
    pending_exit: dict[str, tuple[str, float]] = {}
    day_trades: list[pd.Timestamp] = []
    trades: list[Trade] = []
    eq_index, eq_values = [], []
    is_stock = spec.asset_class == "stocks"
    at_close = spec.fill == "next_close"

    def b2u(sym: str, price: float, ts) -> float:
        """USD value of one unit of the instrument's price movement (quote ccy -> USD)."""
        if not is_stock:
            return usd_per_unit(split_pair(sym)[1], ts, cfg.fx_rates)
        return 1.0

    def equity_now(marks: dict[str, float]) -> float:
        u = 0.0
        for sym, ps in open_pos.items():
            px = marks.get(sym, ps.entry_fill)
            u += ps.p.direction * (px - ps.entry_fill) * ps.p.qty * ps.p.base_to_usd
        return cash + u

    def gross_long_notional(marks) -> float:
        return sum(ps.p.qty * marks.get(s, ps.entry_fill) * ps.p.base_to_usd for s, ps in open_pos.items())

    def recent_day_trades(ts) -> int:
        cutoff = ts - pd.tseries.offsets.BDay(cfg.pdt_window_bdays)
        return sum(1 for d in day_trades if d > cutoff)

    def close(sym: str, ts, mid: float, reason: str, fraction: float = 1.0, force: bool = False) -> bool:
        nonlocal cash
        ps = open_pos[sym]
        p = ps.p
        pdt_violation = False
        if is_stock and cfg.pdt:
            same_day = ts.tz_convert(NY).date() == ps.entry_date
            if same_day and equity_now(last_marks) < cfg.pdt_threshold:
                if recent_day_trades(ts) >= cfg.pdt_max_day_trades:
                    if not force:
                        return False  # defer to a later day
                    pdt_violation = True
                day_trades.append(ts)
        qty = p.qty * fraction
        fill, unit = costs.fill(sym, -p.direction, mid, ts)
        fees = sum(costs.order_fees(sym, -p.direction, qty, fill, ts).values())
        gross = p.direction * (fill - ps.entry_fill) * qty * p.base_to_usd
        share = fraction
        ss = ps.spread_slip * share + (unit["spread"] + unit["slippage"]) * qty * p.base_to_usd
        entry_fees = ps.fees * share
        hold = {k: v * share for k, v in ps.hold.items()}
        hold_total = sum(hold.values())
        net = gross - entry_fees - fees - hold_total
        cash += gross - fees  # entry fees and holding costs were already taken from cash
        trades.append(Trade(
            strategy_id=spec.id, symbol=sym, direction=p.direction, entry_time=p.entry_time,
            entry_price=ps.entry_fill, exit_time=ts, exit_price=fill, qty=qty,
            notional_usd=qty * ps.entry_fill * p.base_to_usd, risk_usd=ps.risk_usd * share,
            gross_pnl=gross, spread_slippage=ss, fees=entry_fees + fees,
            financing=hold.get("financing", 0.0), borrow=hold.get("borrow", 0.0),
            margin_interest=hold.get("margin_interest", 0.0), net_pnl=net,
            r_multiple=net / (ps.risk_usd * share) if ps.risk_usd else 0.0,
            bars_held=p.bars_held, exit_reason=reason, pdt_violation=pdt_violation,
        ))
        if fraction >= 0.999:
            del open_pos[sym]
        else:
            p.qty -= qty
            ps.risk_usd *= 1 - share
            ps.spread_slip *= 1 - share
            ps.fees *= 1 - share
            for k in ps.hold:
                ps.hold[k] *= 1 - share
            p.meta["event_reduced"] = True
        return True

    def fill_exit(sym: str, ts, px: float) -> None:
        if sym in pending_exit and sym in open_pos:
            reason, frac = pending_exit[sym]
            if close(sym, ts, px, reason, frac):
                pending_exit.pop(sym)
        elif sym in pending_exit:
            pending_exit.pop(sym)

    def fill_entry(sym: str, s: dict, i: int, ts, px: float) -> None:
        nonlocal cash
        if sym not in pending_entry or sym in open_pos:
            return
        direction = pending_entry.pop(sym)
        prev_atr = s["atr"][i - 1] if i > 0 else np.nan
        if np.isnan(prev_atr) or prev_atr <= 0:
            return
        eq = equity_now(last_marks)
        fill, unit = costs.fill(sym, direction, px, ts)
        stop_dist = spec.exit.stop_atr * prev_atr
        stop = fill - direction * stop_dist
        target = fill + direction * spec.exit.target_r * stop_dist
        bu = b2u(sym, fill, ts)
        rp = cfg.risk_pct
        if cfg.size_fn is not None:
            rp = cfg.size_fn({"spec": spec, "symbol": sym, "time": ts, "equity": eq})
        rp = min(rp, spec.cap_risk_pct)
        risk_usd = eq * rp / 100.0
        qty = risk_usd / (stop_dist * bu)
        room = eq * cfg.max_leverage - gross_long_notional(last_marks)
        qty = min(qty, max(0.0, room) / (fill * bu))
        qty = float(np.floor(qty))
        heat = sum(p.risk_usd for p in open_pos.values())
        if qty >= 1 and len(open_pos) < cfg.max_positions and \
                heat + qty * stop_dist * bu <= eq * cfg.max_heat_pct / 100.0 + 1e-9:
            fees = sum(costs.order_fees(sym, direction, qty, fill, ts).values())
            cash -= fees
            op = OpenPosition(
                symbol=sym, asset_class=spec.asset_class, strategy_id=spec.id,
                direction=direction, qty=qty, entry_time=ts, entry_price=fill, stop=stop,
                target=target, initial_risk=stop_dist, max_bars=spec.exit.max_bars,
                best_price=fill, base_to_usd=bu,
            )
            open_pos[sym] = _Pos(
                p=op, entry_fill=fill, risk_usd=qty * stop_dist * bu,
                spread_slip=(unit["spread"] + unit["slippage"]) * qty * bu, fees=fees,
                last_accrual=ts, entry_date=ts.tz_convert(NY).date(),
            )

    last_marks: dict[str, float] = {}
    for ts in sorted(timeline):
        for sym, i in timeline[ts]:
            s = S[sym]
            o, h, l, c, atr = s["o"][i], s["h"][i], s["l"][i], s["c"][i], s["atr"][i]
            bar_end = ts + bar_td

            # a, b. exits, then entries, decided at the previous close fill at this open
            # (with ``fill: next_close`` they fill at this bar's close instead, after the stop check)
            if not at_close:
                fill_exit(sym, ts, o)
                fill_entry(sym, s, i, ts, o)

            # c. stop / target inside the bar
            if sym in open_pos:
                ps = open_pos[sym]
                p = ps.p
                d = p.direction
                entered_now = p.entry_time == ts and not at_close
                hit = None
                if not entered_now and d * (o - p.stop) <= 0:
                    hit = ("stop_gap", o)
                elif not entered_now and d * (o - p.target) >= 0:
                    hit = ("target_gap", o)
                else:
                    stop_touched = (l <= p.stop) if d > 0 else (h >= p.stop)
                    tgt_touched = (h >= p.target) if d > 0 else (l <= p.target)
                    if stop_touched:
                        hit = ("stop", p.stop)
                    elif tgt_touched:
                        hit = ("target", p.target)
                if hit:
                    reason, px = hit
                    if p.stop == p.entry_price and reason.startswith("stop"):
                        reason = reason.replace("stop", "breakeven")
                    close(sym, ts, px, reason, force=reason.startswith(("stop", "breakeven")))

            if at_close:
                if sym in open_pos:
                    open_pos[sym].p.bars_held += 1      # held through this bar, entered at an earlier close
                fill_exit(sym, bar_end, c)
                fill_entry(sym, s, i, bar_end, c)

            # d0. intraday strategies are flat at the session close
            if sym in open_pos and s["sess_end"][i]:
                close(sym, bar_end, c, "session_close", force=True)
                pending_exit.pop(sym, None)

            # d. carrying costs and review at the close
            if sym in open_pos:
                ps = open_pos[sym]
                p = ps.p
                notional = p.qty * c * p.base_to_usd
                book = gross_long_notional({**last_marks, sym: c})
                eq = equity_now({**last_marks, sym: c})
                borrowed = notional * max(0.0, 1 - eq / book) if (is_stock and p.direction > 0 and book > eq > 0) else 0.0
                hc = costs.holding_cost(sym, p.direction, p.qty, c, ps.last_accrual, bar_end, p.base_to_usd, borrowed)
                for k, v in hc.items():
                    ps.hold[k] += v
                    cash -= v
                ps.last_accrual = bar_end
                if not at_close:                 # at_close counts the bar before its fills, above
                    p.bars_held += 1
                p.best_price = max(p.best_price, c) if p.direction > 0 else min(p.best_price, c)

                snap = MarketSnapshot(time=bar_end, price=c, atr=atr if not np.isnan(atr) else 0.0, bar_hours=bar_hours)
                if not is_stock and spec.holding.get("crosses_rollover", True):
                    cut, days = next_rollover(bar_end)
                    rate = costs.annual_rate(sym, p.direction, ts)
                    snap.next_rollover_time = cut
                    snap.next_rollover_cost_usd = -p.qty * c * p.base_to_usd * rate * days / 365.0
                if s["events"]:
                    nxt = [e for e in s["events"] if e > bar_end]
                    if nxt:
                        snap.next_event_time, snap.next_event_kind = nxt[0], "earnings"
                for act in reviewer.review(p, snap):
                    if act.kind == ActionKind.MOVE_STOP:
                        p.stop = act.price
                    elif act.kind == ActionKind.CLOSE:
                        pending_exit[sym] = (f"review: {act.reason}", 1.0)
                    elif act.kind == ActionKind.REDUCE:
                        pending_exit[sym] = (f"review: {act.reason}", act.fraction)
                if sym not in pending_exit:
                    if (p.direction > 0 and s["lx"][i]) or (p.direction < 0 and s["sx"][i]):
                        pending_exit[sym] = ("signal_exit", 1.0)

            # e. new entry signals at this close
            if sym not in open_pos and s["trad"][i] and not s["sess_end"][i]:
                le, se = s["le"][i], s["se"][i]
                if le != se:
                    pending_entry[sym] = 1 if le else -1
                else:
                    pending_entry.pop(sym, None)
            else:
                pending_entry.pop(sym, None)
            last_marks[sym] = c

        eq = equity_now(last_marks)
        peak_equity = max(peak_equity, eq)
        eq_index.append(ts)
        eq_values.append(eq)

    # close anything still open at the end of the window at the last close
    for sym in list(open_pos):
        s = S[sym]
        ts_last = eq_index[-1] if eq_index else s["idx"][-1]
        close(sym, ts_last, last_marks.get(sym, open_pos[sym].entry_fill), "end_of_data", force=True)
    if eq_index:
        eq_values[-1] = cash

    tdf = pd.DataFrame([asdict(t) for t in trades])
    equity = pd.Series(eq_values, index=pd.DatetimeIndex(eq_index), name="equity", dtype=float)
    return BacktestResult(spec.id, dict(params or {}), tdf, equity, list(dict.fromkeys(warnings)), cfg)


def session_end_mask(index: pd.DatetimeIndex, bar_td: pd.Timedelta) -> np.ndarray:
    """True on the last bar of each US regular session: the bar ends at or after 16:00 New
    York (DST-aware), or the next bar starts on a later New York date (early closes, gaps)."""
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    start = index.tz_convert(NY)
    end = (index + bar_td).tz_convert(NY)
    day = start.normalize()
    close_16 = day + pd.Timedelta(hours=16)
    at_close = np.asarray(end >= close_16)
    nxt = np.append(np.asarray(day[1:] != day[:-1]), True)
    return at_close | nxt


def _policy_from_spec(spec: StrategySpec) -> ReviewPolicy:
    pol = ReviewPolicy(breakeven_r=spec.exit.breakeven_r, trail_atr=spec.exit.trail_atr)
    if spec.asset_class == "stocks":
        pol.rollover_check = False
    else:
        pol.event_action = "ignore"  # forex events are handled by entry filters for now
    return pol
