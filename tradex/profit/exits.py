"""Exit engine for trade plans (profit spec P1-P3): pure functions, no look-ahead.

A plan leaves the finaliser with a stop, up to two targets and a time stop. This module
adds what happens in between, as a policy (``ExitPolicy``) and two pure functions:

- ``evaluate_exit(policy, state, bars)``: called at a bar close with the CLOSED bars up to
  and including that close. Returns ``Action`` rows (the same type the position reviewer
  uses): move the stop, reduce, close, or hold. Nothing here reads past the last bar given;
  a stop it returns is in force from the NEXT bar, never the one it was computed from.
- ``simulate_exit(policy, state, bars)``: walks bars after the entry, applying intrabar
  stop and target fills (stop first when both are inside one bar, gaps filled at the open)
  and calling ``evaluate_exit`` at every close. This is the one place the counterfactual
  report, the policy comparison and the tests share, so a policy is judged by the same
  rules that would run in the core.

Rules, all optional and set in ``config/exits.yaml``:

1. Breakeven: once a close is at ``breakeven_r`` in profit, the stop goes to entry plus a
   buffer that covers the round-trip cost (the plan's ``cost_r``), so breakeven is
   breakeven net of costs. It also goes there when target 1 has been taken.
2. Trailing: after the best close reached ``trail_start_r``, the stop trails the best price
   by ``trail_atr_mult`` ATR (chandelier), by the lowest low (highest high for shorts) of
   the last ``structure_lookback`` closed bars, or by whichever of the two is tighter.
3. Partial targets: ``partials`` is the share of the ORIGINAL position closed at target 1,
   2, ...; the last target takes whatever is left. A plan with one target exits whole there.
4. Time stop: the plan's ``max_bars`` closes the position at that bar's close; optionally a
   no-progress stop closes early when the best close is still under ``progress_min_r`` after
   ``progress_frac`` of the time.
5. Volatility exits: a bar whose range is ``shock_atr_mult`` times the ATR of the bars
   before it, closing hard against the position, closes it; and when ATR has expanded to
   ``expand_mult`` times its value at entry while in profit, the trail tightens.

A stop only ever moves toward price, never back, and never closer to the last close than
``min_stop_gap_atr`` ATR (a stop that tight is just an exit at the next tick).

Costs: a plan's ``cost_r`` is a round trip. Half is charged at entry and each exit leg pays
its share of the other half, so a single exit costs exactly ``cost_r`` (the same as the
counterfactual tracker). Fixed per-order fees on extra exit legs are ``extra_leg_cost_r``
(default 0; set it from the venue's per-order fee for small accounts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from tradex.core.records import TradePlan
from tradex.positions.review import Action, ActionKind

TRAIL_MODES = ("none", "atr", "structure", "tighter")
WARMUP_BARS = 100                      # history handed to the ATR in simulations; Wilder smoothing forgets older bars


# --- policy ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitPolicy:
    breakeven_r: float | None = 1.0
    breakeven_buffer_r: float | None = None        # None: the plan's cost_r
    breakeven_after_partial: bool = True
    trail: str = "atr"                             # none | atr | structure | tighter
    trail_start_r: float = 1.5
    trail_atr_mult: float = 3.0
    structure_lookback: int = 5
    structure_buffer_atr: float = 0.1
    min_stop_gap_atr: float = 0.25
    partials: tuple[float, ...] = (0.5,)           # share of the original position closed at target 1, 2, ...
    time_stop: bool = True
    progress_frac: float | None = None
    progress_min_r: float = 0.3
    shock_atr_mult: float | None = 3.0
    shock_body_frac: float = 0.5
    expand_mult: float | None = None
    expand_trail_mult: float = 1.5
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.trail not in TRAIL_MODES:
            raise ValueError(f"trail must be one of {TRAIL_MODES}, not {self.trail!r}")
        if any(not 0 < f < 1 for f in self.partials) or sum(self.partials) >= 1:
            raise ValueError("partials must each be in (0, 1) and sum to less than 1")
        if self.atr_period < 2 or self.structure_lookback < 1:
            raise ValueError("atr_period must be at least 2 and structure_lookback at least 1")

    @classmethod
    def baseline(cls) -> "ExitPolicy":
        """What a plan does with no exit engine: the stop, target 1 whole, and the time stop."""
        return cls(breakeven_r=None, breakeven_after_partial=False, trail="none", partials=(),
                   time_stop=True, shock_atr_mult=None, expand_mult=None)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ExitPolicy":
        d = dict(d or {})
        if "partials" in d:
            d["partials"] = tuple(float(x) for x in d["partials"])
        names = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown exit policy keys: {sorted(unknown)}")
        return cls(**d)

    def fractions(self, n_targets: int) -> tuple[float, ...]:
        """Share of the original position closed at each target; the last one takes the rest."""
        k = max(0, n_targets - 1)
        head = self.partials[:k]
        return (*head, 1.0 - sum(head))


def load_exit_policy(path: str | Path = "config/exits.yaml") -> ExitPolicy:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exit policy {p} not found")
    return ExitPolicy.from_dict((yaml.safe_load(p.read_text()) or {}).get("exits"))


# --- state -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitState:
    """What the engine needs to know about one open (or simulated) position."""
    direction: int
    entry_price: float
    initial_stop: float                  # the R unit
    stop: float                          # in force now
    targets: tuple[float, ...]
    entry_time: pd.Timestamp             # open time of the first bar held
    max_bars: int
    cost_r: float = 0.0
    entry_atr: float | None = None
    targets_done: int = 0
    remaining: float = 1.0               # share of the original position still open

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def r(self, price: float) -> float:
        return self.direction * (price - self.entry_price) / self.risk if self.risk else 0.0

    @classmethod
    def from_plan(cls, plan: TradePlan, entry_time: pd.Timestamp | None = None, entry_price: float | None = None,
                  entry_atr: float | None = None) -> "ExitState":
        return cls(direction=plan.direction, entry_price=plan.entry_price if entry_price is None else entry_price,
                   initial_stop=plan.stop, stop=plan.stop, targets=tuple(plan.targets),
                   entry_time=pd.Timestamp(entry_time if entry_time is not None else plan.time),
                   max_bars=int(plan.max_bars), cost_r=plan.cost_r, entry_atr=entry_atr)


# --- indicators (causal: the value at a bar uses that bar and earlier ones only) --------------

def true_range(bars: pd.DataFrame) -> pd.Series:
    pc = bars["close"].shift(1)
    return pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()],
                     axis=1).max(axis=1, skipna=True)


def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR. Row k depends on rows 0..k only."""
    return true_range(bars).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _tighter(direction: int, current: float, candidate: float) -> float:
    return max(current, candidate) if direction > 0 else min(current, candidate)


# --- the evaluation at a bar close ---------------------------------------------------------------

def evaluate_exit(policy: ExitPolicy, state: ExitState, bars: pd.DataFrame) -> list[Action]:
    """Actions at the close of the last bar in ``bars`` (closed bars only; may start before the entry
    so ATR has history). Pure: the same inputs give the same answer, and rows after the last bar
    are not an input."""
    d = state.direction
    held = bars[bars.index >= state.entry_time]
    if held.empty or state.remaining <= 0 or state.risk <= 0:
        return [Action(ActionKind.HOLD, "nothing held yet")]
    n = len(held)
    last = held.iloc[-1]
    close = float(last["close"])
    atr_all = atr_series(bars, policy.atr_period)
    atr_now = float(atr_all.iloc[-1]) if np.isfinite(atr_all.iloc[-1]) else None
    best_close = float(held["close"].max() if d > 0 else held["close"].min())
    best_r = state.r(best_close)

    # 1. time stop, then the no-progress stop
    if policy.time_stop and n >= state.max_bars:
        return [Action(ActionKind.CLOSE, f"time stop: held {n} bars (max {state.max_bars})")]
    if (policy.progress_frac is not None and n >= policy.progress_frac * state.max_bars
            and best_r < policy.progress_min_r):
        return [Action(ActionKind.CLOSE, f"no progress: best close {best_r:+.2f}R after {n}/{state.max_bars} bars")]

    # 2. volatility shock: a wide bar closing hard against the position
    if policy.shock_atr_mult is not None and len(atr_all) >= 2:
        prev_atr = atr_all.iloc[-2]
        rng = float(last["high"] - last["low"])
        body = d * (float(last["close"]) - float(last["open"]))
        if (np.isfinite(prev_atr) and prev_atr > 0 and rng >= policy.shock_atr_mult * prev_atr
                and body < 0 and -body >= policy.shock_body_frac * rng):
            return [Action(ActionKind.CLOSE, f"volatility shock: bar range {rng / prev_atr:.1f} ATR against the position")]

    acts: list[Action] = []

    # 3. a target the last bar reached with no resting order to fill it (limit-order fallback)
    if state.targets_done < len(state.targets):
        t = state.targets[state.targets_done]
        reached = float(last["high"]) >= t if d > 0 else float(last["low"]) <= t
        if reached:
            frac = policy.fractions(len(state.targets))[state.targets_done]
            kind = ActionKind.CLOSE if state.targets_done == len(state.targets) - 1 else ActionKind.REDUCE
            acts.append(Action(kind, f"target {state.targets_done + 1} reached at {t:.5g}", price=t,
                               fraction=None if kind == ActionKind.CLOSE else frac))
            if kind == ActionKind.CLOSE:
                return acts

    # 4. protect profit: breakeven, then the trail; the tightest candidate wins, and only toward price
    cands: list[tuple[float, str]] = []
    buf_r = policy.breakeven_buffer_r if policy.breakeven_buffer_r is not None else state.cost_r
    be = state.entry_price + d * buf_r * state.risk
    if policy.breakeven_r is not None and best_r >= policy.breakeven_r:
        cands.append((be, f"breakeven after {best_r:+.2f}R"))
    if policy.breakeven_after_partial and state.targets_done >= 1:
        cands.append((be, "breakeven after target 1"))
    if policy.trail != "none" and best_r >= policy.trail_start_r and atr_now:
        best_px = float(held["high"].max() if d > 0 else held["low"].min())
        mult = policy.trail_atr_mult
        if policy.expand_mult is not None and state.entry_atr and atr_now / state.entry_atr >= policy.expand_mult \
                and state.r(close) > 0:
            mult = min(mult, policy.expand_trail_mult)
        trails = []
        if policy.trail in ("atr", "tighter"):
            trails.append((best_px - d * mult * atr_now, f"trail {mult:g} ATR"))
        if policy.trail in ("structure", "tighter"):
            look = held.tail(policy.structure_lookback)
            sp = float(look["low"].min() if d > 0 else look["high"].max())
            trails.append((sp - d * policy.structure_buffer_atr * atr_now, f"trail under last {len(look)} bars"))
        if trails:
            cands.append(max(trails, key=lambda c: d * c[0]))
    new_stop, why = state.stop, ""
    gap = policy.min_stop_gap_atr * (atr_now or 0.0)
    for price, reason in cands:
        if d * price > d * new_stop and d * (close - price) >= gap and d * (close - price) > 0:
            new_stop, why = price, reason
    if new_stop != state.stop:
        acts.append(Action(ActionKind.MOVE_STOP, why, price=new_stop))
    return acts or [Action(ActionKind.HOLD, "within plan")]


# --- the walk ------------------------------------------------------------------------------------

@dataclass
class Leg:
    time: pd.Timestamp
    price: float
    fraction: float
    reason: str


@dataclass
class ExitResult:
    r_multiple: float                    # net of costs
    gross_r: float
    reason: str                          # the reason of the last leg
    exit_time: pd.Timestamp | None
    bars_held: int
    legs: list[Leg] = field(default_factory=list)
    stop_moves: list[tuple[pd.Timestamp, float, str]] = field(default_factory=list)
    mfe_r: float = 0.0                   # best price reached, in R
    mae_r: float = 0.0                   # worst price reached, in R (<= 0)
    open_at_end: bool = False            # the data ran out with the position open (marked at the last close)


def simulate_exit(policy: ExitPolicy, state: ExitState, bars: pd.DataFrame, *, gap_aware: bool = True,
                  extra_leg_cost_r: float = 0.0) -> ExitResult:
    """Follow one position through ``bars`` (closed bars, indexed by open time; rows before
    ``state.entry_time`` are ATR history). Stops and targets are checked inside each bar with
    the stop first, using the levels fixed at the previous close; ``evaluate_exit`` then runs at
    the bar's close. With ``gap_aware=False`` a stop or target is filled at its level even when
    the bar opened through it (the counterfactual tracker's convention)."""
    d = state.direction
    fr = policy.fractions(len(state.targets))
    st = state
    legs: list[Leg] = []
    moves: list[tuple[pd.Timestamp, float, str]] = []
    mfe = mae = 0.0
    idx = np.flatnonzero(bars.index >= state.entry_time)
    if len(idx) == 0:
        return ExitResult(0.0, 0.0, "no_bars", None, 0, open_at_end=True)
    first = int(idx[0])
    n_held = 0
    for k in range(first, len(bars)):
        row = bars.iloc[k]
        t = bars.index[k]
        o, h, l, c = (float(row[x]) for x in ("open", "high", "low", "close"))
        n_held += 1
        mfe = max(mfe, st.r(h if d > 0 else l))
        mae = min(mae, st.r(l if d > 0 else h))
        # intrabar: stop first
        if (l <= st.stop) if d > 0 else (h >= st.stop):
            px = (min(o, st.stop) if d > 0 else max(o, st.stop)) if gap_aware else st.stop
            legs.append(Leg(t, px, st.remaining, "stop_gap" if gap_aware and px != st.stop else "stop"))
            st = replace(st, remaining=0.0)
            break
        done_all = False
        while st.targets_done < len(st.targets):
            tg = st.targets[st.targets_done]
            if not ((h >= tg) if d > 0 else (l <= tg)):
                break
            px = (max(o, tg) if d > 0 else min(o, tg)) if gap_aware else tg
            last_target = st.targets_done == len(st.targets) - 1
            share = st.remaining if last_target else min(st.remaining, fr[st.targets_done])
            legs.append(Leg(t, px, share, "target" if last_target else f"target_{st.targets_done + 1}"))
            st = replace(st, remaining=st.remaining - share, targets_done=st.targets_done + 1)
            if st.remaining <= 1e-12 or last_target:
                done_all = True
                break
        if done_all:
            st = replace(st, remaining=0.0)
            break
        # at the close
        window = bars.iloc[max(0, first - WARMUP_BARS): k + 1]
        entry_atr = st.entry_atr
        if entry_atr is None:
            pre = atr_series(bars.iloc[max(0, first - WARMUP_BARS): first], policy.atr_period)
            entry_atr = float(pre.iloc[-1]) if len(pre) and np.isfinite(pre.iloc[-1]) else None
            st = replace(st, entry_atr=entry_atr)
        closed = False
        for a in evaluate_exit(policy, st, window):
            if a.kind == ActionKind.MOVE_STOP and a.price is not None:
                st = replace(st, stop=a.price)
                moves.append((t, a.price, a.reason))
            elif a.kind == ActionKind.CLOSE:
                legs.append(Leg(t, c, st.remaining, _short_reason(a.reason)))
                st = replace(st, remaining=0.0)
                closed = True
        if closed:
            break
    open_end = st.remaining > 1e-12
    if open_end:
        legs.append(Leg(bars.index[-1], float(bars["close"].iloc[-1]), st.remaining, "end_of_data"))
    gross = sum(g.fraction * state.r(g.price) for g in legs)
    cost = 0.5 * state.cost_r + sum(g.fraction * 0.5 * state.cost_r for g in legs) + extra_leg_cost_r * max(0, len(legs) - 1)
    return ExitResult(r_multiple=round(gross - cost, 6), gross_r=round(gross, 6), reason=legs[-1].reason,
                      exit_time=legs[-1].time, bars_held=n_held, legs=legs, stop_moves=moves,
                      mfe_r=round(mfe, 4), mae_r=round(mae, 4), open_at_end=open_end)


def _short_reason(text: str) -> str:
    for key, name in (("time stop", "time_stop"), ("no progress", "no_progress"), ("volatility shock", "vol_shock"),
                      ("target", "target")):
        if text.startswith(key):
            return name
    return "exit"


# --- the plan finaliser's view ------------------------------------------------------------------

@dataclass(frozen=True)
class ExitLadder:
    """The exits a plan will be managed by, as data: what the finaliser can print into the plan's
    invalidation text and the Telegram card."""
    stop: float
    targets: tuple[tuple[float, float], ...]     # (price, share of the original position)
    breakeven_r: float | None
    trail: str
    max_bars: int

    def text(self) -> str:
        parts = [f"stop {self.stop:.5g}"]
        parts += [f"T{i + 1} {p:.5g} ({s:.0%})" for i, (p, s) in enumerate(self.targets)]
        if self.breakeven_r is not None:
            parts.append(f"breakeven at {self.breakeven_r:g}R")
        if self.trail != "none":
            parts.append(f"trail: {self.trail}")
        parts.append(f"time stop {self.max_bars} bars")
        return "; ".join(parts)


def exit_ladder(plan: TradePlan, policy: ExitPolicy) -> ExitLadder:
    fr = policy.fractions(len(plan.targets))
    return ExitLadder(plan.stop, tuple(zip(plan.targets, fr)), policy.breakeven_r, policy.trail, int(plan.max_bars))


# --- ledger-backed evaluation -----------------------------------------------------------------------

BarsFor = Callable[[str, str], "pd.DataFrame | None"]       # (symbol, timeframe) -> closed bars indexed by open time


def plan_outcome(plan: TradePlan, policy: ExitPolicy, bars: pd.DataFrame, gap_aware: bool = True,
                 extra_leg_cost_r: float = 0.0) -> ExitResult | None:
    """A plan followed from the first bar opening at or after its time (the next-open entry the core uses),
    entered at the plan's expected price."""
    t0 = pd.Timestamp(plan.time)
    if not (bars.index >= t0).any():
        return None
    st = ExitState.from_plan(plan, entry_time=t0)
    return simulate_exit(policy, st, bars, gap_aware=gap_aware, extra_leg_cost_r=extra_leg_cost_r)


def summarise(rs: Iterable[float]) -> dict[str, float | int | None]:
    r = np.asarray(list(rs), dtype=float)
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_r": None, "total_r": 0.0, "win_rate": None, "profit_factor": None, "se_r": None}
    gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": n, "mean_r": round(float(r.mean()), 4), "total_r": round(float(r.sum()), 3),
            "win_rate": round(float((r > 0).mean()), 4), "profit_factor": round(float(gw / gl), 3) if gl > 0 else None,
            "se_r": round(float(r.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None}


def compare_policies(plans: list[TradePlan], bars_for: BarsFor, policies: dict[str, ExitPolicy],
                     baseline: str | None = None, extra_leg_cost_r: float = 0.0) -> dict[str, Any]:
    """Run every policy over the same plans and report each one's R, and its paired difference from
    the baseline (same plans, same bars, so the difference is the exits alone). Plans whose bars are
    missing are counted, not guessed."""
    names = list(policies)
    base = baseline or names[0]
    res: dict[str, dict[str, float]] = {n: {} for n in names}
    skipped = 0
    for p in plans:
        bars = bars_for(p.symbol, p.tf) if p.tf else bars_for(p.symbol, "")
        if bars is None or bars.empty:
            skipped += 1
            continue
        outs = {n: plan_outcome(p, pol, bars, extra_leg_cost_r=extra_leg_cost_r) for n, pol in policies.items()}
        if any(o is None or o.open_at_end for o in outs.values()):
            skipped += 1                                  # a trade still open when the data ends has no final R
            continue
        for n, o in outs.items():
            res[n][p.decision_id] = o.r_multiple
    common = sorted(set.intersection(*(set(v) for v in res.values()))) if res else []
    table = {}
    for n in names:
        s = summarise(res[n][i] for i in common)
        if n != base and common:
            diff = np.array([res[n][i] - res[base][i] for i in common])
            se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
            s["delta_mean_r"] = round(float(diff.mean()), 4)
            s["delta_se"] = None if not np.isfinite(se) else round(se, 4)
            s["delta_t"] = None if not np.isfinite(se) or se == 0 else round(float(diff.mean()) / se, 2)
        table[n] = s
    return {"baseline": base, "plans": len(plans), "evaluated": len(common), "skipped_no_bars_or_open": skipped,
            "policies": table, "per_plan": {n: res[n] for n in names}}


def evaluate_on_ledger(ledger, bars_for: BarsFor, policies: dict[str, ExitPolicy], *, which: str = "accepted",
                       book: str = "ensemble", baseline: str | None = None, extra_leg_cost_r: float = 0.0
                       ) -> dict[str, Any]:
    """Compare exit policies on the plans in a ledger: ``accepted`` (a verdict accepted it), ``blocked`` (a
    veto or a rejected verdict) or ``all``. Read-only."""
    plans = [p for p in ledger.records("plan", book=book)]
    veto_ids = {r["decision_id"] for r in ledger.rows(kind="veto")}
    ok_ids = {r["decision_id"] for r in ledger.rows(kind="verdict") if r.get("outcome") == "accepted"} - veto_ids
    pick = {"accepted": lambda p: p.decision_id in ok_ids, "blocked": lambda p: p.decision_id in veto_ids,
            "all": lambda p: True}[which]
    return compare_policies([p for p in plans if pick(p)], bars_for, policies, baseline, extra_leg_cost_r)
