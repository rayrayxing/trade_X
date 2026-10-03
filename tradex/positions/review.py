"""Open-position review: makes sure no position stays open indefinitely.

The same reviewer runs inside the backtester on every bar and, in thread 3, inside
the paper and live loops on a schedule, so backtests already reflect how positions
are actually managed. Every position carries a stop, a target and a time limit from
the moment it opens; the reviewer then asks, each time it runs:

1. Has the hard time limit passed? Close.
2. Is the trade stale (no progress after a share of its allowed time)? Close.
3. Can the target still plausibly be reached in the time left? If not, close.
4. (Forex) Is the next 17:00 New York rollover charge bigger than the expected
   remaining gain? Close before it (three times the bar on Friday).
5. (Stocks) Is an earnings date or other event inside the holding window? Close or cut.
6. Has the trade run far enough to protect it? Move the stop to breakeven or trail it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist

import pandas as pd

_N = NormalDist()


class ActionKind(str, Enum):
    HOLD = "hold"
    MOVE_STOP = "move_stop"
    CLOSE = "close"
    REDUCE = "reduce"


@dataclass
class Action:
    kind: ActionKind
    reason: str
    price: float | None = None       # new stop for MOVE_STOP
    fraction: float | None = None    # share to cut for REDUCE


@dataclass
class OpenPosition:
    symbol: str
    asset_class: str
    strategy_id: str
    direction: int                   # +1 long, -1 short
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    initial_risk: float              # |entry - initial stop| in price units
    max_bars: int
    bars_held: int = 0
    best_price: float | None = None  # most favourable close since entry
    base_to_usd: float = 1.0
    meta: dict = field(default_factory=dict)

    def r_multiple(self, price: float) -> float:
        return self.direction * (price - self.entry_price) / self.initial_risk if self.initial_risk else 0.0


@dataclass
class MarketSnapshot:
    time: pd.Timestamp                       # decision time = close of the bar just finished
    price: float
    atr: float
    bar_hours: float                         # length of one bar in hours
    next_rollover_time: pd.Timestamp | None = None
    next_rollover_cost_usd: float = 0.0      # cost of holding through the next cut (negative = credit)
    next_event_time: pd.Timestamp | None = None
    next_event_kind: str | None = None


@dataclass
class ReviewPolicy:
    hard_max_hours: dict[str, float] = field(default_factory=lambda: {"stocks": 24 * 30, "forex": 24 * 10})
    stale_after_frac: float = 0.5            # share of max_bars after which a flat trade is stale
    stale_band_r: float = 0.25               # "flat" means within this many R of entry
    min_target_prob: float = 0.10            # close when the chance of reaching target falls below this
    prob_check_last_frac: float = 0.3        # only judge reachability in the last 30% of the allowed time
    breakeven_r: float | None = 1.0
    trail_atr: float | None = None
    rollover_check: bool = True
    rollover_window_bars: int = 1            # act when the cut is within this many bars
    event_action: str = "close"              # close | reduce | ignore
    event_window_hours: float = 24.0
    event_reduce_fraction: float = 0.5


def hit_probability(distance: float, sigma_per_bar: float, bars: float) -> float:
    """Chance a driftless random walk touches a level ``distance`` away within ``bars`` bars."""
    if distance <= 0:
        return 1.0
    if sigma_per_bar <= 0 or bars <= 0:
        return 0.0
    return min(1.0, 2.0 * (1.0 - _N.cdf(distance / (sigma_per_bar * math.sqrt(bars)))))


class PositionReviewer:
    def __init__(self, policy: ReviewPolicy | None = None):
        self.policy = policy or ReviewPolicy()

    def review(self, pos: OpenPosition, mkt: MarketSnapshot) -> list[Action]:
        p = self.policy
        actions: list[Action] = []
        held_hours = (mkt.time - pos.entry_time).total_seconds() / 3600
        r_now = pos.r_multiple(mkt.price)
        remaining = pos.max_bars - pos.bars_held

        # 1. hard time limits
        if remaining <= 0:
            return [Action(ActionKind.CLOSE, f"time limit: held {pos.bars_held} bars (max {pos.max_bars})")]
        hard = p.hard_max_hours.get(pos.asset_class)
        if hard and held_hours >= hard:
            return [Action(ActionKind.CLOSE, f"hard ceiling: held {held_hours:.0f}h (max {hard:.0f}h)")]

        # 2. stale trade
        if pos.bars_held >= p.stale_after_frac * pos.max_bars and abs(r_now) < p.stale_band_r:
            return [Action(ActionKind.CLOSE, f"stale: {r_now:+.2f}R after {pos.bars_held}/{pos.max_bars} bars")]

        # 3. target reachability
        sigma = mkt.atr / 1.25 if mkt.atr and mkt.atr > 0 else 0.0  # ATR ~ 1.25 sigma for a normal bar range
        dist_target = pos.direction * (pos.target - mkt.price)
        dist_stop = pos.direction * (mkt.price - pos.stop)
        p_target = hit_probability(dist_target, sigma, remaining)
        if remaining <= p.prob_check_last_frac * pos.max_bars and p_target < p.min_target_prob:
            return [Action(ActionKind.CLOSE, f"target unlikely: {p_target:.0%} chance in {remaining} bars left")]

        # 4. forex rollover economics
        if p.rollover_check and mkt.next_rollover_time is not None and mkt.next_rollover_cost_usd > 0:
            bars_to_cut = (mkt.next_rollover_time - mkt.time).total_seconds() / 3600 / mkt.bar_hours
            if bars_to_cut <= p.rollover_window_bars:
                p_stop = hit_probability(dist_stop, sigma, remaining)
                tot = p_target + p_stop
                pt, ps = (p_target / tot, p_stop / tot) if tot > 0 else (0.0, 0.0)
                exp_gain = (pt * dist_target - ps * dist_stop) * pos.qty * pos.base_to_usd
                if exp_gain < mkt.next_rollover_cost_usd:
                    return [Action(ActionKind.CLOSE,
                                   f"rollover: expected gain ${exp_gain:.2f} < next charge ${mkt.next_rollover_cost_usd:.2f}")]

        # 5. scheduled events inside the window
        if p.event_action != "ignore" and mkt.next_event_time is not None:
            hrs = (mkt.next_event_time - mkt.time).total_seconds() / 3600
            if 0 <= hrs <= p.event_window_hours:
                if p.event_action == "close":
                    return [Action(ActionKind.CLOSE, f"{mkt.next_event_kind or 'event'} in {hrs:.0f}h")]
                if not pos.meta.get("event_reduced"):
                    actions.append(Action(ActionKind.REDUCE, f"{mkt.next_event_kind or 'event'} in {hrs:.0f}h",
                                          fraction=p.event_reduce_fraction))

        # 6. protect open profit
        new_stop = pos.stop
        if p.breakeven_r is not None and r_now >= p.breakeven_r:
            new_stop = _tighter(pos.direction, new_stop, pos.entry_price)
        if p.trail_atr and mkt.atr and r_now > 0:
            best = pos.best_price if pos.best_price is not None else mkt.price
            new_stop = _tighter(pos.direction, new_stop, best - pos.direction * p.trail_atr * mkt.atr)
        if new_stop != pos.stop:
            actions.append(Action(ActionKind.MOVE_STOP, f"protect profit at {r_now:+.2f}R", price=new_stop))
        return actions or [Action(ActionKind.HOLD, "within plan")]


def _tighter(direction: int, current: float, candidate: float) -> float:
    return max(current, candidate) if direction > 0 else min(current, candidate)
