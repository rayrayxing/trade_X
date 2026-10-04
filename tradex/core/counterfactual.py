"""Counterfactual ledger (recommendation 2): what every blocked plan would have done.

Each plan stopped by a gate (the family rule, the calendar, the short checks, the risk
gate) is followed bar by bar with the exits it would have had: stop first inside a bar,
then target 1, then the time stop at the bar close. Its R multiple is net of the plan's
round-trip cost. The rows let the dashboard say what each filter costs or saves, e.g.
"the two-family rule blocked 40 trades this month that would have averaged -0.2R".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradex.core.records import Counterfactual, TradePlan
from tradex.timeframes import duration


@dataclass
class _Ghost:
    plan: TradePlan
    blocked_by: str
    bars: int = 0
    last_close: float | None = None


class CounterfactualTracker:
    def __init__(self, bar: pd.Timedelta | None = None) -> None:
        self.open: list[_Ghost] = []
        self.bar = bar                    # the bars on_bar is fed; a plan's time stop counts its own timeframe

    def _limit(self, p: TradePlan) -> float:
        return p.max_bars * (duration(p.tf) / self.bar) if p.tf and self.bar is not None else p.max_bars

    def track(self, plan: TradePlan, blocked_by: str) -> None:
        self.open.append(_Ghost(plan, blocked_by))

    def on_bar(self, symbol: str, close_t: pd.Timestamp, h: float, l: float, c: float) -> list[Counterfactual]:
        done, keep = [], []
        for g in self.open:
            p = g.plan
            if p.symbol != symbol or close_t.isoformat() <= p.time:
                keep.append(g)
                continue
            g.bars += 1
            g.last_close = c
            d, risk = p.direction, abs(p.entry_price - p.stop)
            t1 = p.targets[0]
            hit = None
            if (l <= p.stop) if d > 0 else (h >= p.stop):
                hit = ("stop", -1.0)
            elif (h >= t1) if d > 0 else (l <= t1):
                hit = ("target", d * (t1 - p.entry_price) / risk)
            elif g.bars >= self._limit(p):
                hit = ("time_stop", d * (c - p.entry_price) / risk)
            if hit:
                done.append(Counterfactual(p.decision_id, p.time, g.blocked_by, close_t.isoformat(), hit[0],
                                           round(hit[1] - p.cost_r, 4)))
            else:
                keep.append(g)
        self.open = keep
        return done

    def finish(self, t: pd.Timestamp) -> list[Counterfactual]:
        out = []
        for g in self.open:
            p = g.plan
            if g.last_close is None:
                continue
            r = p.direction * (g.last_close - p.entry_price) / abs(p.entry_price - p.stop)
            out.append(Counterfactual(p.decision_id, p.time, g.blocked_by, t.isoformat(), "end_of_data",
                                      round(r - p.cost_r, 4)))
        self.open = []
        return out


def filter_report(rows: list[dict]) -> pd.DataFrame:
    """Per blocking gate: how many plans it stopped and what they would have earned in R."""
    if not rows:
        return pd.DataFrame(columns=["blocked_by", "plans", "mean_r", "total_r", "win_rate"])
    df = pd.DataFrame(rows)
    g = df.groupby("blocked_by")["r_multiple"]
    return pd.DataFrame({"plans": g.size(), "mean_r": g.mean().round(3), "total_r": g.sum().round(2),
                         "win_rate": g.apply(lambda s: (s > 0).mean()).round(3)}).reset_index()
