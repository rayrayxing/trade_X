"""Point-in-time liquidity screen: the research universe is chosen by the data, not by hand.

At each rebalance date the universe is the top ``n`` symbols of a candidate pool by
median daily dollar volume (close x volume) over the trailing ``lookback`` bars, using
only bars strictly before that date. A symbol needs a full lookback and a bar within the
last ``stale_days`` to qualify, so names not yet listed (or no longer trading) drop out.

The rebalance dates are the walk-forward test-fold starts, plus yearly dates through
the first training window so the training folds see a point-in-time universe too.
Membership becomes the engine's ``tradable`` mask (entries only; open positions still
exit) and restricts cross-sectional ranks to the members of the day.

The pool itself is today's names (tradex.research.universe): delisted stocks are not
available from OpenD, so a screen over it still carries survivorship bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradex.backtest.validation import WalkForwardConfig, _splits


@dataclass(frozen=True)
class Screen:
    pool: str                 # "stocks" | "etfs" | "all"
    n: int
    lookback: int = 60
    stale_days: int = 10

    @property
    def label(self) -> str:
        return f"top{self.n}-{self.pool}-adv{self.lookback}"


def dollar_volume(bars: pd.DataFrame) -> pd.Series:
    return bars["close"] * bars["volume"]


def rank_at(data: dict[str, pd.DataFrame], at: pd.Timestamp, n: int, lookback: int = 60,
            stale_days: int = 10) -> list[str]:
    """Top ``n`` by trailing median dollar volume from bars strictly before ``at``."""
    scores = {}
    for s, b in data.items():
        past = b[b.index < at]
        if len(past) < lookback or at - past.index[-1] > pd.Timedelta(days=stale_days):
            continue
        scores[s] = float(dollar_volume(past.iloc[-lookback:]).median())
    return sorted(scores, key=lambda s: (-scores[s], s))[:n]


def rebalance_dates(timeline: pd.DatetimeIndex, wf: WalkForwardConfig, embargo: pd.Timedelta,
                    lookback: int = 60, every: str = "365D") -> list[pd.Timestamp]:
    """Fold test starts, plus yearly dates from the first full lookback to the first test start."""
    tests = [te_s for _, _, _, te_s, _ in _splits(timeline, wf, embargo)]
    if not len(timeline):
        return []
    first = timeline[min(lookback, len(timeline) - 1)]
    stop = tests[0] if tests else timeline[-1]
    yearly = list(pd.date_range(first, stop, freq=every, inclusive="left"))
    return sorted(set(yearly) | set(tests))


def membership(data: dict[str, pd.DataFrame], dates: list[pd.Timestamp], screen: Screen) -> dict[pd.Timestamp, list[str]]:
    return {d: rank_at(data, d, screen.n, screen.lookback, screen.stale_days) for d in dates}


def member_frame(members: dict[pd.Timestamp, list[str]], symbols, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bool frame (index x symbols): True while the symbol is in the latest universe; False before the first date."""
    dates = sorted(members)
    m = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=list(symbols))
    for d in dates:
        m.loc[d, [s for s in members[d] if s in m.columns]] = True
    if not len(dates):
        return pd.DataFrame(False, index=index, columns=list(symbols))
    pos = m.index.searchsorted(index, side="right") - 1
    out = m.to_numpy()[np.clip(pos, 0, None)]
    out[pos < 0] = False
    return pd.DataFrame(out, index=index, columns=list(symbols))


def tradable_masks(members: dict[pd.Timestamp, list[str]], symbols) -> dict[str, pd.Series]:
    """Engine ``tradable`` input: per symbol a step series at the rebalance dates."""
    dates = pd.DatetimeIndex(sorted(members))
    return {s: pd.Series([s in members[d] for d in dates], index=dates) for s in symbols}


def ever_members(members: dict[pd.Timestamp, list[str]]) -> list[str]:
    return sorted({s for v in members.values() for s in v})
