"""Timeframe codes and look-ahead-safe multi-timeframe alignment.

Convention used everywhere in tradex: a bar is stamped with its OPEN time in UTC
and covers [ts, ts + duration). A signal computed on bar ``ts`` is only known at
``ts + duration`` and is acted on at the next bar's open.
"""
from __future__ import annotations

import pandas as pd

# code -> (pandas resample rule, bar duration)
TIMEFRAMES: dict[str, tuple[str, pd.Timedelta]] = {
    "M1": ("1min", pd.Timedelta(minutes=1)),
    "M5": ("5min", pd.Timedelta(minutes=5)),
    "M15": ("15min", pd.Timedelta(minutes=15)),
    "M30": ("30min", pd.Timedelta(minutes=30)),
    "H1": ("1h", pd.Timedelta(hours=1)),
    "H4": ("4h", pd.Timedelta(hours=4)),
    "D1": ("1D", pd.Timedelta(days=1)),
    "W1": ("W-MON", pd.Timedelta(days=7)),
}

OHLCV = ["open", "high", "low", "close", "volume"]


def duration(tf: str) -> pd.Timedelta:
    try:
        return TIMEFRAMES[tf][1]
    except KeyError as exc:
        raise ValueError(f"unknown timeframe {tf!r}; known: {sorted(TIMEFRAMES)}") from exc


def resample(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate bars to a higher timeframe, labelled by bar open time."""
    rule = TIMEFRAMES[tf][0]
    kw = {"label": "left", "closed": "left"}
    out = bars.resample(rule, **kw).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "close"])


def align_higher(
    base_index: pd.DatetimeIndex,
    base_tf: str,
    higher: pd.Series,
    higher_tf: str,
) -> pd.Series:
    """Map a higher-timeframe series onto base bars without look-ahead.

    A higher bar opening at T is complete at T + D. It may inform a decision made
    at the close of a base bar opening at t only if T + D <= t + b.
    """
    d, b = duration(higher_tf), duration(base_tf)
    avail = pd.DataFrame({"avail": higher.index + d, "v": higher.to_numpy()}).dropna(subset=["avail"])
    avail = avail.sort_values("avail")
    decide = pd.DataFrame({"decide": base_index + b, "pos": range(len(base_index))})
    merged = pd.merge_asof(decide, avail, left_on="decide", right_on="avail", direction="backward")
    return pd.Series(merged["v"].to_numpy(), index=base_index, name=higher.name)
