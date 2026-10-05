"""Relative-strength momentum: stocks against a benchmark, currencies against each other.

All functions are causal (value at t uses closes at or before t) and return plain Series /
DataFrames that the research builders attach to bars; strategies read them through
``data.column``. Cross-sectional ranks use ``panels.xs_percentile``.

  rs_momentum        change of the stock/benchmark ratio over ``n`` bars, ending ``skip`` bars ago
                     (the usual 12-1 form is n=231, skip=21 on daily bars)
  rs_trend           distance of the ratio above its own moving average
  currency_strength  per currency, the mean n-bar log return of the pairs it appears in, signed so that
                     a rising pair counts for the base and against the quote
  pair_strength_diff base strength minus quote strength for one pair
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rs_ratio(close: pd.Series, bench_close: pd.Series) -> pd.Series:
    return close / bench_close.reindex(close.index, method="ffill")


def rs_momentum(close: pd.Series, bench_close: pd.Series, n: int = 231, skip: int = 21) -> pd.Series:
    """Relative-strength momentum: (ratio ``skip`` bars ago) over (ratio ``skip + n`` bars ago), minus 1."""
    if n < 1 or skip < 0:
        raise ValueError("n must be positive and skip non-negative")
    rs = rs_ratio(close, bench_close)
    return rs.shift(skip) / rs.shift(skip + n) - 1


def rs_trend(close: pd.Series, bench_close: pd.Series, ma: int = 50) -> pd.Series:
    rs = rs_ratio(close, bench_close)
    return rs / rs.rolling(ma).mean() - 1


def currency_strength(closes: dict[str, pd.Series], n: int = 63) -> pd.DataFrame:
    """Strength of each currency from the pairs given (keys like ``EUR_USD``): for a pair BASE_QUOTE the n-bar log
    return counts +1 for BASE and -1 for QUOTE; a currency's strength is the mean over the pairs it is in."""
    px = pd.DataFrame(closes).sort_index()
    logret = np.log(px).diff(n)
    parts: dict[str, list[pd.Series]] = {}
    for pair in px.columns:
        base, quote = pair.split("_")
        parts.setdefault(base, []).append(logret[pair])
        parts.setdefault(quote, []).append(-logret[pair])
    return pd.DataFrame({ccy: pd.concat(s, axis=1).mean(axis=1, skipna=False) for ccy, s in parts.items()})


def pair_strength_diff(pair: str, strength: pd.DataFrame) -> pd.Series:
    base, quote = pair.split("_")
    return strength[base] - strength[quote]
