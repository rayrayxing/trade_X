"""Synthetic bars for tests and smoke runs (never used to judge a real strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradex.timeframes import duration


def synthetic_bars(
    n: int = 1000,
    tf: str = "D1",
    start: str = "2018-01-01",
    seed: int = 0,
    price: float = 100.0,
    vol: float = 0.015,
    drift: float = 0.0002,
    regime_len: int = 120,
    trend_strength: float = 0.002,
    volume: float = 1e6,
    business_days: bool | None = None,
) -> pd.DataFrame:
    """Random walk with alternating trend/range regimes, so strategies have something to find."""
    rng = np.random.default_rng(seed)
    if business_days is None:
        business_days = tf == "D1"
    if business_days:
        idx = pd.bdate_range(start, periods=n, tz="UTC")
    else:
        idx = pd.date_range(start, periods=n, freq=duration(tf), tz="UTC")
    regimes = (np.arange(n) // regime_len) % 3  # 0 up-trend, 1 range, 2 down-trend
    mu = np.where(regimes == 0, trend_strength, np.where(regimes == 2, -trend_strength, 0.0)) + drift
    rets = mu + vol * rng.standard_t(5, size=n) / np.sqrt(5 / 3)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]] * np.exp(rng.normal(0, vol * 0.2, n))
    span = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    vol_series = volume * np.exp(rng.normal(0, 0.4, n)) * (1 + 5 * np.abs(rets))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_series}, index=idx
    )
    df.index = df.index.astype("datetime64[ns, UTC]")
    df.index.name = "ts"
    df.attrs["origin"] = "synthetic"          # provenance tag: paper/live refuse frames that carry it (tradex.data.guard)
    return df
