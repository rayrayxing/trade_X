"""Carry and trend features for FX pairs from an injected rate source.

``carry_columns`` takes any object with ``rates(ccy, index)`` (``sources.RateSource``): a CSV of
official policy rates, Oanda financing history, or a fixture in a test. Nothing here knows a rate.
Rates are looked up as known at each bar's time, so a rate change is used from its effective
date, not before. Pair symbols are ``BASE_QUOTE``; going long earns ``rate(BASE) - rate(QUOTE)``.

  carry      annual rate differential earned by a long position (decimal; 0.02 = 2%)
  carry_vol  carry over the pair's annualised realised volatility (carry per unit of risk)
  rate_chg   change in the differential over ``chg_n`` bars (rate momentum)
  ts_mom     ``trend_n``-bar return of the pair (time-series trend)
  carry_trend  +1 when carry and trend are both positive, -1 when both negative, else 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradex.research.regime import expanding_percentile, realized_vol
from tradex.research.sources import RateSource

CARRY_COLUMNS = ["carry", "carry_vol", "rate_chg", "ts_mom", "carry_trend", "fxvol_pct"]


def carry_columns(bars: pd.DataFrame, pair: str, rates: RateSource, trend_n: int = 100, chg_n: int = 63,
                  vol_n: int = 21) -> pd.DataFrame:
    base, quote = pair.split("_")
    idx = bars.index
    carry = rates.rates(base, idx) - rates.rates(quote, idx)
    vol = realized_vol(bars["close"], vol_n)
    trend = bars["close"].pct_change(trend_n)
    agree = pd.Series(np.where((carry > 0) & (trend > 0), 1.0, np.where((carry < 0) & (trend < 0), -1.0, 0.0)), index=idx)
    agree = agree.where(carry.notna() & trend.notna())
    return pd.DataFrame({
        "carry": carry,
        "carry_vol": carry / vol,
        "rate_chg": carry - carry.shift(chg_n),
        "ts_mom": trend,
        "carry_trend": agree,
        "fxvol_pct": expanding_percentile(vol),
    }, index=idx)[CARRY_COLUMNS]
