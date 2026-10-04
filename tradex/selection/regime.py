"""Market regime labels: trend vs range, and volatility level."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def classify_regime(bars: pd.DataFrame, adx_period: int = 14, adx_trend: float = 25.0,
                    vol_lookback: int = 252) -> pd.DataFrame:
    """Label each bar ``trend``/``range`` and ``low``/``normal``/``high`` volatility, using closed bars only."""
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    adx = pd.Series(talib.ADX(h, l, c, adx_period), index=bars.index)
    atr_pct = pd.Series(talib.ATR(h, l, c, 14), index=bars.index) / bars["close"]
    pct = atr_pct.rolling(vol_lookback, min_periods=60).rank(pct=True)
    trend = np.where(adx >= adx_trend, "trend", "range")
    vol = np.select([pct < 0.33, pct > 0.67], ["low", "high"], "normal")
    out = pd.DataFrame({"trend": trend, "vol": vol, "adx": adx, "atr_pct": atr_pct, "vol_pct": pct}, index=bars.index)
    out.loc[adx.isna() | pct.isna(), ["trend", "vol"]] = "unknown"
    out["label"] = out["trend"] + "/" + out["vol"]
    return out


def daily_labels(regime: pd.DataFrame) -> pd.Series:
    """One regime label per calendar day (last bar of the day)."""
    s = regime["label"].copy()
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")]
