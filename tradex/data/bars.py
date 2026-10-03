"""Bar frame conventions and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradex.timeframes import OHLCV


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame: UTC ns DatetimeIndex, sorted, unique, float columns."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert("UTC").astype("datetime64[ns, UTC]")
    out.index.name = "ts"
    if "volume" not in out.columns:
        out["volume"] = 0.0
    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing columns {missing}")
    out = out[OHLCV].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def validate_bars(df: pd.DataFrame) -> list[str]:
    """Return a list of data problems (empty when clean)."""
    problems = []
    if df.isna().any().any():
        problems.append(f"{int(df.isna().sum().sum())} missing values")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        problems.append(f"{int(bad.sum())} bars where high/low do not bracket open/close")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive prices")
    jumps = np.abs(np.log(df["close"]).diff())
    if (jumps > 0.5).any():
        problems.append(f"{int((jumps > 0.5).sum())} bar-to-bar moves above 50% (unadjusted split?)")
    return problems
