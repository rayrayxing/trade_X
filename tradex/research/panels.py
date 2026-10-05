"""Causal research columns added to real bars: cross-sectional ranks, pair spreads, sessions.

The engine evaluates each symbol on its own bars, so anything that needs other symbols
(a rank across a universe, a pair's spread, a sector ETF) is computed here first and
carried as an extra column, read by specs through the ``data.column`` feature. Every
value at bar t uses only bars at or before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def frame(data: dict[str, pd.DataFrame], col: str = "close") -> pd.DataFrame:
    return pd.DataFrame({s: b[col] for s, b in data.items()}).sort_index()


def xs_percentile(values: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Per-date percentile rank (0..1] across symbols; NaN where fewer than ``min_names`` have a value."""
    pct = values.rank(axis=1, pct=True)
    return pct.where(values.notna().sum(axis=1) >= min_names)


def high_ratio(bars: pd.DataFrame, n: int = 252) -> pd.Series:
    """Close over the n-bar high (George and Hwang's 52-week-high measure)."""
    return bars["close"] / bars["high"].rolling(n, min_periods=n).max()


def momentum(bars: pd.DataFrame, n: int = 252) -> pd.Series:
    return bars["close"] / bars["close"].shift(n) - 1


def relative_strength(stock: pd.DataFrame, bench: pd.DataFrame, n: int = 63, ma: int = 50) -> pd.DataFrame:
    """Stock over benchmark: n-bar change of the ratio, and the ratio's distance above its moving average."""
    rs = stock["close"] / bench["close"].reindex(stock.index)
    return pd.DataFrame({"rs_mom": rs / rs.shift(n) - 1, "rs_above": rs / rs.rolling(ma).mean() - 1})


def pair_zscore(a: pd.DataFrame, b: pd.DataFrame, beta_window: int = 252, z_window: int = 60) -> pd.Series:
    """z-score of log(a) - beta*log(b), beta from a trailing OLS (hedge ratio) on log prices."""
    la, lb = np.log(a["close"]), np.log(b["close"]).reindex(a.index)
    beta = la.rolling(beta_window).cov(lb) / lb.rolling(beta_window).var()
    spread = la - beta * lb
    return (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()


def session_columns(h1: pd.DataFrame, bar: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    """For US intraday bars of length ``bar`` stamped at open time (see tradex.data.opend):

    first_bar  1 on the 09:30 bar
    last_full  1 on the bar that closes at 15:30 (14:30 for 60-minute bars, 15:00 for 30-minute
               bars); acting on it fills at the 15:30 price
    gap        session open over the previous session's close - 1, on the first bar only
    first_ret  first bar's close over its open - 1, on the first bar only
    fh_ret     first bar's close over the previous session's close - 1 (Gao, Han, Li and Zhou's
               first-half-hour return; the first hour on 60-minute bars), on every bar from the first bar
               on: known once the first bar closes, so later bars of the day may read it
    """
    t = h1.index.tz_convert(NY)
    first = (t.hour == 9) & (t.minute == 30)
    day = pd.Series(t.date, index=h1.index)
    prev_close = h1["close"].groupby(day).last().shift(1)
    gap = h1["open"] / day.map(prev_close) - 1
    out = pd.DataFrame(index=h1.index)
    out["first_bar"] = first.astype(float)
    close_min = t.hour * 60 + t.minute + int(bar / pd.Timedelta(minutes=1))
    out["last_full"] = (close_min == 15 * 60 + 30).astype(float)
    out["gap"] = np.where(first, gap, 0.0)
    out["first_ret"] = np.where(first, h1["close"] / h1["open"] - 1, 0.0)
    fh = pd.Series(np.where(first, h1["close"] / day.map(prev_close).to_numpy() - 1, np.nan), index=h1.index)
    out["fh_ret"] = fh.groupby(day).ffill()
    return out.fillna(0.0)


def earnings_columns(d1: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """For daily bars stamped 00:00 New York and an earnings table (tradex.data.earnings):

    earn_day   1 on the reaction day: the release day for a release before or during the
               session, the next session for one after the close; events with unknown
               timing (the SEC filing-date proxy) are skipped
    earn_jump  the reaction day's close over the previous close - 1, on that day only
    """
    out = pd.DataFrame({"earn_day": 0.0, "earn_jump": 0.0}, index=d1.index)
    if events is None or not len(events) or d1.empty:
        return out
    days = d1.index.tz_convert(NY).normalize()
    ret = d1["close"] / d1["close"].shift(1) - 1
    for _, e in events.iterrows():
        if e["timing"] not in ("before", "during", "after"):
            continue
        d = pd.Timestamp(e["date"]).tz_localize(NY) if pd.Timestamp(e["date"]).tzinfo is None \
            else pd.Timestamp(e["date"]).tz_convert(NY).normalize()
        pos = days.searchsorted(d, side="right" if e["timing"] == "after" else "left")
        if pos < len(days) and (days[pos] - d).days <= 5:
            out.iloc[pos, 0] = 1.0
            out.iloc[pos, 1] = ret.iloc[pos] if not np.isnan(ret.iloc[pos]) else 0.0
    return out


def with_columns(bars: pd.DataFrame, cols: pd.DataFrame | dict[str, pd.Series]) -> pd.DataFrame:
    out = bars.copy()
    for k, v in (cols.items() if isinstance(cols, dict) else cols.items()):
        out[k] = v.reindex(bars.index)
    return out
