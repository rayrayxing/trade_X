"""Alpaca IEX vs Massive (consolidated) bar comparison.

Answers the design doc's open question: is the free IEX feed close enough to the
consolidated tape for our signals? It compares prices bar by bar, measures IEX's
share of volume, and checks whether the indicator signals we actually trade on
come out the same on both feeds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import talib

# Pass/fail thresholds for "IEX is good enough for price-based signals".
MAX_MEDIAN_CLOSE_BPS = 5.0
MAX_P95_CLOSE_BPS = 20.0
MIN_SIGNAL_AGREEMENT = 0.97
MAX_VOLUME_SHARE_CV = 0.30


@dataclass
class FeedComparison:
    symbol: str
    timeframe: str
    bars_compared: int
    bars_missing_in_iex: int
    close_bps_median: float
    close_bps_p95: float
    close_bps_max: float
    high_bps_p95: float
    low_bps_p95: float
    iex_volume_share_median: float
    iex_volume_share_cv: float
    signal_agreement: dict[str, float] = field(default_factory=dict)
    price_ok: bool = False
    volume_ok: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _bps(a: pd.Series, b: pd.Series, ref: pd.Series) -> pd.Series:
    return (a - b).abs() / ref * 1e4


def _signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ema20, ema50 = talib.EMA(c, 20), talib.EMA(c, 50)
    rsi = talib.RSI(c, 14)
    macd, sig, _ = talib.MACD(c)
    upper = pd.Series(h).shift(1).rolling(20).max().to_numpy()
    return {
        "ema20_above_ema50": ema20 > ema50,
        "rsi14_below_30": rsi < 30,
        "rsi14_above_70": rsi > 70,
        "macd_above_signal": macd > sig,
        "donchian20_breakout": c > upper,
        "cdl_engulfing": talib.CDLENGULFING(o, h, l, c) != 0,
        "cdl_hammer": talib.CDLHAMMER(o, h, l, c) != 0,
        "cdl_doji": talib.CDLDOJI(o, h, l, c) != 0,
    }


def compare_feeds(iex: pd.DataFrame, ref: pd.DataFrame, symbol: str = "", timeframe: str = "") -> FeedComparison:
    """Compare an IEX bar frame against a consolidated reference frame on shared timestamps."""
    joined = ref.join(iex, how="left", lsuffix="_ref", rsuffix="_iex")
    missing = int(joined["close_iex"].isna().sum())
    j = joined.dropna()
    if len(j) < 30:
        raise ValueError(f"only {len(j)} overlapping bars for {symbol}; need at least 30")
    close_bps = _bps(j["close_iex"], j["close_ref"], j["close_ref"])
    high_bps = _bps(j["high_iex"], j["high_ref"], j["close_ref"])
    low_bps = _bps(j["low_iex"], j["low_ref"], j["close_ref"])
    share = (j["volume_iex"] / j["volume_ref"].replace(0, np.nan)).dropna()

    iex_aligned = j[[f"{k}_iex" for k in ("open", "high", "low", "close")]].set_axis(["open", "high", "low", "close"], axis=1)
    ref_aligned = j[[f"{k}_ref" for k in ("open", "high", "low", "close")]].set_axis(["open", "high", "low", "close"], axis=1)
    s_iex, s_ref = _signals(iex_aligned), _signals(ref_aligned)
    warm = 60
    agreement = {}
    for name in s_ref:
        a, b = s_iex[name][warm:], s_ref[name][warm:]
        if len(a):
            agreement[name] = float(np.mean(a == b))

    res = FeedComparison(
        symbol=symbol, timeframe=timeframe, bars_compared=len(j), bars_missing_in_iex=missing,
        close_bps_median=float(close_bps.median()), close_bps_p95=float(close_bps.quantile(0.95)),
        close_bps_max=float(close_bps.max()), high_bps_p95=float(high_bps.quantile(0.95)),
        low_bps_p95=float(low_bps.quantile(0.95)),
        iex_volume_share_median=float(share.median()) if len(share) else float("nan"),
        iex_volume_share_cv=float(share.std() / share.mean()) if len(share) > 1 else float("nan"),
        signal_agreement=agreement,
    )
    trend_signals = [k for k in agreement if not k.startswith("cdl_")]
    res.price_ok = (
        res.close_bps_median <= MAX_MEDIAN_CLOSE_BPS
        and res.close_bps_p95 <= MAX_P95_CLOSE_BPS
        and min(agreement[k] for k in trend_signals) >= MIN_SIGNAL_AGREEMENT
    )
    res.volume_ok = bool(res.iex_volume_share_cv <= MAX_VOLUME_SHARE_CV)
    if missing:
        res.notes.append(f"{missing} reference bars have no IEX bar (thin IEX trading)")
    cdl = {k: v for k, v in agreement.items() if k.startswith("cdl_")}
    if cdl and min(cdl.values()) < MIN_SIGNAL_AGREEMENT:
        res.notes.append("candlestick detections differ between feeds; calibrate candle scores on the feed you trade")
    if not res.volume_ok:
        res.notes.append("IEX volume share is unstable; use volume relative to IEX's own average, never absolute volume")
    return res


def rth_daily_from_intraday(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday US stock bars into regular-session (09:30-16:00 NY) daily bars."""
    ny = bars.tz_convert("America/New_York")
    t = ny.index.time
    rth = ny[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    daily = rth.groupby(rth.index.date).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    daily.index = pd.to_datetime(daily.index).tz_localize("UTC").astype("datetime64[ns, UTC]")
    return daily


def to_session_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Re-key daily bars by New York calendar date so feeds with different stamps line up."""
    out = df.copy()
    out.index = pd.to_datetime(out.index.tz_convert("America/New_York").date).tz_localize("UTC").astype("datetime64[ns, UTC]")
    return out[~out.index.duplicated(keep="last")]


def render_report(results: list[FeedComparison]) -> str:
    lines = [
        "# Alpaca IEX vs Massive consolidated bars",
        "",
        f"Pass rule for price signals: median close gap <= {MAX_MEDIAN_CLOSE_BPS} bps, "
        f"95th percentile <= {MAX_P95_CLOSE_BPS} bps, trend-signal agreement >= {MIN_SIGNAL_AGREEMENT:.0%}.",
        "",
        "| Symbol | TF | Bars | Close gap median / p95 (bps) | IEX volume share | Worst trend-signal agreement | Price OK | Volume OK |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        trend = [v for k, v in r.signal_agreement.items() if not k.startswith("cdl_")]
        lines.append(
            f"| {r.symbol} | {r.timeframe} | {r.bars_compared} | {r.close_bps_median:.1f} / {r.close_bps_p95:.1f} | "
            f"{r.iex_volume_share_median:.1%} (cv {r.iex_volume_share_cv:.2f}) | {min(trend):.1%} | "
            f"{'yes' if r.price_ok else 'NO'} | {'yes' if r.volume_ok else 'NO'} |"
        )
    lines.append("")
    for r in results:
        for n in r.notes:
            lines.append(f"- {r.symbol} {r.timeframe}: {n}")
    return "\n".join(lines) + "\n"
