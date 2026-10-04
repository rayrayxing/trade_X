"""Forex costs from measured spreads (real-data item 2 of the integration plan).

``OandaFxCosts`` prices every fill with a typical spread per pair, an estimate. Backtests
should instead pay the spread Oanda actually quoted at that bar: the difference between
the ask and bid candles (``price=BA``) at the bar's open, which is where the engine fills
market orders. ``MeasuredFxCosts`` does that and has no fallback: a bar with no measured
spread nearby raises ``MissingSpread``, so a backtest never quietly mixes measured and
typical spreads.

Bid/ask candles come from Oanda practice history (lane B's ``fetch_ba_candles`` or the
research fetcher); both produce ``bid_open``/``ask_open`` columns, which is all this needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradex.costs.models import OandaFxCosts, pip_size


class MissingSpread(LookupError):
    """No measured spread for this pair close enough to the fill time."""


def spreads_from_ba(ba: pd.DataFrame, at: str = "open") -> pd.Series:
    """Quoted spread in price units at each candle's open (or close), from bid/ask candles."""
    for col in (f"bid_{at}", f"ask_{at}"):
        if col not in ba.columns:
            raise ValueError(f"bid/ask candles need a {col} column")
    s = (ba[f"ask_{at}"] - ba[f"bid_{at}"]).astype(float)
    s = s[s.notna() & (s >= 0)].sort_index()
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    return s.rename("spread")


class MeasuredSpreads:
    """Measured spreads per pair: ``{pair: Series of spreads in price units}`` indexed by
    candle open time (UTC). The spread in force at ``ts`` is the latest one at or before
    ``ts``, provided it is no older than ``max_age`` (default 4 days, which spans a
    weekend). Its ``spread_pips(symbol, ts)`` matches the spread half of lane B's live
    ``RateSource``, so the same lookup serves a backtest and a recorded paper session."""

    def __init__(self, spreads: dict[str, pd.Series], max_age: pd.Timedelta = pd.Timedelta(days=4)):
        self.spreads = {k: v.sort_index() for k, v in spreads.items()}
        self.max_age = max_age

    @classmethod
    def from_ba_candles(cls, candles: dict[str, pd.DataFrame], **kw) -> "MeasuredSpreads":
        return cls({pair: spreads_from_ba(df) for pair, df in candles.items()}, **kw)

    def spread_at(self, symbol: str, ts: pd.Timestamp) -> float:
        """Measured spread (price units) in force at ``ts``."""
        s = self.spreads.get(symbol)
        if s is None or s.empty:
            raise MissingSpread(f"no measured spreads for {symbol}")
        i = s.index.searchsorted(ts, side="right")
        if i == 0:
            raise MissingSpread(f"no measured {symbol} spread at or before {ts}")
        if ts - s.index[i - 1] > self.max_age:
            raise MissingSpread(f"latest measured {symbol} spread is from {s.index[i - 1]}, too old for {ts}")
        return float(s.iloc[i - 1])

    def spread_pips(self, symbol: str, ts: pd.Timestamp) -> float:
        return self.spread_at(symbol, ts) / pip_size(symbol)


@dataclass
class MeasuredFxCosts(OandaFxCosts):
    """Oanda costs that pay the measured spread at each fill. Slippage, financing and the
    stress multiplier work as in ``OandaFxCosts``; typical spreads are never used."""

    measured: MeasuredSpreads | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.measured is None:
            raise ValueError("MeasuredFxCosts needs measured spreads")
        self.spread_pips = {}

    @classmethod
    def from_ba_candles(cls, candles: dict[str, pd.DataFrame], max_age: pd.Timedelta = pd.Timedelta(days=4),
                        **kw) -> "MeasuredFxCosts":
        return cls(measured=MeasuredSpreads.from_ba_candles(candles, max_age=max_age), **kw)

    def fill(self, symbol, side, mid, ts):
        pip = pip_size(symbol)
        hs = 0.5 * self.measured.spread_at(symbol, ts) * self.stress
        sl = self.slippage_pips * pip * self.stress
        return mid + side * (hs + sl), {"spread": hs, "slippage": sl}
