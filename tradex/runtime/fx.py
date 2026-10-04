"""FX rate sources: USD per unit of a currency, as of a time.

Paper and live never guess a rate (Ray, 4 Oct 2026): a missing or stale rate raises
``MissingRate`` and the core blocks the trade. Backtest and replay may fall back to the
rough constants in ``tradex.costs.models`` when the data has no USD cross, as before.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from tradex.costs.models import APPROX_USD_PER_UNIT

LIVE_MODES = {"paper", "live"}


class MissingRate(LookupError):
    pass


class RateSource(Protocol):
    def usd_per_unit(self, ccy: str, ts: pd.Timestamp) -> float: ...


def _fallback(ccy: str, mode: str) -> float:
    if mode in LIVE_MODES or ccy not in APPROX_USD_PER_UNIT:
        raise MissingRate(f"no USD rate for {ccy}")
    return APPROX_USD_PER_UNIT[ccy]


class SeriesRates:
    """Rates from recorded series (``{ccy: USD per unit}``, e.g. from XXX_USD closes)."""

    def __init__(self, series: dict[str, pd.Series] | None = None, mode: str = "replay"):
        self.series = series or {}
        self.mode = mode

    def usd_per_unit(self, ccy: str, ts: pd.Timestamp) -> float:
        if ccy == "USD":
            return 1.0
        s = self.series.get(ccy)
        if s is not None:
            i = s.index.searchsorted(ts, side="right")
            if i > 0:
                return float(s.iloc[i - 1])
        return _fallback(ccy, self.mode)


class MarketDataRates:
    """Live rates from the latest closed bars of CCY_USD or USD_CCY in a BarStore. The
    bar must have closed no more than ``max_age`` before ``ts``."""

    def __init__(self, data, mode: str = "paper", max_age: pd.Timedelta = pd.Timedelta(hours=4)):
        self.data, self.mode, self.max_age = data, mode, max_age

    def usd_per_unit(self, ccy: str, ts: pd.Timestamp) -> float:
        if ccy == "USD":
            return 1.0
        for sym, invert in ((f"{ccy}_USD", False), (f"USD_{ccy}", True)):
            b = self.data.bars(sym, ts)
            if len(b):
                closed = b.index[-1] + self.data.bar
                if ts - closed <= self.max_age:
                    px = float(b["close"].iloc[-1])
                    return 1.0 / px if invert else px
        return _fallback(ccy, self.mode)


def rate_snapshot(rates: RateSource, ccys: set[str], ts: pd.Timestamp) -> dict[str, pd.Series]:
    """Rates for ``ccys`` at ``ts`` in the ``{ccy: Series}`` shape the risk code takes. Raises
    MissingRate for any it cannot price, so the gate never falls back to a constant."""
    return {c: pd.Series([rates.usd_per_unit(c, ts)], index=[ts]) for c in sorted(ccys) if c != "USD"}
