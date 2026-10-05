"""Data-provider interfaces for the options lane, and two adapters.

Nothing in ``tradex.options`` fetches data or invents it. Chains, quotes, implied volatility
and Greeks come in through the protocols below; the backtester and the sizing code only
consume them. In paper and live the chain provider is OpenD (read-only quote calls); in a
backtest it is a ``RecordedChainProvider`` over snapshots the paper loop recorded earlier.

**The provider interface (what OpenD has to supply)**

``OptionDataProvider.chain(underlying, ts, at)``
    Every listed contract of ``underlying`` with a two-sided or last quote, as a list of
    ``OptionQuote``: bid, ask, last, volume, open interest, delta, gamma, theta, vega, rho,
    implied volatility (a fraction, 0.45 = 45%) and the underlying's price at that moment.
    OpenD: ``get_option_expiration_date`` then ``get_option_chain`` per expiry, then
    ``get_market_snapshot`` on the codes (snapshot rows carry bid/ask/last, open interest,
    IV and the Greeks). ``quotes_from_opend_frame`` converts such a frame.
``OptionDataProvider.quote(contract, ts, at)``
    The same fields for one contract, or None when the contract has no quote then.

``ts`` is the **bar timestamp** (bar open time, UTC, the convention in ``tradex.timeframes``)
and ``at`` says which moment of that bar: ``"open"`` for the first quote of the bar (where
an order decided at the previous close fills) or ``"close"`` for the last quote of the bar
(marks, exit decisions). A provider must never return a quote from after that moment.

``UnderlyingRiskProvider`` supplies what the naked-call rules need: the next earnings date
and short-interest figures (OpenD does not provide either; they come from the research data
feed). ``None`` means *unknown*, and unknown blocks a naked call (it does not pass it).
``DividendProvider`` is optional and feeds the early-assignment rule.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from tradex.data.guard import RealDataMissing, SyntheticDataRefused, require_real_data
from tradex.execution.checks import ShortInfo
from tradex.options.contract import Greeks, OptionContract, OptionQuote, Right

At = Literal["open", "close"]


@runtime_checkable
class OptionDataProvider(Protocol):
    def chain(self, underlying: str, ts: pd.Timestamp, at: At = "close") -> Sequence[OptionQuote]: ...

    def quote(self, contract: OptionContract, ts: pd.Timestamp, at: At = "close") -> OptionQuote | None: ...


@dataclass(frozen=True)
class EarningsInfo:
    """``next_date`` is the next scheduled report on or after the query date, None when the
    feed knows of none inside its look-ahead window (that is a known absence, not missing data)."""
    next_date: dt.date | None


@runtime_checkable
class UnderlyingRiskProvider(Protocol):
    def next_earnings(self, underlying: str, asof: pd.Timestamp) -> EarningsInfo | None:
        """None means the earnings calendar has no information on this name: unknown."""

    def short_info(self, underlying: str, asof: pd.Timestamp) -> ShortInfo | None:
        """Short interest / borrow data for the name; None means unknown."""


@dataclass(frozen=True)
class Dividend:
    ex_date: dt.date
    amount: float                      # per share


@runtime_checkable
class DividendProvider(Protocol):
    def next_ex_dividend(self, underlying: str, asof: pd.Timestamp) -> Dividend | None: ...


# --- recorded snapshots -------------------------------------------------------------------

QUOTE_COLUMNS = ["ts", "at", "underlying", "expiry", "strike", "right", "bid", "ask", "last", "volume",
                 "open_interest", "delta", "gamma", "theta", "vega", "rho", "iv", "underlying_price"]
_REQUIRED = ["ts", "at", "underlying", "expiry", "strike", "right"]


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


class RecordedChainProvider:
    """Chains recorded from OpenD (or any real feed) and replayed for backtests.

    ``frame`` has one row per contract per snapshot with the columns in ``QUOTE_COLUMNS``
    (``ts`` UTC bar-open timestamp, ``at`` "open" or "close", ``expiry`` a date). Lookups
    are exact on (underlying, ts, at): a missing snapshot is a missing quote, never the
    nearest one, which keeps look-ahead out. Backtest-only: ``require_real_option_data``
    refuses it in paper and live.
    """

    recorded = True

    def __init__(self, frame: pd.DataFrame):
        missing = [c for c in _REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(f"recorded chain frame lacks columns {missing}")
        self._chains: dict[tuple[str, pd.Timestamp, str], dict[OptionContract, OptionQuote]] = {}
        for row in frame.to_dict("records"):
            ts = pd.Timestamp(row["ts"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            at = str(row["at"])
            if at not in ("open", "close"):
                raise ValueError(f"'at' must be open or close, got {at!r}")
            c = OptionContract(str(row["underlying"]), pd.Timestamp(row["expiry"]).date(), float(row["strike"]),
                               Right.parse(row["right"]))
            q = OptionQuote(c, ts, _num(row.get("bid")), _num(row.get("ask")), _num(row.get("last")),
                            _num(row.get("volume")), _num(row.get("open_interest")),
                            Greeks(_num(row.get("delta")), _num(row.get("gamma")), _num(row.get("theta")),
                                   _num(row.get("vega")), _num(row.get("rho")), _num(row.get("iv"))),
                            _num(row.get("underlying_price")))
            self._chains.setdefault((c.underlying, ts, at), {})[c] = q

    def chain(self, underlying: str, ts: pd.Timestamp, at: At = "close") -> list[OptionQuote]:
        return list(self._chains.get((underlying, _utc(ts), at), {}).values())

    def quote(self, contract: OptionContract, ts: pd.Timestamp, at: At = "close") -> OptionQuote | None:
        return self._chains.get((contract.underlying, _utc(ts), at), {}).get(contract)


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# --- OpenD frame conversion ---------------------------------------------------------------

# moomoo snapshot column names for options (moomoo API docs, get_market_snapshot). Unverified
# against a live OpenD session: confirm with one read-only snapshot before relying on them.
OPEND_COLUMNS = {
    "bid": "bid_price", "ask": "ask_price", "last": "last_price", "volume": "volume",
    "open_interest": "option_open_interest", "iv": "option_implied_volatility",
    "delta": "option_delta", "gamma": "option_gamma", "theta": "option_theta", "vega": "option_vega",
    "rho": "option_rho",
}


def quotes_from_opend_frame(df: pd.DataFrame, ts: pd.Timestamp, underlying_price: float | None = None,
                            iv_in_percent: bool = True, columns: dict[str, str] | None = None
                            ) -> list[OptionQuote]:
    """Convert a moomoo snapshot DataFrame (one row per option ``code``) into ``OptionQuote`` s.

    The contract comes from the ``code`` column (``US.AAPL261016C150000``), so no strike or
    expiry column is needed. moomoo reports implied volatility in percent; ``iv_in_percent``
    converts it to a fraction. Rows whose code does not parse are skipped. Pass ``columns`` to
    remap names if OpenD's differ.
    """
    cols = OPEND_COLUMNS | (columns or {})
    out = []
    for row in df.to_dict("records"):
        try:
            c = OptionContract.from_moomoo_code(str(row["code"]))
        except (KeyError, ValueError):
            continue
        g = {k: _num(row.get(cols[k])) for k in cols}
        iv = g["iv"] / 100.0 if (iv_in_percent and g["iv"] is not None) else g["iv"]
        up = _num(row.get("underlying_price")) if "underlying_price" in row else None
        out.append(OptionQuote(c, ts, g["bid"], g["ask"], g["last"], g["volume"], g["open_interest"],
                               Greeks(g["delta"], g["gamma"], g["theta"], g["vega"], g["rho"], iv),
                               up if up is not None else underlying_price))
    return out


# --- guard ---------------------------------------------------------------------------------

def require_real_option_data(mode: str, provider) -> None:
    """Paper and live need a live provider: refuse recorded, synthetic, CSV and cached stand-ins."""
    require_real_data(mode, provider)            # unknown modes, None, synthetic/csv/cached names
    if mode in ("paper", "live") and getattr(provider, "recorded", False):
        raise SyntheticDataRefused(f"{mode} mode refuses recorded option chains ({type(provider).__name__})")


__all__ = ["At", "Dividend", "DividendProvider", "EarningsInfo", "OptionDataProvider", "QUOTE_COLUMNS",
           "RealDataMissing", "RecordedChainProvider", "SyntheticDataRefused", "UnderlyingRiskProvider",
           "quotes_from_opend_frame", "require_real_option_data"]
