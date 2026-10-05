"""Option contract and position model.

One US equity option contract covers 100 shares. Every USD figure in this package is
``premium per share x multiplier x contracts``; forgetting the multiplier is the classic
100x sizing bug, so it is a field on the contract and never a magic number elsewhere
(the only literal 100 is the default ``MULTIPLIER``).

Conventions
- ``contracts`` on a position is signed: positive long, negative short.
- Premiums are per share, as quoted (an ask of 1.25 costs US$125 per contract).
- ``OptionContract`` is frozen and hashable, so it can key dicts and sets.
- Moomoo / OpenD codes look like ``US.AAPL261016C150000`` (strike x 1000, no padding);
  OCC symbols are 21 characters: root padded to 6, YYMMDD, C/P, strike x 1000 in 8 digits.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

MULTIPLIER = 100
NY = "America/New_York"
EXPIRY_CLOSE = dt.time(16, 0)          # US equity options stop trading at 16:00 New York on expiry day


class Right(str, Enum):
    CALL = "C"
    PUT = "P"

    @classmethod
    def parse(cls, value: "str | Right") -> "Right":
        if isinstance(value, Right):
            return value
        v = str(value).strip().upper()
        if v in ("C", "CALL"):
            return cls.CALL
        if v in ("P", "PUT"):
            return cls.PUT
        raise ValueError(f"unknown option right {value!r}")


_MOOMOO = re.compile(r"^(?:US\.)?(?P<root>[A-Z][A-Z0-9.]{0,5}?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<right>[CP])(?P<k>\d+)$")
_OCC = re.compile(r"^(?P<root>[A-Z0-9.]{1,6})\s*(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<right>[CP])(?P<k>\d{8})$")


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiry: dt.date
    strike: float
    right: Right
    multiplier: int = MULTIPLIER

    def __post_init__(self):
        object.__setattr__(self, "right", Right.parse(self.right))
        if isinstance(self.expiry, pd.Timestamp):
            object.__setattr__(self, "expiry", self.expiry.date())
        if not (self.strike > 0 and math.isfinite(self.strike)):
            raise ValueError(f"strike must be positive, got {self.strike!r}")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")

    # --- identifiers ---------------------------------------------------------------------
    @property
    def _strike_milli(self) -> int:
        return int(round(self.strike * 1000))

    @property
    def moomoo_code(self) -> str:
        return f"US.{self.underlying}{self.expiry:%y%m%d}{self.right.value}{self._strike_milli}"

    @property
    def occ_symbol(self) -> str:
        return f"{self.underlying:<6}{self.expiry:%y%m%d}{self.right.value}{self._strike_milli:08d}"

    @classmethod
    def from_moomoo_code(cls, code: str, multiplier: int = MULTIPLIER) -> "OptionContract":
        m = _MOOMOO.match(code.strip().upper())
        if not m:
            raise ValueError(f"not a moomoo option code: {code!r}")
        return cls._from_match(m, int(m["k"]), multiplier)

    @classmethod
    def from_occ(cls, symbol: str, multiplier: int = MULTIPLIER) -> "OptionContract":
        m = _OCC.match(symbol.strip().upper())
        if not m:
            raise ValueError(f"not an OCC option symbol: {symbol!r}")
        return cls._from_match(m, int(m["k"]), multiplier)

    @classmethod
    def _from_match(cls, m: re.Match, strike_milli: int, multiplier: int) -> "OptionContract":
        expiry = dt.date(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]))
        return cls(m["root"], expiry, strike_milli / 1000.0, Right(m["right"]), multiplier)

    # --- value ---------------------------------------------------------------------------
    def intrinsic(self, spot: float) -> float:
        """Intrinsic value per share at ``spot``."""
        if self.right is Right.CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def is_itm(self, spot: float) -> bool:
        return self.intrinsic(spot) > 0.0

    def moneyness(self, spot: float) -> float:
        """strike / spot: above 1 is out of the money for a call, below 1 for a put."""
        return self.strike / spot

    # --- time ----------------------------------------------------------------------------
    @property
    def expiry_time(self) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(self.expiry, EXPIRY_CLOSE)).tz_localize(NY).tz_convert("UTC")

    def days_to_expiry(self, asof: pd.Timestamp | dt.date) -> int:
        """Whole calendar days from the New York date of ``asof`` to expiry (0 on expiry day)."""
        d = asof.tz_convert(NY).date() if isinstance(asof, pd.Timestamp) else asof
        return (self.expiry - d).days

    def years_to_expiry(self, asof: pd.Timestamp) -> float:
        secs = (self.expiry_time - asof).total_seconds()
        return max(secs, 0.0) / (365.0 * 86400.0)

    def expired(self, asof: pd.Timestamp) -> bool:
        return asof >= self.expiry_time


@dataclass(frozen=True)
class Greeks:
    """As reported by the provider (OpenD supplies them); units are the provider's. The backtester
    reads only ``delta`` and ``iv``; ``theta``, ``vega`` etc. are carried for reports."""
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    iv: float | None = None            # implied volatility as a fraction (0.45 = 45%)


@dataclass(frozen=True)
class OptionQuote:
    """One contract's quote at one moment, as injected by an ``OptionDataProvider``."""
    contract: OptionContract
    ts: pd.Timestamp
    bid: float | None
    ask: float | None
    last: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    greeks: Greeks = field(default_factory=Greeks)
    underlying_price: float | None = None

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None and self.ask >= self.bid >= 0 and self.ask > 0

    @property
    def mid(self) -> float | None:
        if self.two_sided:
            return 0.5 * (self.bid + self.ask)
        return self.last if self.last is not None and self.last > 0 else None

    @property
    def spread_pct(self) -> float | None:
        """(ask - bid) / mid; None when the quote is not two-sided."""
        if not self.two_sided or self.mid in (None, 0):
            return None
        return (self.ask - self.bid) / self.mid

    @property
    def delta(self) -> float | None:
        return self.greeks.delta

    @property
    def iv(self) -> float | None:
        return self.greeks.iv


@dataclass
class OptionPosition:
    """An open option position. ``contracts`` > 0 long, < 0 short."""
    contract: OptionContract
    contracts: int
    entry_premium: float               # per share, the fill price
    entry_time: pd.Timestamp
    stop_premium: float | None = None  # long: sell-to-close below this; short: buy-to-close at or above this
    stop_underlying: float | None = None
    target_underlying: float | None = None
    decision_id: str = ""
    entry_underlying: float | None = None
    entry_delta: float | None = None
    entry_iv: float | None = None
    max_bars: int | None = None
    bars_held: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.contracts == 0 or int(self.contracts) != self.contracts:
            raise ValueError("contracts must be a non-zero integer")

    @property
    def direction(self) -> int:
        return 1 if self.contracts > 0 else -1

    @property
    def is_long(self) -> bool:
        return self.contracts > 0

    @property
    def qty(self) -> int:
        """Absolute contract count (the order quantity)."""
        return abs(self.contracts)

    @property
    def is_naked_call(self) -> bool:
        return self.contracts < 0 and self.contract.right is Right.CALL

    @property
    def shares(self) -> int:
        return self.contracts * self.contract.multiplier

    def market_value(self, premium: float) -> float:
        """Signed USD value of the position at ``premium`` (a short is a liability)."""
        return self.shares * premium

    def unrealized_pnl(self, premium: float) -> float:
        """USD P&L versus entry before fees: (premium - entry) x shares."""
        return (premium - self.entry_premium) * self.shares

    def delta_shares(self, delta: float) -> float:
        """Share-equivalent exposure at ``delta`` (negative for a short call or a long put)."""
        return self.shares * delta
