"""Trading cost models: Moomoo SG for US stocks, Oanda SG for forex.

Every number here is a default that can be overridden in config. Live and paper
trading must read margin, borrow and financing rates from the broker APIs at
run time (Ray's decision, 3 Oct 2026); backtests use these estimates instead.

All costs are returned in USD, positive = money paid, negative = money received.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from importlib import resources
from typing import Protocol

import pandas as pd

NY = "America/New_York"


class CostModel(Protocol):
    asset_class: str

    def fill(self, symbol: str, side: int, mid: float, ts: pd.Timestamp) -> tuple[float, dict[str, float]]:
        """Executed price for a market order (side +1 buy, -1 sell) and per-unit spread/slippage."""

    def order_fees(self, symbol: str, side: int, qty: float, price: float, ts: pd.Timestamp) -> dict[str, float]:
        """Fixed and regulatory fees for one order, in USD."""

    def holding_cost(self, symbol: str, direction: int, qty: float, price: float, t0: pd.Timestamp,
                     t1: pd.Timestamp, base_to_usd: float, borrowed_usd: float) -> dict[str, float]:
        """Carrying costs between t0 and t1 for an open position, in USD."""


# --- US stocks at Moomoo SG -------------------------------------------------------------

@dataclass
class MoomooStockCosts:
    """Moomoo SG US-stock costs.

    Platform fee US$0.99 per order and zero commission are from the thread-1 design doc
    (MoneySmart, 3 Oct 2026). The settlement fee and the SEC/FINRA rates are from memory
    and should be checked against Moomoo's fee page and the current SEC Section 31 rate.
    """

    asset_class: str = "stocks"
    commission: float = 0.0
    platform_fee: float = 0.99
    settlement_fee_per_share: float = 0.003
    sec_fee_rate: float = 27.80e-6          # USD per USD of sale proceeds
    finra_taf_per_share: float = 0.000166   # sells only
    finra_taf_max: float = 8.30
    half_spread_bps: dict[str, float] = field(default_factory=dict)
    default_half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_rate_annual: dict[str, float] = field(default_factory=dict)
    default_borrow_rate_annual: float = 0.005
    margin_rate_annual: float = 0.068
    day_count: int = 360
    stress: float = 1.0                     # multiplies spread and slippage in stress tests

    def fill(self, symbol, side, mid, ts):
        hs = self.half_spread_bps.get(symbol, self.default_half_spread_bps) * 1e-4 * mid * self.stress
        sl = self.slippage_bps * 1e-4 * mid * self.stress
        return mid + side * (hs + sl), {"spread": hs, "slippage": sl}

    def order_fees(self, symbol, side, qty, price, ts):
        fees = {"platform": self.platform_fee, "commission": self.commission,
                "settlement": self.settlement_fee_per_share * qty}
        if side < 0:
            fees["sec"] = self.sec_fee_rate * qty * price
            fees["finra_taf"] = min(self.finra_taf_per_share * qty, self.finra_taf_max)
        return fees

    def holding_cost(self, symbol, direction, qty, price, t0, t1, base_to_usd=1.0, borrowed_usd=0.0):
        days = overnight_days(t0, t1)
        if days == 0:
            return {}
        out = {}
        if direction < 0:
            rate = self.borrow_rate_annual.get(symbol, self.default_borrow_rate_annual)
            out["borrow"] = qty * price * rate * days / self.day_count
        if borrowed_usd > 0:
            out["margin_interest"] = borrowed_usd * self.margin_rate_annual * days / self.day_count
        return out


def overnight_days(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """Calendar days between the New York dates of two timestamps (interest accrues per night held)."""
    return max(0, (t1.tz_convert(NY).date() - t0.tz_convert(NY).date()).days)


# --- forex at Oanda SG -------------------------------------------------------------------

PIP = {"JPY": 0.01}

# Typical Oanda standard-account spreads in pips, a starting estimate to replace with
# measured spreads from Oanda practice pricing.
DEFAULT_FX_SPREAD_PIPS = {
    "EUR_USD": 1.4, "USD_JPY": 1.4, "GBP_USD": 2.0, "AUD_USD": 1.4,
    "EUR_JPY": 2.0, "GBP_JPY": 3.0, "USD_CHF": 1.8, "USD_CAD": 2.0, "EUR_GBP": 1.6,
}

# Rough USD value of one unit of each currency, used only when no rate series is supplied.
APPROX_USD_PER_UNIT = {"USD": 1.0, "EUR": 1.10, "GBP": 1.30, "AUD": 0.66, "JPY": 1 / 150, "CHF": 1.15, "CAD": 0.73}


def pip_size(symbol: str) -> float:
    return PIP.get(symbol.split("_")[1], 0.0001)


def split_pair(symbol: str) -> tuple[str, str]:
    base, quote = symbol.split("_")
    return base, quote


class PolicyRates:
    """Step-function policy rates per currency, from the bundled estimate table or a DataFrame."""

    def __init__(self, table: pd.DataFrame | None = None):
        if table is None:
            with resources.files("tradex.costs").joinpath("policy_rates.csv").open() as fh:
                table = pd.read_csv(fh, comment="#")
        table = table.copy()
        table["date"] = pd.to_datetime(table["date"], utc=True)
        self._by_ccy = {c: g.sort_values("date").set_index("date")["rate"] / 100.0 for c, g in table.groupby("currency")}

    def rate(self, ccy: str, ts: pd.Timestamp) -> float:
        s = self._by_ccy.get(ccy)
        if s is None:
            raise KeyError(f"no policy rate history for {ccy}; add it to policy_rates.csv")
        s = s[s.index <= ts]
        return float(s.iloc[-1]) if len(s) else float(self._by_ccy[ccy].iloc[0])


def rollover_days(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """Financing days charged by Oanda for holding over (t0, t1].

    Oanda books financing at 17:00 New York each trading day and charges three days on
    Friday to cover the weekend. Saturday and Sunday 17:00 are not charged.
    """
    a, b = t0.tz_convert(NY), t1.tz_convert(NY)
    days = 0
    d = a.date()
    while d <= b.date():
        cut = pd.Timestamp(dt.datetime.combine(d, dt.time(17, 0))).tz_localize(NY)
        if a < cut <= b:
            wd = cut.weekday()
            days += 3 if wd == 4 else (1 if wd < 4 else 0)
        d += dt.timedelta(days=1)
    return days


def next_rollover(ts: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    """Next 17:00 New York financing cut after ts and how many days it charges."""
    t = ts.tz_convert(NY)
    d = t.date()
    for _ in range(8):
        cut = pd.Timestamp(dt.datetime.combine(d, dt.time(17, 0))).tz_localize(NY)
        if cut > t and cut.weekday() < 5:
            return cut.tz_convert("UTC"), 3 if cut.weekday() == 4 else 1
        d += dt.timedelta(days=1)
    raise RuntimeError("no rollover found")


@dataclass
class OandaFxCosts:
    """Oanda SG standard pricing: spread only, plus daily financing.

    Financing = units x (rate differential - admin fee) x days / 365, converted to USD.
    Oanda's 2.5% admin fee (thread-1 design doc) is applied in full on both sides here,
    which is the conservative reading.
    """

    asset_class: str = "forex"
    spread_pips: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FX_SPREAD_PIPS))
    default_spread_pips: float = 2.5
    slippage_pips: float = 0.2
    admin_fee: float = 0.025
    rates: PolicyRates | None = None
    stress: float = 1.0

    def __post_init__(self):
        if self.rates is None:
            self.rates = PolicyRates()

    def fill(self, symbol, side, mid, ts):
        pip = pip_size(symbol)
        hs = 0.5 * self.spread_pips.get(symbol, self.default_spread_pips) * pip * self.stress
        sl = self.slippage_pips * pip * self.stress
        return mid + side * (hs + sl), {"spread": hs, "slippage": sl}

    def order_fees(self, symbol, side, qty, price, ts):
        return {}

    def annual_rate(self, symbol: str, direction: int, ts: pd.Timestamp) -> float:
        """Annual financing rate received (+) or paid (-) on the base-currency notional."""
        base, quote = split_pair(symbol)
        diff = self.rates.rate(base, ts) - self.rates.rate(quote, ts)
        return (diff if direction > 0 else -diff) - self.admin_fee

    def holding_cost(self, symbol, direction, qty, price, t0, t1, base_to_usd=1.0, borrowed_usd=0.0):
        days = rollover_days(t0, t1)
        if days == 0:
            return {}
        rate = self.annual_rate(symbol, direction, t1)
        return {"financing": -qty * base_to_usd * rate * days / 365.0}


def usd_per_unit(ccy: str, ts: pd.Timestamp | None = None, fx: dict[str, pd.Series] | None = None) -> float:
    """USD value of one unit of ``ccy``; uses supplied series when available, else a rough constant."""
    if ccy == "USD":
        return 1.0
    if fx and ccy in fx:
        s = fx[ccy]
        s = s[s.index <= ts] if ts is not None else s
        if len(s):
            return float(s.iloc[-1])
    return APPROX_USD_PER_UNIT[ccy]


def model_for(asset_class: str, **overrides) -> CostModel:
    if asset_class == "stocks":
        return MoomooStockCosts(**overrides)
    if asset_class == "forex":
        return OandaFxCosts(**overrides)
    raise ValueError(asset_class)
