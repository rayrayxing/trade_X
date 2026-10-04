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

    From moomoo SG's fee page (https://www.moomoo.com/sg/support/topic5_76, checked
    2026-10-04): zero commission, US$0.99 platform fee per order plus 9% GST, settlement
    US$0.003/share capped at 1% of the order's value, CAT US$0.000003/share. SEC Section 31
    rate US$20.60 per million of sale proceeds from 4 Apr 2026 (SEC fee rate advisory
    2026-2); FINRA TAF US$0.000195/share capped at US$9.79 for 2026, rising to
    US$0.000232 and US$11.61 on 1 Jan 2027 (SR-FINRA-2024-019).
    """

    asset_class: str = "stocks"
    commission: float = 0.0
    platform_fee: float = 0.99
    gst: float = 0.09                       # on commission and platform fee
    settlement_fee_per_share: float = 0.003
    settlement_cap_pct: float = 0.01        # of the order's value
    cat_fee_per_share: float = 0.000003
    sec_fee_rate: float = 20.60e-6          # USD per USD of sale proceeds
    finra_taf_per_share: float = 0.000195   # sells only
    finra_taf_max: float = 9.79
    min_regulatory_fee: float = 0.01        # SEC and TAF each
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
        fees = {"platform": self.platform_fee * (1 + self.gst), "commission": self.commission * (1 + self.gst),
                "settlement": min(self.settlement_fee_per_share * qty, self.settlement_cap_pct * qty * price),
                "cat": self.cat_fee_per_share * qty}
        if side < 0:
            fees["sec"] = max(self.sec_fee_rate * qty * price, self.min_regulatory_fee)
            fees["finra_taf"] = max(min(self.finra_taf_per_share * qty, self.finra_taf_max), self.min_regulatory_fee)
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

# BACKTEST-ONLY default spreads (pips). Paper/live never read this table: they take the
# spread from a live RateSource fed by the Oanda stream. For backtests prefer spreads
# measured from Oanda bid/ask candles (tradex.data.oanda.measured_spread_pips).
DEFAULT_FX_SPREAD_PIPS = {
    "EUR_USD": 1.4, "USD_JPY": 1.4, "GBP_USD": 2.0, "AUD_USD": 1.4,
    "EUR_JPY": 2.0, "GBP_JPY": 3.0, "USD_CHF": 1.8, "USD_CAD": 2.0, "EUR_GBP": 1.6,
}

STRICT_MODES = ("paper", "live")


class RateMissing(RuntimeError):
    """A live FX rate or spread is unavailable in paper/live: the caller must block the trade and alert."""


class RateSource(Protocol):
    """Live quotes (from the Oanda stream) behind FX conversion and spreads in paper/live."""

    def usd_per_unit(self, ccy: str, ts: pd.Timestamp | None = None) -> float: ...

    def spread_pips(self, symbol: str, ts: pd.Timestamp | None = None) -> float: ...


_RUN_MODE = "backtest"
_RATE_SOURCE: RateSource | None = None


def configure_run_mode(mode: str, source: RateSource | None = None) -> None:
    """Set the process run mode. In paper/live, `usd_per_unit` uses `source` and never the constants below."""
    global _RUN_MODE, _RATE_SOURCE
    if mode in STRICT_MODES and source is None:
        raise ValueError(f"{mode} mode needs a live RateSource")
    _RUN_MODE, _RATE_SOURCE = mode, source


# BACKTEST-ONLY rough USD value of one unit of each currency, used when no rate series is supplied.
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
    mode: str = "backtest"
    spread_source: RateSource | None = None   # required in paper/live

    def __post_init__(self):
        if self.rates is None:
            self.rates = PolicyRates()
        if self.mode in STRICT_MODES and self.spread_source is None:
            raise ValueError(f"{self.mode} mode needs a live spread_source; defaults are backtest-only")

    def _spread(self, symbol: str, ts) -> float:
        if self.mode in STRICT_MODES:
            return self.spread_source.spread_pips(symbol, ts)   # raises RateMissing; no fallback
        return self.spread_pips.get(symbol, self.default_spread_pips)

    def fill(self, symbol, side, mid, ts):
        pip = pip_size(symbol)
        hs = 0.5 * self._spread(symbol, ts) * pip * self.stress
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
    """USD value of one unit of ``ccy``: supplied series, then (paper/live) the live source, else a backtest constant."""
    if ccy == "USD":
        return 1.0
    if fx and ccy in fx:
        s = fx[ccy]
        s = s[s.index <= ts] if ts is not None else s
        if len(s):
            return float(s.iloc[-1])
    if _RUN_MODE in STRICT_MODES:
        return _RATE_SOURCE.usd_per_unit(ccy, ts)               # raises RateMissing; no fallback
    return APPROX_USD_PER_UNIT[ccy]


def model_for(asset_class: str, **overrides) -> CostModel:
    """Cost model for an asset class. For forex in paper/live pass ``mode=`` and ``spread_source=``."""
    if asset_class == "stocks":
        return MoomooStockCosts(**overrides)
    if asset_class == "forex":
        return OandaFxCosts(**overrides)
    if asset_class == "options":
        return MoomooOptionCosts(**overrides)
    raise ValueError(asset_class)


# --- US options at Moomoo SG -------------------------------------------------------------

@dataclass
class MoomooOptionCosts:
    """Moomoo SG US-options costs, fixed plan (Ray's phase-1 brief, 4 Oct 2026).

    Per order: commission US$0.65/contract (min US$1.99) and platform fee US$0.30/contract
    (min US$0.99), both plus 9% GST; options regulatory fee (ORF) US$0.013/contract; OCC
    clearing US$0.02/contract capped at US$55 per trade; CAT US$0.0003/contract. Sells also
    pay SEC Section 31 (0.0000206 x premium value) and FINRA TAF US$0.00279/contract. About
    US$3.28 per order for one contract, falling to about US$1.07 per contract at ten.
    ``qty`` is contracts and ``price`` the premium per share; one contract covers 100 shares.
    """

    asset_class: str = "options"
    multiplier: int = 100
    commission_per_contract: float = 0.65
    commission_min: float = 1.99
    platform_per_contract: float = 0.30
    platform_min: float = 0.99
    gst: float = 0.09
    orf_per_contract: float = 0.013
    occ_per_contract: float = 0.02
    occ_max: float = 55.0
    cat_per_contract: float = 0.0003
    sec_fee_rate: float = 20.60e-6         # USD per USD of sale proceeds
    finra_taf_per_contract: float = 0.00279
    half_spread_pct: float = 0.02          # of premium; options spreads are wide, measure before trusting
    slippage_pct: float = 0.01
    stress: float = 1.0

    def fill(self, symbol, side, mid, ts):
        hs = self.half_spread_pct * mid * self.stress
        sl = self.slippage_pct * mid * self.stress
        return mid + side * (hs + sl), {"spread": hs, "slippage": sl}

    def order_fees(self, symbol, side, qty, price, ts):
        commission = max(self.commission_per_contract * qty, self.commission_min)
        platform = max(self.platform_per_contract * qty, self.platform_min)
        fees = {"commission": commission, "platform": platform, "gst": self.gst * (commission + platform),
                "orf": self.orf_per_contract * qty, "occ": min(self.occ_per_contract * qty, self.occ_max),
                "cat": self.cat_per_contract * qty}
        if side < 0:
            fees["sec"] = self.sec_fee_rate * qty * price * self.multiplier
            fees["finra_taf"] = self.finra_taf_per_contract * qty
        return fees

    def holding_cost(self, symbol, direction, qty, price, t0, t1, base_to_usd=1.0, borrowed_usd=0.0):
        return {}
