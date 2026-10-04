"""Currency and factor exposure, expected shortfall and stress replays (Ray, 4 Oct 2026).

Three layers, the standard on bank trading desks:

1. **Net open position per currency.** Every position is split into its currency legs in
   USD terms. A USDJPY long of 10,000 USD is +10,000 USD and -10,000 USD-worth of JPY; a
   EURUSD short is -EUR, +USD. The book is in USD, so USD legs carry no currency risk and
   the risk sits in the non-USD legs. Stocks become an equity factor per symbol.
2. **Expected shortfall** (average loss on the worst 2.5% of days, the measure Basel moved
   banks to in place of VaR), by historical simulation over factor returns, or from an
   exponentially weighted covariance when history is short. Each new plan is charged its
   *marginal* expected shortfall: a trade that offsets the book is cheap, a trade that
   repeats an existing bet (USDJPY long plus EURUSD short = one dollar bet) is charged as
   the bigger bet it creates.
3. **Named stress replays.** Shock vectors for the 2024 yen carry unwind, the 2015 franc
   unpeg, March 2020 and a single-stock earnings gap catch the tails the history misses.

Factor names: ``FX:JPY`` is the USD value of one unit of JPY (its return is JPY's move
against USD); ``EQ:AAPL`` is the stock's own return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from tradex.costs.models import split_pair, usd_per_unit

BASE = "USD"
_Z975 = NormalDist().inv_cdf(0.975)
_ES975_NORMAL = NormalDist().pdf(_Z975) / 0.025       # ES/sigma for a normal at 97.5%


@dataclass
class Leg:
    symbol: str
    asset_class: str
    direction: int
    qty: float
    price: float                      # in quote currency
    stop: float | None = None
    account: str = "agent"


def factor_exposures(leg: Leg, fx: dict[str, pd.Series] | None = None, ts: pd.Timestamp | None = None
                     ) -> dict[str, float]:
    """USD exposure of one position to each risk factor (positive = gains when the factor rises)."""
    if leg.asset_class == "stocks":
        return {f"EQ:{leg.symbol}": leg.direction * leg.qty * leg.price}
    base, quote = split_pair(leg.symbol)
    out: dict[str, float] = {}
    base_usd = leg.qty * usd_per_unit(base, ts, fx)
    quote_usd = leg.qty * leg.price * usd_per_unit(quote, ts, fx)
    if base != BASE:
        out[f"FX:{base}"] = out.get(f"FX:{base}", 0.0) + leg.direction * base_usd
    if quote != BASE:
        out[f"FX:{quote}"] = out.get(f"FX:{quote}", 0.0) - leg.direction * quote_usd
    return out


def book_exposures(legs: list[Leg], fx=None, ts=None) -> dict[str, float]:
    tot: dict[str, float] = {}
    for leg in legs:
        for k, v in factor_exposures(leg, fx, ts).items():
            tot[k] = tot.get(k, 0.0) + v
    return {k: v for k, v in tot.items() if abs(v) > 1e-9}


def net_open_position(legs: list[Leg], fx=None, ts=None) -> dict[str, float]:
    """Net USD-equivalent position per currency, USD excluded (the bank-desk net open position)."""
    return {k[3:]: v for k, v in book_exposures(legs, fx, ts).items() if k.startswith("FX:")}


def stop_risk_by_currency(legs: list[Leg], fx=None, ts=None) -> dict[str, float]:
    """Each forex position's loss-to-stop in USD, attributed to its non-USD currencies.

    A cross like EURJPY splits its stop risk equally between EUR and JPY."""
    out: dict[str, float] = {}
    for leg in legs:
        if leg.asset_class != "forex" or leg.stop is None:
            continue
        base, quote = split_pair(leg.symbol)
        risk = max(0.0, leg.direction * (leg.price - leg.stop)) * leg.qty * usd_per_unit(quote, ts, fx)
        ccys = [c for c in (base, quote) if c != BASE]
        for c in ccys:
            out[c] = out.get(c, 0.0) + risk / len(ccys)
    return out


def gross_net_open_position(nop: dict[str, float]) -> float:
    """Basel shorthand: the larger of summed net longs and summed net shorts."""
    longs = sum(v for v in nop.values() if v > 0)
    shorts = -sum(v for v in nop.values() if v < 0)
    return max(longs, shorts)


# --- expected shortfall ----------------------------------------------------------------

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Exponentially weighted covariance (RiskMetrics decay 0.94): recent days count most."""
    r = returns.dropna(how="all").fillna(0.0)
    n = len(r)
    if n == 0:
        return pd.DataFrame(0.0, index=returns.columns, columns=returns.columns)
    w = lam ** np.arange(n)[::-1]
    w /= w.sum()
    x = r.to_numpy() - np.average(r.to_numpy(), axis=0, weights=w)
    cov = (x * w[:, None]).T @ x
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


@dataclass
class ESModel:
    """Expected shortfall at 97.5% over one day, from daily factor returns.

    With at least ``min_hist`` days of history it uses historical simulation over the last
    ``window`` days; otherwise a normal approximation on the EWMA covariance.
    """
    returns: pd.DataFrame                 # daily returns, one column per factor name
    alpha: float = 0.975
    window: int = 500
    min_hist: int = 250
    lam: float = 0.94
    _cov: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def es(self, exposures: dict[str, float]) -> float:
        """Expected shortfall in USD (a positive number is a loss)."""
        cols = [k for k in exposures if k in self.returns.columns]
        if not cols:
            return 0.0
        e = np.array([exposures[k] for k in cols])
        r = self.returns[cols].dropna(how="all").fillna(0.0).tail(self.window)
        if len(r) >= self.min_hist:
            pnl = r.to_numpy() @ e
            k = max(1, int(np.ceil(len(pnl) * (1 - self.alpha))))
            return float(-np.sort(pnl)[:k].mean())
        if self._cov is None:
            self._cov = ewma_cov(self.returns, self.lam)
        cov = self._cov.loc[cols, cols].to_numpy()
        sigma = float(np.sqrt(max(e @ cov @ e, 0.0)))
        return sigma * _ES975_NORMAL

    def marginal(self, book: dict[str, float], add: dict[str, float]) -> float:
        """How much expected shortfall the book gains by adding ``add`` (negative = it diversifies)."""
        merged = dict(book)
        for k, v in add.items():
            merged[k] = merged.get(k, 0.0) + v
        return self.es(merged) - self.es(book)

    def missing(self, exposures: dict[str, float]) -> list[str]:
        return [k for k in exposures if k not in self.returns.columns]


# --- stress replays ----------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    shocks: dict[str, float]              # factor -> return over the episode
    default_equity: float = 0.0           # applied to any EQ: factor not listed
    single_name_gap: float = 0.0          # the largest single stock gaps this much against the book
    note: str = ""

    def pnl(self, exposures: dict[str, float]) -> float:
        out = 0.0
        if self.single_name_gap:
            eq = [abs(v) for k, v in exposures.items() if k.startswith("EQ:")]
            out -= max(eq, default=0.0) * self.single_name_gap
        for k, v in exposures.items():
            if k in self.shocks:
                out += v * self.shocks[k]
            elif k.startswith("EQ:"):
                out += v * self.default_equity
        return out


def worst_stress(exposures: dict[str, float], scenarios: list[Scenario]) -> tuple[str, float]:
    """Name and USD P&L of the worst scenario for this book (negative = loss)."""
    if not scenarios:
        return "none", 0.0
    res = [(s.name, s.pnl(exposures)) for s in scenarios]
    return min(res, key=lambda x: x[1])


def scenarios_from_config(items: list[dict]) -> list[Scenario]:
    return [Scenario(name=i["name"], shocks=dict(i.get("shocks", {})),
                     default_equity=float(i.get("default_equity", 0.0)),
                     single_name_gap=float(i.get("single_name_gap", 0.0)), note=i.get("note", "")) for i in items]


def fx_factor_returns(usd_per_unit_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily returns of each currency against USD, as ``FX:<ccy>`` columns."""
    cols = {f"FX:{c}": s.resample("1D").last().dropna().pct_change() for c, s in usd_per_unit_series.items()}
    return pd.DataFrame(cols).dropna(how="all")
