"""Option sizing: premium at risk for longs, gap-stressed loss for naked short calls, margin hook.

Three rules (Ray, 4 Oct 2026):

1. A long call or put can lose its whole premium, so it is sized on **premium at risk**: the
   debit plus opening and closing fees. A stop on the premium does not shrink the size,
   because options can gap through it (and a stop cannot be placed natively on moomoo US
   options in SIMULATE; the exit is monitored by the runner).
2. A naked short call has no ceiling, so it is sized on a **gap-stressed maximum loss**: the
   buy-to-close stop is assumed to trigger, and then the underlying gaps a further 20% past
   the stop level before the order fills. The loss is that fill against the credit received.
3. Margin is checked through an injected estimator. The default is a Reg-T style estimate for
   backtests; paper and live must inject the broker's own figure.

The gap model: with the underlying at ``S_stop`` when the stop triggers and ``P_stop`` the
premium there, a gap of ``g`` takes it to ``S_gap = S_stop (1 + g)``. With the contract's implied
volatility and time to expiry (``GapVol``) the call is repriced at ``S_gap`` and floored at
``max(S_gap - K, 0)`` plus the time value it still had at the stop (volatility spikes on gaps):
``P_gap = max(BS(S_gap), max(S_gap - K, 0) + max(P_stop - max(S_stop - K, 0), 0))``. Without
volatility inputs it falls back to the hard bound ``P_stop + (S_gap - S_stop)`` (a call gains at
most US$1 per US$1 of underlying), which is always safe and often very loose. Either way the loss
per contract is
``(P_gap x (1 + c) - credit) x multiplier`` plus fees on the open and the buy-to-close, where ``c`` is the
execution cushion (the cost model's half spread plus slippage: the buy-to-close pays the ask, not
the mid).

``tradex/risk/options.py`` in ``patches/options-order-guard.patch`` carries the same formula for the
protected risk gate; ``tests/options/test_patch.py`` checks the two agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from tradex.costs.models import MoomooOptionCosts
from tradex.options.contract import MULTIPLIER, OptionContract, Right
from tradex.options.pricing import bs_price, underlying_for_premium

GAP_PCT_FLOOR = 0.20          # the gap assumed past the stop; callers may raise it, never lower it
STOP_MULT_FLOOR = 1.0         # a buy-to-close stop multiple must exceed this (stop at the credit is no stop)


def default_costs() -> MoomooOptionCosts:
    return MoomooOptionCosts()


def order_fee_usd(costs: MoomooOptionCosts | None, side: int, contracts: int, premium: float,
                  symbol: str = "", ts: pd.Timestamp | None = None) -> float:
    """Total fees in USD for one option order of ``contracts`` at ``premium`` per share."""
    costs = costs or default_costs()
    return float(sum(costs.order_fees(symbol, side, contracts, premium, ts).values()))


# --- long options --------------------------------------------------------------------------

@dataclass(frozen=True)
class LongRisk:
    contracts: int
    premium: float
    premium_usd: float
    open_fees_usd: float
    close_fees_usd: float

    @property
    def max_loss_usd(self) -> float:
        """The most a long can lose: the debit plus both orders' fees."""
        return self.premium_usd + self.open_fees_usd + self.close_fees_usd


def long_premium_at_risk(premium: float, contracts: int, costs: MoomooOptionCosts | None = None,
                         multiplier: int = MULTIPLIER, exit_premium: float = 0.0) -> LongRisk:
    """Premium at risk for ``contracts`` longs bought at ``premium`` per share.

    ``exit_premium`` prices the closing order's fees (it barely matters: fees are almost all per
    contract). Fees on the closing order are included because a stop-out pays them.
    """
    if premium <= 0:
        raise ValueError("premium must be positive")
    if contracts < 1:
        raise ValueError("contracts must be at least 1")
    return LongRisk(contracts, premium, premium * multiplier * contracts,
                    order_fee_usd(costs, +1, contracts, premium), order_fee_usd(costs, -1, contracts, exit_premium))


def max_long_contracts(risk_budget_usd: float, premium: float, costs: MoomooOptionCosts | None = None,
                       multiplier: int = MULTIPLIER, exit_premium: float = 0.0) -> int:
    """Most long contracts whose premium at risk fits ``risk_budget_usd`` (0 when even one does not)."""
    if risk_budget_usd <= 0 or premium <= 0:
        return 0
    n = int(math.floor(risk_budget_usd / (premium * multiplier)))
    while n > 0 and long_premium_at_risk(premium, n, costs, multiplier, exit_premium).max_loss_usd > risk_budget_usd:
        n -= 1
    return n


# --- naked short calls -----------------------------------------------------------------------

@dataclass(frozen=True)
class NakedGapLoss:
    contracts: int
    strike: float
    credit: float                 # premium received per share
    stop_premium: float           # buy-to-close trigger, per share
    underlying_at_stop: float
    gap_pct: float
    underlying_after_gap: float
    premium_after_gap: float      # modelled value of the call after the gap, per share
    multiplier: int
    open_fees_usd: float
    close_fees_usd: float
    exec_cushion_pct: float = 0.0   # buy-to-close pays the ask plus slippage: this share above the modelled value

    @property
    def fill_after_gap(self) -> float:
        """Price the buy-to-close is assumed to fill at after the gap, per share."""
        return self.premium_after_gap * (1.0 + self.exec_cushion_pct)

    @property
    def stop_loss_usd(self) -> float:
        """Loss if the stop fills at its own level (no gap), before fees."""
        return (self.stop_premium - self.credit) * self.multiplier * self.contracts

    @property
    def gap_loss_usd(self) -> float:
        """Loss if the buy-to-close fills after the gap, before fees."""
        return (self.fill_after_gap - self.credit) * self.multiplier * self.contracts

    @property
    def max_loss_usd(self) -> float:
        """Gap-stressed maximum loss including fees: the figure naked calls are sized on."""
        return self.gap_loss_usd + self.open_fees_usd + self.close_fees_usd

    @property
    def per_contract_usd(self) -> float:
        return self.max_loss_usd / self.contracts


@dataclass(frozen=True)
class GapVol:
    """Volatility inputs for repricing the call after the gap: implied vol (a fraction) and years to expiry."""
    iv: float
    years: float
    rate: float = 0.0
    div_yield: float = 0.0


def premium_after_gap(strike: float, stop_premium: float, underlying_at_stop: float, gap_pct: float,
                      vol: GapVol | None = None) -> tuple[float, float]:
    """(underlying after the gap, call premium after the gap) per the module's conservative model."""
    s_gap = underlying_at_stop * (1.0 + gap_pct)
    if vol is None or vol.iv <= 0 or vol.years <= 0:
        return s_gap, stop_premium + (s_gap - underlying_at_stop)          # hard bound, no model needed
    extrinsic_at_stop = max(stop_premium - max(underlying_at_stop - strike, 0.0), 0.0)
    floor = max(s_gap - strike, 0.0) + extrinsic_at_stop
    return s_gap, max(bs_price(s_gap, strike, vol.years, vol.iv, Right.CALL, vol.rate, vol.div_yield), floor)


def naked_call_gap_loss(strike: float, credit: float, stop_premium: float, underlying_at_stop: float,
                        contracts: int = 1, gap_pct: float = GAP_PCT_FLOOR,
                        costs: MoomooOptionCosts | None = None, multiplier: int = MULTIPLIER,
                        vol: GapVol | None = None) -> NakedGapLoss:
    """Gap-stressed maximum loss of ``contracts`` naked short calls (see the module docstring)."""
    if gap_pct < GAP_PCT_FLOOR - 1e-12:
        raise ValueError(f"gap_pct {gap_pct:.2f} is below the {GAP_PCT_FLOOR:.0%} floor")
    if credit <= 0:
        raise ValueError("credit must be positive")
    if stop_premium < credit * STOP_MULT_FLOOR - 1e-12:
        raise ValueError("a buy-to-close stop cannot sit below the credit received")
    if underlying_at_stop <= 0 or contracts < 1:
        raise ValueError("underlying_at_stop must be positive and contracts at least 1")
    s_gap, p_gap = premium_after_gap(strike, stop_premium, underlying_at_stop, gap_pct, vol)
    cm = costs or default_costs()
    cushion = (cm.half_spread_pct + cm.slippage_pct) * cm.stress
    return NakedGapLoss(contracts, strike, credit, stop_premium, underlying_at_stop, gap_pct, s_gap, p_gap,
                        multiplier, order_fee_usd(cm, -1, contracts, credit),
                        order_fee_usd(cm, +1, contracts, p_gap * (1.0 + cushion)), cushion)


def max_naked_contracts(risk_budget_usd: float, strike: float, credit: float, stop_premium: float,
                        underlying_at_stop: float, gap_pct: float = GAP_PCT_FLOOR,
                        costs: MoomooOptionCosts | None = None, multiplier: int = MULTIPLIER,
                        vol: GapVol | None = None) -> int:
    """Most naked short calls whose gap-stressed loss fits ``risk_budget_usd``."""
    if risk_budget_usd <= 0:
        return 0
    one = naked_call_gap_loss(strike, credit, stop_premium, underlying_at_stop, 1, gap_pct, costs, multiplier, vol)
    n = int(math.floor(risk_budget_usd / max(one.max_loss_usd, 1e-9)))
    while n > 0 and naked_call_gap_loss(strike, credit, stop_premium, underlying_at_stop, n, gap_pct, costs,
                                        multiplier, vol).max_loss_usd > risk_budget_usd:
        n -= 1
    return n


def stop_premium_from_multiple(credit: float, multiple: float) -> float:
    """Buy-to-close trigger at ``multiple`` x the credit (2.0 means buy back when the option doubles)."""
    if multiple <= STOP_MULT_FLOOR:
        raise ValueError(f"stop multiple must exceed {STOP_MULT_FLOOR:g}: a stop at or below the credit is no stop")
    return credit * multiple


def effective_naked_stop(contract: OptionContract, credit: float, stop_premium: float, spot: float,
                         iv: float, years: float, underlying_stop: float | None = None,
                         rate: float = 0.0, div_yield: float = 0.0) -> tuple[float, float] | None:
    """(underlying level, premium) at which the buy-to-close stop will actually trigger.

    Two triggers can be set: a premium level and an underlying level. Whichever the market
    reaches first fires, which for a call is the lower underlying level. Returns None when the
    premium trigger cannot be located (no IV, or an unreachable level): the caller must then
    refuse to size the trade, never guess.
    """
    if contract.right is not Right.CALL:
        raise ValueError("naked-call stop helper applies to calls")
    if iv is None or iv <= 0 or years <= 0:
        return None
    s_prem = underlying_for_premium(stop_premium, contract.strike, years, iv, Right.CALL, rate, div_yield)
    if s_prem is None:
        return None
    s_prem = max(s_prem, spot)                      # a stop above the credit cannot trigger below spot
    if underlying_stop is not None and underlying_stop > 0 and underlying_stop < s_prem:
        p_at = max(bs_price(underlying_stop, contract.strike, years, iv, Right.CALL, rate, div_yield), credit)
        return underlying_stop, min(p_at, stop_premium)
    return s_prem, stop_premium


# --- margin hook ---------------------------------------------------------------------------

class MarginEstimator(Protocol):
    """Initial margin in USD for ``contracts`` (signed) of ``contract`` at ``premium`` per share.

    Backtests use ``RegTMarginEstimator``. Paper and live inject the broker's own figure
    through ``BrokerMarginEstimator`` (for moomoo: the OpenD trading-info / max-quantity query
    for the SIMULATE account), so the number that gates a trade is the number the venue enforces.
    """

    def initial_margin(self, contract: OptionContract, contracts: int, premium: float,
                       underlying_price: float) -> float: ...


@dataclass
class RegTMarginEstimator:
    """Reg-T style initial margin estimate (a floor for backtests, not the broker's figure).

    Long options: the premium, paid in full. Short call: the greater of ``naked_pct`` x
    underlying - OTM amount + premium, and ``floor_pct`` x underlying + premium. Short put: the
    same with the OTM amount on the other side and ``floor_pct`` x strike. Per share, times the
    multiplier and the contract count.
    """
    naked_pct: float = 0.20
    floor_pct: float = 0.10

    def initial_margin(self, contract, contracts, premium, underlying_price):
        n, mult = abs(contracts), contract.multiplier
        if contracts > 0:
            return premium * mult * n
        s, k = underlying_price, contract.strike
        if contract.right is Right.CALL:
            otm = max(k - s, 0.0)
            per_share = max(self.naked_pct * s - otm + premium, self.floor_pct * s + premium)
        else:
            otm = max(s - k, 0.0)
            per_share = max(self.naked_pct * s - otm + premium, self.floor_pct * k + premium)
        return per_share * mult * n


@dataclass
class BrokerMarginEstimator:
    """Wraps a callable ``fn(contract, contracts, premium, underlying_price) -> USD`` backed by the broker."""
    fn: Callable[[OptionContract, int, float, float], float]

    def initial_margin(self, contract, contracts, premium, underlying_price):
        return float(self.fn(contract, contracts, premium, underlying_price))


def margin_max_contracts(estimator: MarginEstimator, free_margin_usd: float, contract: OptionContract,
                         premium: float, underlying_price: float, short: bool = True, cap: int = 10_000) -> int:
    """Most contracts the free margin can carry (feeds ``BookState.margin_max_qty`` in the risk gate)."""
    if free_margin_usd <= 0:
        return 0
    sign = -1 if short else 1
    one = estimator.initial_margin(contract, sign, premium, underlying_price)
    if one <= 0:
        return cap
    n = int(math.floor(free_margin_usd / one))
    while n > 0 and estimator.initial_margin(contract, sign * n, premium, underlying_price) > free_margin_usd:
        n -= 1
    return min(n, cap)


# --- one-call sizing -------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionSize:
    contracts: int
    binding: str                      # confidence | margin | none (zero contracts)
    risk_usd: float                   # premium at risk (long) or gap-stressed loss (naked)
    margin_usd: float
    detail: object = None             # LongRisk or NakedGapLoss


def size_long(risk_budget_usd: float, premium: float, costs: MoomooOptionCosts | None = None,
              multiplier: int = MULTIPLIER, estimator: MarginEstimator | None = None,
              contract: OptionContract | None = None, free_margin_usd: float | None = None,
              underlying_price: float | None = None) -> OptionSize:
    n = max_long_contracts(risk_budget_usd, premium, costs, multiplier)
    binding = "confidence"
    if n and estimator is not None and free_margin_usd is not None and contract is not None:
        m = margin_max_contracts(estimator, free_margin_usd, contract, premium, underlying_price or contract.strike, short=False)
        if m < n:
            n, binding = m, "margin"
    if n < 1:
        return OptionSize(0, "none", 0.0, 0.0)
    risk = long_premium_at_risk(premium, n, costs, multiplier)
    margin = estimator.initial_margin(contract, n, premium, underlying_price or 0.0) if estimator and contract else risk.premium_usd
    return OptionSize(n, binding, risk.max_loss_usd, margin, risk)


def size_naked_call(risk_budget_usd: float, contract: OptionContract, credit: float, stop_premium: float,
                    underlying_at_stop: float, underlying_price: float, gap_pct: float = GAP_PCT_FLOOR,
                    costs: MoomooOptionCosts | None = None, estimator: MarginEstimator | None = None,
                    free_margin_usd: float | None = None, vol: GapVol | None = None) -> OptionSize:
    """Contracts of a naked short call that fit the gap-stressed risk budget and the free margin."""
    estimator = estimator or RegTMarginEstimator()
    n = max_naked_contracts(risk_budget_usd, contract.strike, credit, stop_premium, underlying_at_stop,
                            gap_pct, costs, contract.multiplier, vol)
    binding = "confidence"
    if n and free_margin_usd is not None:
        m = margin_max_contracts(estimator, free_margin_usd, contract, credit, underlying_price, short=True)
        if m < n:
            n, binding = m, "margin"
    if n < 1:
        return OptionSize(0, "none", 0.0, 0.0)
    gl = naked_call_gap_loss(contract.strike, credit, stop_premium, underlying_at_stop, n, gap_pct, costs,
                             contract.multiplier, vol)
    return OptionSize(n, binding, gl.max_loss_usd, estimator.initial_margin(contract, -n, credit, underlying_price), gl)
