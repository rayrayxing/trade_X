"""Payoff, P&L, break-evens and expiry / assignment handling.

All USD amounts are ``per-share figure x multiplier x contracts``. Fees are passed in or
computed by the caller from ``MoomooOptionCosts``; nothing here invents a fee. Where the
broker's fee for an exercise or assignment is not known, the settlement says so
(``fee_unknown``) instead of charging a guess.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tradex.options.contract import OptionContract, OptionPosition, Right

# OCC exercise-by-exception: an equity option that is in the money by $0.01 or more at expiry
# is exercised automatically unless the holder instructs otherwise.
AUTO_EXERCISE_THRESHOLD = 0.01
# A short ITM call with this little time value (per share) left is a candidate for early assignment.
MIN_SAFE_EXTRINSIC = 0.05

INF = math.inf


def expiry_value(contract: OptionContract, spot: float) -> float:
    """Value per share at expiry: the intrinsic value."""
    return contract.intrinsic(spot)


def pnl_at_premium(pos: OptionPosition, premium: float, fees: float = 0.0) -> float:
    """USD P&L if the position is closed at ``premium`` per share, net of ``fees`` (a positive cost)."""
    return pos.unrealized_pnl(premium) - fees


def pnl_at_expiry(pos: OptionPosition, spot: float, fees: float = 0.0) -> float:
    """USD P&L if held to expiry with the underlying at ``spot``."""
    return pnl_at_premium(pos, expiry_value(pos.contract, spot), fees)


def pnl_curve(pos: OptionPosition, spots: Iterable[float], fees: float = 0.0) -> pd.Series:
    """Expiry P&L across a range of underlying prices, indexed by price."""
    spots = list(spots)
    return pd.Series([pnl_at_expiry(pos, s, fees) for s in spots], index=pd.Index(spots, name="spot"), name="pnl_usd")


def breakeven(contract: OptionContract, entry_premium: float) -> float:
    """Underlying price at which the position breaks even at expiry, before fees (long or short)."""
    if contract.right is Right.CALL:
        return contract.strike + entry_premium
    return contract.strike - entry_premium


def max_loss(pos: OptionPosition) -> float:
    """Largest possible loss in USD at expiry before fees, as a positive number (``inf`` when unbounded)."""
    c = pos.contract
    if pos.is_long:
        return pos.entry_premium * pos.shares
    if c.right is Right.CALL:
        return INF                                   # a naked short call has no ceiling
    return (c.strike - pos.entry_premium) * abs(pos.shares)    # short put: underlying to zero


def max_profit(pos: OptionPosition) -> float:
    """Largest possible profit in USD at expiry before fees (``inf`` for a long call)."""
    c = pos.contract
    if not pos.is_long:
        return pos.entry_premium * abs(pos.shares)
    if c.right is Right.CALL:
        return INF
    return (c.strike - pos.entry_premium) * pos.shares


# --- expiry and assignment -------------------------------------------------------------

@dataclass(frozen=True)
class ExpirySettlement:
    """What happens to one position at the expiry close.

    ``kind`` is one of expired_worthless, sold_at_intrinsic, exercised, assigned.
    ``shares_delivered`` is signed: positive means the account receives shares. The stock leg
    settles at the strike (``stock_cash_usd``); ``option_pnl_usd`` is the option's own P&L at
    intrinsic before fees; ``stock_pnl_vs_spot_usd`` is what the stock leg is worth against the
    expiry close, so ``option_pnl_usd`` already equals the economic result of the whole event.
    """
    kind: str
    contracts: int
    spot: float
    intrinsic: float
    option_pnl_usd: float
    shares_delivered: int
    stock_cash_usd: float
    stock_pnl_vs_spot_usd: float
    fees_usd: float
    fee_unknown: bool
    note: str = ""

    @property
    def leaves_stock_position(self) -> bool:
        return self.shares_delivered != 0


def settle_expiry(pos: OptionPosition, spot: float, long_itm: str = "sell_at_intrinsic",
                  sell_fees_usd: float | None = None, exercise_fee_per_contract: float | None = None,
                  assignment_fee_per_contract: float | None = None) -> ExpirySettlement:
    """Settle ``pos`` at the expiry close with the underlying at ``spot``.

    - Out of the money (intrinsic below US$0.01): expires worthless, no fee.
    - Long, in the money, ``long_itm="sell_at_intrinsic"`` (default): treated as sold at
      intrinsic with ``sell_fees_usd`` (the closing-order fees the caller computed from the cost
      model). This is how the agent is meant to run: it never wants the shares.
    - Long, in the money, ``long_itm="exercise"``: exercised, shares delivered at the strike;
      fee is ``exercise_fee_per_contract`` x contracts, unknown when None.
    - Short, in the money: assigned. A short call delivers shares (a short stock position at the
      strike); a short put receives them. Fee is ``assignment_fee_per_contract`` x contracts.
    """
    if long_itm not in ("sell_at_intrinsic", "exercise"):
        raise ValueError("long_itm must be 'sell_at_intrinsic' or 'exercise'")
    c = pos.contract
    iv = expiry_value(c, spot)
    n, mult = pos.qty, c.multiplier
    opnl = pnl_at_premium(pos, iv)
    if iv < AUTO_EXERCISE_THRESHOLD:
        return ExpirySettlement("expired_worthless", pos.contracts, spot, iv, pnl_at_premium(pos, 0.0), 0, 0.0, 0.0,
                                0.0, False, "out of the money at expiry")
    if pos.is_long and long_itm == "sell_at_intrinsic":
        unknown = sell_fees_usd is None
        return ExpirySettlement("sold_at_intrinsic", pos.contracts, spot, iv, opnl, 0, 0.0, 0.0,
                                float(sell_fees_usd or 0.0), unknown,
                                "assumed closed at intrinsic; the real sale must be placed before the close")
    # shares change hands at the strike
    if c.right is Right.CALL:
        delivered = n * mult if pos.is_long else -n * mult       # long call receives, short call delivers
    else:
        delivered = -n * mult if pos.is_long else n * mult       # long put delivers, short put receives
    cash = -delivered * c.strike
    stock_vs_spot = delivered * (spot - c.strike)
    fee_pc = exercise_fee_per_contract if pos.is_long else assignment_fee_per_contract
    kind = "exercised" if pos.is_long else "assigned"
    return ExpirySettlement(kind, pos.contracts, spot, iv, opnl, delivered, cash, stock_vs_spot,
                            float(fee_pc * n) if fee_pc is not None else 0.0, fee_pc is None,
                            "naked call assigned: short stock position opened at the strike" if pos.is_naked_call else "")


def early_assignment_risk(pos: OptionPosition, spot: float, premium: float,
                          dividend: float | None = None, ex_dividend_within_days: int | None = None) -> str | None:
    """Reason a short call is at risk of early assignment, else None.

    The standard trigger: a call in the money whose remaining time value is smaller than the
    dividend it would let the holder capture by exercising before the ex-dividend date. A
    short call that is deep in the money with almost no extrinsic value is also at risk.
    """
    c = pos.contract
    if pos.is_long or c.right is not Right.CALL:
        return None
    intrinsic = c.intrinsic(spot)
    if intrinsic <= 0:
        return None
    extrinsic = max(premium - intrinsic, 0.0)
    if dividend and ex_dividend_within_days is not None and ex_dividend_within_days <= 1 and extrinsic < dividend:
        return (f"ITM short call with extrinsic {extrinsic:.2f} below the {dividend:.2f} dividend, "
                f"ex-date within {ex_dividend_within_days} day(s)")
    if extrinsic <= MIN_SAFE_EXTRINSIC:
        return f"ITM short call with only {extrinsic:.3f} of extrinsic value: early assignment likely"
    return None
