"""Pre-trade checks run on every finalised plan before it reaches the broker.

Short-side readiness (recommendation 5): a stock short needs shares to borrow at a sane
fee, no short-sale restriction in force (Rule 201: after a 10% drop, shorts must print
above the best bid, so a market short may not fill), and no squeeze set-up. Sanity
checks catch fat-finger orders. Each check returns a reason string when it blocks.

Live and paper read borrow data from the broker at run time; the replay harness and
virtual books use whatever ``ShortInfo`` table they are given. Unknown borrow data
blocks a live short and is allowed (and labelled) in simulation.

Protected path: thresholds come from config/risk/policy.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShortInfo:
    shortable: bool = True
    borrow_fee_annual: float | None = None      # e.g. 0.005 = 0.5% a year
    ssr_active: bool = False
    short_interest_pct_float: float | None = None
    days_to_cover: float | None = None


@dataclass
class ShortPolicy:
    max_borrow_fee_annual: float = 0.20
    max_short_interest_pct_float: float = 20.0
    max_days_to_cover: float = 5.0
    block_when_ssr: bool = True
    require_known_borrow: bool = True            # live: unknown borrow blocks; simulation sets False


def short_check(asset_class: str, direction: int, info: ShortInfo | None, pol: ShortPolicy) -> str | None:
    if asset_class != "stocks" or direction >= 0:
        return None
    if info is None:
        return "borrow data unknown" if pol.require_known_borrow else None
    if not info.shortable:
        return "no shares available to borrow"
    if info.borrow_fee_annual is not None and info.borrow_fee_annual > pol.max_borrow_fee_annual:
        return f"borrow fee {info.borrow_fee_annual:.1%} a year is above {pol.max_borrow_fee_annual:.0%}"
    if pol.block_when_ssr and info.ssr_active:
        return "short-sale restriction (Rule 201) in force today"
    if info.short_interest_pct_float is not None and info.short_interest_pct_float > pol.max_short_interest_pct_float:
        return f"short interest {info.short_interest_pct_float:.0f}% of float: squeeze risk"
    if info.days_to_cover is not None and info.days_to_cover > pol.max_days_to_cover:
        return f"{info.days_to_cover:.1f} days to cover: squeeze risk"
    return None


@dataclass
class SanityPolicy:
    max_order_notional_x_equity: float = 15.0    # above the 14:1 forex ceiling means a bug, not a trade
    max_price_deviation: float = 0.05            # plan price vs last trade


def sanity_check(qty: float, price: float, base_to_usd: float, equity: float, last_price: float | None,
                 pol: SanityPolicy) -> str | None:
    if qty <= 0 or price <= 0:
        return "non-positive quantity or price"
    notional = qty * price * base_to_usd
    if equity > 0 and notional > pol.max_order_notional_x_equity * equity:
        return f"order notional {notional:,.0f} USD exceeds {pol.max_order_notional_x_equity:g}x equity"
    if last_price and abs(price / last_price - 1) > pol.max_price_deviation:
        return f"plan price {price} is more than {pol.max_price_deviation:.0%} from last {last_price}"
    return None
