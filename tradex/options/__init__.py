"""Options lane: long calls, long puts and naked short calls on US equities.

Contracts and positions (x100 multiplier), payoff and expiry/assignment handling, sizing
(premium at risk for longs, 20%-gap-stressed loss for naked calls, a margin hook), a strategy
spec format built on ``tradex.strategy.spec`` and a backtester that takes chain, IV and Greeks
from injected providers. Nothing here places an order or fetches data; see ``providers.py``
for the interface OpenD has to satisfy and ``patches/README.md`` for the protected-file changes
(risk gate, order guard) that make the lane tradable.
"""
from tradex.options.contract import (MULTIPLIER, Greeks, OptionContract, OptionPosition, OptionQuote,  # noqa: F401
                                     Right)
from tradex.options.payoff import breakeven, pnl_at_expiry, settle_expiry  # noqa: F401
from tradex.options.sizing import (GAP_PCT_FLOOR, long_premium_at_risk, naked_call_gap_loss,  # noqa: F401
                                   size_long, size_naked_call)
from tradex.options.spec import OptionStrategySpec  # noqa: F401
