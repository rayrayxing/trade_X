"""Pyramiding and vol-targeted compounding, as sizing SUGGESTIONS (profit spec P5-P6).

Nothing here sizes a trade. A suggestion is a number the core hands to the risk gate, which
reviews it like any other request and may shrink it (heat, currency, leverage, margin, expected
shortfall, stress, tier). The module imports no gate and no execution code and builds no order;
the only thing that leaves it is a ``SizingSuggestion`` (or, for an add-on, a ``TradePlan`` the
gate reviews in the normal way, with the suggestion as its ceiling).

Pyramiding (adds to winners only)
- an add becomes eligible when the open trade is ``add_at_r[n]`` R in profit on its INITIAL risk;
- the stop must already be at or beyond breakeven, so the base trade can no longer lose;
- no add while volatility has expanded past ``max_atr_expansion`` times its value at entry;
- the add is sized so that, with every unit stopped at the SHARED current stop, the worst case of
  the whole position is a loss no bigger than ``max_total_risk_fraction`` of the base trade's
  initial risk in dollars, and the add alone risks at most ``add_risk_fraction`` of it. Profit
  already locked by the stop counts toward the budget, which is why pyramids get bigger as the
  trail tightens and never grow the worst case.

Vol-targeted compounding
- risk is a percentage of CURRENT equity (compounding is the gate sizing off the live account);
  this adds a multiplier ``target_vol / realised_vol`` clipped to ``[min_scale, max_scale]`` and the
  existing drawdown de-risk. Upsizing above 1.0 needs a calibrated edge (``CalibratedProbability``
  with ``calibrated=True``) and is capped by fractional Kelly on its lower confidence bound; with
  no calibrated edge the scale never exceeds 1.0. With too little equity history it says so and
  returns 1.0.
- the gate today takes a size factor of at most 1.0 (``size_factor`` shrinks, never grows), so
  ``SizingSuggestion.shrink_factor`` is what the core can pass with no gate change; the part
  above 1.0 is informational until the gate patch in patches/ov-exits is applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import pandas as pd

from tradex.core.records import TradePlan
from tradex.profit.calibration import CalibratedProbability
from tradex.risk.sizing import drawdown_scale, kelly_fraction


@dataclass(frozen=True)
class SizingSuggestion:
    kind: str                                   # pyramid | vol_target
    scale: float = 1.0                          # multiple of the gate's own size (vol_target)
    qty: float = 0.0                            # units to add (pyramid)
    risk_usd: float = 0.0                       # worst-case dollars the suggestion puts at risk to the shared stop
    risk_pct: float | None = None               # suggested equity percent at risk (vol_target)
    reasons: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()              # why the suggestion is zero / unchanged
    authority: str = "suggestion"               # the risk gate decides; this is never an order

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def shrink_factor(self) -> float:
        """What the core may pass as the gate's size factor today: never above 1.0."""
        return min(1.0, max(0.0, self.scale))


# --- pyramiding -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PyramidPolicy:
    add_at_r: tuple[float, ...] = (1.0, 2.0)
    max_adds: int = 2
    add_risk_fraction: float = 0.5
    max_total_risk_fraction: float = 1.0
    require_breakeven_stop: bool = True
    max_atr_expansion: float = 2.0


@dataclass(frozen=True)
class OpenTrade:
    direction: int
    entry_price: float                          # average entry of what is held
    initial_stop: float
    stop: float                                 # in force now: shared by any add
    qty: float                                  # held now
    initial_qty: float                          # held when the trade was first opened (the R unit's size)
    price: float                                # last close
    adds_done: int = 0
    atr: float | None = None
    entry_atr: float | None = None
    base_to_usd: float = 1.0                    # USD per price unit per unit of quantity (quote currency rate for forex)


def pyramid_suggestion(trade: OpenTrade, policy: PyramidPolicy | None = None, equity: float | None = None
                       ) -> SizingSuggestion:
    pol = policy or PyramidPolicy()
    d = trade.direction
    risk_unit = abs(trade.entry_price - trade.initial_stop)
    block: list[str] = []
    if risk_unit <= 0 or trade.initial_qty <= 0 or trade.qty <= 0:
        return SizingSuggestion("pyramid", blockers=("the trade has no risk unit to add against",))
    if trade.adds_done >= pol.max_adds or trade.adds_done >= len(pol.add_at_r):
        return SizingSuggestion("pyramid", blockers=(f"{trade.adds_done} adds already made (max {pol.max_adds})",))
    open_r = d * (trade.price - trade.entry_price) / risk_unit
    need = pol.add_at_r[trade.adds_done]
    if open_r < need:
        block.append(f"open profit {open_r:+.2f}R is under the {need:g}R an add needs")
    locked = d * (trade.stop - trade.entry_price)              # price units of profit the stop already holds
    if pol.require_breakeven_stop and locked < 0:
        block.append("the stop is not yet at breakeven")
    if trade.atr and trade.entry_atr and trade.atr / trade.entry_atr > pol.max_atr_expansion:
        block.append(f"volatility is {trade.atr / trade.entry_atr:.1f}x its entry level")
    unit = d * (trade.price - trade.stop)                      # what each added unit loses if stopped
    if unit <= 0:
        block.append("price is not beyond the stop")
    if block:
        return SizingSuggestion("pyramid", blockers=tuple(block))
    init_risk_usd = risk_unit * trade.initial_qty * trade.base_to_usd
    cap_total = pol.max_total_risk_fraction * init_risk_usd + trade.qty * locked * trade.base_to_usd
    cap_add = pol.add_risk_fraction * init_risk_usd
    per_unit = unit * trade.base_to_usd
    q = max(0.0, min(cap_total, cap_add) / per_unit)
    if q <= 0:
        return SizingSuggestion("pyramid", blockers=("no risk budget left for an add",))
    why = (f"open {open_r:+.2f}R, stop {locked / risk_unit:+.2f}R from entry",
           f"adds {q:.4g} units at {unit / risk_unit:.2f}R of stop distance each",
           f"worst case of the whole position {max(0.0, q * per_unit - trade.qty * locked * trade.base_to_usd):.2f} USD "
           f"(budget {pol.max_total_risk_fraction * init_risk_usd:.2f})")
    return SizingSuggestion("pyramid", qty=q, risk_usd=round(q * per_unit, 2),
                            risk_pct=None if not equity else round(100.0 * q * per_unit / equity, 4), reasons=why)


def pyramid_plan(base: TradePlan, trade: OpenTrade, sug: SizingSuggestion, decision_id: str, time: str
                 ) -> TradePlan | None:
    """The add-on as a plan for the gate to review: market entry at the last close, the SHARED stop, the base plan's
    targets that are still ahead. The gate sizes it; the core then caps its quantity at ``sug.qty``."""
    if not sug.ok or sug.qty <= 0:
        return None
    d = base.direction
    ahead = [t for t in base.targets if d * (t - trade.price) > 0]
    if not ahead:
        return None
    unit = d * (trade.price - trade.stop)
    cost_u = base.cost_r * abs(base.entry_price - base.stop)
    t1 = ahead[0]
    rr = (abs(t1 - trade.price) - cost_u) / (unit + cost_u)
    p = base.p_target
    return replace(base, decision_id=decision_id, time=time, entry_type="market", entry_price=trade.price,
                   stop=trade.stop, targets=ahead[:2], reward_risk=round(rr, 4), ev_r=round(p * rr - (1 - p), 4),
                   cost_r=round(cost_u / unit, 4),
                   invalidation=f"add-on to {base.decision_id}: price trades through the shared stop {trade.stop:.5g}")


def cap_to_suggestion(verdict_qty: float, sug: SizingSuggestion) -> float:
    """The gate's quantity, lowered to the suggestion's ceiling. The suggestion can only reduce what the gate allows."""
    return min(verdict_qty, sug.qty) if sug.kind == "pyramid" else verdict_qty


# --- vol targeting and compounding ---------------------------------------------------------------------

@dataclass(frozen=True)
class VolTargetPolicy:
    target_annual_vol: float = 0.10             # of equity
    lookback: int = 60                          # daily observations
    min_obs: int = 30
    min_scale: float = 0.25
    max_scale: float = 1.5
    periods_per_year: int = 252
    drawdown_halve_at: float = 0.20
    kelly_fraction: float = 0.25                # fraction of Kelly (lower bound) that caps the upsizing


def realised_vol(equity: Sequence[float] | pd.Series, lookback: int = 60, periods_per_year: int = 252) -> tuple[float | None, int]:
    """Annualised stdev of the last ``lookback`` simple returns of an equity curve, and how many it used."""
    e = pd.Series(list(equity), dtype=float).dropna()
    r = e.pct_change().dropna().tail(lookback)
    if len(r) < 2:
        return None, len(r)
    return float(r.std(ddof=1) * math.sqrt(periods_per_year)), len(r)


def vol_target_suggestion(equity_curve: Sequence[float] | pd.Series, base_risk_pct: float,
                          policy: VolTargetPolicy | None = None, edge: CalibratedProbability | None = None,
                          reward_risk: float | None = None, peak: float | None = None) -> SizingSuggestion:
    """A multiplier on the gate's own risk percentage. ``base_risk_pct`` is what the gate would risk (its per-trade
    percentage); ``risk_pct`` in the result is ``base_risk_pct * scale``."""
    pol = policy or VolTargetPolicy()
    curve = pd.Series(list(equity_curve), dtype=float).dropna()
    vol, n = realised_vol(curve, pol.lookback, pol.periods_per_year)
    if vol is None or n < pol.min_obs:
        return SizingSuggestion("vol_target", scale=1.0, risk_pct=base_risk_pct,
                                blockers=(f"{n} daily returns of equity history, need {pol.min_obs}: no vol targeting",))
    if vol <= 0:
        return SizingSuggestion("vol_target", scale=1.0, risk_pct=base_risk_pct,
                                blockers=("equity has not moved: realised volatility is zero, nothing to target",))
    why = [f"realised vol {vol:.1%} against a {pol.target_annual_vol:.1%} target"]
    scale = min(pol.max_scale, max(pol.min_scale, pol.target_annual_vol / vol))
    cap = 1.0
    if scale > 1.0:
        if edge is None or not edge.calibrated:
            why.append("no calibrated edge yet: not sized above the gate's own size")
        else:
            cap = pol.max_scale
            if reward_risk and reward_risk > 0:
                f = kelly_fraction(edge.p_low, reward_risk) * pol.kelly_fraction * 100.0
                cap = min(cap, max(1.0, f / base_risk_pct) if base_risk_pct > 0 else 1.0)
                why.append(f"Kelly on the lower bound p={edge.p_low:.2f} allows {f:.2f}% per trade")
        scale = min(scale, cap)
    pk = peak if peak is not None else float(curve.max())
    dd = drawdown_scale(float(curve.iloc[-1]), pk, pol.drawdown_halve_at)
    if dd < 1.0:
        why.append(f"drawdown de-risk x{dd:.2f}")
    scale = round(scale * dd, 4)
    return SizingSuggestion("vol_target", scale=scale, risk_pct=round(base_risk_pct * scale, 4), reasons=tuple(why))
