"""The daily cycle that ties the pieces together. Thread 3 schedules this and wires brokers in.

    morning (before the US session, ~20:30 Singapore time):
      1. scout       -> today's 10-20 stock watchlist
      2. regime      -> trend/range and volatility label from SPY (stocks) or each pair (forex)
      3. select      -> risk budget per live-eligible strategy (allocator)
      4. universe    -> strategies whose universe is "$watchlist" trade today's picks
    every bar / on a schedule while positions are open:
      5. review      -> PositionReviewer actions for every open position
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradex.positions.review import Action, ActionKind, MarketSnapshot, OpenPosition, PositionReviewer
from tradex.scout.base import CompositeScout, Watchlist
from tradex.selection.allocator import AllocationConfig, AllocationSnapshot, StrategyRecord, allocate
from tradex.selection.regime import classify_regime, daily_labels
from tradex.strategy.spec import WATCHLIST_TOKEN, StrategySpec


def resolve_universe(spec: StrategySpec, watchlist: Watchlist | None) -> list[str]:
    syms = [s for s in spec.universe if s != WATCHLIST_TOKEN]
    if spec.uses_watchlist and watchlist is not None:
        syms += [s for s in watchlist.symbols if s not in syms]
    return syms


@dataclass
class MorningPlan:
    asof: str
    watchlist: Watchlist | None
    regime: str
    allocation: AllocationSnapshot
    universes: dict[str, list[str]]


def morning_plan(
    asof: pd.Timestamp,
    specs: list[StrategySpec],
    records: list[StrategyRecord],
    regime_bars: pd.DataFrame,
    scout: CompositeScout | None = None,
    alloc_cfg: AllocationConfig | None = None,
    equity: float | None = None,
    peak_equity: float | None = None,
) -> MorningPlan:
    watchlist = scout.scan(asof) if scout else None
    reg = classify_regime(regime_bars[regime_bars.index < asof])
    label = reg["label"].iloc[-1] if len(reg) else "unknown"
    snap = allocate(records, label, daily_labels(reg), alloc_cfg, equity, peak_equity, asof=str(asof))
    funded = {a.strategy_id for a in snap.allocations if a.risk_pct > 0}
    universes = {s.id: resolve_universe(s, watchlist) for s in specs if s.id in funded}
    return MorningPlan(str(asof), watchlist, label, snap, universes)


def review_positions(positions: list[OpenPosition], snapshots: dict[str, MarketSnapshot],
                     reviewer: PositionReviewer | None = None) -> dict[str, list[Action]]:
    """Run the reviewer over the whole book. Positions with no fresh market snapshot are flagged."""
    reviewer = reviewer or PositionReviewer()
    out = {}
    for p in positions:
        snap = snapshots.get(p.symbol)
        if snap is None:
            out[p.symbol] = [Action(ActionKind.HOLD, "no fresh price; check data feed")]
            continue
        out[p.symbol] = reviewer.review(p, snap)
    return out
