"""Full-stack replay harness (recommendation 6) and the nightly parity check.

``run_replay`` drives the trading core over recorded bars with the replay clock, the
replay market data and the simulated broker, writing every decision to a ledger. Run
the same day twice through the same code and the decision digests must match; replay a
paper or live day and any difference from its ledger is a bug. Agent pull requests must
pass a replay before they can merge.

One run covers strategies that share a signal timeframe. Mixed timeframes in one book
are a follow-up for the runtime loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradex.core.interfaces import ReplayClock, ReplayData
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig, TradingCore
from tradex.costs.models import split_pair
from tradex.events import EventCalendar
from tradex.execution.checks import ShortInfo
from tradex.risk.exposure import ESModel
from tradex.risk.gate import RiskGate
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration


def usd_per_unit_series(frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """USD value of one unit of each currency, from XXX_USD and USD_XXX pairs in the data."""
    out: dict[str, pd.Series] = {}
    for sym, df in frames.items():
        if "_" not in sym:
            continue
        base, quote = split_pair(sym)
        if quote == "USD":
            out[base] = df["close"]
        elif base == "USD":
            out[quote] = 1.0 / df["close"]
    return out


def factor_returns(frames: dict[str, pd.DataFrame], asset_class: dict[str, str]) -> pd.DataFrame:
    """Daily returns per risk factor (``FX:<ccy>`` and ``EQ:<symbol>``) for the ES model."""
    cols: dict[str, pd.Series] = {}
    for ccy, s in usd_per_unit_series(frames).items():
        cols[f"FX:{ccy}"] = s.resample("1D").last().dropna().pct_change()
    for sym, df in frames.items():
        if asset_class.get(sym) == "stocks":
            cols[f"EQ:{sym}"] = df["close"].resample("1D").last().dropna().pct_change()
    return pd.DataFrame(cols).dropna(how="all")


@dataclass
class ReplayResult:
    summary: dict[str, Any]
    digest: str
    chain_ok: bool
    ledger: Ledger


def run_replay(strategies: list[StrategySpec], frames: dict[str, pd.DataFrame], ledger: Ledger,
               start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
               calendar: EventCalendar | None = None, short_info: dict[str, ShortInfo] | None = None,
               cfg: CoreConfig | None = None, policy_path: str | Path | None = None,
               use_es: bool = True) -> ReplayResult:
    tfs = {s.signal_tf for s in strategies}
    if len(tfs) != 1:
        raise ValueError(f"one signal timeframe per replay, got {sorted(tfs)}")
    bar = duration(tfs.pop())
    clock = ReplayClock()
    data = ReplayData(frames, bar, clock)
    ac = {sym: s.asset_class for s in strategies for sym in s.universe}
    fx = usd_per_unit_series(frames) or None
    es = None
    if use_es:
        rets = factor_returns(frames, ac)
        if start is not None:
            rets = rets[rets.index < start]     # the risk model only sees history before the replay
        es = ESModel(rets) if len(rets) >= 20 else None
    gate = RiskGate.from_policy(policy_path, es)
    core = TradingCore(strategies, data, clock, ledger, gate, calendar, short_info, fx, cfg)
    summary = core.run(start, end)
    ok, _ = ledger.verify()
    return ReplayResult(summary, ledger.digest(), ok, ledger)
