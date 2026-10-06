"""Full-stack replay harness (recommendation 6) and the nightly parity check.

``run_replay`` drives the bar-close trading core over recorded bars: a replay clock, a
bar store holding the recorded frames, simulated brokers and recorded FX rates. It is
the same core the live runtime drives from the scheduler, so replaying a paper or live
day must reproduce its ledger; any difference is a bug. Run the same day twice and the
decision digests must match. Agent pull requests must pass a replay before they merge.

Strategies on different timeframes run together: frames are given on the base (finest)
timeframe, higher timeframes are resampled from it, and close events run in time order,
finer timeframes first when closes coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig, TradingCore
from tradex.costs.models import split_pair
from tradex.events import EventCalendar
from tradex.execution.checks import ShortInfo
from tradex.risk.exposure import ESModel
from tradex.risk.gate import RiskGate
from tradex.runtime.fx import SeriesRates
from tradex.runtime.market import BarStore
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


def close_events(data: BarStore, tfs: list[str], start: pd.Timestamp | None = None,
                 end: pd.Timestamp | None = None) -> list[tuple[pd.Timestamp, str]]:
    """(close time, timeframe) for every bar in the store opening in [start, end), in time
    order and finer timeframes first, as the scheduler would emit them."""
    order = {tf: i for i, tf in enumerate(sorted(tfs, key=duration))}
    ev: set[tuple[pd.Timestamp, str]] = set()
    for sym in data.symbols():
        for tf in tfs:
            idx = data.frame(sym, tf).index
            if start is not None:
                idx = idx[idx >= start]
            if end is not None:
                idx = idx[idx < end]
            ev.update((t + duration(tf), tf) for t in idx)
    return sorted(ev, key=lambda e: (e[0], order[e[1]]))


def build_replay_core(strategies: list[StrategySpec], frames: dict[str, pd.DataFrame], ledger: Ledger,
                      start: pd.Timestamp | None = None, calendar: EventCalendar | None = None,
                      short_info: dict[str, ShortInfo] | None = None, cfg: CoreConfig | None = None,
                      policy_path: str | Path | None = None, use_es: bool = True, base_tf: str | None = None,
                      data: BarStore | None = None, clock: ReplayClock | None = None,
                      brokers: dict | None = None, profit=None) -> TradingCore:
    """The core as replay runs it. ``data`` and ``clock`` may be supplied (a live-style store
    the caller appends to); risk-model inputs always come from ``frames`` before ``start``."""
    tfs = sorted({s.signal_tf for s in strategies}, key=duration)
    base_tf = base_tf or tfs[0]
    clock = clock or ReplayClock()
    data = data if data is not None else BarStore(base_tf, frames, clock)
    ac = {sym: s.asset_class for s in strategies for sym in s.universe}
    cfg = cfg or CoreConfig()
    rates = SeriesRates(usd_per_unit_series(frames), cfg.mode)
    es = None
    if use_es:
        rets = factor_returns(frames, ac)
        if start is not None:
            rets = rets[rets.index < start]     # the risk model only sees history before the replay
        es = ESModel(rets) if len(rets) >= 20 else None
    gate = RiskGate.from_policy(policy_path, es)
    return TradingCore(strategies, data, clock, ledger, gate, brokers=brokers, rates=rates, calendar=calendar,
                       short_info=short_info, cfg=cfg, symbols=list(frames), base_tf=base_tf, profit=profit)


def run_replay(strategies: list[StrategySpec], frames: dict[str, pd.DataFrame], ledger: Ledger,
               start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
               calendar: EventCalendar | None = None, short_info: dict[str, ShortInfo] | None = None,
               cfg: CoreConfig | None = None, policy_path: str | Path | None = None,
               use_es: bool = True, base_tf: str | None = None, profit=None) -> ReplayResult:
    """Replay ``frames`` (bars on ``base_tf``, default the finest strategy timeframe)."""
    core = build_replay_core(strategies, frames, ledger, start, calendar, short_info, cfg, policy_path, use_es,
                             base_tf, profit=profit)
    last = None
    for ts, tf in close_events(core.data, core.tfs, start, end):
        core.clock.set(ts)
        core.on_bar_close(tf, ts)
        last = ts
    summary = core.finish(last) if last is not None else core.summary()
    ok, _ = ledger.verify()
    return ReplayResult(summary, ledger.digest(), ok, ledger)
