"""Paper/live wiring: ``build_runtime(mode, ...)`` assembles the core, its data and its book.

What it guarantees before anything runs:

- every data provider (history per asset class, the live stream, the quote book, any
  other live feed) passes ``require_real_data``: a synthetic, CSV or cached provider in
  paper/live refuses to start (SyntheticDataRefused), a missing one too (RealDataMissing);
- the process run mode is set (``configure_run_mode``) with the Oanda quote book behind it,
  forex costs read live spreads from the same book (``OandaFxCosts(mode, spread_source)``),
  and the venues book and the core convert currencies with it; nothing falls back to a
  backtest constant;
- the runner's ``before_close`` hook drains the Oanda stream through the bar builders into
  the bar store the core reads, and a silent stream is a fault.

Venue adapters live in ``tradex/execution`` (protected, Ray's) and do not exist yet, so a
real run refuses to start without them; ``dry=True`` builds everything else, connects no
broker and fetches nothing, and ``Runtime.readiness()`` says what is and is not in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradex.core.interfaces import Broker, Clock, WallClock
from tradex.core.ledger import Ledger
from tradex.core.loop import INACTIVE, CoreConfig, TradingCore
from tradex.core.replay import factor_returns
from tradex.costs.models import OandaFxCosts, configure_run_mode, model_for
from tradex.data.guard import STRICT, RealDataMissing, require_present, require_real_data
from tradex.data.oanda import QuoteBook, instruments_to_stream
from tradex.events import EventCalendar
from tradex.risk.exposure import ESModel
from tradex.risk.gate import RiskGate
from tradex.runtime.config import RuntimeConfig
from tradex.runtime.feed import StreamFeed
from tradex.runtime.fx import LiveQuoteRates
from tradex.runtime.market import BarStore
from tradex.runtime.runner import LiveRunner
from tradex.runtime.schedule import BarCloseScheduler
from tradex.runtime.venues import MultiVenueBook
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

MAX_BUILT_TF = "H1"                      # bar builders make M1..H1; higher timeframes are resampled
SECRETS = {"forex": ("oanda_token", "oanda_account_id"), "stocks": ("moomoo_sim_account_id",),
           "alerts": ("telegram_bot_token", "telegram_chat_id")}


class VenuesMissing(RuntimeError):
    """No venue adapter for an asset class the strategies trade (they need Ray: tradex/execution)."""


@dataclass
class Check:
    name: str
    status: str                          # ok | FAIL | skip
    detail: str = ""


@dataclass
class Runtime:
    mode: str
    dry: bool
    core: TradingCore
    runner: LiveRunner
    store: BarStore
    quotes: QuoteBook
    book: MultiVenueBook
    feed: StreamFeed | None
    feeds: list[Any]
    checks: list[Check] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    def readiness(self) -> str:
        w = max(len(c.name) for c in self.checks)
        lines = [f"{c.status:<4}  {c.name:<{w}}  {c.detail}" for c in self.checks]
        verdict = "READY" if self.ready else "NOT READY"
        return "\n".join(lines + [f"{verdict} ({self.mode}{', dry run: no broker connected, nothing fetched' if self.dry else ''})"])


def _label(x: Any) -> str:
    return type(x).__name__


def build_runtime(mode: str, strategies: list[StrategySpec], ledger: Ledger, *,
                  history: dict[str, Any], stream: Any = None, quotes: QuoteBook | None = None,
                  feeds: dict[str, Any] | None = None, venues: dict[str, Broker] | None = None,
                  clock: Clock | None = None, config: RuntimeConfig | None = None,
                  policy_path: str | Path | None = None, calendar: EventCalendar | None = None,
                  warmup_bars: int = 500, has_secret: Callable[[str], bool] | None = None,
                  dry: bool = False) -> Runtime:
    """``history[asset_class]`` is the bar provider for start-up history; ``stream`` is the
    Oanda price stream (forex); ``feeds[asset_class]`` is any other live bar feed (an object
    with ``before_close(tf, ts)``). ``venues[asset_class]`` are the venue adapters."""
    if mode not in STRICT:
        raise ValueError(f"build_runtime is for paper/live, not {mode!r}; replay uses tradex.core.replay")
    cfg = config or RuntimeConfig.load()
    clock = clock or WallClock()
    feeds = dict(feeds or {})
    specs = [s for s in strategies if s.status not in INACTIVE]
    if not specs:
        raise ValueError("no active strategies")
    classes = sorted({s.asset_class for s in specs})
    checks: list[Check] = [Check("mode", "ok", mode), Check("agents", "ok", f"{cfg.agents_mode} (config/runtime.yaml)")]

    # 1. real data only, for every provider
    for ac in classes:
        require_real_data(mode, history.get(ac))
        checks.append(Check(f"history:{ac}", "ok", _label(history[ac])))
    quotes = quotes if quotes is not None else QuoteBook(max_age_s=cfg.quote_max_age_s)
    fx_syms = sorted({u for s in specs if s.asset_class == "forex" for u in s.universe if not u.startswith("$")})
    if fx_syms:
        require_real_data(mode, stream)
        require_real_data(mode, quotes)
        checks.append(Check("feed:forex", "ok", f"{_label(stream)} -> {_label(quotes)} -> bar builders "
                                                f"({len(instruments_to_stream(fx_syms))} instruments)"))
    for ac in classes:
        if ac == "forex":
            continue
        if ac in feeds:
            require_real_data(mode, feeds[ac])
            checks.append(Check(f"feed:{ac}", "ok", _label(feeds[ac])))
        elif dry:
            checks.append(Check(f"feed:{ac}", "FAIL", "no live bar feed for this asset class yet"))
        else:
            raise RealDataMissing(f"{mode}: no live bar feed for {ac}")

    # 2. one live rate and spread source for everything
    rates = LiveQuoteRates(quotes)
    configure_run_mode(mode, rates)
    costs = {"forex": OandaFxCosts(mode=mode, spread_source=rates), "stocks": model_for("stocks")}
    checks.append(Check("costs", "ok", "forex spreads and FX rates from the live quote book; no defaults"))

    # 3. venues
    if dry:
        venues = {}
        checks.append(Check("broker", "skip", "not connected (dry run)"))
    else:
        missing = [ac for ac in classes if ac not in (venues or {})]
        if missing:
            raise VenuesMissing(f"no venue adapter for {missing}: adapters live in tradex/execution and need Ray")
        checks.append(Check("broker", "ok", ", ".join(f"{ac}={_label(v)}" for ac, v in venues.items())))
    book = MultiVenueBook(dict(venues or {}), rates, clock)

    # 4. bars: history now (not in a dry run), the stream at every close
    tfs = sorted({s.signal_tf for s in specs}, key=duration)
    base_tf = min(tfs[0], MAX_BUILT_TF, key=duration)
    store = BarStore(base_tf, {}, clock)
    symbols = sorted({u for s in specs for u in s.universe if not u.startswith("$")})
    ac_of = {u: s.asset_class for s in specs for u in s.universe}
    es = None
    if dry:
        checks.append(Check("history", "skip", f"would load {warmup_bars} {base_tf} bars for {len(symbols)} symbols"))
    else:
        end = clock.now()
        start = end - duration(base_tf) * warmup_bars
        for sym in symbols:
            df = history[ac_of[sym]].get_bars(sym, base_tf, start.isoformat(), end.isoformat())
            require_present(mode, None if df is None or df.empty else df, f"{sym} {base_tf} history")
            store.append(sym, df)
        rets = factor_returns(store.frames, ac_of)
        es = ESModel(rets) if len(rets) >= 20 else None
        checks.append(Check("history", "ok", f"{warmup_bars} {base_tf} bars for {len(symbols)} symbols"))

    gate = RiskGate.from_policy(policy_path, es)
    core_cfg = CoreConfig(mode=mode, agents_mode=cfg.agents_mode, simulation=False)
    core = TradingCore(specs, store, clock, ledger, gate, brokers={"ensemble": book}, rates=rates,
                       calendar=calendar, cfg=core_cfg, symbols=symbols, base_tf=base_tf, costs=costs)
    feed = None
    if fx_syms:
        feed = StreamFeed(stream, quotes, store, fx_syms, base_tf, core.health,
                          pd.Timedelta(seconds=cfg.stale_after_s), clock.now)
    hooks = ([feed] if feed else []) + [feeds[ac] for ac in classes if ac in feeds]

    def before_close(tf: str, ts: pd.Timestamp) -> None:
        for h in hooks:
            h.before_close(tf, ts)
    sched = BarCloseScheduler(sorted(set(tfs) | {base_tf}, key=duration), ledger,
                              grace=pd.Timedelta(seconds=cfg.grace_s), done_until=clock.now())
    runner = LiveRunner(core, sched, before_close, overrun=pd.Timedelta(seconds=cfg.overrun_s))
    checks.append(Check("strategies", "ok", f"{len(specs)} on {', '.join(tfs)}; base {base_tf}; "
                                            f"{sum(1 for s in specs if s.status in core_cfg.qualified_statuses)} "
                                            "vote in the ensemble"))
    if has_secret is not None:
        for group, names in SECRETS.items():
            if group in ("forex", "stocks") and group not in classes:
                continue
            miss = [n for n in names if not has_secret(n)]
            checks.append(Check(f"secrets:{group}", "FAIL" if miss else "ok",
                                f"missing {', '.join(miss)} (run trade-x setup)" if miss else "set"))
    return Runtime(mode, dry, core, runner, store, quotes, book, feed, hooks, checks)
