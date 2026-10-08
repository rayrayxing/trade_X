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

- every venue account currency that is not USD (the Oanda practice account is in SGD) has
  its USD pair (USD_SGD) on the Oanda price stream, so account equity, margin, fills and
  financing convert at a live quote; a missing quote raises MissingRate (block + fault);
- strategies that read research columns (carry, cross-sectional ranks) get them from
  ``LiveColumns`` (tradex.runtime.columns: Oanda daily candles plus official policy rates,
  the same research functions); a missing pair or a missing/stale rate blocks that
  strategy with a fault; one no live builder can feed is held back, visibly;
- a venue without resting stops (moomoo SIMULATE: ``check_stops``) gets a StopGuardian
  that checks live marks at least once a minute; ``Runtime.start`` runs it in a thread and
  each close checks it is still alive.

Venue adapters live in ``tradex/execution`` (protected, Ray's); ``tradex.runtime.paper``
builds them from config/accounts.yaml. ``dry=True`` builds everything else, connects no
broker and fetches nothing, and ``Runtime.readiness()`` says what is and is not in place.
"""
from __future__ import annotations

import threading
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
from tradex.data.oanda import _USD_PAIR, QuoteBook, instruments_to_stream
from tradex.events import EventCalendar
from tradex.risk.exposure import ESModel
from tradex.risk.gate import RiskGate
from tradex.runtime.columns import LIVE_COLUMNS, LiveColumns, builder_for, research_columns  # noqa: F401 - LIVE_COLUMNS re-exported
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
# moomoo's SIMULATE account ID is optional: config/accounts.yaml lets the adapter auto-select it
SECRETS = {"forex": ("oanda_token", "oanda_account_id"), "stocks": ("alpaca_key_id", "alpaca_secret"),
           "alerts": ("telegram_bot_token", "telegram_chat_id")}
GUARDIAN_INTERVAL_S = 60.0               # the stop guardian checks marks at least this often


def usd_pairs(currencies) -> list[str]:
    """The Oanda instruments that price each account currency in USD (USD_SGD for SGD)."""
    out = set()
    for c in currencies:
        if c == "USD":
            continue
        if c not in _USD_PAIR:
            raise RealDataMissing(f"no Oanda USD pair known for account currency {c}")
        out.add(_USD_PAIR[c])
    return sorted(out)


def stream_instruments(fx_symbols, account_currencies=()) -> list[str]:
    """Everything the Oanda stream must carry: traded pairs, their USD crosses, and the
    USD pair of every non-USD venue account currency."""
    return sorted(set(instruments_to_stream(fx_symbols)) | set(usd_pairs(account_currencies)))


def _default_guardian(venue, marks, on_fault, **kw):
    from tradex.execution.guardian import StopGuardian   # Ray's (protected); imported only when a venue needs it
    return StopGuardian(venue, marks, on_fault, **kw)


class _QuietFaults:
    """The guardian's fault hook: one Health row per distinct fault per ``every`` (it runs
    each minute), and none for missing marks while ``quiet(now)`` (market closed)."""

    def __init__(self, health, clock: Clock, every: pd.Timedelta = pd.Timedelta(minutes=15),
                 quiet: Callable[[pd.Timestamp], bool] | None = None):
        self.health, self.clock, self.every, self.quiet = health, clock, every, quiet
        self._last: dict[str, pd.Timestamp] = {}

    def __call__(self, check: str, detail: str) -> None:
        now = self.clock.now()
        if self.quiet is not None and self.quiet(now) and "mark" in detail:
            return
        seen = self._last.get(detail)
        if seen is not None and now - seen < self.every:
            return
        self._last[detail] = now
        self.health(check, False, detail, now)


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
    guardians: list[Any] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    columns: LiveColumns | None = None
    services: list[Any] = field(default_factory=list)   # background refreshers: ``run(stop_event)``
    watchdog_s: float = 600.0                          # a close running longer than this is a hang
    on_stuck: Callable[[str], None] | None = None      # paper sets this to dump stacks and exit

    def start(self) -> None:
        """Start the background threads: the Oanda stream reader and each stop guardian."""
        if self.dry:
            raise RuntimeError("a dry runtime connects nothing and cannot start")
        if self.feed is not None and self.feed.thread is None:
            self.threads.append(self.feed.start())
        for g in self.guardians:
            t = threading.Thread(target=g.run, args=(self.stop_event,), name=f"stop-guardian-{g.asset_class}",
                                 daemon=True)
            t.start()
            self.threads.append(t)
        for svc in self.services:
            t = threading.Thread(target=svc.run, args=(self.stop_event,), name=f"refresh-{type(svc).__name__}",
                                 daemon=True)
            t.start()
            self.threads.append(t)
        if self.watchdog_s and self.on_stuck is not None:
            t = threading.Thread(target=self.watch, args=(self.stop_event,), name="close-watchdog", daemon=True)
            t.start()
            self.threads.append(t)

    def stuck(self) -> str | None:
        """Why the current close counts as stuck, or None. A close that runs past ``watchdog_s``
        (7 Oct: 13 hours inside one close) never finishes on its own."""
        busy = self.runner.busy
        if busy is None or not self.watchdog_s:
            return None
        tf, ts, t0 = busy
        took = self.runner.timer() - t0
        return f"{tf} close {ts.isoformat()} still running after {took:.0f}s" if took > self.watchdog_s else None

    def watch(self, stop: threading.Event, every_s: float = 30.0) -> None:
        while not stop.wait(every_s):
            why = self.stuck()
            if why is not None:
                self.on_stuck(why)
                return

    def run(self, sleep: Callable[[float], None], stop: Callable[[], bool] | None = None) -> None:
        """The live loop until ``stop()`` (or ``stop_event``); stops the guardians on the way out."""
        def done() -> bool:
            return self.stop_event.is_set() or (stop is not None and stop())
        try:
            self.runner.run(sleep, done)
        finally:
            self.stop_event.set()

    @property
    def ready(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    def readiness(self) -> str:
        w = max(len(c.name) for c in self.checks)
        lines = [f"{c.status:<4}  {c.name:<{w}}  {c.detail}" for c in self.checks]
        verdict = "READY" if self.ready else "NOT READY"
        return "\n".join(lines + [f"{verdict} ({self.mode}{', dry run: no broker connected, nothing fetched' if self.dry else ''})"])


def split_unfed(specs: list[StrategySpec]) -> tuple[list[StrategySpec], dict[str, list[str]]]:
    """Strategies the live runtime can run, and those held back with the columns missing for each.

    A spec that reads a ``data.column`` (research panels such as carry or cross-sectional ranks) needs
    those columns built from real data; live bars only carry OHLCV. ``LiveColumns`` builds them for a
    forex daily spec whose columns one research builder covers; any other is held back, once and
    visibly (running it would fault on every bar close)."""
    run, held = [], {}
    for s in specs:
        need = research_columns(s)
        missing = sorted(need) if need and builder_for(s) is None else []
        if missing:
            held[s.id] = missing
        else:
            run.append(s)
    return run, held


def held_text(held: dict[str, list[str]]) -> str:
    return "; ".join(f"{sid} needs {', '.join(cols)}" for sid, cols in sorted(held.items())) + \
        " (no live builder supplies these research columns for it, so it is not run)"


def _label(x: Any) -> str:
    return type(x).__name__


def build_runtime(mode: str, strategies: list[StrategySpec], ledger: Ledger, *,
                  history: dict[str, Any], stream: Any = None, quotes: QuoteBook | None = None,
                  feeds: dict[str, Any] | None = None, venues: dict[str, Broker] | None = None,
                  clock: Clock | None = None, config: RuntimeConfig | None = None,
                  policy_path: str | Path | None = None, calendar: EventCalendar | None = None,
                  warmup_bars: int = 500, has_secret: Callable[[str], bool] | None = None,
                  dry: bool = False, accounts: dict[str, str] | None = None,
                  account_currencies: set[str] | None = None,
                  marks: dict[str, Callable[[list[str]], dict[str, float]]] | None = None,
                  guardian_factory: Callable[..., Any] | None = None,
                  quiet_marks: Callable[[pd.Timestamp], bool] | None = None,
                  profit=None, policy_rates=None,
                  financing: Callable[[list[str]], dict[str, tuple[float, float]]] | None = None) -> Runtime:
    """``history[asset_class]`` is the bar provider for start-up history; ``stream`` is the
    Oanda price stream; ``feeds[asset_class]`` is any other live bar feed (an object with
    ``before_close(tf, ts)``, and ``bind(store, health)`` if it needs them).
    ``venues[asset_class]`` are the venue adapters (behind the order guard), ``accounts``
    their agent account names. ``account_currencies`` defaults to what the venues report;
    each non-USD one must have its USD pair on the stream. ``marks[asset_class]`` is the
    live mark source for a venue's stop guardian (a venue with ``check_stops``). ``policy_rates`` is the
    official-rate source of the carry columns (default: ``LivePolicyRates``, fetched from the central
    banks); ``financing(pairs)`` returns Oanda's (long, short) financing rates to cross-check it."""
    if mode not in STRICT:
        raise ValueError(f"build_runtime is for paper/live, not {mode!r}; replay uses tradex.core.replay")
    cfg = config or RuntimeConfig.load()
    clock = clock or WallClock()
    feeds = dict(feeds or {})
    active = [s for s in strategies if s.status not in INACTIVE]
    specs, held = split_unfed(active)
    if not specs:
        raise ValueError("no active strategies" if not active else
                         "every active strategy reads research columns the live runtime cannot build: " + held_text(held))
    classes = sorted({s.asset_class for s in specs})
    checks: list[Check] = [Check("mode", "ok", mode), Check("agents", "ok", f"{cfg.agents_mode} (config/runtime.yaml)")]

    # 1. real data only, for every provider
    for ac in classes:
        require_real_data(mode, history.get(ac))
        checks.append(Check(f"history:{ac}", "ok", _label(history[ac])))
    quotes = quotes if quotes is not None else QuoteBook(max_age_s=cfg.quote_max_age_s)
    fx_syms = sorted({u for s in specs if s.asset_class == "forex" for u in s.universe if not u.startswith("$")})
    if not dry and account_currencies is None:
        account_currencies = {v.account().currency for v in (venues or {}).values()}
    acct_pairs = usd_pairs(account_currencies or ())
    if fx_syms or acct_pairs:
        require_real_data(mode, stream)
        require_real_data(mode, quotes)
        need = stream_instruments(fx_syms, account_currencies or ())
        have = getattr(stream, "instruments", None)
        if have is not None and set(need) - set(have):
            raise RealDataMissing(f"{mode}: the Oanda stream lacks {sorted(set(need) - set(have))}")
        checks.append(Check("feed:forex", "ok", f"{_label(stream)} -> {_label(quotes)} -> bar builders "
                                                f"({len(need)} instruments)"))
    if account_currencies is None:
        checks.append(Check("fx:accounts", "skip", "account currencies are read from the venues at start; "
                                                   "each non-USD one gets its USD pair on the stream"))
    else:
        ccys = sorted(account_currencies)
        checks.append(Check("fx:accounts", "ok", ", ".join(ccys) + (f"; {', '.join(acct_pairs)} on the Oanda stream "
                                                                   "(no quote: block and fault)" if acct_pairs else "")))
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
    book = MultiVenueBook(dict(venues or {}), rates, clock, accounts)

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

    fed = [s for s in specs if builder_for(s)]
    columns = LiveColumns(fed, history.get("forex")) if fed else None
    services: list[Any] = []
    if columns is not None:
        if columns.currencies:
            if policy_rates is None:
                from tradex.runtime.policy_rates import LivePolicyRates
                policy_rates = LivePolicyRates(columns.currencies, columns.rate_pairs, clock=clock.now,
                                               financing=financing)
            columns.rates = policy_rates
        if dry:
            checks.append(Check("columns", "ok", columns.describe() + "; nothing fetched (dry run)"))
        else:
            now = clock.now()
            if policy_rates is not None and columns.currencies:
                policy_rates.load_cached()
                if policy_rates.due(now):
                    policy_rates.refresh(now)
                services.append(policy_rates)
            columns.warm(now)
            columns.update(now)
            checks.append(Check("columns", "ok", columns.describe()))
            checks += [Check(*c) for c in columns.checks()]

    gate = RiskGate.from_policy(policy_path, es)
    core_cfg = CoreConfig(mode=mode, agents_mode=cfg.agents_mode, simulation=False)
    core = TradingCore(specs, store, clock, ledger, gate, brokers={"ensemble": book}, rates=rates,
                       calendar=calendar, cfg=core_cfg, symbols=symbols, base_tf=base_tf, costs=costs, profit=profit)
    feed = None
    if fx_syms or acct_pairs:
        feed = StreamFeed(stream, quotes, store, fx_syms, base_tf, core.health,
                          pd.Timedelta(seconds=cfg.stale_after_s), clock.now)
    for ac in classes:
        if ac in feeds and hasattr(feeds[ac], "bind"):
            feeds[ac].bind(store, core.health)
    if columns is not None:
        columns.bind(store, core.health)
    hooks = ([feed] if feed else []) + [feeds[ac] for ac in classes if ac in feeds] + ([columns] if columns else [])

    # stop guardians for venues whose stops the core manages (moomoo SIMULATE)
    guardians, silent = [], set()
    for ac, v in book.venues.items():
        if not hasattr(v, "check_stops"):
            continue
        src = (marks or {}).get(ac)
        if src is None:
            raise RealDataMissing(f"{mode}: {ac} venue needs live marks for its stop guardian")
        make = guardian_factory or _default_guardian
        guardians.append(make(v, src, _QuietFaults(core.health, clock, quiet=quiet_marks),
                              interval_s=GUARDIAN_INTERVAL_S, asset_class=ac, clock=clock.now))
        checks.append(Check(f"stop_guardian:{ac}", "ok", f"{_label(v)} stops checked every "
                                                         f"{GUARDIAN_INTERVAL_S:.0f}s against {_label(src)}"))

    def before_close(tf: str, ts: pd.Timestamp) -> None:
        for h in hooks:
            h.before_close(tf, ts)
        for g in guardians:                       # a guardian that stopped pinging is a fault, once
            alive = g.alive(clock.now())
            if alive == (g.asset_class in silent):
                (silent.discard if alive else silent.add)(g.asset_class)
                core.health("stop_guardian", alive, f"{g.asset_class} stop guardian "
                                                    f"{'back' if alive else 'silent: stops are unwatched'}", ts)
    sched = BarCloseScheduler(sorted(set(tfs) | {base_tf}, key=duration), ledger,
                              grace=pd.Timedelta(seconds=cfg.grace_s), done_until=clock.now())
    runner = LiveRunner(core, sched, before_close, overrun=pd.Timedelta(seconds=cfg.overrun_s))
    if held:
        checks.append(Check("strategies:held_back", "skip", held_text(held)))
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
    return Runtime(mode, dry, core, runner, store, quotes, book, feed, hooks, checks, guardians,
                   columns=columns, services=services)
