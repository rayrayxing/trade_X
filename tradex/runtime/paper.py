"""``tradex run --mode paper``: the agent's paper accounts as venues, then the live loop.

``paper_venues`` reads config/accounts.yaml (the IDs come from the Keychain; moomoo may
auto-select its single US SIMULATE account) and builds, per traded asset class, the venue
adapter for its account: Oanda practice for forex, moomoo SIMULATE for stocks. Every
adapter submits through one OrderGuard (agent-owned accounts, ledger verdicts); anything
that is not a VenueAdapter on that guard is wrapped in a GuardedBroker. The adapters are
Ray's (tradex/execution, protected) and are imported here lazily, so this module and its
tests work without them; a missing adapter module refuses the start.

``run_paper`` checks the secrets before connecting anything, builds the venues, reads each
account's currency (read-only) to put its USD pair on the Oanda stream, builds the runtime
with live marks for the moomoo stop guardian, prints the readiness checks and starts the
loop only when every check passes. Live mode is refused before any of this.
"""
from __future__ import annotations

import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradex.core.interfaces import Broker, Clock, WallClock
from tradex.runtime.build import SECRETS, VenuesMissing, build_runtime, stream_instruments

STATE_DIR = Path("data/state")
MOOMOO_STATE = "moomoo_sim.json"
DEFAULT_LEDGER = "data/ledger/live.sqlite"


class Serialized:
    """One lock around every method call of a venue that two threads use: the core's loop
    and the stop guardian's (the moomoo adapter keeps its order and stop state in memory)."""

    def __init__(self, inner, lock: threading.RLock | None = None):
        self.inner, self.lock = inner, lock or threading.RLock()

    def __getattr__(self, name: str):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*a, **kw):
            with self.lock:
                return attr(*a, **kw)
        return call


@dataclass
class PaperVenues:
    venues: dict[str, Broker]                     # asset class -> adapter behind the order guard
    accounts: dict[str, str]                      # asset class -> agent account name (never the ID)
    guard: Any
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        for c in self.closers:
            try:
                c()
            except Exception:  # noqa: BLE001 - closing a context on the way out must not mask the reason
                pass


def _oanda(acc, guard, ctx: dict):
    try:
        from tradex.execution import oanda
    except ImportError:
        raise VenuesMissing("tradex/execution/oanda.py is not there: commit the venue adapter patch") from None
    return oanda.OandaPracticeAdapter(acc.account_id, guard, account=acc, to_usd=ctx["to_usd"],
                                      http=_logged_http(oanda._http, acc.account_id))


def _logged_http(http: Callable, account_id: str, retry_delays: tuple[float, ...] = (2.0, 5.0, 10.0),
                 sleep: Callable[[float], None] | None = None) -> Callable:
    """Log every non-2xx Oanda REST reply with its path, query and Oanda's own error text.

    The adapter (protected) raises VenueAuthError without Oanda's message, which left the
    hourly 401s on /transactions/sinceid (6-7 Oct) undiagnosable. The account ID is masked
    and the token never reaches this function's output."""
    import logging
    import time
    log = logging.getLogger("tradex.oanda.http")
    sleep = sleep or time.sleep

    def call(method: str, url: str, headers: dict, body: dict | None = None, **kw):
        status, payload = http(method, url, headers, body, **kw)
        if status >= 300:
            where = url.split("/v3/", 1)[-1].replace(account_id, "<account>")
            log.warning("oanda %s /%s -> %s %s %s", method, where, status,
                        payload.get("errorCode", ""), payload.get("errorMessage", ""))
        if status == 401 and method == "GET":
            # A long-running run gets 401s that fresh processes never see (7 Oct). Compare the
            # token this process sent with the Keychain's (8-hex fingerprints only), then retry
            # the read once with the Keychain token: it either heals the call or names the cause.
            # 7 Oct: same token as the Keychain (fingerprints matched), same account, and the very
            # same request returned 200 minutes later. Oanda practice refuses some reads around
            # an hourly close, so a 401 read is retried with backoff before the adapter faults.
            from tradex import secrets
            try:
                fresh = secrets.get("oanda_token")
            except secrets.MissingSecret:
                log.warning("oanda 401: no Keychain token to retry with")
                return status, payload
            sent = headers.get("Authorization", "").removeprefix("Bearer ")
            retry_headers = {**headers, "Authorization": f"Bearer {fresh}"}
            for delay in retry_delays:
                sleep(delay)
                status, payload = http(method, url, retry_headers, body, **kw)
                log.warning("oanda 401 retry after %ss: sent token %s, keychain token %s -> %s %s", delay, _fp(sent),
                            _fp(fresh), status, payload.get("errorMessage", "") if status >= 300 else "ok")
                if status != 401:
                    break
        return status, payload
    return call


def _fp(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:8] if token else "none"


def _moomoo(acc, guard, ctx: dict):
    try:
        from tradex.execution import moomoo
    except ImportError:
        raise VenuesMissing("tradex/execution/moomoo.py is not there: commit the venue adapter patch") from None
    ad = moomoo.MoomooSimAdapter(acc.account_id, guard, account=acc, ctx=ctx.get("moomoo_ctx"),
                                 state_path=Path(ctx["state_dir"]) / MOOMOO_STATE)
    return ad


FACTORIES: dict[str, Callable] = {"oanda": _oanda, "moomoo": _moomoo}


def _load_accounts(path, classes: set[str], ctx: dict):
    """config/accounts.yaml; with stocks traded and the moomoo adapter present, an
    ``auto_select`` account still unresolved asks OpenD for its single SIMULATE account."""
    from tradex.execution.accounts import load_accounts
    if "stocks" in classes:
        try:
            from tradex.execution import moomoo
        except ImportError:
            moomoo = None
        if moomoo is not None:
            octx = ctx["moomoo_ctx"] = moomoo.open_trade_context()
            ctx["closers"].append(octx.close)
            return load_accounts(path, auto={"moomoo": moomoo.auto_resolver(octx)})
    return load_accounts(path)


def paper_venues(classes, ledger, to_usd: Callable[[str], float], *, accounts_path=None,
                 state_dir: str | Path = STATE_DIR, factories: dict[str, Callable] | None = None,
                 load: Callable | None = None) -> PaperVenues:
    """One guarded venue adapter per traded asset class, from the agent accounts."""
    from tradex.execution.accounts import agent_account_ids
    from tradex.execution.guard import GuardedBroker, OrderGuard, VenueAdapter, ledger_verdicts
    classes = set(classes)
    ctx: dict = {"to_usd": to_usd, "state_dir": Path(state_dir), "closers": []}
    try:
        accs = (load or _load_accounts)(accounts_path, classes, ctx)
        guard = OrderGuard(ledger_verdicts(ledger), agent_account_ids(accs))
        venues, names = {}, {}
        for ac in sorted(classes):
            acc = [a for a in accs if ac in a.asset_classes]
            if len(acc) != 1:
                raise VenuesMissing(f"config/accounts.yaml: expected one agent account for {ac}, found {len(acc)}")
            acc = acc[0]
            if not acc.resolved:
                raise VenuesMissing(f"{acc.name}: no account ID (run trade-x setup, or start OpenD for auto-select)")
            make = (factories or FACTORIES).get(acc.venue)
            if make is None:
                raise VenuesMissing(f"{acc.name}: no adapter for venue {acc.venue!r}")
            ad = make(acc, guard, ctx)
            if not (isinstance(ad, VenueAdapter) and ad.guard is guard):
                ad = GuardedBroker(ad, guard, acc.account_id)
            if hasattr(ad, "check_stops"):            # shared with the stop guardian's thread
                ad = Serialized(ad)
            venues[ac], names[ac] = ad, acc.name
    except BaseException:
        PaperVenues({}, {}, None, ctx["closers"]).close()
        raise
    return PaperVenues(venues, names, guard, ctx["closers"])


def missing_secrets(classes, has_secret: Callable[[str], bool]) -> list[str]:
    need = [n for g, names in SECRETS.items() if g == "alerts" or g in classes for n in names]
    return [n for n in need if not has_secret(n)]


@dataclass
class PaperDeps:
    """Everything ``run_paper`` touches outside the process (tests pass fakes)."""
    has_secret: Callable[[str], bool]
    history: Callable[[], dict[str, Any]]
    stream: Callable[[list[str]], Any]
    venues: Callable[..., PaperVenues] = paper_venues
    marks: Callable[[], Any] | None = None
    clock: Clock = field(default_factory=WallClock)
    sleep: Callable[[float], Any] | None = None    # default: wait on the runtime's stop event (SIGTERM wakes it)
    stop: Callable[[], bool] | None = None
    quiet_marks: Callable[[pd.Timestamp], bool] | None = None
    guardian_factory: Callable | None = None
    financing: Callable[[list[str]], dict] | None = None   # Oanda financing rates, to cross-check official rates


def default_deps() -> PaperDeps:
    from tradex import secrets
    from tradex.data.oanda import PriceStream
    from tradex.data.providers import AlpacaProvider, OandaProvider
    from tradex.runtime.marks import OpenDMarks, us_regular_session
    from tradex.data.oanda import fetch_financing
    return PaperDeps(has_secret=secrets.has, history=lambda: {"forex": OandaProvider(), "stocks": AlpacaProvider()},
                     stream=PriceStream, marks=OpenDMarks, quiet_marks=lambda t: not us_regular_session(t),
                     financing=lambda pairs: fetch_financing(pairs))


def run_paper(specs, ledger_path: str | Path, cfg, deps: PaperDeps | None = None, *, accounts_path=None,
              state_dir: str | Path = STATE_DIR, out=None, err=None) -> int:
    """Start paper trading if every readiness check passes; otherwise say what is missing (1)."""
    from tradex.core.ledger import Ledger
    from tradex.data.guard import RealDataMissing, SyntheticDataRefused
    from tradex.data.oanda import QuoteBook
    from tradex.runtime.feed import PolledBarFeed
    from tradex.runtime.fx import LiveQuoteRates
    from tradex.timeframes import duration
    deps = deps or default_deps()
    out, err = out or sys.stdout, err or sys.stderr
    specs = [s for s in specs if s.status not in ("retired", "rejected")]
    if not specs:
        print("not starting: no active strategies", file=err)
        return 1
    classes = {s.asset_class for s in specs}
    miss = missing_secrets(classes, deps.has_secret)
    if miss:
        print(f"not starting: missing secrets {', '.join(miss)} (run trade-x setup)", file=err)
        return 1
    clock = deps.clock
    quotes = QuoteBook(cfg.quote_max_age_s, clock=clock.now)
    rates = LiveQuoteRates(quotes)
    ledger = Ledger(ledger_path, run_id=f"paper-{clock.now():%Y%m%dT%H%M%S}")
    pv = None
    marks = None
    try:
        pv = deps.venues(classes, ledger, lambda ccy: rates.usd_per_unit(ccy, clock.now()),
                         accounts_path=accounts_path, state_dir=state_dir)
        ccys = {v.account().currency for v in pv.venues.values()}          # read-only
        fx = sorted({u for s in specs if s.asset_class == "forex" for u in s.universe if not u.startswith("$")})
        stream = deps.stream(stream_instruments(fx, ccys))
        tfs = sorted({s.signal_tf for s in specs}, key=duration)
        base_tf = min(tfs[0], "H1", key=duration)
        history = deps.history()
        feeds = {}
        if "stocks" in classes:
            stk = sorted({u for s in specs if s.asset_class == "stocks" for u in s.universe if not u.startswith("$")})
            feeds["stocks"] = PolledBarFeed(history["stocks"], stk, base_tf)
            marks = deps.marks() if deps.marks else None
        rt = build_runtime("paper", specs, ledger, history=history, stream=stream, quotes=quotes, feeds=feeds,
                           venues=pv.venues, accounts=pv.accounts, account_currencies=ccys, clock=clock,
                           config=cfg, has_secret=deps.has_secret,
                           marks={"stocks": marks} if marks is not None else None,
                           guardian_factory=deps.guardian_factory, quiet_marks=deps.quiet_marks,
                           financing=deps.financing)
    except (SyntheticDataRefused, RealDataMissing, VenuesMissing, ValueError, LookupError, RuntimeError,
            OSError) as exc:
        print(f"not starting: {type(exc).__name__}: {exc}", file=err)
        _close(pv, marks)
        return 1
    print(rt.readiness(), file=out)
    if not rt.ready:
        print("not starting: fix the FAIL lines above", file=err)
        _close(pv, marks)
        return 1
    def _stop(*_):
        rt.stop_event.set()
    old = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGINT)} if _main_thread() else {}
    try:
        rt.start()
        print(f"paper trading started: ledger {ledger_path}", file=out)
        rt.run(deps.sleep or rt.stop_event.wait, deps.stop)
    finally:
        rt.stop_event.set()
        for s, h in old.items():
            signal.signal(s, h)
        _close(pv, marks)
    return 0


def _main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _close(pv: PaperVenues | None, marks) -> None:
    if pv is not None:
        pv.close()
    close = getattr(marks, "close", None)
    if close is not None:
        close()
