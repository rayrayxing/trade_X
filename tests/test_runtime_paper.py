"""`tradex run --mode paper` wiring, with fake venues, stream, history and marks only.

Nothing here connects to Oanda or OpenD: the venue adapters are fakes implementing the
VenueAdapter interface, built by injected factories."""
import sys
import threading
from dataclasses import replace

import pandas as pd
import pytest

import tradex.costs.models as cm
from test_runtime_build import START, RecordedHistory
from test_runtime_core import _mixed_specs
from tradex.cli import main
from tradex.core.interfaces import AccountInfo, ReplayClock
from tradex.core.ledger import Ledger
from tradex.data.guard import RealDataMissing
from tradex.data.oanda import QuoteBook, Tick
from tradex.execution.accounts import AgentAccount
from tradex.execution.guard import GuardedBroker, OrderGuard, VenueAdapter
from tradex.execution.sim import SimBroker
from tradex.runtime import paper
from tradex.runtime.build import VenuesMissing, build_runtime, stream_instruments
from tradex.runtime.config import RuntimeConfig
from tradex.runtime.feed import PolledBarFeed
from tradex.runtime.fx import MissingRate
from tradex.runtime.marks import OpenDMarks, us_regular_session


@pytest.fixture(autouse=True)
def _reset_run_mode():
    yield
    cm.configure_run_mode("backtest")


ACCOUNTS = [AgentAccount("oanda-practice", "oanda", "paper", ["forex"], "101-001", "oanda_account_id", "practice"),
            AgentAccount("moomoo-simulate", "moomoo", "paper", ["stocks"], "998877", "moomoo_sim_account_id",
                         "SIMULATE")]


class FakeOanda(VenueAdapter):
    """A VenueAdapter like OandaPracticeAdapter: a simulated fill engine behind ``place``, the
    account kept in SGD, and every SGD amount converted with the injected ``to_usd``."""
    venue = "oanda"

    def __init__(self, account_id, guard, to_usd):
        super().__init__(account_id, guard)
        self.sim, self.to_usd = SimBroker(10_000, account_id=account_id), to_usd
        self.simulated = True
        self.convert_fills = False

    def _submit(self, req):
        return self.sim.place(req)

    def on_bar(self, *a):
        return self.sim.on_bar(*a)

    def cancel(self, coid):
        return self.sim.cancel(coid)

    def amend_stop(self, did, stop):
        self.sim.amend_stop(did, stop)

    def positions(self, account="agent"):
        return self.sim.positions(account)

    def fills(self, since=None):
        if self.convert_fills:
            self.to_usd("SGD")                       # like oanda._usd: raises MissingRate without a quote
        return self.sim.fills(since)

    def order_status(self, coid):
        return self.sim.order_status(coid)

    def account(self):
        a = self.sim.account()
        return AccountInfo(self.account_id, "oanda", "SGD", a.equity, a.cash, a.margin_used, a.buying_power)


class FakeMoomoo(SimBroker):
    """Not a VenueAdapter (so paper_venues must wrap it in a GuardedBroker); core-managed stops."""

    def __init__(self, account_id):
        super().__init__(10_000, account_id=account_id)
        self.venue = "moomoo"

    def stop_levels(self):
        return {}

    def check_stops(self, marks):
        return []

    def mark_fired(self, did, coid):
        pass


class FakeGuardian:
    def __init__(self, venue, marks, on_fault, *, interval_s, asset_class, clock):
        self.venue, self.marks, self.on_fault = venue, marks, on_fault
        self.interval_s, self.asset_class, self.clock = interval_s, asset_class, clock
        self.ticks, self.last_ping = 0, None
        self.ticked = threading.Event()

    def tick(self):
        self.ticks += 1
        self.last_ping = self.clock()
        self.ticked.set()

    def alive(self, now=None):
        return self.last_ping is not None and (now or self.clock()) - self.last_ping <= pd.Timedelta(
            seconds=2 * self.interval_s)

    def run(self, stop):
        while not stop.is_set():
            self.tick()
            stop.wait(0.01)


class FakeStream:
    heartbeats = 0

    def __init__(self, instruments):
        self.instruments = list(instruments)

    def ticks(self):
        return iter(())


def _factories():
    return {"oanda": lambda acc, guard, ctx: FakeOanda(acc.account_id, guard, ctx["to_usd"]),
            "moomoo": lambda acc, guard, ctx: FakeMoomoo(acc.account_id)}


def _venues(classes, ledger, to_usd, accounts=ACCOUNTS, **kw):
    return paper.paper_venues(classes, ledger, to_usd, factories=_factories(), load=lambda *a: accounts, **kw)


def _stock_spec():
    s = _mixed_specs()[0]
    return replace(s, id="stk-h1", asset_class="stocks", universe=["AAA"])


class StockHistory(RecordedHistory):
    def __init__(self):
        super().__init__()
        self.frames["AAA"] = self.frames["EUR_USD"] * 100
        self.polls = []

    def get_bars(self, symbol, tf, start, end):
        self.polls.append((symbol, end))
        return super().get_bars(symbol, tf, start, end)


# --- venues from config/accounts.yaml -------------------------------------------------------

def test_paper_venues_put_every_adapter_behind_one_order_guard():
    led = Ledger(":memory:", git_commit="t")
    pv = _venues({"forex", "stocks"}, led, lambda c: 1.0)
    fx, stk = pv.venues["forex"], pv.venues["stocks"]
    assert isinstance(fx, FakeOanda) and fx.guard is pv.guard and isinstance(pv.guard, OrderGuard)
    assert isinstance(stk, paper.Serialized) and isinstance(stk.inner, GuardedBroker) and stk.guard is pv.guard
    assert stk.account().account_id == "998877" and stk.simulated     # calls and attributes pass through
    assert pv.guard.agent_accounts == {"101-001", "998877"}
    assert pv.accounts == {"forex": "oanda-practice", "stocks": "moomoo-simulate"}


def test_unresolved_or_missing_accounts_refuse_the_start():
    led = Ledger(":memory:", git_commit="t")
    unresolved = [replace(ACCOUNTS[0], account_id=""), ACCOUNTS[1]]
    with pytest.raises(VenuesMissing, match="trade-x setup"):
        _venues({"forex"}, led, lambda c: 1.0, accounts=unresolved)
    with pytest.raises(VenuesMissing, match="stocks"):
        _venues({"stocks"}, led, lambda c: 1.0, accounts=ACCOUNTS[:1])


def block_adapters(monkeypatch):
    """Make the adapter modules unimportable (as before Ray's patch lands), even if another
    test already imported them: nothing can then reach a real venue."""
    import tradex.execution
    for name in ("oanda", "moomoo"):
        monkeypatch.setitem(sys.modules, f"tradex.execution.{name}", None)
        monkeypatch.delattr(tradex.execution, name, raising=False)


def test_missing_adapter_modules_refuse_the_start(monkeypatch):
    block_adapters(monkeypatch)
    led = Ledger(":memory:", git_commit="t")
    for classes in ({"forex"}, {"stocks"}):
        with pytest.raises(VenuesMissing, match="venue adapter patch"):
            paper.paper_venues(classes, led, lambda c: 1.0, load=lambda *a: ACCOUNTS)


# --- account currency: USD_SGD on the stream, live conversion everywhere ------------------------

def _build(clock, quotes, venues, stream=None, specs=None, **kw):
    specs = specs or _mixed_specs()
    fx = sorted({u for s in specs if s.asset_class == "forex" for u in s.universe})
    stream = stream or FakeStream(stream_instruments(fx, {"SGD"}))
    args = dict(history={"forex": RecordedHistory()}, stream=stream, quotes=quotes, clock=clock,
                config=RuntimeConfig(), warmup_bars=400, venues=venues)
    args.update(kw)
    return build_runtime("paper", specs, Ledger(":memory:", git_commit="t"), **args)


def test_sgd_account_needs_usd_sgd_on_the_stream_and_converts_at_the_live_quote():
    assert stream_instruments(["EUR_USD"], {"SGD", "USD"}) == ["EUR_USD", "USD_SGD"]
    clock = ReplayClock(START)
    quotes = QuoteBook(30, clock=clock.now)
    led = Ledger(":memory:", git_commit="t")
    from tradex.runtime.fx import LiveQuoteRates
    rates = LiveQuoteRates(quotes)
    pv = _venues({"forex"}, led, lambda c: rates.usd_per_unit(c, clock.now()))
    with pytest.raises(RealDataMissing, match="USD_SGD"):
        _build(clock, quotes, pv.venues, stream=FakeStream(["EUR_USD", "USD_JPY"]))
    rt = _build(clock, quotes, pv.venues, accounts=pv.accounts)
    assert "USD_SGD on the Oanda stream" in rt.readiness()
    with pytest.raises(MissingRate):                              # no USD_SGD quote yet: never a constant
        rt.book.account()
    rt.feed.on_tick(Tick("USD_SGD", START, 1.2999, 1.3001))       # the stream fills the quote book
    assert rt.book.account().equity == pytest.approx(10_000 / 1.3)
    assert pv.venues["forex"].to_usd("SGD") == pytest.approx(1 / 1.3)
    assert rt.book.accounts == {"forex": "oanda-practice"}


def test_missing_account_rate_in_paper_is_a_fault_not_a_guess():
    clock = ReplayClock(START)
    quotes = QuoteBook(30, clock=clock.now)
    from tradex.runtime.fx import LiveQuoteRates
    rates = LiveQuoteRates(quotes)
    pv = _venues({"forex"}, Ledger(":memory:"), lambda c: rates.usd_per_unit(c, clock.now()))
    rt = _build(clock, quotes, pv.venues)
    pv.venues["forex"].convert_fills = True
    t = START + pd.Timedelta(hours=1)
    clock.set(t)
    rt.store.append("EUR_USD", RecordedHistory().frames["EUR_USD"].loc[[START]])
    rt.core.on_bar_close("H1", t)
    faults = [h for h in rt.core.ledger.rows(kind="health") if not h["ok"]]
    assert faults and any("SGD" in h["detail"] for h in faults)
    assert not rt.core.ledger.rows(kind="order")


# --- the stop guardian for moomoo -----------------------------------------------------------------

def _mixed_with_stocks():
    return _mixed_specs() + [_stock_spec()]


def test_moomoo_gets_a_stop_guardian_that_runs_in_a_thread_and_is_watched():
    clock = ReplayClock(START)
    quotes = QuoteBook(30, clock=clock.now)
    pv = _venues({"forex", "stocks"}, Ledger(":memory:"), lambda c: 1.0)
    hist = {"forex": RecordedHistory(), "stocks": StockHistory()}
    feeds = {"stocks": PolledBarFeed(hist["stocks"], ["AAA"], "H1")}
    with pytest.raises(RealDataMissing, match="marks"):
        _build(clock, quotes, pv.venues, specs=_mixed_with_stocks(), history=hist, feeds=feeds,
               guardian_factory=FakeGuardian)
    rt = _build(clock, quotes, pv.venues, specs=_mixed_with_stocks(), history=hist, feeds=feeds,
                guardian_factory=FakeGuardian, marks={"stocks": lambda syms: {}})
    (g,) = rt.guardians
    assert g.asset_class == "stocks" and g.interval_s <= 60 and g.venue is pv.venues["stocks"]
    assert "stop_guardian:stocks" in rt.readiness()
    rt.before_close = None
    rt.runner.before_close("H1", START)                           # never pinged yet: one loud fault
    rt.runner.before_close("H1", START)
    silent = [h for h in rt.core.ledger.rows(kind="health") if h["check"] == "stop_guardian"]
    assert [h["ok"] for h in silent] == [False]
    rt.start()
    assert g.ticked.wait(2)
    rt.runner.before_close("H1", START)
    assert [h["ok"] for h in rt.core.ledger.rows(kind="health") if h["check"] == "stop_guardian"] == [False, True]
    rt.stop_event.set()
    for t in rt.threads:
        t.join(2)
    assert not any(t.is_alive() for t in rt.threads)


def test_guardian_faults_are_throttled_and_quiet_when_the_market_is_closed():
    clock = ReplayClock(START)
    pv = _venues({"forex", "stocks"}, Ledger(":memory:"), lambda c: 1.0)
    hist = {"forex": RecordedHistory(), "stocks": StockHistory()}
    closed = {"v": False}
    rt = _build(clock, QuoteBook(30, clock=clock.now), pv.venues, specs=_mixed_with_stocks(), history=hist,
                feeds={"stocks": PolledBarFeed(hist["stocks"], ["AAA"], "H1")}, guardian_factory=FakeGuardian,
                marks={"stocks": lambda s: {}}, quiet_marks=lambda t: closed["v"])
    fault = rt.guardians[0].on_fault
    fault("stop_guardian", "AAA: no live mark; its stop is unwatched this minute")
    fault("stop_guardian", "AAA: no live mark; its stop is unwatched this minute")       # a minute later: same
    clock.set(START + pd.Timedelta(minutes=16))
    fault("stop_guardian", "AAA: no live mark; its stop is unwatched this minute")
    closed["v"] = True
    clock.set(START + pd.Timedelta(minutes=40))
    fault("stop_guardian", "AAA: no live mark; its stop is unwatched this minute")       # market closed
    fault("stop_guardian", "X stop exit not sent: OrderRejected")                       # still loud
    rows = [h["detail"] for h in rt.core.ledger.rows(kind="health")]
    assert rows.count("AAA: no live mark; its stop is unwatched this minute") == 2
    assert "X stop exit not sent: OrderRejected" in rows


def test_ray_stop_guardian_plugs_in_when_his_patch_is_committed():
    guardian = pytest.importorskip("tradex.execution.guardian")
    clock = ReplayClock(START)
    pv = _venues({"forex", "stocks"}, Ledger(":memory:"), lambda c: 1.0)
    hist = {"forex": RecordedHistory(), "stocks": StockHistory()}
    rt = _build(clock, QuoteBook(30, clock=clock.now), pv.venues, specs=_mixed_with_stocks(), history=hist,
                feeds={"stocks": PolledBarFeed(hist["stocks"], ["AAA"], "H1")}, marks={"stocks": lambda s: {}})
    (g,) = rt.guardians
    assert isinstance(g, guardian.StopGuardian) and g.interval_s <= 60
    assert g.tick() == [] and g.alive(clock.now())


# --- the polled stock feed and live marks ---------------------------------------------------------

def test_polled_feed_appends_only_closed_bars_once_per_close_and_faults_per_symbol():
    clock = ReplayClock(START)
    pv = _venues({"forex", "stocks"}, Ledger(":memory:"), lambda c: 1.0)
    hist = StockHistory()
    feed = PolledBarFeed(hist, ["AAA"], "H1")
    rt = _build(clock, QuoteBook(30, clock=clock.now), pv.venues, specs=_mixed_with_stocks(),
                history={"forex": RecordedHistory(), "stocks": hist}, feeds={"stocks": feed},
                guardian_factory=FakeGuardian, marks={"stocks": lambda s: {}})
    assert rt.store.frames["AAA"].index[-1] == START - pd.Timedelta(hours=1)
    n = len(hist.polls)
    t = START + pd.Timedelta(hours=1)
    feed.before_close("H1", t)
    feed.before_close("H4", t)                                   # same close, other timeframe: no second poll
    assert len(hist.polls) == n + 1 and rt.store.frames["AAA"].index[-1] == START
    hist.get_bars = lambda *a: (_ for _ in ()).throw(OSError("timeout"))
    feed.before_close("H1", t + pd.Timedelta(hours=1))
    assert [(h["check"], h["ok"]) for h in rt.core.ledger.rows(kind="health")][-1] == ("feed", False)


def test_opend_marks_use_fresh_snapshots_in_the_regular_session_only():
    class Ctx:
        def get_market_snapshot(self, codes):
            return 0, pd.DataFrame([{"code": "US.AAA", "last_price": 101.5, "update_time": "2026-10-05 10:59:30"},
                                    {"code": "US.BBB", "last_price": 50.0, "update_time": "2026-10-05 10:50:00"}])
    now = pd.Timestamp("2026-10-05 11:00", tz="America/New_York").tz_convert("UTC")
    assert us_regular_session(now) and not us_regular_session(now + pd.Timedelta(hours=6))
    m = OpenDMarks(Ctx(), clock=lambda: now)
    assert m(["AAA", "BBB"]) == {"AAA": 101.5}                    # BBB is 10 minutes old: left out
    assert OpenDMarks(Ctx(), clock=lambda: now + pd.Timedelta(hours=6))(["AAA"]) == {}


# --- run_paper and the CLI ------------------------------------------------------------------------

def _deps(stops_after=1, has_secret=lambda n: True, venues=None):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > stops_after
    clock = ReplayClock(START)
    hist = {"forex": RecordedHistory(), "stocks": StockHistory()}
    deps = paper.PaperDeps(has_secret=has_secret, history=lambda: hist, stream=FakeStream,
                           venues=venues or (lambda classes, ledger, to_usd, **kw: _venues(classes, ledger, to_usd)),
                           marks=lambda: (lambda syms: {}), clock=clock, sleep=lambda s: None, stop=stop,
                           guardian_factory=FakeGuardian)
    return deps, calls


def test_run_paper_starts_the_loop_when_every_check_passes(tmp_path, capsys):
    deps, calls = _deps()
    rc = paper.run_paper(_mixed_with_stocks(), tmp_path / "live.sqlite", RuntimeConfig(), deps,
                         state_dir=tmp_path / "state")
    out = capsys.readouterr().out
    assert rc == 0 and calls["n"] == 2
    assert "READY (paper)" in out and "USD_SGD on the Oanda stream" in out and "paper trading started" in out
    assert Ledger(tmp_path / "live.sqlite").verify() == (True, None)


def test_run_paper_says_what_is_missing_and_connects_nothing(tmp_path, capsys):
    built = []
    deps, _ = _deps(has_secret=lambda n: n != "oanda_token",
                    venues=lambda *a, **k: built.append(1))
    assert paper.run_paper(_mixed_with_stocks(), tmp_path / "l.sqlite", RuntimeConfig(), deps) == 1
    assert "missing secrets oanda_token" in capsys.readouterr().err and not built

    def refuse(*a, **k):
        raise VenuesMissing("tradex/execution/oanda.py is not there: commit the venue adapter patch")
    deps, calls = _deps(venues=refuse)
    assert paper.run_paper(_mixed_specs(), tmp_path / "l.sqlite", RuntimeConfig(), deps) == 1
    assert "venue adapter patch" in capsys.readouterr().err and calls["n"] == 0


def test_run_paper_does_not_start_when_a_readiness_check_fails(tmp_path, capsys):
    deps, calls = _deps()
    deps.has_secret = lambda n: n != "telegram_chat_id"
    assert paper.run_paper(_mixed_specs(), tmp_path / "l.sqlite", RuntimeConfig(), deps) == 1
    io = capsys.readouterr()
    assert "telegram_chat_id" in io.err and calls["n"] == 0


def test_cli_live_mode_is_refused_and_paper_goes_through_run_paper(tmp_path, capsys, monkeypatch):
    from test_runtime_build import _write_fx_strategy
    _write_fx_strategy(tmp_path / "s")
    assert main(["run", "--mode", "live", "--strategies", str(tmp_path / "s")]) == 2
    assert "live mode is not allowed" in capsys.readouterr().err
    seen = {}

    def fake_run_paper(specs, ledger, cfg):
        seen.update(specs=[s.id for s in specs], ledger=ledger)
        return 0
    monkeypatch.setattr(paper, "run_paper", fake_run_paper)
    assert main(["run", "--mode", "paper", "--strategies", str(tmp_path / "s"),
                 "--ledger", str(tmp_path / "x.sqlite")]) == 0
    assert seen == {"specs": ["fx-t"], "ledger": str(tmp_path / "x.sqlite")}
