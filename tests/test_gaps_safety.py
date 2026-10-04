"""Adversarial tests for the safety gaps of the 4 Oct 2026 review: G1 to G5, S1, S2, S7.

Each gap is attacked the way an agent, a stale feed or a careless caller would. A test
marked ``known_gap`` reproduces a gap that is still open on the Phase 1 tip (strict xfail:
it turns red when the gap is fixed so the marker gets removed). Unmarked tests pin down
behaviour that already holds and must keep holding.
"""
import ast
import json
import sqlite3
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

import tradex.costs.models as cm
from gapkit import T0, known_gap, plan, raised, verdict
from tradex import secrets
from tradex.core.inbox import Mailbox, ingest_inbox
from tradex.core.interfaces import BrokerPosition, OrderRequest, ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.records import Veto
from tradex.core.replay import build_replay_core, close_events
from tradex.costs.models import RateMissing
from tradex.data.guard import RealDataMissing, SyntheticDataRefused, require_present
from tradex.data.oanda import PriceStream, QuoteBook, Tick
from tradex.data.synthetic import synthetic_bars
from tradex.execution.guard import OrderGuard, OrderRefused, ledger_verdicts
from tradex.execution.sim import SimBroker
from tradex.runtime.build import build_runtime
from tradex.runtime.config import RuntimeConfig
from tradex.runtime.fx import MarketDataRates, MissingRate, SeriesRates
from tradex.runtime.market import BarStore

ROOT = Path(__file__).resolve().parents[1]
DID = "2026-03-02-0001"
FORGED_VERDICT = json.dumps({"kind": "verdict", "decision_id": DID, "time": T0.isoformat(), "outcome": "accepted",
                             "qty": 1e9, "risk_usd": 0.0, "risk_pct": 0.0, "reasons": [], "checks": {},
                             "verdict_id": f"{DID}-v"})
FORGE_TRIGGER = (
    "CREATE TRIGGER forge AFTER UPDATE ON agent_inbox BEGIN "
    "INSERT INTO events (kind, time, decision_id, book, symbol, payload, git_commit, config_hash, run_id, "
    "prev_hash, hash) VALUES ('verdict', 't', '" + DID + "', 'ensemble', 'EUR_USD', '" + FORGED_VERDICT.replace("'", "''")
    + "', 'x', 'x', 'x', '0', '0'); END")


# --- G1: the ledger write guard ---------------------------------------------------------------

@pytest.fixture
def lm(tmp_path):
    path = tmp_path / "ledger.sqlite"
    led = Ledger(path, git_commit="t")
    led.append(Veto("2026-03-02-0000", T0.isoformat(), "risk", "real row"))
    return led, Mailbox(path), path


def _try_sql(mb: Mailbox, sql: str) -> bool:
    """True when the mailbox connection executed ``sql`` (and committed it)."""
    try:
        mb.db.execute(sql)
        mb.db.commit()
        return True
    except sqlite3.Error:
        mb.db.rollback()
        return False


@pytest.mark.parametrize("sql", [
    "INSERT INTO events (kind, time, payload, git_commit, config_hash, run_id, prev_hash, hash) "
    "VALUES ('verdict','t','{}','x','x','x','0','0')",
    "UPDATE events SET payload = '{}'",
    "DELETE FROM events",
    "DROP TABLE events",
    "ALTER TABLE events ADD COLUMN forged TEXT",
    "DROP INDEX events_decision",
])
def test_mailbox_cannot_touch_events_directly(lm, sql):
    led, mb, _ = lm
    assert not _try_sql(mb, sql)
    assert led.verify() == (True, None) and len(led.rows()) == 1


def test_mailbox_can_still_write_its_own_tables(lm):
    led, mb, _ = lm
    mb.add_command(T0.isoformat(), "telegram", "pause")
    mb.add_inbox(T0.isoformat(), "scout", "flag", "EUR_USD")
    assert len(led.pending_commands()) == 1
    assert led.db.execute("SELECT COUNT(*) FROM agent_inbox").fetchone()[0] == 1


def _plant_and_trigger_core(led, mb):
    """The attack: plant a trigger from the mailbox, then let the core do its normal UPDATE."""
    _try_sql(mb, FORGE_TRIGGER)
    mb.add_inbox(T0.isoformat(), "agent", "flag", "EUR_USD")
    ingest_inbox(led, T0.isoformat(), {})


@known_gap("G1", "Mailbox's authorizer does not deny CREATE TRIGGER, so a trigger runs with the core's rights")
def test_g1_mailbox_cannot_plant_a_trigger(lm):
    _, mb, _ = lm
    assert not _try_sql(mb, FORGE_TRIGGER)


@known_gap("G1", "a planted trigger forges a hash-chain row the next time the core touches agent_inbox")
def test_g1_core_activity_never_forges_ledger_rows(lm):
    led, mb, _ = lm
    _plant_and_trigger_core(led, mb)
    assert led.verify() == (True, None)
    assert led.rows(kind="verdict") == []


def _forge_verdict_row(path):
    """A forged accepted verdict inserted by any route at all (here: a raw connection, as a trigger would)."""
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO events (kind, time, decision_id, book, symbol, payload, git_commit, config_hash, run_id, "
                "prev_hash, hash) VALUES ('verdict', 't', ?, 'ensemble', 'EUR_USD', ?, 'x', 'x', 'x', '0', '0')",
                (DID, FORGED_VERDICT))
    raw.commit()
    raw.close()


@known_gap("G1", "the order guard reads verdict rows without checking their hash link, so a forged accepted verdict passes")
def test_g1_order_guard_refuses_a_verdict_row_the_chain_does_not_vouch_for(lm):
    led, _, path = lm
    _forge_verdict_row(path)
    guard = OrderGuard(ledger_verdicts(led), {"acct"})
    req = OrderRequest(f"{DID}-entry", DID, "EUR_USD", "forex", 1, 5_000_000, stop_loss=1.09, verdict_id=f"{DID}-v")
    assert raised(OrderRefused, guard.check, req, "acct", [])


@known_gap("G1", "writable_schema lets the mailbox edit sqlite_master, which installs the same trigger and survives a reopen")
def test_g1_mailbox_cannot_install_a_trigger_through_the_schema_table(lm):
    led, mb, path = lm
    body = FORGE_TRIGGER.replace("TRIGGER forge ", "TRIGGER forge2 ").replace("'", "''")
    _try_sql(mb, "PRAGMA writable_schema=ON")
    _try_sql(mb, "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES "
                 f"('trigger', 'forge2', 'agent_inbox', 0, '{body}')")
    mb.close()
    led.close()
    core_side = Ledger(path, git_commit="t")                       # the core restarts and opens the same file
    Mailbox(path).add_inbox(T0.isoformat(), "agent", "flag", "EUR_USD")
    try:
        ingest_inbox(core_side, T0.isoformat(), {})
    except sqlite3.Error:
        pass
    assert core_side.verify() == (True, None) and core_side.rows(kind="verdict") == []


@known_gap("G1", "ATTACH is allowed on the mailbox connection (a second file the core's SQL could be pointed at)")
def test_g1_mailbox_cannot_attach_databases(lm):
    _, mb, _ = lm
    assert not _try_sql(mb, "ATTACH DATABASE ':memory:' AS other")


# --- G2: the order guard (tradex/execution/guard.py is protected: patch for Ray) -------------------

def _guard(*records):
    led = Ledger(":memory:", git_commit="t")
    for r in records:
        led.append(r)
    return OrderGuard(ledger_verdicts(led), {"acct"})


def _entry(coid="e1", qty=1000.0, symbol="EUR_USD", side=1, stop=1.09, verdict_id=f"{DID}-v", did=DID, **kw):
    return OrderRequest(coid, did, symbol, "forex", side, qty, stop_loss=stop, verdict_id=verdict_id, **kw)


def test_guard_baseline_accepts_one_order_inside_the_verdict():
    _guard(plan(), verdict(qty=1000)).check(_entry(qty=1000), "acct", [])


@pytest.mark.parametrize("make, why", [
    (lambda: (_entry(qty=1000.5), "acct"), "quantity above the verdict"),
    (lambda: (_entry(did="2026-03-02-0009"), "acct"), "another decision's id"),
    (lambda: (_entry(verdict_id=""), "acct"), "no verdict cited"),
    (lambda: (_entry(verdict_id="nope-v"), "acct"), "unknown verdict"),
    (lambda: (_entry(), "ray-main"), "account that is not agent-owned"),
    (lambda: (_entry(account="ray"), "acct"), "Ray's own account"),
    (lambda: (_entry(qty=0), "acct"), "zero quantity"),
])
def test_guard_refuses_the_basic_abuses(make, why):
    req, acct = make()
    assert raised(OrderRefused, _guard(plan(), verdict(qty=1000)).check, req, acct, []), why


def test_guard_refuses_an_entry_citing_a_rejected_verdict():
    g = _guard(plan(), verdict(qty=1000, outcome="rejected"))
    assert raised(OrderRefused, g.check, _entry(), "acct", [])


def _pos(qty=1000.0, direction=1, did=DID, symbol="EUR_USD"):
    return BrokerPosition(did, symbol, "forex", direction, qty, 1.10, T0, 1.09, None, "agent")


@pytest.mark.parametrize("req, positions", [
    (OrderRequest("x1", DID, "EUR_USD", "forex", -1, 2000.0, purpose="exit"), [_pos(1000)]),                 # too big
    (OrderRequest("x2", DID, "EUR_USD", "forex", 1, 500.0, purpose="exit"), [_pos(1000)]),                   # adds, not reduces
    (OrderRequest("x3", DID, "GBP_USD", "forex", -1, 500.0, purpose="exit"), [_pos(1000)]),                  # wrong symbol
    (OrderRequest("x4", "2026-03-02-0007", "EUR_USD", "forex", -1, 500.0, purpose="exit"), [_pos(1000)]),    # other decision
    (OrderRequest("x5", DID, "EUR_USD", "forex", -1, 500.0, purpose="exit"), []),                            # nothing open
])
def test_guard_exits_must_reduce_an_open_agent_position(req, positions):
    assert raised(OrderRefused, _guard().check, req, "acct", positions)


def test_guard_never_blocks_a_flatten_for_want_of_a_verdict():
    _guard().check(OrderRequest("x6", DID, "EUR_USD", "forex", -1, 1000.0, purpose="exit"), "acct", [_pos(1000)])


@known_gap("G2", "a risk verdict can be reused: every order citing it is checked against the full size again")
def test_g2_one_verdict_cannot_size_several_orders():
    g = _guard(plan(), verdict(qty=1000))
    placed = [_entry(coid=f"e{i}", qty=1000.0) for i in range(3)]
    ok = []
    for r in placed:
        if not raised(OrderRefused, g.check, r, "acct", []):
            ok.append(r)                                           # the core would have placed it
    assert sum(r.qty for r in ok) <= 1000.0


@known_gap("G2", "the guard never compares the order's symbol with the plan the verdict belongs to")
def test_g2_entry_must_match_the_symbol_of_its_plan():
    g = _guard(plan(symbol="EUR_USD"), verdict(qty=1000))
    assert raised(OrderRefused, g.check, _entry(symbol="USD_JPY", stop=149.0), "acct", [])


@known_gap("G2", "the guard never compares the order's side with the plan's direction")
def test_g2_entry_must_match_the_direction_of_its_plan():
    g = _guard(plan(direction=1), verdict(qty=1000))
    assert raised(OrderRefused, g.check, _entry(side=-1, stop=1.11), "acct", [])


@known_gap("G2", "an entry with no stop loss passes the guard")
def test_g2_entry_without_a_stop_is_refused():
    g = _guard(plan(), verdict(qty=1000))
    assert raised(OrderRefused, g.check, _entry(stop=None), "acct", [])


@known_gap("G2", "the entry's stop is not compared with the plan's stop (a far stop passes)")
def test_g2_entry_stop_must_be_the_plans_stop_or_tighter():
    g = _guard(plan(entry=1.10, stop=1.09), verdict(qty=1000))
    assert raised(OrderRefused, g.check, _entry(stop=0.50), "acct", [])


# --- G3: flatten and pause ----------------------------------------------------------------------

def _stock_frames():
    return {"AAA": synthetic_bars(500, seed=3, price=60.0), "BBB": synthetic_bars(500, seed=4, price=40.0)}


def _specs():
    from test_spine import _two_family_specs
    return _two_family_specs()


class NeverFillsBroker:
    """A venue whose entries stay open (a limit away from the market): the core must not feed it bars."""

    simulated = False

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_flatten_cancels_an_entry_that_has_not_filled_and_nothing_fills_afterwards():
    frames = _stock_frames()
    start = frames["AAA"].index[260]
    led = Ledger(":memory:", git_commit="t")
    base = SimBroker(10_000, bar=pd.Timedelta(days=1))
    core = build_replay_core(_specs(), frames, led, start, brokers={"ensemble": NeverFillsBroker(base)},
                             use_es=False)
    events = close_events(core.data, core.tfs, start)
    pending, flatten_at = [], None
    for i, (ts, tf) in enumerate(events):
        core.clock.set(ts)
        core.on_bar_close(tf, ts)
        pending = core.pending_entries()
        if pending:
            flatten_at = i
            break
    assert pending, "the fixture must leave at least one unfilled ensemble entry"
    did = pending[0]
    led.add_command((events[flatten_at][0]).isoformat(), "telegram", "flatten")
    ts, tf = events[flatten_at + 1]
    core.clock.set(ts)
    core.on_bar_close(tf, ts)
    assert core.pending_entries() == [] and core.paused
    assert [v["source"] for v in led.rows(kind="veto", decision_id=did)] == ["command"]
    for ts, tf in events[flatten_at + 2:]:
        core.clock.set(ts)
        core.on_bar_close(tf, ts)
    assert not [f for f in led.rows(kind="fill") if f["book"] == "ensemble"]
    cut = events[flatten_at][0].isoformat()
    assert not [o for o in led.rows(kind="order") if o["book"] == "ensemble" and o["time"] > cut]


def test_pause_stops_new_entries_but_keeps_virtual_books_running():
    frames = _stock_frames()
    start = frames["AAA"].index[260]
    led = Ledger(":memory:", git_commit="t")
    led.add_command(start.isoformat(), "telegram", "pause")
    core = build_replay_core(_specs(), frames, led, start, use_es=False)
    for ts, tf in close_events(core.data, core.tfs, start):
        core.clock.set(ts)
        core.on_bar_close(tf, ts)
    assert core.paused and not [o for o in led.rows(kind="order") if o["book"] == "ensemble"]
    assert [o for o in led.rows(kind="order") if o["book"].startswith("virtual:")]


@known_gap("G3", "commands are applied inside on_bar_close after the 'no bars closed' early return: a dead feed blocks /flatten")
def test_g3_pause_is_applied_even_when_no_bar_arrives():
    frames = _stock_frames()
    start = frames["AAA"].index[260]
    led = Ledger(":memory:", git_commit="t")
    core = build_replay_core(_specs(), frames, led, start, use_es=False)
    events = close_events(core.data, core.tfs, start)
    ts, tf = events[0]
    core.clock.set(ts)
    core.on_bar_close(tf, ts)
    led.add_command(ts.isoformat(), "telegram", "pause")
    dead = frames["AAA"].index[-1] + pd.Timedelta(days=30)          # the feed is dead: no bar closes at this time
    core.clock.set(dead)
    core.on_bar_close(tf, dead)
    assert led.pending_commands() == [] and core.paused


# --- G4: the real-data guard ----------------------------------------------------------------------

@pytest.fixture
def _reset_run_mode():
    yield
    cm.configure_run_mode("backtest")


class OandaPracticeHistory:
    """Named like the real thing, serves synthetic bars: the name-based guard cannot tell."""

    def __init__(self):
        from test_runtime_core import _fx_frames
        self.frames = _fx_frames()

    def get_bars(self, symbol, tf, start, end):
        df = self.frames[symbol]
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


class _Stream:
    heartbeats = 0

    def ticks(self):
        return iter(())


def _live_build(history):
    from test_runtime_core import _mixed_specs
    clock = ReplayClock(pd.Timestamp("2025-02-05 10:00", tz="UTC"))
    return build_runtime("live", _mixed_specs(), Ledger(":memory:", git_commit="t"), history={"forex": history},
                         stream=_Stream(), quotes=QuoteBook(30, clock=clock.now), clock=clock,
                         venues={"forex": SimBroker(10_000, account_id="sim")}, config=RuntimeConfig(),
                         warmup_bars=400)


def test_live_refuses_a_provider_whose_class_name_says_synthetic(_reset_run_mode):
    class SyntheticHistory(OandaPracticeHistory):
        pass
    with pytest.raises(SyntheticDataRefused):
        _live_build(SyntheticHistory())


@known_gap("G4", "the guard checks provider names only: synthetic frames behind an innocent class name run live")
def test_g4_live_refuses_synthetic_frames_from_an_innocently_named_provider(_reset_run_mode):
    assert raised((SyntheticDataRefused, RealDataMissing), _live_build, OandaPracticeHistory())


@known_gap("G4", "require_present only tests for empty/NaN, not for provenance: synthetic frames count as real data")
def test_g4_require_present_rejects_synthetic_frames_in_live():
    df = synthetic_bars(50, seed=1)
    assert raised((SyntheticDataRefused, RealDataMissing), require_present, "live", df, "AAA history")


@known_gap("G4", "BarStore.append accepts any frame in live mode; nothing carries the data's origin")
def test_g4_bar_store_in_live_refuses_synthetic_appends():
    store = BarStore("D1", {}, ReplayClock(T0))
    store.live = True
    assert raised((SyntheticDataRefused, RealDataMissing), store.append, "AAA", synthetic_bars(50, seed=1))


# --- G5: stale FX rates ---------------------------------------------------------------------------

NOW = pd.Timestamp("2026-10-04 12:00", tz="UTC")
OLD = pd.Series([1.10], index=[pd.Timestamp("2020-01-02", tz="UTC")])


class _FreshSource:
    def usd_per_unit(self, ccy, ts=None):
        raise RateMissing(f"no fresh quote for {ccy}")

    def spread_pips(self, symbol, ts=None):
        raise RateMissing(f"no fresh quote for {symbol}")


@known_gap("G5", "SeriesRates returns the last point of a series however old it is, in live mode too")
@pytest.mark.parametrize("mode", ["paper", "live"])
def test_g5_series_rates_refuse_a_six_year_old_point_in_live(mode):
    assert raised((MissingRate, RateMissing), SeriesRates({"EUR": OLD}, mode).usd_per_unit, "EUR", NOW)


@known_gap("G5", "costs.models.usd_per_unit takes the last supplied-series point at any age even in paper/live")
def test_g5_supplied_series_is_not_trusted_when_it_is_years_old(_reset_run_mode):
    cm.configure_run_mode("live", _FreshSource())
    assert raised((MissingRate, RateMissing), cm.usd_per_unit, "EUR", NOW, {"EUR": OLD})


def test_live_rates_without_any_source_never_use_a_constant(_reset_run_mode):
    cm.configure_run_mode("live", _FreshSource())
    assert raised(RateMissing, cm.usd_per_unit, "EUR", NOW)                 # no series: the live source answers (raises)
    assert raised(MissingRate, SeriesRates({}, "live").usd_per_unit, "EUR", NOW)
    assert SeriesRates({}, "replay").usd_per_unit("EUR", NOW) == pytest.approx(1.10)   # replay may use the constant


def test_quote_book_and_market_data_rates_reject_stale_inputs():
    clock = {"t": NOW}
    qb = QuoteBook(30, clock=lambda: clock["t"])
    qb.update(Tick("EUR_USD", NOW, 1.1, 1.1002))
    assert qb.usd_per_unit("EUR") == pytest.approx(1.1001)
    clock["t"] = NOW + pd.Timedelta(seconds=31)
    assert raised(RateMissing, qb.usd_per_unit, "EUR")
    assert raised(RateMissing, qb.spread_pips, "EUR_USD")

    store = BarStore("H1", {"EUR_USD": synthetic_bars(10, tf="H1", price=1.1, business_days=False, start="2026-10-04")},
                     ReplayClock(NOW))
    rates = MarketDataRates(store, "paper", pd.Timedelta(hours=4))
    far = store.frames["EUR_USD"].index[-1] + pd.Timedelta(hours=6)
    assert raised(MissingRate, rates.usd_per_unit, "EUR", far)


# --- S1: the Oanda stream ------------------------------------------------------------------------------

class _Runaway(Exception):
    pass


@known_gap("S1", "a 401 from the stream is an OSError, so PriceStream retries a bad token forever")
def test_s1_stream_stops_on_an_auth_failure():
    def connect():
        raise urllib.error.HTTPError("https://stream-fxpractice.oanda.com", 401, "Unauthorized", {}, None)

    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) > 25:
            raise _Runaway()

    stream = PriceStream(["EUR_USD"], "acct", "bad-token", connect=connect, sleep=sleep)
    try:
        list(stream.ticks())
    except _Runaway:
        raise AssertionError("PriceStream kept retrying a rejected token") from None
    except Exception:  # noqa: BLE001 - any real error that ends the loop is the fix
        pass
    assert len(sleeps) <= 5


# --- S2: logs and secrets ---------------------------------------------------------------------------------

@known_gap("S2", "setup_logging (the secret scrubber) is never called outside tests: production logs are unscrubbed")
def test_s2_the_cli_entry_point_switches_the_scrubber_on():
    calls = []
    for py in (ROOT / "tradex").rglob("*.py"):
        if py.name == "log.py" and py.parent.name == "ops":
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "setup_logging":
                calls.append(py.name)
    assert calls


class _Keyring:
    def __init__(self, d):
        self.d = d

    def get_password(self, service, name):
        return self.d.get(name)


@known_gap("S2", "any CI env var switches secrets.get to environment variables, so a stray CI=1 on Ray's Mac overrides the Keychain")
def test_s2_a_bare_ci_variable_does_not_override_the_keychain(monkeypatch):
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("TRADEX_ALLOW_ENV_SECRETS", raising=False)
    monkeypatch.setenv("TRADEX_OANDA_TOKEN", "from-the-environment")
    monkeypatch.setattr(secrets, "_keyring", lambda: _Keyring({"oanda_token": "from-the-keychain"}))
    assert secrets.get("oanda_token") == "from-the-keychain"


# --- S7: the protected-path check -----------------------------------------------------------------------------

@known_gap("S7", "tools/check_protected_paths.py only applies to agent/ branches: claude/ lane PRs touch risk files with green CI")
def test_s7_protected_path_check_covers_claude_branches(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_protected_paths", ROOT / "tools" / "check_protected_paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "changed_files", lambda base: ["tradex/risk/gate.py", "README.md"])
    assert mod.main(["--base", "origin/main", "--head-ref", "claude/p1-gap-wire"]) == 1
    assert mod.main(["--base", "origin/main", "--head-ref", "agent/researcher"]) == 1       # the existing rule holds


def test_protected_path_check_flags_agent_branches_and_lists_the_files(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_protected_paths", ROOT / "tools" / "check_protected_paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "changed_files", lambda base: ["tradex/execution/guard.py", "tradex/core/loop.py",
                                                             "config/gates/thresholds.yaml"])
    assert mod.main(["--base", "origin/main", "--head-ref", "agent/designer"]) == 1
    out = capsys.readouterr().out
    assert "tradex/execution/guard.py" in out and "config/gates/thresholds.yaml" in out and "loop.py" not in out
