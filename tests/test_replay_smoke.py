"""Full-stack replay smoke test: recorded bars -> Oanda-style ticks -> quote book and bar builders ->
bar store -> scheduler -> trading core -> gates -> venue adapter behind the order guard -> ledger.

Everything the live runtime wires is real here except the network: the stream is replayed from
frozen fixtures (tests/fixtures/replay, see make_replay_fixtures.py), the venue is a SimBroker
standing in for Oanda behind the production OrderGuard, and the clock is a ReplayClock. No order
leaves the process. A change that breaks the wiring, the ledger chain, the guard contract or
determinism fails here before it can reach a paper run.
"""
from pathlib import Path

import pandas as pd
import pytest

import tradex.costs.models as cm
from test_runtime_core import _mixed_specs
from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.data.oanda import QuoteBook, Tick
from tradex.execution.guard import GuardedBroker, OrderGuard, ledger_verdicts
from tradex.execution.sim import SimBroker
from tradex.runtime.build import build_runtime
from tradex.runtime.config import RuntimeConfig

FIX = Path(__file__).parent / "fixtures" / "replay"
WARM = 700                                             # bars of history before the replay starts
SPREAD = {"EUR_USD": 0.00014, "USD_JPY": 0.014}        # 1.4 pips, as the practice stream shows
HOUR = pd.Timedelta(hours=1)


@pytest.fixture(autouse=True)
def _reset_run_mode():
    yield
    cm.configure_run_mode("backtest")


def load_fixture() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in SPREAD:
        df = pd.read_csv(FIX / f"{sym}_H1.csv", index_col="ts", parse_dates=True)
        df.index = df.index.astype("datetime64[ns, UTC]")
        out[sym] = df.astype(float)
    return out


class History:
    """Start-up history: the recorded bars before the replay starts, sliced like Oanda's candles endpoint."""

    def __init__(self, frames):
        self.frames = frames

    def get_bars(self, symbol, tf, start, end):
        df = self.frames[symbol]
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


class Silent:
    heartbeats = 0

    def ticks(self):
        return iter(())


def bar_ticks(sym: str, t: pd.Timestamp, row: pd.Series):
    """Four ticks that rebuild one H1 bar exactly: open, high, low, close (mid prices)."""
    half = SPREAD[sym] / 2
    for off, px in ((0, row["open"]), (600, row["high"]), (1200, row["low"]), (3590, row["close"])):
        yield Tick(sym, t + pd.Timedelta(seconds=off), px - half, px + half)


def run_stack(commands: dict[int, str] | None = None, bars: int = 400):
    """Replay ``bars`` hours through the runtime. ``commands`` maps a bar number to a command Ray sends
    (as the Telegram service would) during that bar. Returns (runtime, ledger, frames, start)."""
    frames = load_fixture()
    start = frames["EUR_USD"].index[WARM]
    clock = ReplayClock(start)
    ledger = Ledger(":memory:", git_commit="smoke", run_id="smoke")
    venue = GuardedBroker(SimBroker(10_000, account_id="sim"), OrderGuard(ledger_verdicts(ledger), {"sim"}))
    rt = build_runtime("paper", _mixed_specs(), ledger, history={"forex": History(frames)}, stream=Silent(),
                       quotes=QuoteBook(30, clock=clock.now), venues={"forex": venue}, clock=clock,
                       config=RuntimeConfig(), warmup_bars=WARM)
    for i in range(bars):
        t = start + i * HOUR
        for sym in SPREAD:
            for tick in bar_ticks(sym, t, frames[sym].loc[t]):
                clock.set(tick.time)
                rt.feed.on_tick(tick)
        if commands and i in commands:
            ledger.add_command((t + pd.Timedelta(minutes=30)).isoformat(), "telegram", commands[i])
        clock.set(t + HOUR + pd.Timedelta(seconds=6))               # past the close and the 5 s grace
        rt.runner.tick()
    return rt, ledger, frames, start


@pytest.fixture(scope="module")
def stack():
    return run_stack()


def test_stream_rebuilds_the_recorded_bars_exactly(stack):
    rt, _, frames, start = stack
    for sym in SPREAD:
        built = rt.store.frames[sym].loc[start:]
        want = frames[sym].loc[start:]
        assert len(built) >= 399
        pd.testing.assert_frame_equal(built[["open", "high", "low", "close"]],
                                      want.loc[built.index, ["open", "high", "low", "close"]], rtol=1e-9, atol=1e-9,
                                      check_freq=False)


def test_the_ledger_chain_is_intact_and_no_fault_was_raised(stack):
    _, ledger, _, _ = stack
    assert ledger.verify() == (True, None)
    faults = [h for h in ledger.rows(kind="health") if not h["ok"]]
    assert not faults, faults[:3]


def test_the_stack_actually_trades(stack):
    _, ledger, _, _ = stack
    kinds = {r["kind"] for r in ledger.rows()}
    assert {"config_version", "vote", "plan", "verdict", "order", "fill", "close"} <= kinds
    assert [o for o in ledger.rows(kind="order") if o["book"] == "ensemble" and o["purpose"] == "entry"]


def test_every_ensemble_entry_cites_an_accepted_verdict_it_fits_inside(stack):
    _, ledger, _, _ = stack
    entries = [o for o in ledger.rows(kind="order") if o["book"] == "ensemble" and o["purpose"] == "entry"]
    verdicts = {v["verdict_id"]: v for v in ledger.rows(kind="verdict")}
    used = {}
    for o in entries:
        v = verdicts[f"{o['decision_id']}-v"]
        assert v["outcome"] == "accepted" and o["qty"] <= v["qty"] + 1e-9
        used[v["verdict_id"]] = used.get(v["verdict_id"], 0) + 1
    assert max(used.values()) == 1, "a verdict sized more than one entry order"


def test_the_decision_chain_is_complete_for_every_decision(stack):
    _, ledger, _, _ = stack
    ens = [p for p in ledger.rows(kind="plan") if p["book"] == "ensemble"]
    assert ens
    for p in ens:
        kinds = [r["kind"] for r in ledger.why(p["decision_id"])]
        assert kinds[0] == "plan" and "vote" in kinds
        if "order" in kinds:
            assert "verdict" in kinds
        if "close" in kinds:
            assert "fill" in kinds and kinds.index("order") < kinds.index("fill") < kinds.index("close")
    fills = {f["decision_id"] for f in ledger.rows(kind="fill") if f["book"] == "ensemble"}
    orders = {o["decision_id"] for o in ledger.rows(kind="order") if o["book"] == "ensemble"}
    assert fills <= orders


def test_each_scheduled_close_ran_exactly_once(stack):
    _, ledger, _, start = stack
    jobs = ledger.jobs("scheduler")
    closes = [(j["payload"]["tf"], j["payload"]["close"]) for j in jobs]
    assert len(closes) == len(set(closes))
    assert all(j["status"] == "done" for j in jobs)
    assert {tf for tf, _ in closes} == {"H1", "H4"}


def test_equity_snapshots_match_the_venue(stack):
    rt, ledger, _, _ = stack
    snaps = [s for s in ledger.rows(kind="snapshot") if s["book"] == "ensemble"]
    assert snaps and all(s["equity_usd"] > 0 for s in snaps)
    assert all(s["config_hash"] == snaps[0]["config_hash"] for s in snaps)


def test_the_same_replay_twice_gives_the_same_decisions():
    a = run_stack(bars=120)[1]
    b = run_stack(bars=120)[1]
    assert a.digest() == b.digest()
    assert a.digest() != Ledger(":memory:").digest()


def test_a_pause_command_stops_new_entries_but_not_the_book():
    rt, ledger, _, start = run_stack(commands={100: "pause"}, bars=200)
    cmd = ledger.db.execute("SELECT * FROM commands").fetchone()
    assert cmd["applied_at"] is not None and "no new entries" in cmd["result"]
    after = (start + 101 * HOUR).isoformat()
    late = [o for o in ledger.rows(kind="order") if o["book"] == "ensemble" and o["purpose"] == "entry" and o["time"] > after]
    assert not late
    assert [o for o in ledger.rows(kind="order") if o["book"].startswith("virtual:") and o["time"] > after]
    assert rt.core.paused and ledger.verify() == (True, None)


def test_a_flatten_command_closes_a_live_position_and_leaves_nothing_pending(stack):
    ledger = stack[1]
    first = {}
    for f in ledger.rows(kind="fill"):
        if f["book"] == "ensemble":
            first.setdefault(f["decision_id"], pd.Timestamp(f["time"]))
    closed = {c["decision_id"]: pd.Timestamp(c["time"]) for c in ledger.rows(kind="close") if c["book"] == "ensemble"}
    held = sorted(t for d, t in first.items() if closed.get(d, pd.Timestamp.max.tz_localize("UTC")) - t >= 5 * HOUR)
    assert held, "the fixture must hold at least one ensemble position for five hours"
    start = load_fixture()["EUR_USD"].index[WARM]
    i = int((held[0] - start) / HOUR) + 2                             # two bars after the fill, position still open
    rt, led, _, _ = run_stack(commands={i: "flatten"}, bars=i + 6)
    res = led.db.execute("SELECT result FROM commands").fetchone()["result"]
    assert "exits sent for 1 of 1" in res or "exits sent for 2 of 2" in res, res
    assert rt.core.paused and rt.core.agent_positions() == [] and rt.core.pending_entries() == []
    assert any(o["purpose"].startswith("exit") and o["client_order_id"].endswith("-flatten") for o in led.rows(kind="order"))
    assert led.verify() == (True, None)
