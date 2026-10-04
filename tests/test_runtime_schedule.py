"""Bar-close scheduler with a fake clock, and the live runner driving the core."""
import pandas as pd
import pytest

from test_runtime_core import _decisions, _fx_frames, _mixed_specs
from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.replay import build_replay_core, run_replay
from tradex.execution.guard import GuardedBroker, OrderGuard, ledger_verdicts
from tradex.runtime.market import BarStore
from tradex.runtime.runner import LiveRunner
from tradex.runtime.schedule import AGENT, BarCloseScheduler, boundary

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def test_boundaries_match_the_bar_store_bins():
    t = T("2026-03-04 13:27")                                # a Wednesday
    assert boundary(t, "H1") == T("2026-03-04 13:00")
    assert boundary(t, "H4") == T("2026-03-04 12:00")
    assert boundary(t, "D1") == T("2026-03-04")
    assert boundary(t, "W1") == T("2026-03-02")              # Monday
    assert boundary(T("2026-03-04 12:00"), "H4") == T("2026-03-04 12:00")


class _Log:
    def __init__(self):
        self.calls = []

    def __call__(self, tf, ts):
        self.calls.append((tf, ts))


def test_one_close_per_timeframe_at_its_boundary_finer_first():
    led = Ledger(":memory:")
    s = BarCloseScheduler(["H4", "H1"], led, done_until=T("2026-03-04 10:00"))
    log = _Log()
    for minute in range(0, 4 * 60 + 1, 15):                 # 10:00 .. 14:00 every 15 minutes
        s.run_due(T("2026-03-04 10:00") + pd.Timedelta(minutes=minute), log)
    assert log.calls == [("H1", T("2026-03-04 11:00")), ("H1", T("2026-03-04 12:00")), ("H4", T("2026-03-04 12:00")),
                         ("H1", T("2026-03-04 13:00")), ("H1", T("2026-03-04 14:00"))]
    assert s.run_due(T("2026-03-04 14:00"), log) == []         # same instant again: nothing
    assert s.next_wakeup(T("2026-03-04 14:20")) == T("2026-03-04 15:00")


def test_missed_closes_after_sleep_run_once_never_twice():
    led = Ledger(":memory:")
    s = BarCloseScheduler(["H1", "H4"], led, done_until=T("2026-03-04 12:00"))
    log = _Log()
    s.run_due(T("2026-03-04 19:40"), log)                     # laptop slept from 12:00 to 19:40
    assert log.calls == [("H4", T("2026-03-04 16:00")), ("H1", T("2026-03-04 19:00"))]
    jobs = {(j["payload"]["tf"], j["status"]): j["result"] for j in led.jobs(AGENT)}
    assert jobs[("H1", "done")] == "caught up: skipped 6 earlier closes"
    assert jobs[("H4", "done")] == ""
    s.run_due(T("2026-03-04 19:59"), log)
    assert len(log.calls) == 2
    again = BarCloseScheduler(["H1", "H4"], led)              # restart reads the jobs table back
    again.run_due(T("2026-03-04 19:41"), log)
    assert len(log.calls) == 2


def test_a_close_claimed_before_a_crash_is_not_run_again():
    led = Ledger(":memory:")
    assert led.claim_job(AGENT, {"tf": "H1", "close": T("2026-03-04 12:00").isoformat()}, "x") is not None
    log = _Log()
    BarCloseScheduler(["H1"], led).run_due(T("2026-03-04 12:30"), log)
    assert log.calls == []
    assert led.claim_job(AGENT, {"close": T("2026-03-04 12:00").isoformat(), "tf": "H1"}, "y") is None


def test_a_failed_close_is_recorded_and_not_retried():
    led = Ledger(":memory:")
    s = BarCloseScheduler(["H1"], led, done_until=T("2026-03-04 11:00"))

    def boom(tf, ts):
        raise RuntimeError("feed down")
    with pytest.raises(RuntimeError):
        s.run_due(T("2026-03-04 12:00"), boom)
    assert led.jobs(AGENT)[0]["status"] == "failed" and "feed down" in led.jobs(AGENT)[0]["result"]
    log = _Log()
    BarCloseScheduler(["H1"], led).run_due(T("2026-03-04 12:10"), log)
    assert log.calls == []


def test_fresh_scheduler_runs_the_latest_close_once():
    log = _Log()
    BarCloseScheduler(["D1"], Ledger(":memory:")).run_due(T("2026-03-04 13:00"), log)
    assert log.calls == [("D1", T("2026-03-04"))]


def test_scheduler_driven_live_runner_matches_replay():
    """The live runtime end to end with a fake clock: feed appends each closed bar, the
    scheduler fires closes, the ensemble sits behind the order guard."""
    frames = _fx_frames()
    start = frames["EUR_USD"].index[700]
    clock = ReplayClock()
    store = BarStore("H1", {s: df[df.index < start] for s, df in frames.items()}, clock)
    led = Ledger(":memory:", git_commit="t")
    core = build_replay_core(_mixed_specs(), frames, led, start, use_es=False, data=store, clock=clock)
    core.brokers["ensemble"] = GuardedBroker(core.brokers["ensemble"], OrderGuard(ledger_verdicts(led), {"sim"}))

    def feed(tf, ts):                                         # the provider poll: top the store up to ts
        for sym, df in frames.items():
            have = store.frames[sym].index[-1]
            new = df[(df.index > have) & (df.index + store.bar <= ts)]
            if len(new):
                store.append(sym, new)
    runner = LiveRunner(core, BarCloseScheduler(core.tfs, led, done_until=start), before_close=feed)
    end = frames["EUR_USD"].index[-1] + store.bar
    t = start
    while t < end:                                            # wake every 20 minutes
        t += pd.Timedelta(minutes=20)
        clock.set(min(t, end))
        runner.tick()
    core.finish(end)
    rep = run_replay(_mixed_specs(), frames, Ledger(":memory:", git_commit="t"), start, use_es=False)
    assert _decisions(led) == _decisions(rep.ledger)
    assert led.rows(kind="order") and led.rows(kind="close")
    closes = [j["payload"] for j in led.jobs(AGENT)]
    assert len(closes) == len({(c["tf"], c["close"]) for c in closes})
