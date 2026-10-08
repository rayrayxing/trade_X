"""7 Oct: the paper core hung for 13 hours inside one close (an OpenD call that never
returned). A venue call past its deadline, or a close past the watchdog, now ends the
process so launchd restarts it; and a restarted run shows its account at the first close."""
import threading
import time

import pandas as pd
import pytest

from tradex.runtime.paper import Serialized


class Venue:
    def __init__(self):
        self.release = threading.Event()

    def positions(self, account="agent"):
        return ["p"]

    def hang(self):
        self.release.wait(5)

    def boom(self):
        raise ValueError("venue said no")


def test_a_venue_call_past_its_deadline_goes_to_on_hang():
    v, hung = Venue(), []
    s = Serialized(v, timeout_s=0.2, on_hang=hung.append)
    assert s.positions() == ["p"]
    with pytest.raises(TimeoutError):
        s.hang()
    assert hung and "Venue.hang: no reply" in hung[0]
    v.release.set()


def test_venue_errors_still_reach_the_caller_and_free_the_lock():
    s = Serialized(Venue(), timeout_s=1.0, on_hang=lambda why: pytest.fail(why))
    with pytest.raises(ValueError, match="venue said no"):
        s.boom()
    assert s.positions() == ["p"]


def test_the_watchdog_names_a_close_that_runs_too_long():
    from tradex.runtime.build import Runtime
    from tradex.runtime.runner import LiveRunner
    now = [100.0]
    runner = LiveRunner(core=None, scheduler=None, timer=lambda: now[0])
    rt = Runtime.__new__(Runtime)
    rt.runner, rt.watchdog_s = runner, 600.0
    assert rt.stuck() is None
    runner.busy = ("H1", pd.Timestamp("2026-10-07 15:00", tz="UTC"), 100.0)
    now[0] = 650.0
    assert rt.stuck() is None
    now[0] = 701.0
    assert "H1 close 2026-10-07T15:00:00+00:00 still running after 601s" == rt.stuck()


def test_the_runner_clears_busy_even_when_a_close_fails():
    from tradex.runtime.runner import LiveRunner

    class Core:
        def on_bar_close(self, tf, ts):
            raise RuntimeError("bad close")

    r = LiveRunner(Core(), scheduler=None)
    with pytest.raises(RuntimeError):
        r._handle("H1", pd.Timestamp("2026-10-07 15:00", tz="UTC"))
    assert r.busy is None
