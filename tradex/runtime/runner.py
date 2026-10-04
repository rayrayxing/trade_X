"""The live loop: wake at each bar boundary, run due closes through the core, sleep.

Everything with side effects is injected (clock, sleep, timer, the core's brokers and
data), so tests drive it with a fake clock. Before each close the feed must have appended
that bar to the bar store; ``before_close`` is the hook for it (drain the stream into the
bar builders, top up the store). A close whose bars are not there yet does nothing in the
core, so the grace period should cover the feed's usual delay.

Scheduler faults are Health rows (ok=False, loud on Telegram): a close that took longer
than ``overrun`` and closes missed while the process was asleep or behind.
"""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from tradex.runtime.schedule import BarCloseScheduler, CloseEvent


class LiveRunner:
    def __init__(self, core, scheduler: BarCloseScheduler,
                 before_close: Callable[[str, pd.Timestamp], None] | None = None,
                 overrun: pd.Timedelta | None = None, timer: Callable[[], float] = time.monotonic):
        self.core, self.scheduler = core, scheduler
        self.before_close = before_close
        self.overrun, self.timer = overrun, timer

    def _handle(self, tf: str, ts: pd.Timestamp) -> None:
        t0 = self.timer()
        if self.before_close is not None:
            self.before_close(tf, ts)
        self.core.on_bar_close(tf, ts)
        took = self.timer() - t0
        if self.overrun is not None and took > self.overrun.total_seconds():
            self.core.health("scheduler", False, f"{tf} close {ts.isoformat()} took {took:.1f}s "
                             f"(limit {self.overrun.total_seconds():.0f}s)", ts)

    def tick(self) -> list[CloseEvent]:
        ran = self.scheduler.run_due(self.core.clock.now(), self._handle)
        for ev in ran:
            if ev.skipped:
                self.core.health("scheduler", False, f"missed {ev.skipped} {ev.tf} closes before "
                                 f"{ev.ts.isoformat()}; ran the latest only", ev.ts)
        return ran

    def run(self, sleep: Callable[[float], None], stop: Callable[[], bool]) -> None:
        while not stop():
            self.tick()
            now = self.core.clock.now()
            sleep(max(0.0, (self.scheduler.next_wakeup(now) - now).total_seconds()))
