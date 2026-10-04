"""The live loop: wake at each bar boundary, run due closes through the core, sleep.

Everything with side effects is injected (clock, sleep, the core's brokers and data), so
tests drive it with a fake clock. Before each close the feed must have appended that
bar to the bar store; ``before_close`` is the hook for it (poll the provider, top up the
store). A close whose bars are not there yet does nothing in the core, so the grace
period should cover the feed's usual delay.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from tradex.runtime.schedule import BarCloseScheduler, CloseEvent


class LiveRunner:
    def __init__(self, core, scheduler: BarCloseScheduler,
                 before_close: Callable[[str, pd.Timestamp], None] | None = None):
        self.core, self.scheduler = core, scheduler
        self.before_close = before_close

    def _handle(self, tf: str, ts: pd.Timestamp) -> None:
        if self.before_close is not None:
            self.before_close(tf, ts)
        self.core.on_bar_close(tf, ts)

    def tick(self) -> list[CloseEvent]:
        return self.scheduler.run_due(self.core.clock.now(), self._handle)

    def run(self, sleep: Callable[[float], None], stop: Callable[[], bool]) -> None:
        while not stop():
            self.tick()
            now = self.core.clock.now()
            sleep(max(0.0, (self.scheduler.next_wakeup(now) - now).total_seconds()))
