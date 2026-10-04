"""The bar-close scheduler: one close event per timeframe at each bar boundary.

Strategies subscribe to their own timeframe through the core (``on_bar_close(tf, ts)``
only runs strategies on ``tf``); higher timeframes are visible to lower ones only once
closed, because the bar store hides a bar until ``open + duration <= now``.

Boundaries follow the bar store's resampling: every timeframe up to D1 is aligned to
00:00 UTC, W1 to Monday 00:00 UTC. When closes coincide, finer timeframes go first, as in
replay.

Each close runs at most once, ever. The ledger's jobs table records every claimed close
(agent ``scheduler``); a restart reads it back, and a job left ``running`` by a crash is
not run again. After a sleep, missed closes of a timeframe collapse into one run of the
latest close, with the number skipped in the job's result. A fresh scheduler with no jobs
treats closes up to ``done_until`` (default: none) as done, so the latest close runs once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tradex.timeframes import duration

AGENT = "scheduler"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
ANCHORS = {"W1": pd.Timestamp("1970-01-05", tz="UTC")}       # a Monday, like resample's W-MON bins


def boundary(t: pd.Timestamp, tf: str) -> pd.Timestamp:
    """The latest bar boundary of ``tf`` at or before ``t``."""
    anchor, d = ANCHORS.get(tf, EPOCH), duration(tf)
    return anchor + ((t - anchor) // d) * d


@dataclass(frozen=True)
class CloseEvent:
    tf: str
    ts: pd.Timestamp                   # close time = boundary
    skipped: int = 0                   # earlier closes of this timeframe missed and not run


class BarCloseScheduler:
    def __init__(self, tfs: list[str], ledger, grace: pd.Timedelta = pd.Timedelta(0),
                 done_until: pd.Timestamp | None = None):
        self.tfs = sorted(set(tfs), key=duration)
        self.ledger = ledger
        self.grace = grace                                     # wait this long after a boundary for the last bar
        self.last: dict[str, pd.Timestamp | None] = {tf: None for tf in self.tfs}
        for job in ledger.jobs(AGENT):
            tf, ts = job["payload"].get("tf"), pd.Timestamp(job["payload"].get("close"))
            if tf in self.last and (self.last[tf] is None or ts > self.last[tf]):
                self.last[tf] = ts
        if done_until is not None:
            for tf in self.tfs:
                if self.last[tf] is None:
                    self.last[tf] = boundary(done_until, tf)

    def due(self, now: pd.Timestamp) -> list[CloseEvent]:
        out = []
        for tf in self.tfs:
            b = boundary(now - self.grace, tf)
            last = self.last[tf]
            if last is None or b > last:
                skipped = 0 if last is None else int((b - last) / duration(tf)) - 1
                out.append(CloseEvent(tf, b, skipped))
        order = {tf: i for i, tf in enumerate(self.tfs)}
        return sorted(out, key=lambda e: (e.ts, order[e.tf]))

    def run_due(self, now: pd.Timestamp, handler: Callable[[str, pd.Timestamp], None]) -> list[CloseEvent]:
        """Run every due close once through ``handler(tf, ts)``; returns the events run."""
        ran = []
        for ev in self.due(now):
            self.last[ev.tf] = ev.ts
            jid = self.ledger.claim_job(AGENT, {"tf": ev.tf, "close": ev.ts.isoformat()}, now.isoformat())
            if jid is None:
                continue                                       # already run (or claimed by another process)
            try:
                handler(ev.tf, ev.ts)
            except Exception as exc:
                self.ledger.finish_job(jid, "failed", f"{type(exc).__name__}: {exc}")
                raise
            note = f"caught up: skipped {ev.skipped} earlier closes" if ev.skipped else ""
            self.ledger.finish_job(jid, "done", note)
            ran.append(ev)
        return ran

    def next_wakeup(self, now: pd.Timestamp) -> pd.Timestamp:
        return min(boundary(now - self.grace, tf) + duration(tf) for tf in self.tfs) + self.grace
