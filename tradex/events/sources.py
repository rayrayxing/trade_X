"""Shared pieces of the calendar importers: the error they raise and the YAML they write.

The importers (``jobs_report``, ``earnings``) turn a source's rows into ``Event`` objects for
the ``EventCalendar``. The rule for all of them (Ray, 4 Oct 2026): real data or nothing. A
source that is unreachable, returns nothing for the window asked, or returns a row we cannot
read raises ``CalendarDataMissing``. They never fall back to an older file, a default date or an
estimate, because a calendar that silently misses an event lets the bot trade through it.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from tradex.events import Event


class CalendarDataMissing(RuntimeError):
    """A calendar source gave no usable data for what was asked. Nothing is guessed in its place."""


def to_utc(ts, what: str = "time") -> pd.Timestamp:
    """A timezone-aware UTC timestamp; a naive or unparsable value raises (we never assume a zone here)."""
    try:
        t = pd.Timestamp(ts)
    except (ValueError, TypeError) as exc:
        raise CalendarDataMissing(f"unreadable {what}: {ts!r}") from exc
    if pd.isna(t):
        raise CalendarDataMissing(f"missing {what}")
    if t.tzinfo is None:
        raise CalendarDataMissing(f"{what} {ts!r} has no timezone")
    return t.tz_convert("UTC")


def check_window(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    s, e = to_utc(start, "window start"), to_utc(end, "window end")
    if e <= s:
        raise ValueError(f"window end {e} is not after its start {s}")
    return s, e


def dedupe(events: Iterable[Event]) -> list[Event]:
    """One event per (time, kind, scope), sorted by time."""
    seen: dict[tuple, Event] = {}
    for e in events:
        seen.setdefault((e.time, e.kind, e.scope), e)
    return sorted(seen.values(), key=lambda e: (e.time, e.kind, e.scope))


def write_events_yaml(events: list[Event], path: str | Path, header: str = "") -> Path:
    """Write ``events`` in the format ``EventCalendar.add_file`` reads. Atomic: a failed write never leaves a
    half-written calendar that a later run would load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"time": e.time.strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": e.kind, "scope": e.scope, "note": e.note}
            for e in sorted(events, key=lambda e: e.time)]
    text = "".join(f"# {ln}\n" for ln in header.splitlines()) + yaml.safe_dump({"events": rows}, sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
