"""Earnings dates for a list of symbols, from an injected source (the OpenD adapter in production).

``import_earnings`` returns ``Event(kind="earnings", scope=<symbol>)`` rows for the
``EventCalendar``, whose earnings rule blocks entries held into the announcement. A symbol with
no announcement found in the window raises ``CalendarDataMissing`` by default: a missing
earnings date is exactly what the calendar exists to prevent, so it is never treated as "none".
Pass ``require_all=False`` only for a window known to be shorter than a reporting cycle.
"""
from __future__ import annotations

from typing import Iterable, Protocol

import pandas as pd

from tradex.events import Event
from tradex.events.opend_calendar import EarningsRecord, short_symbol
from tradex.events.sources import CalendarDataMissing, check_window, dedupe


class EarningsSource(Protocol):
    def earnings(self, symbols: Iterable[str], start, end) -> list[EarningsRecord]: ...


def import_earnings(source: EarningsSource, symbols: Iterable[str], start, end, *, require_all: bool = True
                    ) -> list[Event]:
    want = sorted({short_symbol(str(s)).upper() for s in symbols})
    if not want:
        raise ValueError("no symbols asked for")
    s, e = check_window(start, end)
    try:
        got = source.earnings(want, s, e)
    except CalendarDataMissing:
        raise
    except Exception as exc:  # noqa: BLE001 - a source that breaks is missing data, not a reason to carry on
        raise CalendarDataMissing(f"earnings source failed: {type(exc).__name__}") from exc
    events = []
    for r in got:
        sym = short_symbol(r.symbol).upper()
        if sym not in want or not (s <= r.time <= e):
            continue                                    # extra rows from a chatty source are not ours to add
        when = r.pub_type if r.pub_type not in ("", "N/A") else "time of day not given"
        events.append(Event(r.time.tz_convert("UTC"), "earnings", sym,
                            f"{sym} earnings {r.period}".strip() + f" ({when}) [OpenD]"))
    events = dedupe(events)
    found = {e_.scope for e_ in events}
    if not events:
        raise CalendarDataMissing(f"no earnings found for any of {len(want)} symbols between {s:%Y-%m-%d} and {e:%Y-%m-%d}")
    absent = [x for x in want if x not in found]
    if absent and require_all:
        raise CalendarDataMissing(f"no earnings date found between {s:%Y-%m-%d} and {e:%Y-%m-%d} for: {', '.join(absent)}")
    return events


def coverage(events: list[Event], symbols: Iterable[str], now: pd.Timestamp, horizon_days: int = 100) -> list[str]:
    """Symbols with no earnings event within ``horizon_days`` after ``now``: the calendar cannot protect these."""
    end = now + pd.Timedelta(days=horizon_days)
    have = {e.scope for e in events if e.kind == "earnings" and now <= e.time <= end}
    return sorted({short_symbol(str(s)).upper() for s in symbols} - have)
