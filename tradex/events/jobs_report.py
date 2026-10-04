"""US jobs report (BLS "Employment Situation") release times as ``nfp`` events.

Primary source: the BLS release schedule calendar file, https://www.bls.gov/schedule/news_release/bls.ics,
read through an injected client (``fetch_ics() -> str``). Second source: the OpenD economic
calendar (``OpenDCalendar.economic``), used on its own when no BLS client is given, or, when both
are, to cross-check the BLS dates (BLS is authoritative; any disagreement raises).

Real data or nothing: an empty or unreadable file, an Employment Situation entry with no usable
time, a time zone we do not know, a window with no release, or a cross-check that disagrees all
raise ``CalendarDataMissing``. Floating times (no zone in the file) are refused unless the caller
names the zone with ``assume_tz``; nothing is assumed silently.

Not verified here (needs a machine that can reach bls.gov): the exact layout of the live ``bls.ics``.
The parser follows RFC 5545 (line unfolding, ``DTSTART;TZID=...`` and ``Z`` forms) and the tests use
a hand-built sample in that layout, not a recording. Replace it with a download from Ray's Mac.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from tradex.events import Event
from tradex.events.opend_calendar import EconomicRecord
from tradex.events.sources import CalendarDataMissing, check_window, dedupe

BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
SUMMARY_MATCH = re.compile(r"employment\s+situation", re.I)
# Titles the OpenD economic calendar may use for the report (English and Chinese UI languages).
ECONOMIC_TITLES = re.compile(r"non[\s-]?farm\s+payrolls?|非农", re.I)
US_COUNTRIES = {"united states", "us", "usa", "u.s.", "美国"}
TZID_ALIASES = {"US-EASTERN": "America/New_York", "US/EASTERN": "America/New_York", "EASTERN": "America/New_York"}
CROSS_CHECK_TOLERANCE = pd.Timedelta(minutes=30)


class BlsScheduleClient(Protocol):
    def fetch_ics(self) -> str: ...


class HttpBlsClient:
    """Fetches the schedule over HTTPS. ``get(url, headers) -> text`` is injectable; the default uses urllib.

    BLS rejects anonymous clients, so pass a ``contact`` (an email) that goes into the User-Agent.
    """

    def __init__(self, contact: str, get: Callable[[str, dict], str] | None = None, url: str = BLS_ICS_URL):
        if not contact or "@" not in contact:
            raise ValueError("BLS asks API clients to identify themselves: pass a contact email")
        self.url, self.contact, self._get = url, contact, get or self._urllib_get

    @staticmethod
    def _urllib_get(url: str, headers: dict) -> str:
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:  # noqa: S310
            return resp.read().decode("utf-8")

    def fetch_ics(self) -> str:
        try:
            return self._get(self.url, {"User-Agent": f"tradex-calendar ({self.contact})"})
        except Exception as exc:  # noqa: BLE001
            raise CalendarDataMissing(f"could not fetch the BLS schedule: {type(exc).__name__}") from exc


@dataclass(frozen=True)
class JobsRelease:
    time: pd.Timestamp          # UTC
    summary: str


def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _prop(line: str) -> tuple[str, dict[str, str], str]:
    head, _, value = line.partition(":")
    name, *params = head.split(";")
    return name.upper(), {k.upper(): v.strip('"') for k, _, v in (p.partition("=") for p in params)}, value.strip()


def _parse_dt(value: str, params: dict[str, str], assume_tz: str | None, summary: str) -> pd.Timestamp:
    if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", value):
        raise CalendarDataMissing(f"BLS entry '{summary}' has a date but no time of day")
    m = re.fullmatch(r"(\d{8}T\d{6})(Z?)", value)
    if not m:
        raise CalendarDataMissing(f"BLS entry '{summary}' has an unreadable start: {value!r}")
    naive = pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S"))
    if m.group(2):
        return naive.tz_localize("UTC")
    tzid = params.get("TZID")
    name = TZID_ALIASES.get((tzid or "").upper(), tzid) if tzid else assume_tz
    if not name:
        raise CalendarDataMissing(f"BLS entry '{summary}' has a floating time with no zone; pass assume_tz to say which")
    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CalendarDataMissing(f"BLS entry '{summary}' uses a time zone we do not know: {name!r}") from exc
    try:
        return naive.tz_localize(zone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    except Exception as exc:  # noqa: BLE001 - a time inside a DST jump is bad data, not ours to resolve
        raise CalendarDataMissing(f"BLS entry '{summary}' has a local time that does not exist or repeats: {value}") from exc


def parse_bls_ics(text: str, assume_tz: str | None = None) -> list[JobsRelease]:
    """Every Employment Situation entry in an iCalendar file, UTC, sorted."""
    if not text or "BEGIN:VCALENDAR" not in text:
        raise CalendarDataMissing("the BLS schedule is empty or is not an iCalendar file")
    out: list[JobsRelease] = []
    block: list[str] | None = None
    for line in _unfold(text):
        if line.strip().upper() == "BEGIN:VEVENT":
            block = []
        elif line.strip().upper() == "END:VEVENT":
            if block is not None:
                props = [_prop(ln) for ln in block if ":" in ln]
                summary = next((v for n, _, v in props if n == "SUMMARY"), "")
                if SUMMARY_MATCH.search(summary):
                    start = next(((p, v) for n, p, v in props if n == "DTSTART"), None)
                    if start is None:
                        raise CalendarDataMissing(f"BLS entry '{summary}' has no start time")
                    out.append(JobsRelease(_parse_dt(start[1], start[0], assume_tz, summary), summary))
            block = None
        elif block is not None:
            block.append(line)
    if not out:
        raise CalendarDataMissing("the BLS schedule has no Employment Situation entries (wrong file, or the layout changed)")
    return sorted(out, key=lambda r: r.time)


def _from_bls(client: BlsScheduleClient, assume_tz: str | None) -> list[JobsRelease]:
    try:
        text = client.fetch_ics()
    except CalendarDataMissing:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CalendarDataMissing(f"the BLS schedule could not be read: {type(exc).__name__}") from exc
    return parse_bls_ics(text, assume_tz)


def _payroll_rows(rows: list[EconomicRecord]) -> list[EconomicRecord]:
    return [r for r in rows if ECONOMIC_TITLES.search(r.title) and r.country.strip().lower() in US_COUNTRIES]


def cross_check(releases: list[JobsRelease], rows: list[EconomicRecord], start: pd.Timestamp, end: pd.Timestamp
                ) -> list[str]:
    """Disagreements between BLS release times and the OpenD economic calendar inside ``[start, end]``."""
    payroll = _payroll_rows(rows)
    problems = []
    for rel in releases:
        near = [r for r in payroll if abs(r.time - rel.time) <= CROSS_CHECK_TOLERANCE]
        if not near:
            same_day = [r for r in payroll if r.time.date() == rel.time.date()]
            problems.append(f"BLS has a jobs report at {rel.time:%Y-%m-%d %H:%M} UTC; OpenD "
                            + (f"has it at {same_day[0].time:%H:%M} UTC" if same_day else "has no payrolls entry that day"))
    for r in payroll:
        if not any(abs(r.time - rel.time) <= CROSS_CHECK_TOLERANCE for rel in releases) and start <= r.time <= end:
            problems.append(f"OpenD lists payrolls at {r.time:%Y-%m-%d %H:%M} UTC; the BLS schedule does not")
    return problems


def import_jobs_reports(bls: BlsScheduleClient | None = None, opend=None, *, start, end,
                        assume_tz: str | None = None) -> list[Event]:
    """``nfp`` events for the jobs reports between ``start`` and ``end`` (both timezone-aware).

    ``bls``: a ``BlsScheduleClient``; ``opend``: an ``OpenDCalendar`` (anything with ``economic(start, end)``).
    At least one is required. With both, BLS decides and OpenD must agree.
    """
    if bls is None and opend is None:
        raise ValueError("give a BLS client, an OpenD calendar, or both")
    s, e = check_window(start, end)
    events: list[Event] = []
    rows: list[EconomicRecord] | None = None
    if opend is not None:
        try:
            rows = opend.economic(s - pd.Timedelta(days=2), e + pd.Timedelta(days=2))
        except CalendarDataMissing:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CalendarDataMissing(f"the OpenD economic calendar could not be read: {type(exc).__name__}") from exc
    if bls is not None:
        rels = [r for r in _from_bls(bls, assume_tz) if s <= r.time <= e]
        if not rels:
            raise CalendarDataMissing(f"the BLS schedule has no jobs report between {s:%Y-%m-%d} and {e:%Y-%m-%d}")
        if rows is not None:
            problems = cross_check(rels, rows, s, e)
            if problems:
                raise CalendarDataMissing("BLS and OpenD disagree: " + "; ".join(problems))
        events = [Event(r.time, "nfp", "USD", f"{r.summary} [BLS schedule]") for r in rels]
    else:
        payroll = [r for r in _payroll_rows(rows or []) if s <= r.time <= e]
        if not payroll:
            raise CalendarDataMissing(f"the OpenD economic calendar has no nonfarm payrolls between {s:%Y-%m-%d} and {e:%Y-%m-%d}")
        events = [Event(r.time, "nfp", "USD", f"{r.title} [OpenD economic calendar]") for r in payroll]
    return dedupe(events)
