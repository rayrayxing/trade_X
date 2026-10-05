"""moomoo OpenD earnings and economic calendar, as an adapter over an injected quote context.

Source: the OpenQuoteContext methods ``get_earnings_calendar`` and ``get_economic_calendar``
(moomoo-api 10.11.7108; their docstrings give the arguments and the result columns used here).
This module only ever uses a QUOTE context that the caller opens and passes in; it never
imports the SDK, opens a trade context or places anything. Tests hand it a fake context that
returns frames of the documented columns.

What the SDK documents, and what the adapter does about it:

- ``get_earnings_calendar(market, begin_date, end_date)``: the interval may not exceed 7 days,
  so a longer window is fetched in 7-day pieces. Columns used: ``security`` ("US.AAPL"),
  ``earnings_date`` ("yyyy-MM-dd"), ``earnings_timestamp`` (epoch seconds), ``pub_type``,
  ``period_text``. A row without a usable timestamp for a symbol we asked about raises.
- ``get_economic_calendar(begin_date, end_date, market_list, importance, count, next_page)``:
  paged (``count`` at most 100, ``next_page`` / ``has_more``). Dates are read in the OpenD
  host's own time zone, so each window is widened by a day on both sides and cut again on the
  exact timestamps. Columns used: ``title``, ``timestamp`` (epoch seconds), ``country``,
  ``star``, ``previous``, ``consensus``, ``actual``.
- A failed call (``ret != RET_OK``) raises ``CalendarDataMissing``; so does a page loop that
  does not end. A partial result is never returned.

Not verified against a live OpenD (needs Ray's Mac): how many rows a busy earnings week
returns in one call, and the exact English titles and country names the economic calendar uses.
Both are handled defensively (see ``jobs_report.ECONOMIC_TITLES``) and fail loudly when they do
not match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from tradex.events.sources import CalendarDataMissing, check_window

RET_OK = 0                      # moomoo.RET_OK; kept here so this module needs no SDK import
MAX_EARNINGS_SPAN_DAYS = 7      # documented: begin and end at most 7 days apart
ECONOMIC_PAGE = 100             # documented maximum page size
MAX_PAGES = 200                 # a loop that long is a bug or a runaway cursor: raise, never truncate
NY = "America/New_York"


@dataclass(frozen=True)
class EarningsRecord:
    symbol: str                 # "AAPL", not "US.AAPL"
    time: pd.Timestamp          # announcement time, UTC
    date: str                   # market-local date as OpenD reports it
    pub_type: str               # BEFORE | AFTER | REGULAR | N/A
    period: str = ""


@dataclass(frozen=True)
class EconomicRecord:
    title: str
    time: pd.Timestamp          # release time, UTC
    country: str
    importance: str             # LOW | MEDIUM | HIGH
    previous: str = ""
    consensus: str = ""
    actual: str = ""


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() in ("", "N/A", "--"))


def _epoch(v: Any, what: str) -> pd.Timestamp:
    if _missing(v):
        raise CalendarDataMissing(f"OpenD returned no {what}")
    try:
        return pd.Timestamp(float(v), unit="s", tz="UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalendarDataMissing(f"OpenD returned an unreadable {what}: {v!r}") from exc


def _date_str(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d")


def short_symbol(security: str) -> str:
    return security.split(".", 1)[1] if "." in security else security


class OpenDCalendar:
    """Read-only calendar queries on an open quote context (``OpenQuoteContext``)."""

    def __init__(self, quote, market: str = "US", ok: int = RET_OK):
        self.quote, self.market, self.ok = quote, market, ok

    # --- earnings ---------------------------------------------------------------------------

    def earnings(self, symbols: Iterable[str], start, end) -> list[EarningsRecord]:
        """Earnings announcements for ``symbols`` between ``start`` and ``end`` (inclusive of the days touched).

        Symbols with no announcement in the window are simply absent here; whether that is an
        error is the caller's call (``earnings.import_earnings`` makes it one by default)."""
        want = {short_symbol(s) for s in symbols}
        if not want:
            raise ValueError("no symbols asked for")
        s, e = check_window(start, end)
        out: dict[tuple[str, pd.Timestamp], EarningsRecord] = {}
        day, stop = (s - pd.Timedelta(days=1)).normalize(), (e + pd.Timedelta(days=1)).normalize()
        while day <= stop:                  # OpenD reads dates in the market's day, so cover one day more each side
            last = min(day + pd.Timedelta(days=MAX_EARNINGS_SPAN_DAYS - 1), stop)
            for rec in self._earnings_window(day, last, want):
                if s <= rec.time <= e:
                    out[(rec.symbol, rec.time)] = rec
            day = last + pd.Timedelta(days=1)
        return sorted(out.values(), key=lambda r: (r.time, r.symbol))

    def _earnings_window(self, begin: pd.Timestamp, end: pd.Timestamp, want: set[str]) -> list[EarningsRecord]:
        ret, data = self.quote.get_earnings_calendar(self.market, begin_date=_date_str(begin), end_date=_date_str(end))
        if ret != self.ok:
            raise CalendarDataMissing(f"OpenD earnings calendar {_date_str(begin)}..{_date_str(end)} failed: {data}")
        if data is None:
            raise CalendarDataMissing(f"OpenD earnings calendar {_date_str(begin)}..{_date_str(end)} returned no frame")
        out = []
        for row in data.to_dict("records"):
            sym = short_symbol(str(row.get("security", "")))
            if sym not in want:
                continue
            t = _epoch(row.get("earnings_timestamp"), f"earnings timestamp for {sym} ({row.get('earnings_date')})")
            date = str(row.get("earnings_date", ""))
            if not _missing(date):
                gap = abs((t.tz_convert(NY).normalize().tz_localize(None) - pd.Timestamp(date)).days)
                if gap > 1:
                    raise CalendarDataMissing(f"OpenD earnings date {date} and timestamp {t.isoformat()} for {sym} disagree")
            pub = row.get("pub_type")
            out.append(EarningsRecord(sym, t, date, "N/A" if _missing(pub) else str(pub),
                                      "" if _missing(row.get("period_text")) else str(row["period_text"])))
        return out

    # --- economic calendar --------------------------------------------------------------------

    def economic(self, start, end, importance: str = "ALL") -> list[EconomicRecord]:
        s, e = check_window(start, end)
        begin, stop = _date_str(s - pd.Timedelta(days=1)), _date_str(e + pd.Timedelta(days=1))
        out: list[EconomicRecord] = []
        page = None
        for _ in range(MAX_PAGES):
            ret, df, nxt, more = self.quote.get_economic_calendar(
                begin, stop, market_list=[self.market], importance=importance, count=ECONOMIC_PAGE, next_page=page)
            if ret != self.ok:
                raise CalendarDataMissing(f"OpenD economic calendar {begin}..{stop} failed: {df}")
            for row in (df.to_dict("records") if df is not None else []):
                t = _epoch(row.get("timestamp"), f"timestamp of '{row.get('title')}'")
                if s <= t <= e:
                    out.append(EconomicRecord(str(row.get("title", "")), t, str(row.get("country", "")),
                                              str(row.get("star", "")), *(("" if _missing(row.get(k)) else str(row[k]))
                                                                          for k in ("previous", "consensus", "actual"))))
            if not more:
                return sorted(out, key=lambda r: (r.time, r.title))
            if _missing(nxt):
                raise CalendarDataMissing("OpenD economic calendar says there are more pages but gave no cursor")
            page = nxt
        raise CalendarDataMissing(f"OpenD economic calendar did not end after {MAX_PAGES} pages")
