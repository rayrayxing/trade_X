"""Calendar importers: US jobs report (BLS schedule, OpenD economic calendar) and earnings (OpenD).

The fixtures in tests/fixtures/events are FORMAT SAMPLES built from the documented layouts (the
RFC 5545 file BLS publishes, the columns of moomoo-api 10.11.7108's get_earnings_calendar and
get_economic_calendar), with 2031 test dates. They are not recordings; swap in captures from
Ray's Mac when available. The clients are fakes, so nothing here touches a network or OpenD.
"""
import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from tradex.events import EventCalendar
from tradex.events.earnings import coverage, import_earnings
from tradex.events.jobs_report import (HttpBlsClient, cross_check, import_jobs_reports, parse_bls_ics)
from tradex.events.opend_calendar import MAX_PAGES, OpenDCalendar
from tradex.events.refresh import refresh
from tradex.events.sources import CalendarDataMissing, write_events_yaml

FIX = Path(__file__).parent / "fixtures" / "events"
ICS = (FIX / "bls_schedule_layout_sample.ics").read_text()
EARN_COLS = ["security", "name", "earnings_date", "earnings_timestamp", "pub_type", "period_text", "eps_actual",
             "eps_predict", "revenue_actual", "revenue_predict", "ebit_actual", "ebit_predict", "option_volume", "iv",
             "iv_rank", "iv_percentile", "market_cap", "price"]
ECO_COLS = ["title", "timestamp", "country", "star", "previous", "consensus", "actual"]
UTC = "UTC"


def ts(s, tz=UTC):
    return pd.Timestamp(s, tz=tz)


def rows(name):
    return json.loads((FIX / name).read_text())["rows"]


class FakeQuote:
    """A quote context that answers like the SDK methods (same signatures and return shapes) from the samples."""

    def __init__(self, earnings=None, economic=None, page=4, fail=None):
        self.earn = earnings if earnings is not None else rows("opend_earnings_calendar_format_sample.json")
        self.eco = economic if economic is not None else rows("opend_economic_calendar_format_sample.json")
        self.page, self.fail, self.calls = page, fail, []

    def get_earnings_calendar(self, market, sort_type=None, begin_date=None, end_date=None, filter_list=None):
        self.calls.append(("earnings", market, begin_date, end_date))
        if self.fail == "earnings":
            return -1, "OpenD is not logged in"
        b, e = pd.Timestamp(begin_date), pd.Timestamp(end_date)
        if (e - b).days > 7:
            return -1, "begin and end are more than 7 days apart"
        hit = [r for r in self.earn if b <= pd.Timestamp(r["earnings_date"]) <= e]
        return 0, pd.DataFrame(hit, columns=EARN_COLS)

    def get_economic_calendar(self, begin_date, end_date=None, market_list=None, importance=None, count=None,
                              next_page=None):
        self.calls.append(("economic", market_list, begin_date, end_date, count, next_page))
        if self.fail == "economic":
            return -1, "quota exceeded", None, None
        if self.fail == "runaway":
            return 0, pd.DataFrame([], columns=ECO_COLS), "again", True
        if self.fail == "no_cursor":
            return 0, pd.DataFrame([], columns=ECO_COLS), None, True
        b, e = pd.Timestamp(begin_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)
        hit = [r for r in self.eco if b <= pd.Timestamp(r["timestamp"], unit="s", tz=UTC).tz_convert("America/New_York")
               .tz_localize(None) < e]
        off = int(next_page or 0)
        chunk = hit[off:off + self.page]
        more = off + self.page < len(hit)
        return 0, pd.DataFrame(chunk, columns=ECO_COLS), (str(off + self.page) if more else None), more


class BlsText:
    def __init__(self, text=ICS):
        self.text = text

    def fetch_ics(self):
        return self.text


JAN = (ts("2030-12-20"), ts("2031-02-10"))
YEAR = (ts("2030-12-01"), ts("2031-12-31"))


# --- BLS schedule ----------------------------------------------------------------------------------

def test_bls_ics_parses_zones_folding_and_ignores_other_releases():
    got = parse_bls_ics(ICS)
    assert [(r.time, r.summary) for r in got] == [
        (ts("2031-01-03 13:30"), "Employment Situation for December 2030"),          # EST
        (ts("2031-06-06 12:30"), "Employment Situation for May2031"),                # EDT; folded line joined
        (ts("2031-09-05 12:30"), "Employment Situation for August 2031")]            # given in UTC


def test_jobs_events_for_a_window_are_nfp_usd_and_cut_to_the_window():
    ev = import_jobs_reports(BlsText(), start=ts("2030-12-20"), end=ts("2031-07-01"))
    assert [(e.time, e.kind, e.scope) for e in ev] == [(ts("2031-01-03 13:30"), "nfp", "USD"),
                                                       (ts("2031-06-06 12:30"), "nfp", "USD")]
    assert all("BLS" in e.note for e in ev)


@pytest.mark.parametrize("text, why", [
    ("", "empty file"),
    ("<html>Access Denied</html>", "not an iCalendar file"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Consumer Price Index\nDTSTART:20310114T133000Z\nEND:VEVENT\nEND:VCALENDAR\n",
     "no Employment Situation entry"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART;VALUE=DATE:20310103\nEND:VEVENT\nEND:VCALENDAR\n",
     "date without a time"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nEND:VEVENT\nEND:VCALENDAR\n", "no start"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART;TZID=Mars/Olympus:20310103T083000\n"
     "END:VEVENT\nEND:VCALENDAR\n", "unknown zone"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART:20310103T083000\nEND:VEVENT\nEND:VCALENDAR\n",
     "floating time"),
    ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART:soon\nEND:VEVENT\nEND:VCALENDAR\n", "garbage"),
])
def test_unreadable_bls_data_raises_and_returns_no_events(text, why):
    with pytest.raises(CalendarDataMissing):
        import_jobs_reports(BlsText(text), start=YEAR[0], end=YEAR[1])


def test_a_floating_time_is_only_read_when_the_caller_names_the_zone():
    floating = ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART:20310103T083000\n"
                "END:VEVENT\nEND:VCALENDAR\n")
    ev = import_jobs_reports(BlsText(floating), start=JAN[0], end=JAN[1], assume_tz="America/New_York")
    assert ev[0].time == ts("2031-01-03 13:30")


def test_no_release_in_the_window_raises_instead_of_returning_nothing():
    with pytest.raises(CalendarDataMissing, match="no jobs report between"):
        import_jobs_reports(BlsText(), start=ts("2031-02-01"), end=ts("2031-05-01"))


def test_a_broken_bls_client_raises_calendar_data_missing():
    class Down:
        def fetch_ics(self):
            raise TimeoutError("proxy")
    with pytest.raises(CalendarDataMissing):
        import_jobs_reports(Down(), start=JAN[0], end=JAN[1])

    def get(url, headers):
        raise OSError("403")
    with pytest.raises(CalendarDataMissing, match="could not fetch"):
        HttpBlsClient("ray@example.com", get=get).fetch_ics()
    with pytest.raises(ValueError):
        HttpBlsClient("")


def test_the_bls_client_identifies_itself_and_asks_for_the_schedule_file():
    seen = {}

    def get(url, headers):
        seen.update(url=url, **headers)
        return ICS
    assert HttpBlsClient("ray@example.com", get=get).fetch_ics() == ICS
    assert seen["url"].endswith("/schedule/news_release/bls.ics") and "ray@example.com" in seen["User-Agent"]


def test_some_source_is_required():
    with pytest.raises(ValueError):
        import_jobs_reports(start=JAN[0], end=JAN[1])
    with pytest.raises(CalendarDataMissing, match="no timezone"):
        import_jobs_reports(BlsText(), start=ts("2031-02-01").tz_localize(None), end=ts("2031-03-01"))


# --- OpenD economic calendar as a jobs source --------------------------------------------------------

def test_opend_alone_gives_the_us_payrolls_and_nothing_else():
    ev = import_jobs_reports(opend=OpenDCalendar(FakeQuote()), start=ts("2030-12-20"), end=ts("2031-07-01"))
    assert [(e.time, e.kind, e.scope) for e in ev] == [(ts("2031-01-03 13:30"), "nfp", "USD"),
                                                       (ts("2031-06-06 12:30"), "nfp", "USD")]   # no UK decoy, no ADP, no CPI


def test_economic_calendar_is_paged_and_windowed_by_exact_timestamps():
    q = FakeQuote(page=2)
    recs = OpenDCalendar(q).economic(ts("2031-01-03 00:00"), ts("2031-01-03 23:59"))
    assert sorted(r.title for r in recs) == ["Nonfarm Payrolls", "Nonfarm Payrolls", "Unemployment Rate"]
    assert {r.time for r in recs} == {ts("2031-01-03 13:30")}                  # the Jan 1 and Jan 14 rows are cut away
    pages = [c for c in q.calls if c[0] == "economic"]
    assert all(c[4] == 100 and c[1] == ["US"] for c in pages)
    q2 = FakeQuote(page=2)
    OpenDCalendar(q2).economic(ts("2031-01-01"), ts("2031-03-31"))
    assert len([c for c in q2.calls if c[0] == "economic"]) > 1 and [c[5] for c in q2.calls][0] is None


def test_bls_and_opend_that_agree_pass_the_cross_check():
    ev = import_jobs_reports(BlsText(), OpenDCalendar(FakeQuote()), start=ts("2030-12-20"), end=ts("2031-07-01"))
    assert len(ev) == 2 and all("BLS" in e.note for e in ev)                    # BLS is authoritative


def _is_us_payrolls(r):
    return r["title"] == "Nonfarm Payrolls" and r["country"] == "United States"


def _a_day_late(rs):
    return [dict(r, timestamp=r["timestamp"] + 86400) if _is_us_payrolls(r) and r["timestamp"] < 1.95e9 else r for r in rs]


def _without_payrolls(rs):
    return [r for r in rs if not _is_us_payrolls(r)]


def _with_an_unscheduled_release(rs):
    first = next(r for r in rs if _is_us_payrolls(r))
    return rs + [dict(first, timestamp=first["timestamp"] + 20 * 86400)]


@pytest.mark.parametrize("edit, match", [(_a_day_late, "OpenD has it at|disagree"), (_without_payrolls, "no payrolls entry"),
                                         (_with_an_unscheduled_release, "the BLS schedule does not")])
def test_a_disagreeing_opend_calendar_raises(edit, match):
    q = FakeQuote(economic=edit(rows("opend_economic_calendar_format_sample.json")))
    with pytest.raises(CalendarDataMissing, match=match):
        import_jobs_reports(BlsText(), OpenDCalendar(q), start=ts("2030-12-20"), end=ts("2031-07-01"))


def test_cross_check_reports_each_problem_in_words():
    rels = parse_bls_ics(ICS)[:1]
    assert cross_check(rels, [], JAN[0], JAN[1])[0].startswith("BLS has a jobs report at 2031-01-03 13:30 UTC")


@pytest.mark.parametrize("fail", ["economic", "runaway", "no_cursor"])
def test_opend_economic_failures_raise(fail):
    with pytest.raises(CalendarDataMissing):
        OpenDCalendar(FakeQuote(fail=fail)).economic(JAN[0], JAN[1])


def test_the_page_loop_is_bounded():
    q = FakeQuote(fail="runaway")
    with pytest.raises(CalendarDataMissing, match="did not end"):
        OpenDCalendar(q).economic(JAN[0], JAN[1])
    assert len(q.calls) == MAX_PAGES


# --- earnings ------------------------------------------------------------------------------------------------

E_WIN = (ts("2031-01-20"), ts("2031-02-10"))


def test_earnings_events_carry_the_symbol_and_the_announcement_time_in_utc():
    q = FakeQuote()
    ev = import_earnings(OpenDCalendar(q), ["AAPL", "NVDA", "AMD", "MSFT"], *E_WIN)
    assert [(e.scope, e.time, e.kind) for e in ev] == [
        ("MSFT", ts("2031-01-27 21:05"), "earnings"),          # 16:05 New York, EST
        ("AAPL", ts("2031-01-29 21:30"), "earnings"),
        ("AMD", ts("2031-02-04 21:00"), "earnings"),
        ("NVDA", ts("2031-02-04 21:20"), "earnings")]
    assert all("OpenD" in e.note for e in ev)


def test_long_windows_are_fetched_in_pieces_of_at_most_seven_days_without_gaps():
    q = FakeQuote()
    import_earnings(OpenDCalendar(q), ["AAPL"], ts("2031-01-01"), ts("2031-03-01"))
    spans = [(pd.Timestamp(b), pd.Timestamp(e)) for kind, m, b, e in q.calls if kind == "earnings"]
    assert all((e - b).days <= 7 for b, e in spans) and len(spans) >= 8
    assert all(b2 == e1 + pd.Timedelta(days=1) for (b1, e1), (b2, e2) in zip(spans, spans[1:]))
    assert spans[0][0] <= pd.Timestamp("2030-12-31") and spans[-1][1] >= pd.Timestamp("2031-03-02")


def test_a_symbol_with_no_announcement_in_the_window_raises_by_default():
    with pytest.raises(CalendarDataMissing, match="GOOG"):
        import_earnings(OpenDCalendar(FakeQuote()), ["AAPL", "GOOG"], *E_WIN)
    ev = import_earnings(OpenDCalendar(FakeQuote()), ["AAPL", "GOOG"], *E_WIN, require_all=False)
    assert [e.scope for e in ev] == ["AAPL"]


def test_nothing_found_at_all_raises_even_when_partial_results_are_allowed():
    with pytest.raises(CalendarDataMissing, match="no earnings found"):
        import_earnings(OpenDCalendar(FakeQuote(earnings=[])), ["AAPL"], *E_WIN, require_all=False)


def test_a_row_without_a_timestamp_for_a_requested_symbol_raises():
    with pytest.raises(CalendarDataMissing, match="QQQQ"):
        import_earnings(OpenDCalendar(FakeQuote()), ["AAPL", "QQQQ"], *E_WIN)
    # but a symbol nobody asked about with the same problem is irrelevant
    import_earnings(OpenDCalendar(FakeQuote()), ["AAPL"], *E_WIN)


def test_a_date_and_timestamp_that_disagree_raise():
    bad = [dict(r, earnings_timestamp=r["earnings_timestamp"] + 40 * 86400) if r["security"] == "US.AAPL" else r
           for r in rows("opend_earnings_calendar_format_sample.json")]
    with pytest.raises(CalendarDataMissing, match="disagree"):
        OpenDCalendar(FakeQuote(earnings=bad)).earnings(["AAPL"], ts("2031-01-20"), ts("2031-03-20"))


def test_opend_errors_and_bad_input_raise():
    with pytest.raises(CalendarDataMissing, match="not logged in"):
        import_earnings(OpenDCalendar(FakeQuote(fail="earnings")), ["AAPL"], *E_WIN)
    with pytest.raises(ValueError):
        import_earnings(OpenDCalendar(FakeQuote()), [], *E_WIN)
    with pytest.raises(CalendarDataMissing):
        import_earnings(OpenDCalendar(FakeQuote()), ["AAPL"], E_WIN[0].tz_localize(None), E_WIN[1])
    with pytest.raises(ValueError):
        import_earnings(OpenDCalendar(FakeQuote()), ["AAPL"], E_WIN[1], E_WIN[0])

    class Boom:
        def earnings(self, *a):
            raise KeyError("x")
    with pytest.raises(CalendarDataMissing):
        import_earnings(Boom(), ["AAPL"], *E_WIN)


def test_extra_rows_from_a_chatty_source_are_ignored():
    class Chatty:
        def earnings(self, symbols, start, end):
            return OpenDCalendar(FakeQuote()).earnings(["AAPL", "NVDA", "JPM", "ZZZZ"], ts("2031-01-01"), ts("2031-03-01"))
    ev = import_earnings(Chatty(), ["AAPL"], *E_WIN)
    assert [e.scope for e in ev] == ["AAPL"]


def test_coverage_names_symbols_the_calendar_cannot_protect():
    ev = import_earnings(OpenDCalendar(FakeQuote()), ["AAPL", "NVDA"], *E_WIN)
    assert coverage(ev, ["AAPL", "NVDA", "AMD"], ts("2031-01-20"), 30) == ["AMD"]
    assert coverage(ev, ["AAPL", "NVDA"], ts("2031-01-20"), 5) == ["AAPL", "NVDA"]


# --- the adapter's boundaries --------------------------------------------------------------------------------------

def test_the_opend_adapter_is_read_only_and_never_imports_the_sdk():
    src = (Path(__file__).parents[1] / "tradex" / "events" / "opend_calendar.py")
    tree = ast.parse(src.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.startswith("moomoo") for m in imported)
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not called & {"place_order", "modify_order", "unlock_trade", "OpenSecTradeContext", "accinfo_query"}
    assert "OpenSecTradeContext" not in src.read_text().replace("never opens a trade context", "")


# --- from importers to the EventCalendar ------------------------------------------------------------------------------

def test_written_calendars_load_into_the_event_calendar_and_block_what_they_should(tmp_path):
    files = refresh(start=ts("2030-12-20"), end=ts("2031-02-10"), symbols=["AAPL", "NVDA"], bls=BlsText(),
                    opend=OpenDCalendar(FakeQuote()), out_dir=tmp_path)
    cal = EventCalendar.load(list(files.values()))
    assert {e.kind for e in cal.events} == {"nfp", "earnings"}
    aapl = ts("2031-01-29 21:30")
    veto, _, hits = cal.check("AAPL", "stocks", aapl - pd.Timedelta(days=1), aapl - pd.Timedelta(hours=1) + pd.Timedelta(days=2))
    assert veto and "earnings" in veto and hits
    assert cal.check("MSFT", "stocks", aapl - pd.Timedelta(days=1), aapl)[0] is None        # not in the written set
    nfp = ts("2031-01-03 13:30")
    assert cal.check("EUR_USD", "forex", nfp - pd.Timedelta(hours=1), nfp)[0] is not None
    assert cal.check("EUR_USD", "forex", nfp - pd.Timedelta(hours=12), nfp - pd.Timedelta(hours=11))[0] is None


def test_refresh_is_all_or_nothing(tmp_path):
    good = refresh(start=ts("2030-12-20"), end=ts("2031-02-10"), symbols=["AAPL"], bls=BlsText(),
                   opend=OpenDCalendar(FakeQuote()), out_dir=tmp_path)
    before = {k: p.read_text() for k, p in good.items()}
    with pytest.raises(CalendarDataMissing):                                    # GOOG has no date: earnings step fails
        refresh(start=ts("2030-12-20"), end=ts("2031-02-10"), symbols=["AAPL", "GOOG"], bls=BlsText(),
                opend=OpenDCalendar(FakeQuote()), out_dir=tmp_path)
    with pytest.raises(CalendarDataMissing):                                    # BLS down: jobs step fails
        refresh(start=ts("2030-12-20"), end=ts("2031-02-10"), symbols=["AAPL"], bls=BlsText(""),
                opend=OpenDCalendar(FakeQuote()), out_dir=tmp_path)
    assert {k: p.read_text() for k, p in good.items()} == before
    assert not list(tmp_path.glob("*.tmp"))


def test_yaml_writer_is_atomic_and_round_trips(tmp_path, monkeypatch):
    ev = import_jobs_reports(BlsText(), start=JAN[0], end=JAN[1])
    path = write_events_yaml(ev, tmp_path / "x.yaml", "line one\nline two")
    assert path.read_text().startswith("# line one\n# line two\n")
    assert EventCalendar.load([path]).events == ev
    import os
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        write_events_yaml(ev, tmp_path / "y.yaml")
    assert not (tmp_path / "y.yaml").exists() and not list(tmp_path.glob("y.yaml.*"))
