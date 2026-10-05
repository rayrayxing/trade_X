import json
from pathlib import Path

import pandas as pd
import pytest

from tradex.data import earnings as E
from tradex.events import EventCalendar
from tradex.strategy.spec import FILTERS

FIX = Path(__file__).parent / "fixtures" / "earnings"


def sec_http(url):
    """Recorded SEC responses (trimmed): Apple's submissions JSON and its older-filings page."""
    if url == E.TICKERS_URL:
        return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"}}
    name = url.rsplit("/", 1)[1]
    return json.loads((FIX / f"sec_{name}").read_text())


class FakeOpenD:
    def __init__(self, ret=0):
        self.ret, self.codes = ret, []

    def get_financials_earnings_price_move(self, code, period_count=None):
        self.codes.append((code, period_count))
        if self.ret:
            return -1, "This API only supports equities."
        return 0, pd.DataFrame(json.loads((FIX / "opend_AAPL_price_move.json").read_text()))


def test_parse_opend_one_row_per_report_with_timing():
    t = E.parse_opend(pd.DataFrame(json.loads((FIX / "opend_AAPL_price_move.json").read_text())))
    assert t["date"].is_unique and len(t) == 3
    assert set(t["timing"]) == {"after"} and set(t["source"]) == {"opend"}
    assert t["date"].iloc[-1] == "2026-07-30"


def test_parse_sec_keeps_8k_item_202_and_falls_back_to_10q_in_years_without():
    docs = [sec_http("https://data.sec.gov/submissions/CIK0000320193.json"),
            sec_http("https://data.sec.gov/submissions/CIK0000320193-submissions-001.json")]
    t = E.parse_sec(docs)
    k8 = t[t["source"] == "sec_8k_2.02"]["date"].tolist()
    assert "2026-07-30" in k8 and "2007-01-17" in k8 and "2006-10-18" in k8
    assert "2026-04-20" not in k8 and "2007-04-24" not in k8           # 5.02 / 8.01 only
    q = t[t["source"] == "sec_10q_10k"]["date"].tolist()
    assert q == ["2003-02-10", "2003-05-13"]                           # 2003 has no item-2.02 8-K
    assert set(t["timing"]) == {"unknown"}


def test_merge_prefers_opend_and_fills_gaps_with_sec():
    op = pd.DataFrame({"date": ["2015-01-27", "2015-04-27"], "timing": "after", "source": "opend"})
    sec = pd.DataFrame({"date": ["2014-10-20", "2015-01-27", "2015-04-28", "2015-07-21"], "timing": "unknown",
                        "source": "sec_8k_2.02"})
    m = E.merge(op, sec)
    assert m["date"].tolist() == ["2014-10-20", "2015-01-27", "2015-04-27", "2015-07-21"]
    assert m.set_index("date").loc["2015-01-27", "source"] == "opend"


def test_event_time_is_conservative_for_the_next_open_fill():
    assert E.event_time("2026-07-30", "after") == pd.Timestamp("2026-07-30 16:00", tz="America/New_York")
    assert E.event_time("2026-01-15", "before") == pd.Timestamp("2026-01-15 00:00", tz="America/New_York")
    assert E.event_time("2026-01-15", "unknown") == E.event_time("2026-01-15", "before")


def test_fetch_caches_and_falls_back_to_sec_when_opend_refuses(tmp_path):
    e = E.Earnings(tmp_path, opend_ctx=FakeOpenD(), sec_http=sec_http, pause_s=0)
    t = e.fetch("AAPL")
    assert e.ctx.codes == [("US.AAPL", 50)]
    assert set(t["source"]) == {"opend", "sec_8k_2.02"} and "2026-07-30" in t["date"].tolist()
    assert t["date"].min() >= "2005-01-01"
    pd.testing.assert_frame_equal(e.load("AAPL"), t.astype(str))
    e2 = E.Earnings(tmp_path / "b", opend_ctx=FakeOpenD(ret=-1), sec_http=sec_http, pause_s=0)
    assert set(e2.fetch("AAPL")["source"]) == {"sec_8k_2.02"}
    assert E.Earnings(tmp_path, sec_http=sec_http).cik("BRK.B") == 1067983


def test_sec_host_guard():
    with pytest.raises(ValueError):
        E._sec_get("https://example.com/submissions/CIK0000320193.json")


def test_filter_dates_feed_the_no_earnings_filter_per_symbol(tmp_path):
    E.Earnings(tmp_path, opend_ctx=FakeOpenD(), sec_http=sec_http, pause_s=0).fetch("AAPL")
    d = E.filter_dates(["AAPL", "SPY", "MSFT"], etfs=["SPY"], cache_dir=tmp_path)
    assert len(d["AAPL"]) > 0 and len(d["SPY"]) == 0 and "MSFT" not in d
    idx = pd.date_range("2026-07-20", "2026-08-05", freq="D", tz="America/New_York").tz_convert("UTC")
    bars = pd.DataFrame({"close": 1.0}, index=idx)
    allow = FILTERS["no_earnings_3d"](bars, {"earnings": d, "symbol": "AAPL"})
    blocked = allow[~allow].index.tz_convert("America/New_York").strftime("%m-%d").tolist()
    assert blocked == ["07-28", "07-29", "07-30"]          # after-close report on 07-30: 3 days before it
    ctx = {"earnings": d, "symbol": "SPY"}
    assert FILTERS["no_earnings_3d"](bars, ctx).all() and not ctx.get("warnings")
    ctx = {"earnings": d, "symbol": "MSFT"}
    assert FILTERS["no_earnings_3d"](bars, ctx).all() and "inactive" in ctx["warnings"][0]


def test_calendar_events_veto_trades_holding_through_a_report(tmp_path):
    E.Earnings(tmp_path, opend_ctx=FakeOpenD(), sec_http=sec_http, pause_s=0).fetch("AAPL")
    cal = EventCalendar(events=E.calendar_events(["AAPL"], tmp_path))
    veto, _, hits = cal.check("AAPL", "stocks", pd.Timestamp("2026-07-28 14:00", tz="UTC"),
                              pd.Timestamp("2026-07-31 20:00", tz="UTC"))
    assert veto and hits[0].kind == "earnings" and "opend" in hits[0].note
    veto, _, _ = cal.check("MSFT", "stocks", pd.Timestamp("2026-07-28", tz="UTC"), pd.Timestamp("2026-07-31", tz="UTC"))
    assert veto is None
