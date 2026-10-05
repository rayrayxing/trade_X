"""Official macro history parsers on responses recorded from the publishers on 5 Oct 2026
(tests/fixtures/macro, trimmed to a few weeks around known changes). No network."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tradex.data import macro_history as mh
from tradex.research import builders, panels
from tradex.research.sources import CsvRateSource, DataUnavailable, FileEventCalendar

FIX = Path(__file__).parent / "fixtures" / "macro"


def text(name):
    return (FIX / name).read_bytes().decode("utf-8-sig")


def at(s, day):
    """The value in force on ``day`` of a change-point series."""
    return float(s[s.index <= pd.Timestamp(day)].iloc[-1])


def test_each_publisher_format_parses_to_the_published_change():
    usd = mh.currency_series("USD", FIX)
    assert at(usd, "2008-12-15") == 1.0 and at(usd, "2008-12-16") == 0.125   # single target, then range midpoint
    assert at(usd, "2015-12-16") == 0.375
    assert at(mh.currency_series("EUR", FIX), "2024-06-12") == 3.75
    assert at(mh.currency_series("GBP", FIX), "2024-08-01") == 5.0
    assert at(mh.currency_series("CAD", FIX), "2024-06-05") == 5.0 and at(mh.currency_series("CAD", FIX), "2024-06-06") == 4.75
    aud = mh.parse_rba_f1(text("rba_f1.csv"))
    assert at(aud, "2026-09-30") == 4.6 and at(aud, "2026-09-29") == 4.35
    chf = mh.currency_series("CHF", FIX)
    assert at(chf, "2015-01-31") == -0.75          # target-range midpoint, dated at the month's end
    assert at(chf, "2019-06-13") == -0.75 and at(chf, "2022-09-23") == 0.5
    jp = mh.currency_series("JPY", FIX)
    assert at(jp, "2016-09-21") == -0.1 and at(jp, "2024-03-21") == 0.05
    assert mh.parse_bis(text("bis_NZ.csv")).loc["2024-03-20"] == 5.5


def test_rates_table_feeds_the_carry_rate_source_and_cites_its_sources(tmp_path):
    out = mh.write_rates(FIX, tmp_path / "policy_rates.csv")
    body = out.read_text()
    assert "fred.stlouisfed.org" in body and "data.snb.ch" in body and "stats.bis.org" in body
    src = CsvRateSource(out)
    assert src.currencies() == sorted(mh.RATE_FILES)
    idx = pd.DatetimeIndex(["2024-06-04", "2024-06-06"], tz="UTC")
    assert list(src.rates("CAD", idx)) == pytest.approx([0.05, 0.0475])


def test_fomc_pages_give_scheduled_meetings_only_with_the_release_rule():
    h2008 = mh.parse_fomc_historical(text("fed_fomchistorical2008.htm"))
    assert [r["date"].strftime("%m-%d") for r in h2008] == ["01-30", "03-18", "04-30", "06-25", "08-05", "09-16",
                                                            "10-29", "12-16"]       # no conference calls
    h2020 = [r["date"].strftime("%m-%d") for r in mh.parse_fomc_historical(text("fed_fomchistorical2020.htm"))]
    assert "03-03" not in h2020 and "03-15" not in h2020 and "03-18" not in h2020 and len(h2020) == 7
    h2012 = {r["date"].strftime("%m-%d"): r["press_conference"] for r in mh.parse_fomc_historical(text("fed_fomchistorical2012.htm"))}
    assert h2012["08-01"] is False and h2012["01-25"] is True                 # "July 31-August 1" meeting
    cal = [r["date"].strftime("%Y-%m-%d") for r in mh.parse_fomc_calendar(text("fed_fomccalendars.htm"))]
    assert "2024-05-01" in cal and "2025-01-29" in cal and "2025-08-22" not in cal   # Apr/May; notation vote dropped
    assert mh.release_time(pd.Timestamp("2024-05-01"), True)[0] == pd.Timestamp("2024-05-01 18:00", tz="UTC")
    assert mh.release_time(pd.Timestamp("2012-01-25"), True)[0] == pd.Timestamp("2012-01-25 17:30", tz="UTC")
    assert mh.release_time(pd.Timestamp("2008-01-30"), False)[0] == pd.Timestamp("2008-01-30 19:15", tz="UTC")


def test_fomc_table_reads_as_the_event_calendar(tmp_path, monkeypatch):
    for n in ("fed_fomccalendars.htm", "fed_fomchistorical2020.htm"):
        shutil.copy(FIX / n, tmp_path / n)
    monkeypatch.setattr(mh, "FOMC_FIRST_YEAR", 2020)
    monkeypatch.setitem(mh.RAW, "fed_fomccalendars.htm", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    # the calendar fixture starts in 2024, so 2020-2023 would be read from historical pages: keep only 2020
    for y in (2021, 2022, 2023):
        (tmp_path / f"fed_fomchistorical{y}.htm").write_text("")
    p = mh.write_fomc(tmp_path)
    times = FileEventCalendar(p).times()
    assert len(times) == 7 + 16 and times[0] == pd.Timestamp("2020-01-29 19:00", tz="UTC")


def test_vix_is_attached_to_its_own_session(tmp_path, monkeypatch):
    shutil.copy(FIX / "cboe_VIX_History.csv", tmp_path / "cboe_VIX_History.csv")
    p = mh.write_vix(tmp_path)
    monkeypatch.setattr(builders, "VIX_FILE", p)
    idx = pd.DatetimeIndex([pd.Timestamp("2020-03-16", tz="America/New_York"),
                            pd.Timestamp("2020-03-17", tz="America/New_York")]).tz_convert("UTC")
    v = builders.vix_column(idx)
    assert v.iloc[0] == pytest.approx(82.69) and v.iloc[1] < v.iloc[0]
    monkeypatch.setattr(builders, "VIX_FILE", tmp_path / "none.csv")
    with pytest.raises(DataUnavailable, match="VIX"):
        builders.vix_column(idx)


def test_only_official_hosts_are_fetched():
    with pytest.raises(ValueError, match="official host"):
        mh.http_get("https://example.com/rates.csv")


def test_session_columns_on_30_minute_bars_flag_the_bar_closing_at_1530():
    t = pd.date_range("2024-03-04 09:30", "2024-03-04 15:30", freq="30min", tz="America/New_York")
    t = t.append(pd.date_range("2024-03-05 09:30", "2024-03-05 15:30", freq="30min", tz="America/New_York"))
    b = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}, index=t.tz_convert("UTC"))
    b.iloc[13, b.columns.get_loc("close")] = 101.0      # 2024-03-05 09:30 bar closes up 1% on the previous close
    c = panels.session_columns(b, pd.Timedelta(minutes=30))
    flagged = c.index[c["last_full"] > 0].tz_convert("America/New_York").strftime("%H:%M")
    assert list(flagged) == ["15:00", "15:00"]
    assert c["fh_ret"].iloc[13:].tolist() == pytest.approx([0.01] * 13)
