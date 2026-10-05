import urllib.parse

import pandas as pd
import pytest

from agent_fakes import StubFeed, fixture, item
from tradex.agents.news import AlpacaNewsFeed, CombinedNewsFeed, MassiveNewsFeed, NewsError

START = pd.Timestamp("2026-10-02T00:00:00Z")
END = pd.Timestamp("2026-10-04T00:00:00Z")
SECRETS = {"alpaca_key_id": "AK-ID-SECRETVALUE", "alpaca_secret": "AK-SEC-SECRETVALUE", "massive_key": "MASSIVE-SECRETVALUE"}


def secret(n):
    return SECRETS[n]


class Recorder:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        return self.pages[len(self.calls) - 1]


def q(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def test_alpaca_request_shape_paging_and_filters():
    get = Recorder([fixture("alpaca_news_page1.json"), fixture("alpaca_news_page2.json")])
    slept = []
    feed = AlpacaNewsFeed(get=get, secret=secret, sleep=slept.append)
    items = feed.fetch(START, END, ["NVDA", "AMD", "TSLA", "AAPL", "MSFT"])
    url1, hdr1 = get.calls[0]
    assert url1.startswith("https://data.alpaca.markets/v1beta1/news?")
    p1 = q(url1)
    assert p1["symbols"] == "NVDA,AMD,TSLA,AAPL,MSFT" and p1["sort"] == "asc"
    assert p1["start"] == "2026-10-02T00:00:00Z" and p1["end"] == "2026-10-04T00:00:00Z"
    assert hdr1 == {"APCA-API-KEY-ID": "AK-ID-SECRETVALUE", "APCA-API-SECRET-KEY": "AK-SEC-SECRETVALUE"}
    assert q(get.calls[1][0])["page_token"] == "PAGE2TOKEN" and slept == [0.35]
    got = [(i.symbol, i.headline) for i in items]
    # roundup with 7 symbols skipped; duplicate TSLA headline from a second wire counted once;
    # AAPL item stamped after END dropped even though the fake server returned it; OTHR not requested
    assert ("NVDA", "NVDA raises data center outlook after supplier checks") in got
    assert sum(1 for s, hl in got if s == "TSLA") == 1
    assert not any(s in ("AAPL", "OTHR") for s, _ in got)
    assert {s for s, hl in got if "12 movers" in hl} == set()
    assert [s for s, hl in got if "export licence" in hl] == ["AMD", "NVDA"]
    assert all(i.published <= END and i.published.tzinfo is not None for i in items)
    assert [i.published for i in items] == sorted(i.published for i in items)
    assert items[0].source == "alpaca:benzinga" and items[0].url.endswith("41001")


def test_alpaca_chunks_symbols_and_caps_pages():
    page = {"news": [], "next_page_token": "again"}
    get = Recorder([page] * 20)
    feed = AlpacaNewsFeed(get=get, secret=secret, symbols_per_call=2, max_pages=2, sleep=lambda s: None)
    feed.fetch(START, END, ["A", "B", "C"])
    assert len(get.calls) == 4                          # 2 chunks x 2 pages
    assert q(get.calls[0][0])["symbols"] == "A,B" and q(get.calls[2][0])["symbols"] == "C"


def test_alpaca_error_does_not_leak_keys_or_url():
    def boom(url, headers):
        raise RuntimeError(f"failed {url} {headers}")
    with pytest.raises(NewsError) as e:
        AlpacaNewsFeed(get=boom, secret=secret).fetch(START, END, ["NVDA"])
    assert "SECRETVALUE" not in str(e.value) and "http" not in str(e.value)


def test_massive_window_query_local_filter_sentiment_and_throttle():
    get = Recorder([fixture("massive_news_page1.json"), fixture("massive_news_page2.json")])
    slept, clock = [], iter([0.0, 3.0, 3.0])
    feed = MassiveNewsFeed(get=get, secret=secret, sleep=slept.append, clock=lambda: next(clock))
    items = feed.fetch(START, END, ["NVDA", "AMD", "TSLA"])
    u1 = get.calls[0][0]
    p1 = q(u1)
    assert u1.startswith("https://api.massive.com/v2/reference/news?") and p1["apiKey"] == "MASSIVE-SECRETVALUE"
    assert p1["published_utc.gte"] == "2026-10-02T00:00:00Z" and p1["published_utc.lte"] == "2026-10-04T00:00:00Z"
    assert "ticker" not in p1                                  # one market-wide query, not one call per ticker
    assert get.calls[1][0] == "https://api.massive.com/v2/reference/news?cursor=CURSOR2&apiKey=MASSIVE-SECRETVALUE"
    assert slept == [pytest.approx(12.5 - 3.0)]               # second call waits out the 5/min free tier
    by = {(i.symbol, i.headline): i for i in items}
    assert by[("NVDA", "Why NVDA stock jumped today")].sentiment == 1.0
    assert by[("AMD", "AMD and TSLA on the move")].sentiment == 0.0
    assert by[("TSLA", "AMD and TSLA on the move")].sentiment == -1.0
    assert by[("NVDA", "Why NVDA stock jumped today")].source == "massive:The Motley Fool"
    assert not any(i.symbol == "ZZZZ" for i in items)         # not in the requested universe
    assert not any("cut-off" in i.headline for i in items)    # published after END


def test_massive_error_does_not_leak_key():
    def boom(url, headers):
        raise RuntimeError(url)
    with pytest.raises(NewsError) as e:
        MassiveNewsFeed(get=boom, secret=secret, sleep=lambda s: None).fetch(START, END, None)
    assert "SECRETVALUE" not in str(e.value)


def test_combined_dedupes_and_survives_a_dead_feed():
    a = StubFeed([item("NVDA", "2026-10-02 13:00", "NVDA Beats  Estimates"), item("AMD", "2026-10-02 12:00", "AMD up")])
    b = StubFeed([item("NVDA", "2026-10-02 13:01", "nvda beats estimates")])
    dead = StubFeed([], raises=TimeoutError("slow"))
    comb = CombinedNewsFeed([dead, a, b])
    out = comb.fetch(START, END, None)
    assert [(i.symbol) for i in out] == ["AMD", "NVDA"]
    assert len(comb.errors) == 1 and "TimeoutError" in comb.errors[0]
