import pandas as pd
import pytest

from tradex.data.oanda_history import HostRefused, OandaHistory, _http_get, parse_candles, spread_series


def candle(t, bo, ao, complete=True):
    """Shape of a v20 price=BA candle (as documented and as returned by the practice host)."""
    b = {"o": f"{bo:.5f}", "h": f"{bo + 0.0010:.5f}", "l": f"{bo - 0.0010:.5f}", "c": f"{bo + 0.0002:.5f}"}
    a = {"o": f"{ao:.5f}", "h": f"{ao + 0.0010:.5f}", "l": f"{ao - 0.0010:.5f}", "c": f"{ao + 0.0002:.5f}"}
    return {"complete": complete, "volume": 1234, "time": t, "bid": b, "ask": a}


RECORDED = {
    None: {"instrument": "EUR_USD", "granularity": "H1", "candles": [
        candle("2016-10-04T00:00:00.000000000Z", 1.12100, 1.12114),
        candle("2016-10-04T01:00:00.000000000Z", 1.12150, 1.12162)]},
    "2016-10-04T01:00:01Z": {"instrument": "EUR_USD", "granularity": "H1", "candles": [
        candle("2016-10-04T02:00:00.000000000Z", 1.12200, 1.12215),
        candle("2016-10-04T03:00:00.000000000Z", 1.12250, 1.12265, complete=False)]},
}


class FakeHttp:
    def __init__(self):
        self.urls = []

    def __call__(self, url, headers):
        assert headers["Authorization"] == "Bearer test-token"
        self.urls.append(url)
        frm = pd.Timestamp(dict(x.split("=") for x in url.split("?")[1].split("&"))["from"].replace("%3A", ":"))
        return RECORDED[None] if frm < pd.Timestamp("2016-10-04T00:30Z") else RECORDED["2016-10-04T01:00:01Z"]


def test_parse_keeps_complete_candles_with_bid_ask_and_mid():
    df = parse_candles(RECORDED["2016-10-04T01:00:01Z"]["candles"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["bid_open"] == 1.122 and row["ask_open"] == 1.12215
    assert row["open"] == pytest.approx((1.122 + 1.12215) / 2)
    assert spread_series(df).iloc[0] == pytest.approx(0.00015)


def test_fetch_pages_caches_and_uses_practice_host(tmp_path):
    http = FakeHttp()
    h = OandaHistory(cache_dir=tmp_path, token="test-token", http=http, page=2, pause_s=0)
    df = h.fetch("EUR_USD", "H1", years=10, end="2016-10-05")
    assert len(df) == 3 and df.index[0] == pd.Timestamp("2016-10-04 00:00", tz="UTC")
    assert all(u.startswith("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?") for u in http.urls)
    assert all("price=BA" in u and "granularity=H1" in u for u in http.urls)
    assert len(http.urls) == 2
    cached = h.load("EUR_USD", "H1")
    pd.testing.assert_frame_equal(cached, df, check_freq=False)


def test_live_host_refused():
    with pytest.raises(HostRefused):
        _http_get("https://api-fxtrade.oanda.com/v3/instruments/EUR_USD/candles", {})
