import json

import pandas as pd
import pytest

from tradex.costs.models import (OandaFxCosts, RateMissing, configure_run_mode, model_for, usd_per_unit)
from tradex.data.barbuilder import BarBuilder, to_frame
from tradex.data.oanda import (LiveHostRefused, PriceStream, QuoteBook, Tick, check_host, fetch_ba_candles,
                               instruments_to_stream, measured_spread_pips, parse_stream_line)

T = pd.Timestamp


def candle(ts, bid, ask, complete=True):
    px = lambda v: {"o": str(v), "h": str(v + 0.0002), "l": str(v - 0.0002), "c": str(v + 0.0001)}
    return {"complete": complete, "volume": 10, "time": ts, "bid": px(bid), "ask": px(ask)}


def price_line(inst, t, bid, ask, tradeable=True):
    return json.dumps({"type": "PRICE", "instrument": inst, "time": t, "tradeable": tradeable,
                       "bids": [{"price": str(bid), "liquidity": 1}], "asks": [{"price": str(ask), "liquidity": 1}]})


HB = json.dumps({"type": "HEARTBEAT", "time": "2024-01-02T00:00:05.000000000Z"})


def test_host_guard():
    check_host("https://api-fxpractice.oanda.com/v3/x")
    check_host("https://stream-fxpractice.oanda.com/v3/x")
    for bad in ("https://api-fxtrade.oanda.com/v3/x", "https://stream-fxtrade.oanda.com/", "https://evil.example/"):
        with pytest.raises(LiveHostRefused):
            check_host(bad)


def test_candles_paging_bid_ask_mid():
    t0 = T("2024-01-02", tz="UTC")
    times = [(t0 + pd.Timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z") for i in range(5)]
    pages = {0: [candle(times[i], 1.1000, 1.1002) for i in range(3)],
             1: [candle(times[3], 1.1000, 1.1002), candle(times[4], 1.1, 1.1002, complete=False)]}
    calls = []

    def http(url, headers):
        calls.append(url)
        assert "price=BA" in url and headers["Authorization"] == "Bearer tok"
        return {"candles": pages[len(calls) - 1] if len(calls) <= 2 else []}

    df = fetch_ba_candles("EUR_USD", "M1", "2024-01-02", "2024-01-02T01:00:00", token="tok", http=http, page=3)
    assert len(calls) == 2 and len(df) == 3 + 1            # incomplete candle excluded
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["ask_open"].iloc[0] > df["bid_open"].iloc[0]
    assert df["open"].iloc[0] == pytest.approx(1.1001)
    assert measured_spread_pips(df, "EUR_USD") == pytest.approx(2.0)
    with pytest.raises(RateMissing):
        measured_spread_pips(df.iloc[0:0], "EUR_USD")


def test_parse_stream_lines():
    tk = parse_stream_line(price_line("EUR_USD", "2024-01-02T00:00:01.123456789Z", 1.1, 1.1002))
    assert isinstance(tk, Tick) and tk.mid == pytest.approx(1.1001)
    assert parse_stream_line(HB) == "heartbeat"
    assert parse_stream_line(price_line("EUR_USD", "2024-01-02T00:00:01Z", 1.1, 1.1002, tradeable=False)) is None
    assert parse_stream_line("") is None


def test_stream_reconnects_after_drop_and_silence():
    streams = [
        iter([price_line("EUR_USD", "2024-01-02T00:00:01Z", 1.1, 1.1002), HB]),     # ends: drop
        (x for x in [price_line("EUR_USD", "2024-01-02T00:00:02Z", 1.1, 1.1002)] + [TimeoutError("silent")]),
        iter([price_line("EUR_USD", "2024-01-02T00:00:03Z", 1.1, 1.1002)]),
    ]

    def gen(src):
        for x in src:
            if isinstance(x, Exception):
                raise x
            yield x

    it = iter(streams)
    sleeps = []
    ps = PriceStream(["EUR_USD"], connect=lambda: gen(next(it)), sleep=sleeps.append)
    got = list(ps.ticks(max_reconnects=2))
    assert [t.time.second for t in got] == [1, 2, 3]
    assert ps.heartbeats == 1 and ps.reconnects == 3
    assert sum(sleeps) + 7.0 < 10                  # silence window + total backoff stays inside 10 s
    assert max(sleeps) <= 2.0 and sleeps[0] == 0.0


def test_stream_never_touches_live_host(monkeypatch):
    ps = PriceStream(["EUR_USD"], account_id="a", token="t")
    monkeypatch.setattr("tradex.data.oanda.STREAM_HOST", "stream-fxtrade.oanda.com")
    with pytest.raises(LiveHostRefused):
        list(ps.ticks(max_reconnects=0))


def feed(bb, specs):
    out = []
    for s, p in specs:
        out += bb.on_tick(T(s, tz="UTC"), p, p + 0.0002)
    return out


def test_bar_closes_once_late_ticks_and_gaps(caplog):
    bb = BarBuilder("EUR_USD", "M1")
    bars = feed(bb, [("2024-01-02 00:00:05", 1.10), ("2024-01-02 00:00:30", 1.11), ("2024-01-02 00:00:50", 1.09),
                     ("2024-01-02 00:01:01", 1.10)])
    assert len(bars) == 1
    b = bars[0]
    assert (b.open, b.volume) == (pytest.approx(1.1001), 3.0) and b.high > b.low and not b.after_gap
    with caplog.at_level("WARNING", logger="tradex.data.bars"):
        assert feed(bb, [("2024-01-02 00:00:59", 1.5)]) == []        # late: dropped and logged
    assert bb.late_ticks == 1 and "late tick" in caplog.text
    assert bb.flush(T("2024-01-02 00:01:30", tz="UTC")) == []        # bar not over yet
    closed = bb.flush(T("2024-01-02 00:02:00", tz="UTC"))
    assert len(closed) == 1 and bb.flush(T("2024-01-02 00:05:00", tz="UTC")) == []   # exactly once
    assert feed(bb, [("2024-01-02 00:01:10", 1.2)]) == []            # cannot reopen a closed bar
    assert bb.late_ticks == 2
    out = feed(bb, [("2024-01-02 00:05:10", 1.10), ("2024-01-02 00:06:00", 1.10)])
    assert len(out) == 1 and out[0].ts == T("2024-01-02 00:05", tz="UTC") and out[0].after_gap
    assert [(g.start, g.end) for g in bb.gaps] == [(T("2024-01-02 00:02", tz="UTC"), T("2024-01-02 00:05", tz="UTC"))]
    assert len(to_frame(bars + closed + out)) == 3                    # no filler bars were invented


def test_m5_alignment_and_unsupported_tf():
    bb = BarBuilder("EUR_USD", "M5")
    bars = feed(bb, [("2024-01-02 00:03:00", 1.1), ("2024-01-02 00:07:00", 1.1)])
    assert bars[0].ts == T("2024-01-02 00:00", tz="UTC")
    with pytest.raises(ValueError):
        BarBuilder("EUR_USD", "D1")


def test_quote_book_and_strict_costs():
    now = T("2024-01-02 00:00:10", tz="UTC")
    qb = QuoteBook(max_age_s=30, clock=lambda: now)
    qb.update(Tick("EUR_USD", T("2024-01-02 00:00:05", tz="UTC"), 1.1000, 1.1002))
    qb.update(Tick("USD_JPY", T("2024-01-02 00:00:05", tz="UTC"), 150.00, 150.02))
    qb.update(Tick("GBP_USD", T("2023-12-31", tz="UTC"), 1.27, 1.2702))           # stale
    assert qb.spread_pips("EUR_USD") == pytest.approx(2.0)
    assert qb.usd_per_unit("JPY") == pytest.approx(1 / 150.01)
    for bad in (lambda: qb.usd_per_unit("GBP"), lambda: qb.spread_pips("AUD_USD"), lambda: qb.usd_per_unit("CHF")):
        with pytest.raises(RateMissing):
            bad()
    assert instruments_to_stream(["EUR_JPY"]) == ["EUR_JPY", "EUR_USD", "USD_JPY"]

    with pytest.raises(ValueError):
        OandaFxCosts(mode="paper")                                    # no source, no defaults
    c = model_for("forex", mode="paper", spread_source=qb)
    price, parts = c.fill("EUR_USD", 1, 1.1, now)
    assert parts["spread"] == pytest.approx(0.5 * 2.0 * 0.0001)
    with pytest.raises(RateMissing):
        c.fill("AUD_USD", 1, 0.66, now)                               # no AUD_USD quote: no default spread
    assert OandaFxCosts().fill("AUD_USD", 1, 0.66, now)[1]["spread"] > 0   # backtest keeps its labelled defaults

    try:
        configure_run_mode("paper", qb)
        assert usd_per_unit("JPY") == pytest.approx(1 / 150.01)
        with pytest.raises(RateMissing):
            usd_per_unit("AUD")                                       # APPROX table is never used
        assert usd_per_unit("USD") == 1.0
    finally:
        configure_run_mode("backtest")
    assert usd_per_unit("AUD") == pytest.approx(0.66)
    with pytest.raises(ValueError):
        configure_run_mode("live")
