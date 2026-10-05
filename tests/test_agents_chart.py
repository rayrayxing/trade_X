import json

import pandas as pd
import pytest

from agent_fakes import FakeGateway, failed
from tradex.agents.chart_agent import ChartAgentConfig, ChartPlan, ChartReader, chart_facts, knowable_bars
from tradex.agents.common import InboxSink, ReplayGateway, ShadowStore
from tradex.core.inbox import Mailbox
from tradex.core.ledger import Ledger
from tradex.data.synthetic import synthetic_bars

BARS = synthetic_bars(200, seed=3, start="2026-01-01")        # synthetic bars: tests only
ASOF = BARS.index[150] + pd.Timedelta(days=1)                 # exactly the close of bar 150
DID = "2026-10-05-0003"


def read_reply(bias="bearish", conf=0.8, **kw):
    return {"pattern": "evening star", "bias": bias, "confidence": conf, "support": None, "resistance": None,
            "invalidation": None, "note": "reversal at highs", **kw}


def make(tmp_path, reply, **kw):
    Ledger(tmp_path / "l.db")
    mb, store = Mailbox(tmp_path / "l.db"), ShadowStore(tmp_path / "s.db")
    gw = FakeGateway(reply)
    return ChartReader(gw, store, InboxSink(mb, store), **kw), store, mb, gw


def inbox(mb):
    return mb.read("SELECT * FROM agent_inbox")


def test_only_closed_bars_are_knowable():
    k = knowable_bars(BARS, "D1", ASOF)
    assert k.index[-1] == BARS.index[150]                      # bar 150 closes at ASOF, so it counts
    k2 = knowable_bars(BARS, "D1", ASOF - pd.Timedelta(hours=1))
    assert k2.index[-1] == BARS.index[149]                     # one hour earlier it is still forming
    h1 = synthetic_bars(50, tf="H1", seed=1)
    assert knowable_bars(h1, "H1", h1.index[10] + pd.Timedelta(minutes=59)).index[-1] == h1.index[9]
    assert knowable_bars(BARS, "D1", ASOF.tz_localize(None)).index[-1] == BARS.index[150]


def test_prompt_and_facts_ignore_bars_after_asof(tmp_path):
    reader, store, mb, gw = make(tmp_path, read_reply("neutral"))
    reader.read(ASOF, "NVDA", BARS)
    reader.read(ASOF, "NVDA", BARS.iloc[:151])                  # same history, no future bars at all
    reader.read(ASOF, "NVDA", BARS.iloc[:200])
    assert gw.calls[0][1] == gw.calls[1][1] == gw.calls[2][1]
    # and a bar still forming at asof is not shown either
    reader.read(ASOF - pd.Timedelta(hours=1), "NVDA", BARS)
    assert gw.calls[3][1] != gw.calls[0][1] and BARS.index[150].strftime("%Y-%m-%d") not in gw.calls[3][1].split("BARS")[1]
    f_full, f_cut = chart_facts(knowable_bars(BARS, "D1", ASOF)), chart_facts(BARS.iloc[:151])
    assert f_full == f_cut


def test_veto_when_confident_read_contradicts_the_plan(tmp_path):
    reader, store, mb, gw = make(tmp_path, read_reply("bearish", 0.8))
    r = reader.read(ASOF, "NVDA", BARS, ChartPlan("NVDA", 1, DID, 100.0))
    assert r.status == "proposed" and r.action == "veto"
    (row,) = inbox(mb)
    body = json.loads(row["body"])
    assert (row["source"], row["action"], row["target"]) == ("chart_reader", "veto", DID)
    assert body["shadow"] and body["expected_direction"] == -1 and body["plan_direction"] == 1
    assert body["bar_open"] == BARS.index[150].isoformat() and body["atr"] > 0 and body["close"] == pytest.approx(BARS["close"].iloc[150])
    assert "evening star" in body["reason"] and body["bars"] == 1
    assert "considering a long here at 100" in gw.calls[0][1]


def test_shrink_for_moderate_contradiction_and_nothing_when_it_agrees(tmp_path):
    reader, store, mb, _ = make(tmp_path, read_reply("bearish", 0.65))
    assert reader.read(ASOF, "NVDA", BARS, ChartPlan("NVDA", 1, DID)).action == "shrink"
    body = json.loads(inbox(mb)[0]["body"])
    assert body["factor"] == 0.5 and inbox(mb)[0]["target"] == DID
    reader, store, mb, _ = make(tmp_path / "x", read_reply("bearish", 0.5))
    r = reader.read(ASOF, "NVDA", BARS, ChartPlan("NVDA", 1, DID))
    assert r.status == "neutral" and inbox(mb) == []              # too weak to act on
    reader, store, mb, _ = make(tmp_path / "y", read_reply("bullish", 0.95))
    r = reader.read(ASOF, "NVDA", BARS, ChartPlan("NVDA", 1, DID))
    assert r.status == "agree" and inbox(mb) == []                 # agreement never adds size: it just logs
    assert store.decisions("chart_reader")[0]["status"] == "agree"


def test_standalone_read_flags_only_confident_directional_reads(tmp_path):
    reader, store, mb, _ = make(tmp_path, read_reply("bullish", 0.7))
    r = reader.read(ASOF, "NVDA", BARS)
    assert r.action == "flag"
    row = inbox(mb)[0]
    assert row["target"] == "NVDA" and json.loads(row["body"])["expected_direction"] == 1
    for reply in (read_reply("neutral", 0.99), read_reply("bullish", 0.4)):
        reader, store, mb, _ = make(tmp_path / str(reply["bias"]), reply)
        assert reader.read(ASOF, "NVDA", BARS).status == "neutral" and inbox(mb) == []


def test_levels_outside_a_sane_range_are_nulled(tmp_path):
    last = float(BARS["close"].iloc[150])
    reader, store, mb, _ = make(tmp_path, read_reply("bullish", 0.9, support=last * 0.95, resistance=last * 40,
                                                     invalidation="soon"))
    r = reader.read(ASOF, "NVDA", BARS)
    assert r.levels == {"support": round(last * 0.95, 4), "resistance": None, "invalidation": None}


def test_bad_replies_and_failures_write_nothing(tmp_path):
    for reply, status in (("no idea", "invalid"), ({"bias": "sideways", "confidence": 0.9}, "invalid"),
                          ({"bias": "bullish", "confidence": "high"}, "invalid"), (failed("chart_reader"), "skipped")):
        reader, store, mb, _ = make(tmp_path / status / str(len(str(reply))), reply)
        assert reader.read(ASOF, "NVDA", BARS).status == status and inbox(mb) == []


def test_short_history_skips_without_a_model_call(tmp_path):
    reader, store, mb, gw = make(tmp_path, read_reply())
    r = reader.read(BARS.index[30] + pd.Timedelta(days=1), "NVDA", BARS)
    assert r.status == "skipped" and gw.calls == []


def test_plan_and_symbol_must_be_consistent(tmp_path):
    reader, *_ = make(tmp_path, read_reply())
    with pytest.raises(ValueError):
        reader.read(ASOF, "NVDA", BARS, ChartPlan("AMD", 1, DID))
    with pytest.raises(ValueError):
        reader.read(ASOF, "NVDA", BARS, ChartPlan("NVDA", 1, "not-a-decision-id"))
    with pytest.raises(ValueError):
        reader.read(ASOF, "nvda; drop table", BARS)


def test_prompt_contains_candle_and_swing_facts(tmp_path):
    reader, store, mb, gw = make(tmp_path, read_reply("neutral"))
    reader.read(ASOF, "NVDA", BARS)
    p = gw.calls[0][1]
    for s in ("ATR14", "20-bar range", "candlestick detectors on the last bar", "confirmed swing points",
              "double top/bottom"):
        assert s in p
    assert p.count("o=") == ChartAgentConfig().bars_shown


def test_replay(tmp_path):
    reader, store, mb, _ = make(tmp_path, read_reply("bearish", 0.9))
    plan = ChartPlan("NVDA", 1, DID)
    live = reader.read(ASOF, "NVDA", BARS, plan)
    store2 = ShadowStore(tmp_path / "r.db")
    again = ChartReader(ReplayGateway(store), store2, InboxSink(None, store2)).read(ASOF, "NVDA", BARS, plan)
    assert (again.action, again.bias, again.confidence) == (live.action, live.bias, live.confidence)
