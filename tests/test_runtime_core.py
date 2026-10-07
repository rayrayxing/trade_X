"""The bar-close core: live-style driving and replay write the same decision rows."""
import pandas as pd
import pytest

from gapkit import recorded
from test_spine import _frames, _two_family_specs
from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.replay import build_replay_core, run_replay
from tradex.data.synthetic import synthetic_bars
from tradex.execution.guard import GuardedBroker, OrderGuard, ledger_verdicts
from tradex.runtime.market import BarStore
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

DECISION_KINDS = ("plan", "vote", "veto", "verdict", "order", "fill", "close", "counterfactual")


def _decisions(led: Ledger) -> list[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in led.rows() if r["kind"] in DECISION_KINDS]


def run_live(strategies, frames, start, cfg=None, base_tf=None, use_es=True):
    """Drive the core the way the live runtime does: an empty bar store that receives each
    bar only once it has closed, a clock at the bar close, one close event per timeframe at
    its boundary, and the ensemble book behind the order guard like a venue adapter."""
    base_tf = base_tf or min((s.signal_tf for s in strategies), key=duration)
    bar = duration(base_tf)
    clock = ReplayClock()
    store = BarStore(base_tf, {s: df[df.index < start] for s, df in frames.items()}, clock)
    led = Ledger(":memory:", git_commit="t")
    core = build_replay_core(strategies, frames, led, start, cfg=cfg, base_tf=base_tf, data=store, clock=clock,
                             use_es=use_es)
    core.brokers["ensemble"] = GuardedBroker(core.brokers["ensemble"], OrderGuard(ledger_verdicts(led), {"sim"}))
    opens = sorted({t for df in frames.values() for t in df.index if t >= start})
    for t in opens:
        for sym, df in frames.items():
            if t in df.index:
                store.append(sym, df.loc[[t]])
        close = t + bar
        clock.set(close)
        for tf in core.tfs:
            d = duration(tf)
            if (close - pd.Timestamp(0, tz="UTC")) % d == pd.Timedelta(0):
                core.on_bar_close(tf, close)
    core.finish(opens[-1] + bar)
    return led


def test_live_core_and_replay_core_write_identical_decision_rows():
    frames = _frames()
    start = frames["AAA"].index[260]
    live = run_live(_two_family_specs(), frames, start)
    rep = run_replay(_two_family_specs(), frames, Ledger(":memory:", git_commit="t"), start)
    assert live.digest() == rep.digest
    assert _decisions(live) == _decisions(rep.ledger)
    assert any(r["kind"] == "order" and r["book"] == "ensemble" for r in _decisions(live))


def _mixed_specs():
    base = {"version": 1, "asset_class": "forex", "universe": ["EUR_USD", "USD_JPY"], "status": "paper",
            "holding": {"expected_hours": 24, "crosses_rollover": True},
            "exit": {"stop_atr": 1.5, "target_r": 2.5, "max_bars": 20}}
    h1 = StrategySpec.from_dict(base | {"id": "h1-trend", "family": "trend", "timeframes": {"signal": "H1"},
                                        "features": {"ema": {"fn": "talib.EMA", "period": 30},
                                                     "ema_h4": {"fn": "talib.EMA", "period": 10, "tf": "H4"}},
                                        "entry": {"long": "close > ema and close > ema_h4",
                                                  "short": "close < ema and close < ema_h4"}})
    h4 = StrategySpec.from_dict(base | {"id": "h4-mom", "family": "momentum", "timeframes": {"signal": "H4"},
                                        "features": {"roc": {"fn": "talib.ROC", "period": 6}},
                                        "entry": {"long": "roc > 0", "short": "roc < 0"}})
    return [h1, h4]


def _fx_frames(n=1_100):
    kw = dict(tf="H1", vol=0.002, trend_strength=0.0004, regime_len=200, start="2025-01-06", business_days=False)
    return {"EUR_USD": recorded(synthetic_bars(n, seed=5, price=1.10, **kw)),
            "USD_JPY": recorded(synthetic_bars(n, seed=6, price=150.0, **kw))}


def test_mixed_timeframes_run_together_and_match_live():
    frames = _fx_frames()
    start = frames["EUR_USD"].index[700]
    rep = run_replay(_mixed_specs(), frames, Ledger(":memory:", git_commit="t"), start, use_es=False)
    live = run_live(_mixed_specs(), frames, start, use_es=False)
    assert _decisions(live) == _decisions(rep.ledger)
    votes = rep.ledger.rows(kind="vote")
    h4 = [pd.Timestamp(v["time"]) for v in votes if v["strategy_id"] == "h4-mom"]
    h1 = [pd.Timestamp(v["time"]) for v in votes if v["strategy_id"] == "h1-trend"]
    assert h4 and h1
    assert all(t.hour % 4 == 0 and t.minute == 0 for t in h4)          # H4 votes only at H4 closes
    assert any(t.hour % 4 for t in h1)


def test_higher_timeframe_bar_is_invisible_until_it_closes():
    frames = _fx_frames(48)
    store = BarStore("H1", frames, ReplayClock())
    t0 = frames["EUR_USD"].index[0]
    assert len(store.bars("EUR_USD", t0 + pd.Timedelta(hours=3), "H4")) == 0
    h4 = store.bars("EUR_USD", t0 + pd.Timedelta(hours=4), "H4")
    assert len(h4) == 1 and h4["close"].iloc[0] == frames["EUR_USD"]["close"].iloc[3]


def test_missing_fx_rate_in_paper_blocks_instead_of_guessing():
    frames = _fx_frames()
    frames = {"EUR_JPY": frames["USD_JPY"]}                             # no USD cross: JPY cannot be priced
    specs = _mixed_specs()
    for s in specs:
        s.universe = ["EUR_JPY"]
    led = Ledger(":memory:")
    run_replay(specs, frames, led, frames["EUR_JPY"].index[1000], cfg=CoreConfig(mode="paper"),
               use_es=False)
    vetoes = [v for v in led.rows(kind="veto") if v["source"] == "data"]
    assert vetoes and "JPY" in vetoes[0]["reason"]
    assert not led.rows(kind="order")
    assert any(not h["ok"] for h in led.rows(kind="health"))
    led2 = Ledger(":memory:")                                           # replay may still use the rough constant
    run_replay(specs, frames, led2, frames["EUR_JPY"].index[1000], use_es=False)
    assert not [v for v in led2.rows(kind="veto") if v["source"] == "data"]


def test_mixed_timeframe_votes_form_one_plan_with_the_shortest_time_stop():
    """An H4 vote stays valid until the next H4 close, so H1 closes in between combine it
    with fresh H1 votes; the time stop is the shorter in time, counted in H1 bars, and the
    entry fills at the next H1 open."""
    frames = _fx_frames()
    start = frames["EUR_USD"].index[700]

    def specs():
        s = _mixed_specs()
        s[1].exit.max_bars = 3                                         # 3 H4 bars = 12 H1 bars < 20 H1 bars
        return s
    rep = run_replay(specs(), frames, Ledger(":memory:", git_commit="t"), start, use_es=False)
    live = run_live(specs(), frames, start, use_es=False)
    assert _decisions(live) == _decisions(rep.ledger)
    led = rep.ledger
    both = [p for p in led.rows(kind="plan") if p["book"] == "ensemble" and set(p["strategies"]) == {"h1-trend", "h4-mom"}]
    assert both and all(p["families"] == ["momentum", "trend"] for p in both)
    assert all(p["tf"] == "H1" and p["max_bars"] == 12 for p in both)
    assert any(pd.Timestamp(p["time"]).hour % 4 for p in both)        # formed between H4 closes
    for p in both:
        t = pd.Timestamp(p["time"])
        h4 = [v for v in led.rows(kind="vote", decision_id=p["decision_id"]) if v["strategy_id"] == "h4-mom"]
        assert len(h4) == 1 and t - pd.Timedelta(hours=4) < pd.Timestamp(h4[0]["time"]) <= t
    traded = [p for p in both if led.rows(kind="fill", decision_id=p["decision_id"])]
    assert traded
    for p in traded:
        assert led.rows(kind="fill", decision_id=p["decision_id"])[0]["time"] == p["time"]   # next H1 open


def test_votes_expire_at_their_timeframes_next_close():
    frames = _fx_frames()
    start = frames["EUR_USD"].index[700]
    led = run_replay(_mixed_specs(), frames, Ledger(":memory:", git_commit="t"), start, use_es=False).ledger
    for p in led.rows(kind="plan"):
        t = pd.Timestamp(p["time"])
        for v in led.rows(kind="vote", decision_id=p["decision_id"]):
            assert t < pd.Timestamp(v["time"]) + duration(v["tf"]) and pd.Timestamp(v["time"]) <= t
