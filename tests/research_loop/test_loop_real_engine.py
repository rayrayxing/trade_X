"""The loop on the real backtest engine, walk-forward, trial ledger and gate code (fixture bars, small settings)."""
import pandas as pd

from tradex.backtest.validation import Thresholds, WalkForwardConfig
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.evaluators import EngineEvaluator, current_params
from tradex.strategy.spec import StrategySpec

from loopkit import HOLDOUT_START, FakeData, make_frames, make_rig, spec_dict

SID = "stk-rsi-test"
UNIVERSE = ("AAA", "BBB", "CCC", "DDD")
LENIENT = Thresholds(min_trades=1, min_profit_factor=0.0, min_dsr=0.0, max_drawdown=-1.0, min_positive_folds=0.0)
WF = WalkForwardConfig(n_folds=3, grid_points=2)
CFG = LoopConfig(screen_min_bars=100, screen_min_trades=5, screen_min_profit_factor=0.0, holdout_min_trades=1, apply=True)


def engine_rig(tmp_path, thresholds):
    data = FakeData(make_frames(UNIVERSE, n=1500, seed0=4))
    rig = make_rig(tmp_path, cfg=CFG, data=data, evaluator=EngineEvaluator(thresholds=thresholds, wf=WF))
    rig.add_spec(spec_dict(SID, universe=UNIVERSE))
    return rig


def test_end_to_end_on_the_real_engine_with_lenient_thresholds(tmp_path):
    rig = engine_rig(tmp_path, LENIENT)
    r = rig.run()
    assert rig.status(SID) == "paper", {k: v.items for k, v in r.items()}
    screen = rig.state.latest_evidence(SID, 1, "screen")["data"]
    assert screen["trades"] >= 5 and screen["bars"] > 1000 and screen["params"] == {"stop_atr": 2.0}
    wf = rig.state.latest_evidence(SID, 1, "walk_forward")["data"]
    assert wf["folds"] == 3 and wf["n_trials"] >= wf["ledger_count"] >= 2
    assert wf["data_key"].startswith("fixture-bars:D1:AAA,BBB,CCC,DDD:2017-01-02..")
    assert pd.Timestamp(wf["data_key"].split("..")[1], tz="UTC") < HOLDOUT_START          # the walk-forward data ends before the holdout
    ho = rig.state.latest_evidence(SID, 1, "holdout")["data"]
    assert ho["trades"] >= 1 and ho["days"] > 0 and ho["holdout_start"].startswith("2022-01-03")
    b = rig.state.baseline(SID, 1)
    assert b["n"] > 100 and b["std"] > 0
    assert rig.state.verify_audit() == (True, None)


def test_the_protected_thresholds_reject_a_synthetic_strategy_and_the_look_is_never_used(tmp_path):
    rig = engine_rig(tmp_path, Thresholds.from_config())                       # config/gates/thresholds.yaml
    rig.run()
    assert rig.status(SID) == "rejected"
    wf = rig.state.latest_evidence(SID, 1, "walk_forward")
    assert wf["passed"] == 0 and wf["data"]["failing"]
    assert wf["data"]["thresholds"]["min_dsr"] == 0.95 and wf["data"]["thresholds"]["min_trades"] == 100
    assert not rig.holdout.looks and rig.state.latest_evidence(SID, 1, "holdout") is None


def test_the_trial_ledger_counts_the_screen_and_every_walk_forward_grid_point(tmp_path):
    rig = engine_rig(tmp_path, LENIENT)
    rig.run(stages=("propose", "screen"))
    assert rig.trials.count(SID) == 1                                           # the screen's own parameter set
    rig.run(run_id="2026-W42", stages=("walk_forward",))
    n = rig.trials.count(SID)
    grid = StrategySpec.from_dict(spec_dict(SID, universe=UNIVERSE)).param_grid(WF.grid_points)
    assert n == len(grid) + (0 if {"stop_atr": 2.0} in grid else 1)
    assert {s["strategy_id"] for s in rig.trials.summary()} == {SID}
    rows = rig.trials._conn().execute("SELECT DISTINCT source FROM trials").fetchall()
    assert {r[0] for r in rows} == {"loop_screen", "walk_forward"}


def test_current_params_reads_exit_rules_and_dotted_feature_paths():
    s = StrategySpec.from_dict(spec_dict(search_space={"stop_atr": [1.0, 3.0], "features.ema50.period": [20, 80]}))
    assert current_params(s) == {"stop_atr": 2.0, "features.ema50.period": 50}


def test_the_holdout_backtest_only_trades_inside_the_holdout(tmp_path):
    rig = engine_rig(tmp_path, LENIENT)
    ev = rig.ev
    frames = make_frames(UNIVERSE, n=1500, seed0=4)
    spec = StrategySpec.from_dict(spec_dict(SID, universe=UNIVERSE))
    out = ev.holdout(spec, {"stop_atr": 2.0}, frames, HOLDOUT_START)
    assert out["trades"] >= 1
    from tradex.backtest.engine import EngineConfig, run_backtest
    res = run_backtest(spec, frames, ev.costs(spec), EngineConfig(start=HOLDOUT_START), params={"stop_atr": 2.0})
    assert res.trades["entry_time"].min() >= HOLDOUT_START
