from tradex.backtest.validation import WalkForwardConfig, walk_forward
from tradex.research.trials import TrialLedger, params_hash
from tradex.strategy.spec import StrategySpec


def test_ledger_counts_distinct_param_sets(tmp_path):
    led = TrialLedger(tmp_path / "t.sqlite")
    led.record("s", 1, {"a": 1, "b": 2}, run_id="r1", sharpe=0.1)
    led.record("s", 2, {"b": 2, "a": 1}, run_id="r2", sharpe=0.2)   # same set, other order and version
    led.record("s", 1, {"a": 2}, run_id="r2")
    led.record("other", 1, {"a": 1}, run_id="r3")
    assert led.count("s") == 2 and led.runs("s") == 2 and led.count("other") == 1 and led.count("none") == 0
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert {r["strategy_id"]: r["param_sets"] for r in led.summary()} == {"other": 1, "s": 2}


def test_walk_forward_appends_and_deflates_with_history(stock_data, tmp_path):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    led = TrialLedger(tmp_path / "wf.sqlite")
    small = walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=2), trials=led)
    assert small.n_trials == 4 and small.n_trials_this_run == 4 and led.count(spec.id) == 4
    # A second, wider search adds new parameter sets; N now counts both searches.
    wide = walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=3), trials=led)
    assert wide.n_trials_this_run == 9
    assert wide.n_trials == led.count(spec.id) == len({params_hash(p) for p in
                                                       spec.param_grid(2) + spec.param_grid(3)})
    # Re-running the small search alone still deflates by everything tried so far.
    again = walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=2), trials=led)
    assert again.n_trials == wide.n_trials > again.n_trials_this_run


def test_walk_forward_uses_global_ledger_by_default(stock_data, tmp_path):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=2, grid_points=2))
    assert TrialLedger(tmp_path / "trials.sqlite").count(spec.id) == 4   # conftest points the env var here


def test_trial_group_counts_every_test_of_the_same_hypothesis(stock_data, tmp_path):
    led = TrialLedger(tmp_path / "g.sqlite")
    led.record("a", 1, {"x": 1}, run_id="r1")
    led.record("b", 1, {"x": 1}, run_id="r2")      # same params under another id: another trial
    led.record("b", 1, {"x": 2}, run_id="r2")
    assert led.count_group(["a", "b", "a"]) == 3 and led.count_group(["c"]) == 0
    base = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    walk_forward(base, stock_data, wf=WalkForwardConfig(n_folds=2, grid_points=2), trials=led)
    sib = StrategySpec.from_dict({**base.raw, "id": "sibling",
                                  "provenance": {"shares_trials_with": [base.id]}})
    rep = walk_forward(sib, stock_data, wf=WalkForwardConfig(n_folds=2, grid_points=2), trials=led)
    assert rep.n_trials == 8 and led.count(sib.id) == 4
