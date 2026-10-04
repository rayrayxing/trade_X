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


def test_sharpe_variance_comes_from_every_set_tried(tmp_path):
    led = TrialLedger(tmp_path / "v.sqlite")
    w = ("2024-01-01", "2024-06-01")
    for i, sr in enumerate([0.0, 0.1, 0.2]):
        led.record("s", 1, {"a": i}, run_id="r1", fold=0, window=w, sharpe=sr)
    assert led.sharpe_variance("s") > 0 and led.sharpe_variance("none") == 0.0
    # a later run of one set in the same window updates that set's Sharpe instead of adding a spread of its own
    led.record("s", 1, {"a": 0}, run_id="r2", fold=0, window=w, sharpe=0.0)
    assert led.sharpe_variance("s") == __import__("statistics").variance([0.0, 0.1, 0.2])
    # a window holding a single set contributes nothing
    led.record("s", 1, {"a": 0}, run_id="r3", fold=1, window=("2024-06-01", "2024-12-01"), sharpe=5.0)
    assert led.sharpe_variance("s") == __import__("statistics").variance([0.0, 0.1, 0.2])


def test_narrow_rerun_keeps_the_deflation_of_the_wide_search(stock_data, tmp_path):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    led = TrialLedger(tmp_path / "rerun.sqlite")
    wide = walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=3), trials=led)
    assert wide.oos["expected_max_sharpe_annual"] > 0
    only_best = spec.with_params(wide.recommended_params)
    narrow = walk_forward(only_best, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=1), trials=led)
    # one parameter set this run (V=0 on its own), yet the ledger's spread still deflates it
    assert narrow.n_trials_this_run == 1 and narrow.n_trials >= wide.n_trials
    assert narrow.oos["expected_max_sharpe_annual"] > 0
