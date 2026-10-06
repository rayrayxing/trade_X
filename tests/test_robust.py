import numpy as np
import pandas as pd
import pytest

from tradex.backtest.validation import WalkForwardConfig, walk_forward
from tradex.costs.models import model_for
from tradex.backtest.engine import EngineConfig
from tradex.research import robust
from tradex.strategy.spec import StrategySpec


def _returns(mu, sd, n=2000, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n, tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=idx)


def test_bootstrap_interval_brackets_the_sample_sharpe_and_separates_signs():
    up = _returns(0.001, 0.01)
    b = robust.stationary_bootstrap_sharpe(up, n=1000)
    sr = up.mean() / up.std() * np.sqrt(252)
    lo, mid, hi = (b["percentiles"][k] for k in ("p5", "p50", "p95"))
    assert lo < sr < hi and abs(mid - sr) < 0.3 and b["p_sharpe_le_0"] < 0.05
    flat = robust.stationary_bootstrap_sharpe(_returns(0.0, 0.01, seed=1), n=1000)
    assert flat["percentiles"]["p5"] < 0 < flat["percentiles"]["p95"]
    assert robust.stationary_bootstrap_sharpe(up.iloc[:5])["p_sharpe_le_0"] is None


def test_subperiod_keeps_only_returns_and_trades_from_the_start():
    r = _returns(0.001, 0.01)
    trades = pd.DataFrame({"entry_time": [r.index[10], r.index[-10]], "net_pnl": [5.0, -2.0], "gross_pnl": [5.0, -2.0],
                           "spread_slippage": 0.0, "fees": 0.0, "financing": 0.0, "borrow": 0.0, "margin_interest": 0.0,
                           "r_multiple": [1.0, -0.4], "bars_held": 3, "exit_reason": "target"})
    s = robust.subperiod(r, trades, "2016-01-01")
    assert s["trades"] == 1 and s["profit_factor"] == 0.0
    assert s["total_return"] == pytest.approx((1 + r[r.index >= "2016-01-01"]).prod() - 1, abs=1e-3)


def test_rerun_folds_at_normal_costs_reproduces_the_walk_forward(stock_data):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    costs = model_for("stocks")
    rep = walk_forward(spec, stock_data, costs=costs, wf=WalkForwardConfig(n_folds=3, grid_points=2))
    again = robust.rerun_folds(spec, stock_data, rep.folds, costs, EngineConfig())
    assert again["trades"] == rep.oos["trades"]
    assert again["fold_returns"] == [round(f.test_return, 3) for f in rep.folds]
    worse = robust.rerun_folds(spec, stock_data, rep.folds, model_for("stocks", stress=3.0), EngineConfig())
    assert worse["total_return"] <= again["total_return"]
