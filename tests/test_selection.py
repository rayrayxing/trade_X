import numpy as np
import pandas as pd
import pytest

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig
from tradex.backtest.validation import WalkForwardConfig, walk_forward
from tradex.data.synthetic import synthetic_bars
from tradex.risk.sizing import (StressCase, conservative_kelly, drawdown_scale, kelly_fraction, leverage_gate,
                                ruin_probability)
from tradex.selection.allocator import AllocationConfig, StrategyRecord, allocate, blend_report, correlation_clusters
from tradex.selection.regime import classify_regime
from tradex.strategy.spec import StrategySpec


def test_walk_forward_folds_are_disjoint_and_embargoed(stock_data):
    spec = StrategySpec.load("strategies/seeds/stk-rsi2-meanrev.yaml")
    rep = walk_forward(spec, stock_data, wf=WalkForwardConfig(n_folds=3, grid_points=2))
    assert rep.n_trials == 4
    for f in rep.folds:
        assert pd.Timestamp(f.train_end) <= pd.Timestamp(f.test_start) - pd.Timedelta(days=spec.exit.max_bars)
    for a, b in zip(rep.folds, rep.folds[1:]):
        assert pd.Timestamp(a.test_end) <= pd.Timestamp(b.test_start)
    assert rep.status in ("validated", "rejected")
    assert set(rep.oos) >= {"dsr", "sharpe", "trades", "profit_factor"}


def test_noise_strategy_rejected():
    data = {s: synthetic_bars(1200, seed=40 + i, trend_strength=0.0, drift=0.0) for i, s in enumerate("ABC")}
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    rep = walk_forward(spec, data, wf=WalkForwardConfig(n_folds=3, grid_points=3))
    assert rep.status == "rejected"


def test_dsr_penalises_more_trials():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 750))
    assert metrics.deflated_sharpe(r, 1, 0.0) > metrics.deflated_sharpe(r, 50, 0.0004) > metrics.deflated_sharpe(r, 500, 0.0004)


def test_wilson_lower_below_point_estimate():
    assert metrics.wilson_lower(60, 100) < 0.6
    assert metrics.wilson_lower(6, 10) < metrics.wilson_lower(60, 100)


def test_kelly_and_ruin():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_fraction(0.3, 1.0) < 0
    rng = np.random.default_rng(0)
    r = np.where(rng.random(300) < 0.45, 2.0, -1.0)
    low = ruin_probability(r, 0.01, paths=2000)
    high = ruin_probability(r, 0.01, leverage=10, paths=2000)
    stressed = ruin_probability(r, 0.01, leverage=3, paths=2000, stress=StressCase())
    assert low < ruin_probability(r, 0.01, leverage=3, paths=2000) <= stressed
    assert high > 0.5
    assert leverage_gate(0.2, "validated", 14) == 0
    assert leverage_gate(0.001, "arbitrage", 20) == pytest.approx(14)
    assert leverage_gate(0.01, "validated", 20) == 10
    assert drawdown_scale(80, 100) == pytest.approx(0.5)


def _record(id_, rets, status="validated", dsr=0.97, symbols=("X",), win=0.55):
    rng = np.random.default_rng(abs(hash(id_)) % 2**32)
    r_mult = np.where(rng.random(200) < win, 2.0, -1.0)
    return StrategyRecord(id_, "stocks", status, rets, pd.DataFrame({"r_multiple": r_mult}), dsr=dsr,
                          symbols=list(symbols))


def test_allocator_caps_clusters_and_exclusions():
    idx = pd.bdate_range("2024-01-01", periods=400, tz="UTC")
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0.001, 0.01, 400), index=idx)
    recs = [
        _record("a", base),
        _record("a_clone", base + rng.normal(0, 0.001, 400)),                    # correlated with a
        _record("b", pd.Series(rng.normal(0.0012, 0.01, 400), index=idx), symbols=("Y",)),
        _record("loser", pd.Series(rng.normal(-0.002, 0.01, 400), index=idx), symbols=("Z",), dsr=0.1),
        _record("draft", base, status="proposed"),
    ]
    cfg = AllocationConfig(total_risk_pct=12, per_strategy_cap_pct=3, per_cluster_cap_pct=4)
    snap = allocate(recs, "trend/normal", None, cfg)
    alloc = {a.strategy_id: a for a in snap.allocations}
    assert "draft" in snap.excluded and "loser" in snap.excluded
    assert alloc["a"].cluster == alloc["a_clone"].cluster != alloc["b"].cluster
    assert alloc["a"].risk_pct + alloc["a_clone"].risk_pct <= 4 + 1e-9
    assert all(a.risk_pct <= 3 + 1e-9 for a in snap.allocations)
    assert snap.total_risk_pct <= 12
    rep = blend_report(recs, snap)
    assert "blend" in rep


def test_allocator_respects_kelly_and_drawdown():
    idx = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    rets = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, 300), index=idx)
    no_edge = _record("thin", rets, win=0.25)
    snap = allocate([no_edge], "trend/normal")
    assert "thin" in snap.excluded
    good = _record("good", rets, win=0.6)
    full = allocate([good], "x").total_risk_pct
    half = allocate([good], "x", equity=80, peak_equity=100).total_risk_pct
    assert half <= full


def test_correlation_unknown_overlap_same_symbols_grouped():
    a = _record("a", pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10, tz="UTC")))
    b = _record("b", pd.Series([0.01] * 10, index=pd.bdate_range("2025-01-01", periods=10, tz="UTC")))
    assert len(set(correlation_clusters([a, b], 0.6, 60).values())) == 1


def test_regime_labels():
    reg = classify_regime(synthetic_bars(600, seed=2))
    assert set(reg["trend"].unique()) <= {"trend", "range", "unknown"}
    assert (reg["label"].iloc[-100:] != "unknown").all()
