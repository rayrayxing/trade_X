import numpy as np
import pandas as pd
import pytest

from tradex.data.compare import compare_feeds, rth_daily_from_intraday
from tradex.data.synthetic import synthetic_bars
from tradex.pipeline import morning_plan, resolve_universe
from tradex.scout.base import CompositeScout, ScoutConfig, TechnicalSource, scout_mask
from tradex.selection.allocator import StrategyRecord
from tradex.strategy import expr
from tradex.strategy.spec import StrategySpec, load_dir

from conftest import simple_spec


def test_expression_sandbox():
    idx = pd.RangeIndex(3)
    env = {"close": pd.Series([1.0, 2.0, 3.0]), "ema": pd.Series([2.0, 2.0, 2.0])}
    assert expr.evaluate("close > ema", env, idx).tolist() == [False, False, True]
    assert expr.evaluate("cross_above(close, ema)", env, idx).tolist() == [False, False, True]
    for bad in ["__import__('os')", "close.__class__", "[x for x in close]", "open('f')", "close if ema else 1"]:
        with pytest.raises((expr.ExprError, SyntaxError)):
            expr.evaluate(bad, env, idx)


def test_schema_check_catches_problems():
    s = simple_spec(long="close > nope")
    assert any("undefined" in e for e in s.validate())
    raw = dict(s.raw)
    raw["features"] = {"x": {"fn": "talib.NOPE"}}
    raw["exit"] = {"stop_atr": 0, "target_r": 2, "max_bars": 5}
    errs = StrategySpec.from_dict(raw).validate()
    assert any("not registered" in e for e in errs) and any("time limit" in e for e in errs)


def test_param_overrides():
    s = StrategySpec.load("strategies/seeds/stk-rsi2-meanrev.yaml")
    v = s.with_params({"stop_atr": 3.0, "features.ema200.period": 100})
    assert v.exit.stop_atr == 3.0 and v.features["ema200"]["period"] == 100
    assert s.exit.stop_atr == 2.5


def _universe(n=30):
    out = {}
    for i in range(n):
        b = synthetic_bars(400, seed=100 + i, price=20 + 5 * i, volume=2e6 + 1e5 * i, vol=0.01 + 0.001 * (i % 7))
        out[f"S{i:02d}"] = b
    return out


def test_technical_scout_picks_10_to_20_without_lookahead():
    bars = _universe()
    asof = bars["S00"].index[300]
    scout = CompositeScout(list(bars), [TechnicalSource(bars=bars)], ScoutConfig(always_on=[]))
    wl = scout.scan(asof)
    assert 10 <= len(wl.items) <= 20
    assert all(i.reasons for i in wl.items)
    future_changed = {k: v.copy() for k, v in bars.items()}
    for v in future_changed.values():
        v.loc[v.index >= asof, ["close", "volume"]] *= 3
    wl2 = CompositeScout(list(bars), [TechnicalSource(bars=future_changed)], ScoutConfig(always_on=[])).scan(asof)
    assert wl.symbols == wl2.symbols


def test_scout_mask_and_watchlist_universe():
    bars = _universe(25)
    scout = CompositeScout(list(bars), [TechnicalSource(bars=bars)], ScoutConfig(always_on=[]))
    masks = scout_mask(scout, bars["S00"].index[250:260])
    assert set(masks) == set(bars)
    daily_counts = pd.DataFrame(masks).sum(axis=1)
    assert ((daily_counts >= 10) & (daily_counts <= 20)).all()
    spec = StrategySpec.load("strategies/seeds/stk-breakout-volume.yaml")
    wl = scout.scan(bars["S00"].index[255])
    uni = resolve_universe(spec, wl)
    assert "$watchlist" not in uni and set(wl.symbols) <= set(uni) and "NVDA" in uni


def test_morning_plan_runs():
    idx = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    rets = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300), index=idx)
    trades = pd.DataFrame({"r_multiple": np.where(np.random.default_rng(1).random(200) < 0.55, 2.0, -1.0)})
    specs = load_dir("strategies")
    recs = [StrategyRecord("stk-rsi2-meanrev", "stocks", "validated", rets, trades, dsr=0.97, symbols=["SPY"])]
    spy = synthetic_bars(500, seed=9)
    plan = morning_plan(spy.index[-1], specs, recs, spy)
    assert plan.allocation.allocations and "stk-rsi2-meanrev" in plan.universes


def test_feed_comparison_identical_and_noisy():
    ref = synthetic_bars(400, seed=5)
    same = compare_feeds(ref.assign(volume=ref.volume * 0.03), ref, "X", "D1")
    assert same.price_ok and same.close_bps_median == 0
    assert same.iex_volume_share_median == pytest.approx(0.03)
    rng = np.random.default_rng(0)
    noisy = ref.copy()
    noisy[["open", "high", "low", "close"]] *= (1 + rng.normal(0, 0.004, (len(ref), 1)))
    bad = compare_feeds(noisy, ref, "X", "D1")
    assert not bad.price_ok and bad.close_bps_p95 > 20


def test_rth_daily_aggregation():
    idx = pd.date_range("2026-01-05 13:00", "2026-01-05 22:00", freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)
    d = rth_daily_from_intraday(bars)
    assert len(d) == 1 and d.volume.iloc[0] == 10.0 * 26  # 09:30-16:00 NY = 26 quarter hours
