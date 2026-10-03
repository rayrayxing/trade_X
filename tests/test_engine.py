import numpy as np
import pandas as pd
import pytest

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.costs.models import MoomooStockCosts, OandaFxCosts
from tradex.strategy.spec import StrategySpec, compute_signals, load_dir

from conftest import flat_bars, simple_spec

ZERO_COST = dict(platform_fee=0.0, settlement_fee_per_share=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0,
                 default_half_spread_bps=0.0, slippage_bps=0.0)


def test_seed_strategies_have_no_lookahead(stock_data, fx_data):
    """Signals at bar t must not change when future bars are removed."""
    for spec in load_dir("strategies"):
        data = stock_data if spec.asset_class == "stocks" else fx_data
        sym = next(iter(data))
        bars = data[sym]
        if spec.signal_tf == "H4":
            from tradex.timeframes import resample
            bars = resample(bars, "H4")
        full = compute_signals(spec, bars)
        cut = int(len(bars) * 0.7)
        part = compute_signals(spec, bars.iloc[:cut])
        for name in ("long_entry", "short_entry", "long_exit", "short_exit"):
            a = getattr(full, name).iloc[:cut]
            b = getattr(part, name)
            assert (a == b).all(), f"{spec.id}.{name} changes when future bars are removed"


def test_entry_next_open_and_gap_through_stop():
    bars = flat_bars(30)
    bars.iloc[17, :4] = [95.0, 95.5, 94.0, 95.0]  # gap down well below the stop
    spec = simple_spec(long="close > 0")
    res = run_backtest(spec, {"X": bars}, MoomooStockCosts(**ZERO_COST), EngineConfig(risk_pct=1.0))
    t = res.trades.iloc[0]
    sig_bar = 14  # first bar with ATR(14)
    assert t.entry_time == bars.index[sig_bar + 1]
    assert t.exit_reason == "stop_gap"
    assert t.exit_price == pytest.approx(95.0)
    assert t.r_multiple < -1  # a gap loses more than 1R


def test_stop_assumed_before_target_when_both_touched():
    bars = flat_bars(30)
    bars.iloc[16, 1:3] = [110.0, 90.0]
    res = run_backtest(simple_spec(), {"X": bars}, MoomooStockCosts(**ZERO_COST))
    assert res.trades.iloc[0].exit_reason == "stop"


def test_target_hit_and_costs_reduce_net():
    bars = flat_bars(30)
    bars.iloc[17, 1] = 110.0
    gross = run_backtest(simple_spec(), {"X": bars}, MoomooStockCosts(**ZERO_COST)).trades.iloc[0]
    net = run_backtest(simple_spec(), {"X": bars}, MoomooStockCosts()).trades.iloc[0]
    assert gross.exit_reason == net.exit_reason == "target"
    assert gross.r_multiple == pytest.approx(2.0, rel=0.02)
    assert net.net_pnl < gross.net_pnl
    assert net.fees >= 2 * 0.99


def test_time_limit_closes_position():
    bars = flat_bars(60, rng=1.0)
    res = run_backtest(simple_spec(max_bars=5), {"X": bars}, MoomooStockCosts(**ZERO_COST))
    assert res.trades.bars_held.max() <= 5
    assert res.trades.exit_reason.str.startswith("review").any()


def test_forex_rollover_charged_on_long_carry_negative():
    idx = pd.date_range("2025-03-03", periods=24 * 5, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": 1.08, "high": 1.0802, "low": 1.0798, "close": 1.08, "volume": 100.0}, index=idx)
    # long EUR/USD in 2025: EUR ~2.75% vs USD ~4.375% => negative carry, paid every 17:00 NY
    spec = simple_spec("forex", tf="H1", stop_atr=50, target_r=50, max_bars=100)
    from tradex.positions.review import ReviewPolicy
    cfg = EngineConfig(review=ReviewPolicy(rollover_check=False, stale_after_frac=2.0, breakeven_r=None))
    res = run_backtest(spec, {"EUR_USD": bars}, OandaFxCosts(), cfg)
    assert res.trades.financing.sum() > 0


def test_rollover_rule_exits_flat_negative_carry_trade_before_cut():
    idx = pd.date_range("2025-03-03", periods=24 * 5, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": 1.08, "high": 1.0802, "low": 1.0798, "close": 1.08, "volume": 100.0}, index=idx)
    spec = simple_spec("forex", tf="H1", stop_atr=50, target_r=50, max_bars=100)
    res = run_backtest(spec, {"EUR_USD": bars}, OandaFxCosts())
    assert res.trades.exit_reason.str.contains("rollover").any()
    assert res.trades.financing.sum() == pytest.approx(0.0)


def test_day_trade_cap_when_enabled():
    idx = pd.date_range("2026-01-05 14:30", periods=7 * 30, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    bars = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e5}, index=idx)
    spec = simple_spec(tf="H1", stop_atr=20, target_r=0.02, max_bars=50)
    cfg = EngineConfig(pdt=True)
    res = run_backtest(spec, {"X": bars}, MoomooStockCosts(**ZERO_COST), cfg)
    t = res.trades
    same_day = t[t.entry_time.dt.tz_convert("America/New_York").dt.date == t.exit_time.dt.tz_convert("America/New_York").dt.date]
    days = same_day.exit_time.sort_values()
    for d in days:
        window = days[(days > d - pd.tseries.offsets.BDay(5)) & (days <= d)]
        assert len(window) <= 3
    off = run_backtest(spec, {"X": bars}, MoomooStockCosts(**ZERO_COST), EngineConfig(pdt=False))
    assert len(off.trades) > len(res.trades)


def test_heat_and_position_limits(stock_data):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    res = run_backtest(spec, stock_data, cfg=EngineConfig(max_positions=1))
    t = res.trades.sort_values("entry_time")
    for a, b in zip(t.itertuples(), t.iloc[1:].itertuples()):
        assert b.entry_time >= a.exit_time or b.symbol == a.symbol


def test_metrics_summary_fields(stock_data):
    spec = StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")
    res = run_backtest(spec, stock_data)
    m = metrics.summarize(res.equity, res.trades)
    assert m["trades"] == len(res.trades) > 0
    assert m["costs_usd"] > 0
    assert -1 < m["max_drawdown"] <= 0
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.trades.net_pnl.sum(), rel=1e-6)
