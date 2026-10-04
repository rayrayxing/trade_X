import pandas as pd
import pytest

from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.costs.measured import MeasuredFxCosts, MeasuredSpreads, MissingSpread, spreads_from_ba
from tradex.costs.models import OandaFxCosts
from tradex.positions.review import ReviewPolicy
from tests.conftest import simple_spec


def ba_candles(idx, mid=1.08, spread=0.00012):
    df = pd.DataFrame({"open": mid, "high": mid + 0.0002, "low": mid - 0.0002, "close": mid,
                       "volume": 100.0}, index=idx)
    for k in ("open", "high", "low", "close"):
        df[f"bid_{k}"] = df[k] - spread / 2
        df[f"ask_{k}"] = df[k] + spread / 2
    return df


IDX = pd.date_range("2025-03-03", periods=24 * 5, freq="h", tz="UTC")


def test_spread_is_the_latest_measured_one_at_or_before_the_fill():
    ba = ba_candles(IDX)
    ba.loc[IDX[10]:, ["bid_open"]] -= 0.0001          # spread widens from bar 10
    m = MeasuredSpreads.from_ba_candles({"EUR_USD": ba})
    assert m.spread_at("EUR_USD", IDX[5]) == pytest.approx(0.00012)
    assert m.spread_at("EUR_USD", IDX[12] + pd.Timedelta(minutes=30)) == pytest.approx(0.00022)
    assert m.spread_pips("EUR_USD", IDX[5]) == pytest.approx(1.2)


def test_no_fallback_to_typical_spreads():
    m = MeasuredSpreads.from_ba_candles({"EUR_USD": ba_candles(IDX)})
    with pytest.raises(MissingSpread):
        m.spread_at("GBP_USD", IDX[5])                    # pair never measured
    with pytest.raises(MissingSpread):
        m.spread_at("EUR_USD", IDX[0] - pd.Timedelta(hours=1))   # before the first candle
    with pytest.raises(MissingSpread):
        m.spread_at("EUR_USD", IDX[-1] + pd.Timedelta(days=5))   # stale
    with pytest.raises(ValueError):
        MeasuredFxCosts()                                 # measured spreads are required


def test_jpy_pips_and_fill_price():
    ba = ba_candles(IDX, mid=150.0, spread=0.014)
    c = MeasuredFxCosts.from_ba_candles({"USD_JPY": ba}, slippage_pips=0.0)
    assert c.measured.spread_pips("USD_JPY", IDX[3]) == pytest.approx(1.4)
    px, unit = c.fill("USD_JPY", +1, 150.0, IDX[3])
    assert px == pytest.approx(150.007)
    assert unit["spread"] == pytest.approx(0.007)


def test_spreads_from_ba_requires_bid_ask_columns():
    with pytest.raises(ValueError):
        spreads_from_ba(pd.DataFrame({"open": [1.0]}, index=IDX[:1]))


def test_backtest_pays_the_measured_spread():
    ba = ba_candles(IDX, spread=0.0004)                   # 4 pips, wider than the 1.4 typical
    bars = ba[["open", "high", "low", "close", "volume"]]
    spec = simple_spec("forex", tf="H1", stop_atr=50, target_r=50, max_bars=20)
    cfg = EngineConfig(review=ReviewPolicy(rollover_check=False, stale_after_frac=2.0, breakeven_r=None))
    typical = run_backtest(spec, {"EUR_USD": bars}, OandaFxCosts(slippage_pips=0.0), cfg).trades
    measured = run_backtest(spec, {"EUR_USD": bars},
                            MeasuredFxCosts.from_ba_candles({"EUR_USD": ba}, slippage_pips=0.0), cfg).trades
    assert len(measured) == len(typical) > 0
    per_unit = lambda t: (t.spread_slippage / t.qty).iloc[0]  # noqa: E731
    assert per_unit(measured) == pytest.approx(0.0004, rel=1e-6)   # half on entry, half on exit
    assert per_unit(typical) == pytest.approx(0.00014, rel=1e-6)


def test_backtest_without_measured_spreads_for_a_bar_fails_loudly():
    ba = ba_candles(IDX[:24])                             # only the first day measured
    bars = ba_candles(IDX)[["open", "high", "low", "close", "volume"]]
    spec = simple_spec("forex", tf="H1", stop_atr=50, target_r=50, max_bars=200)
    costs = MeasuredFxCosts.from_ba_candles({"EUR_USD": ba}, max_age=pd.Timedelta(hours=2))
    cfg = EngineConfig(review=ReviewPolicy(rollover_check=False, stale_after_frac=2.0, breakeven_r=None))
    with pytest.raises(MissingSpread):
        run_backtest(spec, {"EUR_USD": bars}, costs, cfg)
