"""Research data gaps: liquidity screen, session-close exit, intraday/earnings columns,
trial variants, measured FX spreads, and the single Oanda candle downloader."""
import numpy as np
import pandas as pd
import pytest

from tradex.backtest.engine import EngineConfig, run_backtest, session_end_mask
from tradex.backtest.validation import WalkForwardConfig, walk_forward
from tradex.costs.models import MoomooStockCosts, OandaFxCosts
from tradex.data.synthetic import synthetic_bars
from tradex.research import panels, screen as scr
from tradex.research.trials import TrialLedger
from tradex.strategy.spec import StrategySpec

from conftest import flat_bars, simple_spec

NY = "America/New_York"
ZERO_COST = dict(platform_fee=0.0, settlement_fee_per_share=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0,
                 default_half_spread_bps=0.0, slippage_bps=0.0)
HOURS = ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:00")   # OpenD 60-minute stamps (open time)


def h1_bars(days, price=100.0):
    ny = pd.DatetimeIndex([f"{d} {h}" for d in days for h in HOURS], tz=NY)
    idx = ny.tz_convert("UTC").astype("datetime64[ns, UTC]")
    c = price + np.arange(len(idx), dtype=float) * 0.1
    return pd.DataFrame({"open": c - 0.05, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)


# --- liquidity screen ----------------------------------------------------------------------

def _dv_panel():
    idx = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    mk = lambda vol: pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": vol}, index=idx)
    a, b, c = mk(100.0), mk(50.0), mk(10.0)
    b.loc[b.index >= idx[120], "volume"] = 1000.0     # B becomes the most liquid from bar 120
    return {"A": a, "B": b, "C": c.iloc[:90]}         # C stops trading after bar 90


def test_screen_ranks_with_past_bars_only_and_drops_stale_names():
    data = _dv_panel()
    idx = data["A"].index
    assert scr.rank_at(data, idx[80], 2) == ["A", "B"]
    assert scr.rank_at(data, idx[120], 2, lookback=60) == ["A", "B"]          # bar 120 itself is not used
    assert scr.rank_at(data, idx[199], 1, lookback=60) == ["B"]
    assert scr.rank_at(data, idx[40], 3, lookback=60) == []                    # no full lookback yet
    assert "C" not in scr.rank_at(data, idx[150], 3)                           # stale
    assert scr.rank_at(data, idx[85], 3, lookback=60)[-1] == "C"


def test_screen_masks_follow_rebalance_dates():
    data = _dv_panel()
    idx = data["A"].index
    members = {idx[70]: ["A"], idx[150]: ["B"]}
    m = scr.member_frame(members, ["A", "B"], idx)
    assert not m.iloc[69].any() and m.iloc[70]["A"] and m.iloc[149]["A"] and m.iloc[150]["B"] and not m.iloc[150]["A"]
    t = scr.tradable_masks(members, ["A", "B", "C"])
    assert t["A"].tolist() == [True, False] and t["C"].tolist() == [False, False]
    dates = scr.rebalance_dates(idx, WalkForwardConfig(n_folds=2), pd.Timedelta(days=5), lookback=20, every="30D")
    assert dates == sorted(dates) and dates[0] == idx[20] and idx[80] in dates   # first test fold at 40%


def test_tradable_mask_blocks_entries_outside_the_universe():
    bars = flat_bars(60)
    spec = simple_spec(long="close > 0")
    allowed = pd.Series([False, True], index=[bars.index[0], bars.index[40]])
    res = run_backtest(spec, {"X": bars}, MoomooStockCosts(**ZERO_COST), tradable={"X": allowed})
    assert res.trades.entry_time.min() > bars.index[40]


# --- session close ---------------------------------------------------------------------------

def test_session_end_mask_is_dst_aware_and_catches_early_closes():
    b = h1_bars(["2026-01-15", "2026-09-29"])                 # EST and EDT
    m = session_end_mask(b.index, pd.Timedelta(hours=1))
    assert m.tolist() == [False] * 6 + [True] + [False] * 6 + [True]
    early = pd.DatetimeIndex(["2026-11-27 11:30", "2026-11-27 12:00", "2026-11-30 09:30"], tz=NY).tz_convert("UTC")
    assert session_end_mask(early, pd.Timedelta(hours=1)).tolist() == [False, True, True]


def _intraday_spec(**exit_kw):
    d = {"id": "intra", "version": 1, "asset_class": "stocks", "universe": ["X"], "timeframes": {"signal": "H1"},
         "features": {"lf": {"fn": "data.column", "name": "last_full"}}, "entry": {"long": "lf > 0"},
         "exit": {"stop_atr": 50.0, "target_r": 50.0, "max_bars": 10} | exit_kw, "holding": {"expected_hours": 0.5}}
    return StrategySpec.from_dict(d)


def test_session_close_exit_flattens_at_1600_new_york():
    days = [f"2026-03-{d:02d}" for d in range(2, 21) if pd.Timestamp(f"2026-03-{d:02d}").weekday() < 5]
    b = h1_bars(days)
    b = panels.with_columns(b, panels.session_columns(b))
    res = run_backtest(_intraday_spec(session_close=True), {"X": b}, MoomooStockCosts(**ZERO_COST))
    t = res.trades
    assert len(t) and (t.exit_reason == "session_close").all()
    entry_ny, exit_ny = t.entry_time.dt.tz_convert(NY), t.exit_time.dt.tz_convert(NY)
    assert (entry_ny.dt.strftime("%H:%M") == "15:00").all()          # the 15:30-16:00 bar's open
    assert (exit_ny.dt.strftime("%H:%M") == "16:00").all() and (entry_ny.dt.date == exit_ny.dt.date).all()
    closes = b["close"].reindex(t.entry_time).to_numpy()
    assert t.exit_price.to_numpy() == pytest.approx(closes)
    held = run_backtest(_intraday_spec(), {"X": b}, MoomooStockCosts(**ZERO_COST)).trades
    assert (held.exit_reason != "session_close").all()


# --- research columns ----------------------------------------------------------------------

def test_first_hour_return_is_known_from_the_first_bar_on():
    b = h1_bars(["2026-09-29", "2026-09-30"])
    c = panels.session_columns(b)
    first_close, prev_close = b["close"].iloc[7], b["close"].iloc[6]
    assert c["fh_ret"].iloc[7:].tolist() == pytest.approx([first_close / prev_close - 1] * 7)
    assert (c["fh_ret"].iloc[:7] == 0).all()                         # no previous session in the data
    part = panels.session_columns(b.iloc[:9])
    pd.testing.assert_frame_equal(part, c.iloc[:9])


def test_earnings_columns_mark_the_reaction_day():
    idx = pd.DatetimeIndex(pd.date_range("2026-07-27", periods=6, freq="B"), tz=NY).tz_convert("UTC")
    d1 = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": [100, 100, 100, 100, 110, 111.0],
                       "volume": 1.0}, index=idx)
    ev = pd.DataFrame({"date": ["2026-07-30", "2026-07-28", "2026-07-27"], "timing": ["after", "unknown", "before"],
                       "source": "opend"})
    c = panels.earnings_columns(d1, ev)
    assert c["earn_day"].tolist() == [1, 0, 0, 0, 1, 0]               # unknown timing skipped
    assert c["earn_jump"].iloc[4] == pytest.approx(0.10)


# --- trials and costs ------------------------------------------------------------------------

def test_variant_adds_trials_for_the_same_parameters(tmp_path):
    spec = simple_spec(long="close > open")
    spec = StrategySpec.from_dict({**spec.raw, "search_space": {"stop_atr": [1.0, 2.0]}})
    data = {"X": synthetic_bars(600, seed=3)}
    led = TrialLedger(tmp_path / "t.sqlite")
    wf = WalkForwardConfig(n_folds=2, grid_points=2)
    assert walk_forward(spec, data, wf=wf, trials=led).n_trials == 2
    assert walk_forward(spec, data, wf=wf, trials=led).n_trials == 2
    assert walk_forward(spec, data, wf=wf, trials=led, variant="top30").n_trials == 4


def test_fx_costs_use_the_measured_spread_at_the_fill():
    ts = pd.DatetimeIndex(["2026-01-05 00:00", "2026-01-05 01:00"], tz="UTC")
    c = OandaFxCosts(measured_spreads={"EUR_USD": pd.Series([0.6, 3.0], index=ts)}, slippage_pips=0.0)
    fill, unit = c.fill("EUR_USD", 1, 1.1, ts[0] + pd.Timedelta(minutes=30))
    assert unit["spread"] == pytest.approx(0.3 * 0.0001)
    assert c.fill("EUR_USD", 1, 1.1, ts[1])[1]["spread"] == pytest.approx(1.5 * 0.0001)
    assert c.fill("GBP_USD", 1, 1.3, ts[1])[1]["spread"] == pytest.approx(1.0 * 0.0001)   # default table


def test_one_oanda_candle_downloader():
    from tradex.data import oanda, oanda_history
    assert oanda_history.HostRefused is oanda.LiveHostRefused
    assert oanda_history.PRACTICE_HOST == oanda.REST_HOST
    assert "OandaHistory" in oanda.fetch_ba_candles.__code__.co_names
