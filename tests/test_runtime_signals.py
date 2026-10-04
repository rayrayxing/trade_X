"""Incremental signals: a rolling window of >= 3 lookbacks gives the full-history signals."""
import numpy as np
import pandas as pd
import pytest

from tradex.data.synthetic import synthetic_bars
from tradex.runtime.market import BarStore
from tradex.runtime.signals import SignalCache, feature_lookbacks, strategy_lookback, window_bars
from tradex.strategy.spec import StrategySpec, compute_signals, load_dir
from tradex.timeframes import duration

SEEDS = {s.id: s for s in load_dir("strategies/seeds")}
LENGTH = {"D1": 4_000, "H1": 6_000, "H4": 10_000}


def _series(tf: str, seed: int) -> pd.DataFrame:
    if tf == "D1":
        return synthetic_bars(LENGTH[tf], tf="D1", seed=seed, start="2000-01-03")
    return synthetic_bars(LENGTH[tf], tf=tf, seed=seed, price=1.1, vol=0.002, trend_strength=0.0003,
                          regime_len=300, start="2020-01-01", business_days=False)


@pytest.mark.parametrize("sid", sorted(SEEDS))
def test_rolling_window_signals_equal_full_history_on_seeds(sid):
    spec = SEEDS[sid]
    tf = spec.signal_tf
    df = _series(tf, seed=len(sid))
    full = compute_signals(spec, df)
    cache = SignalCache(BarStore(tf, {"X": df}))
    n = cache.window(spec)
    assert n is None or n >= 3 * strategy_lookback(spec)
    tail = range(len(df) - 2_000, len(df))
    fired = [i for i in tail if full.long_entry.iloc[i] or full.short_entry.iloc[i]]
    for i in sorted(set(fired) | set(tail[::23])):
        close_t = df.index[i] + duration(tf)
        pt = cache.at(spec, "X", close_t)
        assert (pt.long_entry, pt.short_entry) == (bool(full.long_entry.iloc[i]), bool(full.short_entry.iloc[i])), i
        assert (pt.long_exit, pt.short_exit) == (bool(full.long_exit.iloc[i]), bool(full.short_exit.iloc[i])), i
        assert pt.atr == pytest.approx(float(full.atr.iloc[i]), rel=1e-9)
        if n is not None and i % 5 == 0:                    # the features themselves, not only the booleans
            win = compute_signals(spec, df.iloc[max(0, i + 1 - n):i + 1])
            for k, v in full.features.items():
                a, b = float(win.features[k].iloc[-1]), float(v.iloc[i])
                assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), (k, i)


def test_window_is_shorter_than_history_for_most_seeds():
    """Otherwise the parity test above would prove nothing."""
    short = [s for s in SEEDS.values() if (window_bars(s) or 10**9) < LENGTH[s.signal_tf]]
    assert len(short) >= 5


def test_lookback_counts_recursive_indicators_and_higher_timeframes():
    spec = SEEDS["fx-trend-pullback-engulfing"]
    lb = feature_lookbacks(spec)
    assert lb["ema_fast"] == 5 * 20                       # EMA: 5 periods
    assert lb["ema_slow"] >= 5 * 50 * 24                  # D1 EMA seen from H1 bars
    assert lb["pullback"] == lb["ema_fast"] + 10 * 14     # zone touch adds Wilder ATR to its reference
    assert feature_lookbacks(SEEDS["stk-double-bottom-breakout"])["dbl"] == float("inf")
    assert window_bars(SEEDS["stk-double-bottom-breakout"]) is None


def test_no_signal_when_the_symbol_has_no_bar_at_that_close():
    spec = SEEDS["stk-rsi2-meanrev"]
    df = _series("D1", 1)
    cache = SignalCache(BarStore("D1", {"X": df}))
    assert cache.at(spec, "X", df.index[100] + pd.Timedelta(hours=5)) is None
