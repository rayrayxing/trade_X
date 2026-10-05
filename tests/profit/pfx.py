"""Hand-built bars and plans for the profit-module tests (test data only)."""
import numpy as np
import pandas as pd

from tradex.core.records import TradePlan

T0 = pd.Timestamp("2026-03-02 00:00", tz="UTC")
H = pd.Timedelta(hours=1)


def idx(n, start=T0, step=H):
    return pd.DatetimeIndex([start + i * step for i in range(n)]).astype("datetime64[ns, UTC]")


def history(n=30, price=100.0, rng=1.0, start=T0):
    """Quiet bars with a constant true range of ``rng``: ATR(14) is exactly ``rng``."""
    return pd.DataFrame({"open": price, "high": price + rng / 2, "low": price - rng / 2, "close": price,
                         "volume": 1.0}, index=idx(n, start)).astype(float)


def bars_from(rows, start):
    """rows: (open, high, low, close) tuples, one bar each from ``start``."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx(len(rows), start))
    df["volume"] = 1.0
    return df.astype(float)


def with_history(rows, n_hist=30, price=100.0, rng=1.0):
    h = history(n_hist, price, rng)
    entry_t = h.index[-1] + H
    return pd.concat([h, bars_from(rows, entry_t)]), entry_t


def mirror(bars):
    m = bars.copy()
    m["open"], m["close"] = -bars["open"], -bars["close"]
    m["high"], m["low"] = -bars["low"], -bars["high"]
    return m


def plan(direction=1, entry=100.0, stop=98.0, targets=(104.0, 108.0), max_bars=10, cost_r=0.0, t=None,
         decision_id="2026-03-02-0001", symbol="EUR_USD", tf="H1", book="ensemble", **kw):
    return TradePlan(decision_id=decision_id, time=(t or T0).isoformat(), symbol=symbol, asset_class="forex",
                     direction=direction, entry_type="market", entry_price=entry, stop=stop, targets=list(targets),
                     max_bars=max_bars, invalidation="", families=["trend", "breakout"], strategies=["a", "b"],
                     score=0.5, p_target=kw.pop("p_target", 0.4), p_source="base_rate",
                     reward_risk=kw.pop("reward_risk", 2.0), ev_r=kw.pop("ev_r", 0.2), cost_r=cost_r, book=book,
                     tf=tf, **kw)


def random_walk_bars(n, seed, price=100.0, vol=0.8, start=T0):
    rng = np.random.default_rng(seed)
    c = price + np.cumsum(rng.normal(0, vol, n))
    o = np.concatenate([[price], c[:-1]]) + rng.normal(0, vol * 0.2, n)
    hi = np.maximum(o, c) + np.abs(rng.normal(0, vol * 0.5, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, vol * 0.5, n))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1.0}, index=idx(n, start))
