"""How solid is a near miss? Report-only checks on a walk-forward result, never part of the gate.

A strategy that clears every stage-3 rung except the deflated Sharpe is not tweaked until it
passes; instead these numbers show how much of its result survives scrutiny:

- per fold: each test window's return, Sharpe and trade count with the parameters chosen on
  its training window (already in the walk-forward report);
- a subperiod: the same out-of-sample returns and trades from a start date on (no rerun);
- doubled spread and slippage: each fold's test window rerun with the parameters that fold
  chose, under the stressed cost model. No parameter is selected, so it is not a new trial
  and the trial ledger is not touched;
- a bootstrap confidence interval of the annualised out-of-sample Sharpe (stationary
  bootstrap, Politis and Romano 1994, so runs of correlated days stay together).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig, run_backtest


def _r(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else (x if np.isinf(x) else round(x, nd))


def stats(returns: pd.Series, trades: pd.DataFrame | None) -> dict:
    """Trades, profit factor, annualised Sharpe, total return and max drawdown of a daily-return series."""
    r = returns.dropna()
    eq = (1 + r).cumprod()
    t = metrics.trade_stats(trades if trades is not None else pd.DataFrame())
    curve = pd.Series(np.r_[1.0, eq.to_numpy()]) if len(eq) else pd.Series([1.0])
    return {"trades": int(t.get("trades", 0)), "profit_factor": _r(t.get("profit_factor")),
            "sharpe": _r(metrics.sharpe(r)) if len(r) > 2 else None,
            "total_return": _r(curve.iloc[-1] - 1), "max_drawdown": _r(metrics.max_drawdown(curve))}


def subperiod(returns: pd.Series, trades: pd.DataFrame | None, start: str) -> dict:
    """Out-of-sample results from ``start`` on: returns dated on or after it, trades entered on or after it."""
    t0 = pd.Timestamp(start, tz="UTC")
    r = returns[returns.index >= t0]
    tr = trades[pd.to_datetime(trades["entry_time"], utc=True) >= t0] if trades is not None and len(trades) else trades
    return {"from": start} | stats(r, tr)


def stationary_bootstrap_sharpe(returns: pd.Series, n: int = 5000, mean_block: float = 10.0, seed: int = 0,
                                levels=(0.05, 0.5, 0.95)) -> dict:
    """Percentiles of the annualised Sharpe over stationary-bootstrap resamples of the daily returns,
    and the share of resamples at or below zero."""
    x = returns.dropna().to_numpy(float)
    m = len(x)
    if m < 20:
        return {"n": n, "mean_block_days": mean_block, "percentiles": {}, "p_sharpe_le_0": None}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    steps = np.arange(m)
    for k in range(n):
        start = rng.integers(m, size=m)
        new = rng.random(m) < 1.0 / mean_block
        new[0] = True
        # each position continues the block begun at the last "new" position: start there, step forward
        last = np.maximum.accumulate(np.where(new, steps, 0))
        idx = (start[last] + steps - last) % m
        s = x[idx]
        sd = s.std(ddof=1)
        out[k] = s.mean() / sd * np.sqrt(metrics.PERIODS_PER_YEAR) if sd > 0 else 0.0
    return {"n": n, "mean_block_days": mean_block,
            "percentiles": {f"p{int(round(q * 100))}": _r(np.quantile(out, q)) for q in levels},
            "p_sharpe_le_0": _r(float(np.mean(out <= 0)))}


def rerun_folds(spec, data, folds, costs, cfg: EngineConfig, tradable=None, filter_ctx=None) -> dict:
    """Each fold's test window again, with that fold's chosen parameters, under ``costs``. No selection."""
    parts, trades, per = [], [], []
    for f in folds:
        run_cfg = EngineConfig(**{**cfg.__dict__, "start": pd.Timestamp(f.test_start), "end": pd.Timestamp(f.test_end)})
        res = run_backtest(spec.with_params(f.best_params), data, costs, run_cfg, tradable, filter_ctx)
        parts.append(res.daily_returns)
        if len(res.trades):
            trades.append(res.trades)
        per.append(float(res.equity.iloc[-1] / res.equity.iloc[0] - 1) if len(res.equity) > 1 else 0.0)
    r = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    r = r[~r.index.duplicated()]
    out = stats(r, pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())
    return out | {"folds_profitable": f"{sum(x > 0 for x in per)}/{len(per)}", "fold_returns": [_r(x) for x in per]}
