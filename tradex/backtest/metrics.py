"""Performance statistics, including the probabilistic and deflated Sharpe ratios.

Deflated Sharpe follows Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (JPM 2014),
eqs. (1)-(2): the expected maximum Sharpe of N unskilled trials is
    SR0 = sqrt(V) * ((1 - g) * Zinv(1 - 1/N) + g * Zinv(1 - 1/(N e)))
with g the Euler-Mascheroni constant and V the variance of the trial Sharpes, and
    DSR = PSR(SR0) = Z((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2)),
kurt being raw (not excess) kurtosis. All Sharpe ratios inside PSR/DSR are per period
(daily), not annualised. Checked against the papers' worked examples in tests/test_dsr.py.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649
PERIODS_PER_YEAR = 252


def sharpe(returns: pd.Series, annualise: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    return float(sr * math.sqrt(PERIODS_PER_YEAR)) if annualise else float(sr)


def sortino(returns: pd.Series) -> float:
    r = returns.dropna()
    down = r[r < 0]
    if len(r) < 2 or len(down) < 2 or down.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / down.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def psr_from_moments(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """PSR from summary statistics (Bailey and Lopez de Prado 2012, eq. 11). ``kurt`` is raw kurtosis."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if t < 2 or denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


def min_track_record_length(sr: float, sr_benchmark: float, skew: float, kurt: float, prob: float = 0.95) -> float:
    """Observations needed for PSR(sr_benchmark) to reach ``prob`` (2012 paper, eq. 13)."""
    if sr <= sr_benchmark:
        return math.inf
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr * sr) * (_N.inv_cdf(prob) / (sr - sr_benchmark)) ** 2


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Probability the true per-period Sharpe exceeds ``sr_benchmark`` (per-period units)."""
    r = returns.dropna()
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas gives excess kurtosis
    return psr_from_moments(float(sr), sr_benchmark, t, skew, kurt)


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected best per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1 or trial_sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(trial_sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_variance: float) -> float:
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, trial_sr_variance))


def dsr_from_moments(sr: float, t: int, skew: float, kurt: float, n_trials: int, trial_sr_variance: float) -> float:
    return psr_from_moments(sr, expected_max_sharpe(n_trials, trial_sr_variance), t, skew, kurt)


def wilson_lower(wins: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% lower bound on a win rate."""
    if n == 0:
        return 0.0
    p = wins / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - adj) / den)


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": 0.0, "win_rate_lower": 0.0, "profit_factor": 0.0,
                "avg_win_loss": 0.0, "expectancy_r": 0.0, "avg_bars_held": 0.0,
                "costs_usd": 0.0, "costs_share_of_gross": 0.0, "pdt_violations": 0}
    net = trades["net_pnl"]
    wins, losses = net[net > 0], net[net <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    cost_cols = ["spread_slippage", "fees", "financing", "borrow", "margin_interest"]
    costs = float(trades[cost_cols].sum().sum())
    gross_abs = float(trades["gross_pnl"].abs().sum() + trades["spread_slippage"].sum())
    return {
        "trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "win_rate_lower": wilson_lower(len(wins), len(trades)),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "avg_win_loss": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else 0.0,
        "expectancy_r": float(trades["r_multiple"].mean()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "costs_usd": costs,
        "costs_share_of_gross": costs / gross_abs if gross_abs else 0.0,
        "cost_breakdown": {c: float(trades[c].sum()) for c in cost_cols},
        "exit_reasons": trades["exit_reason"].str.split(":").str[0].value_counts().to_dict(),
        "pdt_violations": int(trades.get("pdt_violation", pd.Series(dtype=bool)).sum()),
    }


def summarize(equity: pd.Series, trades: pd.DataFrame, daily_returns: pd.Series | None = None) -> dict:
    if daily_returns is None:
        daily_returns = equity.resample("1D").last().dropna().pct_change().dropna() if len(equity) else pd.Series(dtype=float)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9) if len(equity) > 1 else 0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    out = {
        "start": str(equity.index[0]) if len(equity) else None,
        "end": str(equity.index[-1]) if len(equity) else None,
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if years and total > -1 else -1.0,
        "sharpe": sharpe(daily_returns),
        "sortino": sortino(daily_returns),
        "max_drawdown": max_drawdown(equity),
        "psr": probabilistic_sharpe(daily_returns),
    }
    out.update(trade_stats(trades))
    return out
