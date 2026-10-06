"""Strategy health: a lower CUSUM on paper returns, and a drawdown check, against the walk-forward baseline.

The baseline is what the strategy earned out of sample when it passed the gate: mean and standard
deviation of daily returns, and the worst drawdown. Walk-forward picks the best parameters on each
training window, so its mean is optimistic; the reference mean is shrunk (``mean_shrink``) before
paper returns are compared with it.

    z_t = (r_t - mu0) / sigma0
    S_t = min(0, S_{t-1} + z_t + k)         alarm when S_t <= -h

A run of returns that average ``k`` standard deviations below the reference drifts S down to the alarm;
ordinary noise keeps it near zero. These functions are pure: arrays in, numbers out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Baseline:
    mean: float
    std: float
    n: int
    max_drawdown: float          # negative or zero, e.g. -0.12


@dataclass(frozen=True)
class Health:
    n: int
    cusum_min: float
    cusum_last: float
    drawdown: float
    reference_mean: float
    sigma: float
    cusum_alarm: bool
    cusum_warn: bool
    drawdown_alarm: bool

    @property
    def alarm(self) -> bool:
        return self.cusum_alarm or self.drawdown_alarm


def baseline_from_returns(oos_returns: pd.Series) -> Baseline:
    r = pd.Series(oos_returns, dtype=float).dropna()
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    return Baseline(mean=float(r.mean()) if len(r) else 0.0, std=float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                    n=int(len(r)), max_drawdown=dd)


def lower_cusum(returns: pd.Series | np.ndarray, mu0: float, sigma0: float, k: float) -> np.ndarray:
    if sigma0 <= 0:
        raise ValueError("baseline standard deviation must be positive")
    z = (np.asarray(returns, dtype=float) - mu0) / sigma0
    s, out = 0.0, np.empty(len(z))
    for i, zi in enumerate(z):
        s = min(0.0, s + zi + k)
        out[i] = s
    return out


def drawdown(returns: pd.Series | np.ndarray) -> float:
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0


def assess(returns: pd.Series, base: Baseline, *, k: float, h: float, warn_frac: float, mean_shrink: float,
           dd_multiple: float) -> Health:
    mu0 = base.mean * mean_shrink
    s = lower_cusum(returns, mu0, base.std, k)
    dd = drawdown(returns)
    return Health(
        n=int(len(returns)), cusum_min=float(s.min()) if len(s) else 0.0, cusum_last=float(s[-1]) if len(s) else 0.0,
        drawdown=dd, reference_mean=mu0, sigma=base.std,
        cusum_alarm=bool(len(s) and s.min() <= -h),
        cusum_warn=bool(len(s) and s.min() <= -h * warn_frac),
        drawdown_alarm=bool(base.max_drawdown < 0 and dd < base.max_drawdown * dd_multiple),
    )
