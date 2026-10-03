"""Confidence-based sizing, ruin simulation and the leverage gate (design doc, Risk self-assessment)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradex.backtest.metrics import wilson_lower


def kelly_fraction(p: float, b: float) -> float:
    """Kelly fraction f* = p - (1 - p) / b. Zero or negative means do not trade."""
    if b <= 0:
        return 0.0
    return p - (1 - p) / b


def conservative_kelly(trades: pd.DataFrame, fraction: float = 0.25) -> dict:
    """Fractional Kelly from the LOWER bound of the win rate, so thin evidence gives small size."""
    if trades is None or trades.empty:
        return {"p": 0.0, "p_lower": 0.0, "b": 0.0, "kelly": 0.0, "risk_pct": 0.0}
    r = trades["r_multiple"].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    p_lower = wilson_lower(len(wins), len(r))
    b = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else 0.0
    f = kelly_fraction(p_lower, b)
    return {"p": len(wins) / len(r), "p_lower": p_lower, "b": b, "kelly": f,
            "risk_pct": max(0.0, f * fraction * 100)}


def drawdown_scale(equity: float, peak: float, halve_at: float = 0.20) -> float:
    """Size multiplier that shrinks linearly with drawdown: 1.0 at peak, 0.5 at ``halve_at``."""
    dd = max(0.0, 1 - equity / peak) if peak > 0 else 0.0
    return max(0.0, 1 - dd * 0.5 / halve_at)


@dataclass
class StressCase:
    loss_multiplier: float = 1.3   # gaps past the stop make losses bigger than 1R
    extra_cost_r: float = 0.05     # spreads three times wider, expressed in R per trade
    worse_fill_r: float = 0.10     # fills one ATR worse on a 1.5 ATR stop is ~0.67R; applied to a share of trades
    worse_fill_share: float = 0.15


def ruin_probability(
    r_multiples: np.ndarray,
    risk_per_trade: float,
    leverage: float = 1.0,
    horizon: int = 200,
    paths: int = 10_000,
    block: int = 5,
    ruin_drawdown: float = 0.5,
    stress: StressCase | None = None,
    seed: int = 0,
) -> float:
    """Share of block-bootstrapped paths whose equity falls below (1 - ruin_drawdown) x its running peak.

    ``risk_per_trade`` is the equity fraction lost at -1R before leverage.
    Blocks keep losing streaks together.
    """
    r = np.asarray(r_multiples, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < block:
        return 1.0
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(r) - block + 1, size=(paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)[:, :horizon]
    sims = r[idx]
    if stress:
        sims = np.where(sims < 0, sims * stress.loss_multiplier, sims) - stress.extra_cost_r
        bad = rng.random(sims.shape) < stress.worse_fill_share
        sims = sims - bad * stress.worse_fill_r
    step = np.clip(1 + sims * risk_per_trade * leverage, 0, None)
    eq = np.cumprod(step, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), eq], axis=1), axis=1)[:, 1:]
    ruined = (eq <= peak * (1 - ruin_drawdown)).any(axis=1)
    return float(ruined.mean())


def leverage_gate(ruin_p_stressed: float, evidence: str, broker_max: float) -> float:
    """Maximum leverage allowed (design doc table). ``evidence``: none | positive | validated | arbitrage."""
    if ruin_p_stressed > 0.10:
        return 0.0          # size down or skip
    if ruin_p_stressed > 0.02:
        return 1.0 if evidence in ("positive", "validated", "arbitrage") else 0.0
    if ruin_p_stressed > 0.005:
        return 0.5 * broker_max if evidence in ("validated", "arbitrage") else 1.0
    if evidence == "arbitrage":
        return 0.7 * broker_max
    return 0.5 * broker_max if evidence == "validated" else 1.0
