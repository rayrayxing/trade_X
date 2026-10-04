"""Cross-asset regime features from daily bars of several instruments.

These are the simple, causal stand-ins for the catalog's regime entries (hidden-Markov
allocation, realised-covariance regime detection, regime-switching volatility): realised
volatility percentile, average pairwise correlation and the absorption ratio of a basket,
a risk-on score across equities, bonds and gold, and the momentum-crash state. No model is
fitted; every window trails, and percentiles are against the expanding history, so the
value at t uses only data up to t. The strategies read the result through ``data.column``.

Column names produced by ``regime_columns``:

  rg_vol_pct     expanding percentile (0..1) of the market's 21-day realised volatility
  rg_dd          market close over its 252-day high, minus 1 (0 at a high, -0.2 = 20% below)
  rg_corr        average pairwise correlation of the basket's 63-day returns
  rg_absorb      absorption ratio: variance share of the top eigenvectors of the basket's 250-day covariance
  rg_absorb_z    15-day mean of rg_absorb minus its 250-day mean, in 250-day standard deviations
  rg_score       risk-on score 0..4: equity trend, equity beats bonds, equity beats gold, calm volatility
  rg_crash       1 when the market is in a bear state (negative 504-day return) and volatility is high
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

REGIME_COLUMNS = ["rg_vol_pct", "rg_dd", "rg_corr", "rg_absorb", "rg_absorb_z", "rg_score", "rg_crash"]


def realized_vol(close: pd.Series, n: int = 21, periods: int = 252) -> pd.Series:
    """Annualised standard deviation of the last ``n`` daily log returns."""
    return np.log(close).diff().rolling(n).std() * np.sqrt(periods)


def expanding_percentile(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Share of earlier-or-equal values at or below today's, using only data up to today (0..1; NaN until warm)."""
    out = np.full(len(s), np.nan)
    seen: list[float] = []
    for i, v in enumerate(s.to_numpy(float)):
        if np.isnan(v):
            continue
        bisect.insort(seen, v)
        if len(seen) >= min_periods:
            out[i] = bisect.bisect_right(seen, v) / len(seen)
    return pd.Series(out, index=s.index)


def drawdown_from_high(close: pd.Series, n: int = 252) -> pd.Series:
    return close / close.rolling(n, min_periods=n).max() - 1


def avg_pairwise_corr(returns: pd.DataFrame, n: int = 63) -> pd.Series:
    """Mean off-diagonal correlation of the basket over the trailing ``n`` returns."""
    r = returns.to_numpy(float)
    k = r.shape[1]
    out = np.full(len(r), np.nan)
    iu = np.triu_indices(k, 1)
    for t in range(n - 1, len(r)):
        w = r[t - n + 1:t + 1]
        if np.isnan(w).any():
            continue
        c = np.corrcoef(w, rowvar=False)
        out[t] = np.nanmean(c[iu])
    return pd.Series(out, index=returns.index)


def absorption_ratio(returns: pd.DataFrame, n: int = 250, top: int | None = None) -> pd.Series:
    """Kritzman et al.'s absorption ratio: share of total variance carried by the top ``top`` eigenvectors
    (default a fifth of the basket, at least one) of the trailing ``n``-return covariance. High = assets move as one."""
    r = returns.to_numpy(float)
    k = r.shape[1]
    top = top or max(1, round(k / 5))
    out = np.full(len(r), np.nan)
    for t in range(n - 1, len(r)):
        w = r[t - n + 1:t + 1]
        if np.isnan(w).any():
            continue
        ev = np.linalg.eigvalsh(np.cov(w, rowvar=False))
        out[t] = ev[-top:].sum() / ev.sum()
    return pd.Series(out, index=returns.index)


def absorption_shift(ar: pd.Series, short: int = 15, long: int = 250) -> pd.Series:
    """Standardised shift in the absorption ratio: (short mean - long mean) / long standard deviation."""
    return (ar.rolling(short).mean() - ar.rolling(long).mean()) / ar.rolling(long).std()


def risk_on_score(equity: pd.Series, bond: pd.Series, gold: pd.Series | None = None, vol_pct: pd.Series | None = None,
                  trend_n: int = 200, rel_n: int = 63, calm_below: float = 0.7) -> pd.Series:
    """Cross-asset risk-on score, one point each for: equity above its ``trend_n``-day mean; equity's ``rel_n``-day
    return above the bond's; above gold's (skipped without gold); volatility percentile below ``calm_below``
    (skipped without ``vol_pct``). NaN until every available component is defined."""
    comps = [(equity > equity.rolling(trend_n).mean()).where(equity.rolling(trend_n).mean().notna()),
             (equity.pct_change(rel_n) > bond.reindex(equity.index).pct_change(rel_n))
             .where(equity.pct_change(rel_n).notna() & bond.reindex(equity.index).pct_change(rel_n).notna())]
    if gold is not None:
        g = gold.reindex(equity.index).pct_change(rel_n)
        comps.append((equity.pct_change(rel_n) > g).where(g.notna() & equity.pct_change(rel_n).notna()))
    if vol_pct is not None:
        comps.append((vol_pct < calm_below).where(vol_pct.notna()))
    frame = pd.concat([c.astype(float) for c in comps], axis=1)
    return frame.sum(axis=1, min_count=len(comps)) if len(comps) else pd.Series(np.nan, index=equity.index)


def momentum_crash_state(market_close: pd.Series, vol_pct: pd.Series, bear_n: int = 504, high_vol: float = 0.8) -> pd.Series:
    """Daniel and Moskowitz's momentum-crash condition: the market's ``bear_n``-day return is negative and volatility
    is in the top part of its history. 1.0 / 0.0, NaN until defined."""
    bear = market_close.pct_change(bear_n)
    flag = ((bear < 0) & (vol_pct > high_vol)).astype(float)
    return flag.where(bear.notna() & vol_pct.notna())


def regime_columns(closes: dict[str, pd.Series], market: str = "SPY", bond: str = "TLT", gold: str | None = "GLD",
                   basket: list[str] | None = None) -> pd.DataFrame:
    """The regime column set (module docstring) on the market's daily index, from close series by symbol.

    ``basket`` are the instruments whose joint behaviour defines the correlation and absorption
    columns (default: every series except bond and gold)."""
    px = pd.DataFrame(closes).sort_index()
    if market not in px or bond not in px:
        raise KeyError(f"regime columns need {market} and {bond} closes; have {sorted(px)}")
    basket = basket or [c for c in px if c not in (bond, gold)]
    mkt = px[market].dropna()
    vol_pct = expanding_percentile(realized_vol(mkt))
    rets = np.log(px[basket]).diff().dropna(how="any")
    ar = absorption_ratio(rets)
    out = pd.DataFrame(index=mkt.index)
    out["rg_vol_pct"] = vol_pct
    out["rg_dd"] = drawdown_from_high(mkt)
    out["rg_corr"] = avg_pairwise_corr(rets).reindex(mkt.index)
    out["rg_absorb"] = ar.reindex(mkt.index)
    out["rg_absorb_z"] = absorption_shift(ar).reindex(mkt.index)
    out["rg_score"] = risk_on_score(mkt, px[bond], px[gold] if gold in px else None, vol_pct)
    out["rg_crash"] = momentum_crash_state(mkt, vol_pct)
    return out[REGIME_COLUMNS]
