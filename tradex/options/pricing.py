"""Black-Scholes-Merton helpers (European, continuous dividend yield).

Live and replay data carry the provider's own Greeks and implied volatility; this module
exists for three jobs the providers cannot do: (1) find the underlying level at which an
option's premium hits a stop (the input to gap-stress sizing), (2) check put-call parity
and monotonicity in tests, (3) a sanity cross-check of a provider's Greeks. US equity
options are American, so for dividend payers early exercise adds value this ignores.
No third-party dependencies: the normal CDF comes from ``math.erf``.
"""
from __future__ import annotations

import math

from tradex.options.contract import Right

SQRT2 = math.sqrt(2.0)
MIN_T = 1e-9


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, t: float, iv: float, r: float, q: float) -> tuple[float, float]:
    vol_sqrt_t = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(spot: float, strike: float, t: float, iv: float, right: Right | str,
             r: float = 0.0, q: float = 0.0) -> float:
    """Option price per share. At or past expiry (or zero vol) it is the discounted intrinsic."""
    right = Right.parse(right)
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= MIN_T or iv <= 0:
        fwd = spot * math.exp(-q * max(t, 0.0)) - strike * math.exp(-r * max(t, 0.0))
        return max(fwd, 0.0) if right is Right.CALL else max(-fwd, 0.0)
    d1, d2 = _d1_d2(spot, strike, t, iv, r, q)
    df_r, df_q = math.exp(-r * t), math.exp(-q * t)
    if right is Right.CALL:
        return spot * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2)
    return strike * df_r * norm_cdf(-d2) - spot * df_q * norm_cdf(-d1)


def bs_greeks(spot: float, strike: float, t: float, iv: float, right: Right | str,
              r: float = 0.0, q: float = 0.0) -> dict[str, float]:
    """delta, gamma, vega (per 1.00 of vol), theta (per calendar day), rho (per 1.00 of rate)."""
    right = Right.parse(right)
    if t <= MIN_T or iv <= 0:
        itm = (spot > strike) if right is Right.CALL else (spot < strike)
        return {"delta": (1.0 if right is Right.CALL else -1.0) if itm else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(spot, strike, t, iv, r, q)
    df_r, df_q = math.exp(-r * t), math.exp(-q * t)
    pdf = norm_pdf(d1)
    gamma = df_q * pdf / (spot * iv * math.sqrt(t))
    vega = spot * df_q * pdf * math.sqrt(t)
    common = -spot * df_q * pdf * iv / (2.0 * math.sqrt(t))
    if right is Right.CALL:
        delta = df_q * norm_cdf(d1)
        theta = common - r * strike * df_r * norm_cdf(d2) + q * spot * df_q * norm_cdf(d1)
        rho = strike * t * df_r * norm_cdf(d2)
    else:
        delta = -df_q * norm_cdf(-d1)
        theta = common + r * strike * df_r * norm_cdf(-d2) - q * spot * df_q * norm_cdf(-d1)
        rho = -strike * t * df_r * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365.0, "rho": rho}


def implied_vol(price: float, spot: float, strike: float, t: float, right: Right | str,
                r: float = 0.0, q: float = 0.0, lo: float = 1e-4, hi: float = 8.0) -> float | None:
    """Implied volatility by bisection; None when ``price`` is outside the no-arbitrage range."""
    right = Right.parse(right)
    if t <= MIN_T:
        return None
    if price < bs_price(spot, strike, t, lo, right, r, q) - 1e-12 or price > bs_price(spot, strike, t, hi, right, r, q):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, t, mid, right, r, q) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def underlying_for_premium(premium: float, strike: float, t: float, iv: float, right: Right | str,
                           r: float = 0.0, q: float = 0.0, lo: float | None = None,
                           hi: float | None = None) -> float | None:
    """Underlying level at which the option is worth ``premium`` (same time and vol), by bisection.

    A call's price rises with spot and a put's falls, so the root is unique. Returns None if
    ``premium`` is not reachable inside the search bracket (default: 1% to 100x the strike).
    """
    right = Right.parse(right)
    lo = strike * 0.01 if lo is None else lo
    hi = strike * 100.0 if hi is None else hi
    sign = 1.0 if right is Right.CALL else -1.0
    f = lambda s: sign * (bs_price(s, strike, t, iv, right, r, q) - premium)   # noqa: E731 - increasing in s
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
