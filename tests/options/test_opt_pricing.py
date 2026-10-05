import math

import pytest

from tradex.options.contract import Right
from tradex.options.pricing import bs_greeks, bs_price, implied_vol, norm_cdf, underlying_for_premium

CASES = [(100, 100, 0.25, 0.30, 0.03, 0.01), (100, 110, 0.5, 0.45, 0.0, 0.0), (50, 40, 0.1, 0.60, 0.04, 0.0),
         (200, 230, 1.0, 0.25, 0.02, 0.02)]


def test_known_value():
    # S=K=100, T=1, vol 20%, r=5%: textbook call value 10.4506
    assert bs_price(100, 100, 1.0, 0.20, "C", 0.05) == pytest.approx(10.4506, abs=1e-3)
    assert bs_price(100, 100, 1.0, 0.20, "P", 0.05) == pytest.approx(5.5735, abs=1e-3)


@pytest.mark.parametrize("s,k,t,iv,r,q", CASES)
def test_put_call_parity(s, k, t, iv, r, q):
    c, p = bs_price(s, k, t, iv, "C", r, q), bs_price(s, k, t, iv, "P", r, q)
    assert c - p == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-9)


@pytest.mark.parametrize("s,k,t,iv,r,q", CASES)
def test_price_bounds_and_monotonic(s, k, t, iv, r, q):
    c = bs_price(s, k, t, iv, "C", r, q)
    assert max(s - k, 0) * 0 <= c <= s
    assert bs_price(s * 1.05, k, t, iv, "C", r, q) > c
    assert bs_price(s, k, t, iv * 1.2, "C", r, q) > c           # vega positive


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("s,k,t,iv,r,q", CASES)
def test_greeks_match_finite_differences(right, s, k, t, iv, r, q):
    g = bs_greeks(s, k, t, iv, right, r, q)
    h = s * 1e-4
    up, dn = bs_price(s + h, k, t, iv, right, r, q), bs_price(s - h, k, t, iv, right, r, q)
    mid = bs_price(s, k, t, iv, right, r, q)
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-4, abs=1e-6)
    assert g["gamma"] == pytest.approx((up - 2 * mid + dn) / h ** 2, rel=1e-3, abs=1e-6)
    dv = 1e-5
    assert g["vega"] == pytest.approx((bs_price(s, k, t, iv + dv, right, r, q) - bs_price(s, k, t, iv - dv, right, r, q)) / (2 * dv), rel=1e-4)
    dt_ = 1e-5
    theta = -(bs_price(s, k, t + dt_, iv, right, r, q) - bs_price(s, k, t - dt_, iv, right, r, q)) / (2 * dt_)
    assert g["theta"] == pytest.approx(theta / 365.0, rel=1e-3, abs=1e-6)


def test_call_delta_between_zero_and_one_put_negative():
    assert 0 < bs_greeks(100, 105, 0.3, 0.3, "C")["delta"] < 1
    assert -1 < bs_greeks(100, 105, 0.3, 0.3, "P")["delta"] < 0


def test_expiry_and_zero_vol_collapse_to_intrinsic():
    assert bs_price(110, 100, 0.0, 0.3, "C") == 10 and bs_price(90, 100, 0.0, 0.3, "C") == 0
    assert bs_price(90, 100, 0.0, 0.3, "P") == 10
    assert bs_price(110, 100, 0.5, 0.0, "C") == 10
    assert bs_greeks(110, 100, 0.0, 0.3, "C")["delta"] == 1.0
    assert bs_greeks(90, 100, 0.0, 0.3, "C")["delta"] == 0.0 and bs_greeks(90, 100, 0.0, 0.3, "P")["delta"] == -1.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bs_price(0, 100, 1, 0.2, "C")
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.2, "X")


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_implied_vol_round_trip(right):
    px = bs_price(100, 105, 0.4, 0.37, right, 0.02)
    assert implied_vol(px, 100, 105, 0.4, right, 0.02) == pytest.approx(0.37, abs=1e-6)


def test_implied_vol_out_of_range_is_none():
    assert implied_vol(0.0, 100, 100, 0.5, "C") is None
    assert implied_vol(150.0, 100, 100, 0.5, "C") is None
    assert implied_vol(5.0, 100, 100, 0.0, "C") is None


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_underlying_for_premium_round_trip(right):
    target = 3.2
    s = underlying_for_premium(target, 100, 0.2, 0.35, right, 0.0)
    assert bs_price(s, 100, 0.2, 0.35, right, 0.0) == pytest.approx(target, abs=1e-6)


def test_underlying_for_premium_unreachable():
    assert underlying_for_premium(1e9, 100, 0.2, 0.35, "C") is None
    assert underlying_for_premium(-1.0, 100, 0.2, 0.35, "C") is None


def test_norm_cdf_sanity():
    assert norm_cdf(0) == 0.5 and norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
