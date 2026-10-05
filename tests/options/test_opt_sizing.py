import datetime as dt

import pytest

from tradex.costs.models import MoomooOptionCosts, model_for
from tradex.options.contract import OptionContract, Right
from tradex.options.pricing import bs_price
from tradex.options.sizing import (GAP_PCT_FLOOR, BrokerMarginEstimator, GapVol, RegTMarginEstimator,
                                   effective_naked_stop, long_premium_at_risk, margin_max_contracts,
                                   max_long_contracts, max_naked_contracts, naked_call_gap_loss, order_fee_usd,
                                   premium_after_gap, size_long, size_naked_call, stop_premium_from_multiple)

COSTS = MoomooOptionCosts()
EXP = dt.date(2026, 10, 16)


def fee(side, n, px):
    return sum(COSTS.order_fees("X", side, n, px, None).values())


def call(k=105.0):
    return OptionContract("XYZ", EXP, k, Right.CALL)


# --- fees -------------------------------------------------------------------------------------

def test_moomoo_option_fee_schedule_matches_the_brief():
    # US$0.65 commission + US$0.30 platform per contract (minimums US$1.99 / US$0.99), 9% GST on both
    assert order_fee_usd(COSTS, +1, 1, 1.00) == pytest.approx(3.2815, abs=1e-4)
    assert order_fee_usd(COSTS, +1, 10, 1.00) / 10 == pytest.approx(1.0688, abs=1e-3)
    one = COSTS.order_fees("X", +1, 10, 1.0, None)
    assert one["commission"] == pytest.approx(6.5) and one["platform"] == pytest.approx(3.0)
    assert one["gst"] == pytest.approx(0.09 * 9.5)
    assert model_for("options").order_fees("X", 1, 1, 1.0, None) == COSTS.order_fees("X", 1, 1, 1.0, None)


def test_sell_fees_add_sec_and_finra():
    buy, sell = order_fee_usd(COSTS, +1, 5, 2.0), order_fee_usd(COSTS, -1, 5, 2.0)
    assert sell - buy == pytest.approx(COSTS.sec_fee_rate * 5 * 2.0 * 100 + COSTS.finra_taf_per_contract * 5)


# --- long premium at risk ---------------------------------------------------------------------

def test_long_premium_at_risk_uses_the_multiplier():
    r = long_premium_at_risk(2.50, 2, COSTS)
    assert r.premium_usd == pytest.approx(500)                      # 2.50 x 100 x 2, not 5
    assert r.open_fees_usd == pytest.approx(fee(+1, 2, 2.50))
    assert r.close_fees_usd == pytest.approx(fee(-1, 2, 0.0))
    assert r.max_loss_usd == pytest.approx(500 + r.open_fees_usd + r.close_fees_usd)


def test_long_premium_at_risk_validates():
    for args in ((0, 1), (-1, 1), (1.0, 0)):
        with pytest.raises(ValueError):
            long_premium_at_risk(*args)


def test_max_long_contracts_fits_budget_exactly():
    budget = 1000.0
    n = max_long_contracts(budget, 2.50, COSTS)
    assert n >= 1
    assert long_premium_at_risk(2.50, n, COSTS).max_loss_usd <= budget
    assert long_premium_at_risk(2.50, n + 1, COSTS).max_loss_usd > budget


def test_max_long_contracts_fees_can_cost_a_contract():
    # 4 contracts at 2.50 cost exactly 1000 before fees, so fees push the answer to 3
    assert max_long_contracts(1000.0, 2.50, COSTS) == 3


def test_max_long_contracts_zero_when_one_does_not_fit():
    assert max_long_contracts(200.0, 2.50, COSTS) == 0
    assert max_long_contracts(0.0, 1.0) == 0 and max_long_contracts(-5.0, 1.0) == 0 and max_long_contracts(100, 0) == 0


# --- naked call gap-stressed loss ---------------------------------------------------------------

def test_gap_loss_hand_calculation_without_vol_inputs():
    # K=105, credit 1.50, stop at 3.00 (2x) where the stock is 103; gap 20% -> 123.6
    gl = naked_call_gap_loss(105, 1.50, 3.00, 103.0, contracts=1, costs=COSTS)
    assert gl.underlying_after_gap == pytest.approx(123.6)
    # hard bound: 3.00 + (123.6 - 103) = 23.6
    assert gl.premium_after_gap == pytest.approx(23.6)
    cushion = (COSTS.half_spread_pct + COSTS.slippage_pct)
    assert gl.fill_after_gap == pytest.approx(23.6 * (1 + cushion))
    assert gl.gap_loss_usd == pytest.approx((23.6 * (1 + cushion) - 1.50) * 100)
    assert gl.stop_loss_usd == pytest.approx((3.0 - 1.5) * 100)
    assert gl.open_fees_usd == pytest.approx(fee(-1, 1, 1.50))
    assert gl.close_fees_usd == pytest.approx(fee(+1, 1, 23.6 * (1 + cushion)))
    assert gl.max_loss_usd == pytest.approx(gl.gap_loss_usd + gl.open_fees_usd + gl.close_fees_usd)
    assert gl.per_contract_usd == pytest.approx(gl.max_loss_usd)


def test_gap_loss_scales_with_contracts_and_multiplier():
    one = naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, costs=COSTS)
    five = naked_call_gap_loss(105, 1.5, 3.0, 103.0, 5, costs=COSTS)
    assert five.gap_loss_usd == pytest.approx(5 * one.gap_loss_usd)
    assert five.max_loss_usd < 5 * one.max_loss_usd                  # per-contract fees fall with size
    assert five.stop_loss_usd == pytest.approx(5 * one.stop_loss_usd)


def test_gap_loss_with_vol_inputs_is_tighter_but_never_below_floor():
    vol = GapVol(iv=0.30, years=30 / 365)
    loose = naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, costs=COSTS)
    tight = naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, costs=COSTS, vol=vol)
    assert tight.premium_after_gap < loose.premium_after_gap
    s_gap = 103.0 * 1.2
    assert tight.premium_after_gap >= s_gap - 105 + 3.0 - 1e-9       # intrinsic + time value left at the stop


@pytest.mark.parametrize("k,s_stop,iv,years", [(100, 90, 0.30, 0.1), (100, 98, 0.50, 0.25), (100, 100, 0.25, 0.05),
                                               (100, 110, 0.40, 0.2), (50, 45, 0.80, 0.5)])
def test_modelled_gap_price_covers_black_scholes_repricing_and_stays_under_the_hard_bound(k, s_stop, iv, years):
    p_stop = bs_price(s_stop, k, years, iv, "C")
    s_gap, p_gap = premium_after_gap(k, p_stop, s_stop, 0.20, GapVol(iv, years))
    assert p_gap >= bs_price(s_gap, k, years, iv, "C") - 1e-9
    assert p_gap <= p_stop + (s_gap - s_stop) + 1e-9
    _, hard = premium_after_gap(k, p_stop, s_stop, 0.20)
    assert hard == pytest.approx(p_stop + (s_gap - s_stop))


def test_gap_loss_monotonic_in_gap_and_stop():
    base = naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, 0.20, COSTS).max_loss_usd
    assert naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, 0.30, COSTS).max_loss_usd > base
    assert naked_call_gap_loss(105, 1.5, 4.5, 106.0, 1, 0.20, COSTS).max_loss_usd > base


def test_gap_floor_cannot_be_lowered():
    assert GAP_PCT_FLOOR == 0.20
    with pytest.raises(ValueError, match="floor"):
        naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, gap_pct=0.19)
    naked_call_gap_loss(105, 1.5, 3.0, 103.0, 1, gap_pct=0.20)


def test_stop_below_credit_is_refused_and_multiple_must_exceed_one():
    with pytest.raises(ValueError):
        naked_call_gap_loss(105, 1.5, 1.2, 103.0)
    with pytest.raises(ValueError):
        stop_premium_from_multiple(1.5, 1.0)
    with pytest.raises(ValueError):
        stop_premium_from_multiple(1.5, 0.5)
    assert stop_premium_from_multiple(1.5, 2.0) == 3.0


@pytest.mark.parametrize("bad", [dict(credit=0), dict(underlying_at_stop=0), dict(contracts=0)])
def test_gap_loss_input_validation(bad):
    args = dict(strike=105, credit=1.5, stop_premium=3.0, underlying_at_stop=103.0, contracts=1) | bad
    with pytest.raises(ValueError):
        naked_call_gap_loss(**args)


def test_max_naked_contracts_fits_budget():
    kw = dict(strike=105, credit=1.5, stop_premium=3.0, underlying_at_stop=103.0, costs=COSTS)
    one = naked_call_gap_loss(contracts=1, **kw).max_loss_usd
    budget = one * 3.5
    n = max_naked_contracts(budget, **kw)
    assert n == 3
    assert naked_call_gap_loss(contracts=n, **kw).max_loss_usd <= budget
    assert naked_call_gap_loss(contracts=n + 1, **kw).max_loss_usd > budget
    assert max_naked_contracts(one * 0.99, **kw) == 0 and max_naked_contracts(0, **kw) == 0


# --- effective stop ---------------------------------------------------------------------------

def test_effective_stop_premium_trigger_located_by_black_scholes():
    c = call(105)
    iv, yrs = 0.30, 30 / 365
    credit = bs_price(100, 105, yrs, iv, "C")
    s, p = effective_naked_stop(c, credit, 2 * credit, 100.0, iv, yrs)
    assert p == pytest.approx(2 * credit) and s > 100
    assert bs_price(s, 105, yrs, iv, "C") == pytest.approx(2 * credit, abs=1e-6)


def test_effective_stop_underlying_trigger_wins_when_it_comes_first():
    c = call(105)
    iv, yrs = 0.30, 30 / 365
    credit = bs_price(100, 105, yrs, iv, "C")
    s_prem, _ = effective_naked_stop(c, credit, 2 * credit, 100.0, iv, yrs)
    s, p = effective_naked_stop(c, credit, 2 * credit, 100.0, iv, yrs, underlying_stop=101.0)
    assert s == 101.0 and s < s_prem and credit <= p < 2 * credit
    s2, _ = effective_naked_stop(c, credit, 2 * credit, 100.0, iv, yrs, underlying_stop=s_prem + 5)
    assert s2 == pytest.approx(s_prem)                  # a looser underlying stop does not matter


def test_effective_stop_is_none_without_iv_or_time():
    c = call()
    assert effective_naked_stop(c, 1.5, 3.0, 100.0, None, 0.1) is None
    assert effective_naked_stop(c, 1.5, 3.0, 100.0, 0.3, 0.0) is None
    assert effective_naked_stop(c, 1.5, 1e6, 100.0, 0.3, 0.1) is None          # trigger unreachable
    with pytest.raises(ValueError):
        effective_naked_stop(OptionContract("X", EXP, 100, "P"), 1.5, 3.0, 100.0, 0.3, 0.1)


# --- margin ------------------------------------------------------------------------------------

def test_regt_margin_naked_call_hand_calculation():
    est = RegTMarginEstimator()
    c = call(105)
    # OTM 5: max(0.20*100 - 5 + 1.5, 0.10*100 + 1.5) = max(16.5, 11.5) = 16.5 per share
    assert est.initial_margin(c, -2, 1.5, 100.0) == pytest.approx(16.5 * 100 * 2)
    # ITM call: 0.20*120 + 16.5 premium... K=105 S=120 premium 16: max(24 - 0 + 16, 12 + 16) = 40
    assert est.initial_margin(c, -1, 16.0, 120.0) == pytest.approx(40 * 100)
    # far OTM hits the 10% floor: max(0.2*100 - 30 + 0.2, 10.2) = 10.2
    assert est.initial_margin(call(130), -1, 0.2, 100.0) == pytest.approx(10.2 * 100)


def test_regt_margin_long_is_premium_and_short_put():
    est = RegTMarginEstimator()
    assert est.initial_margin(call(), 3, 2.0, 100.0) == pytest.approx(600)
    p = OptionContract("XYZ", EXP, 95.0, Right.PUT)
    # OTM 5 for the put: max(20 - 5 + 1, 0.10*95 + 1) = 16
    assert est.initial_margin(p, -1, 1.0, 100.0) == pytest.approx(1600)


def test_margin_max_contracts():
    est = RegTMarginEstimator()
    c = call(105)
    per = 16.5 * 100
    assert margin_max_contracts(est, per * 3.2, c, 1.5, 100.0, short=True) == 3
    assert margin_max_contracts(est, per * 0.9, c, 1.5, 100.0) == 0
    assert margin_max_contracts(est, 0, c, 1.5, 100.0) == 0
    assert margin_max_contracts(est, 1000, c, 2.0, 100.0, short=False) == 5          # long: premium 200 each


def test_broker_margin_estimator_defers_to_the_broker():
    calls = []

    def broker(contract, contracts, premium, spot):
        calls.append((contract, contracts))
        return 1234.0 * abs(contracts)
    est = BrokerMarginEstimator(broker)
    assert est.initial_margin(call(), -3, 1.0, 100.0) == 3702.0 and calls == [(call(), -3)]
    assert margin_max_contracts(est, 5000.0, call(), 1.0, 100.0) == 4


# --- one-call sizing ---------------------------------------------------------------------------

def test_size_long_confidence_and_margin_binding():
    c = call(105)
    s = size_long(1000.0, 2.50, COSTS)
    assert s.contracts == 3 and s.binding == "confidence" and s.risk_usd == pytest.approx(s.detail.max_loss_usd)
    capped = size_long(1000.0, 2.50, COSTS, estimator=RegTMarginEstimator(), contract=c, free_margin_usd=500.0,
                       underlying_price=100.0)
    assert capped.contracts == 2 and capped.binding == "margin"
    assert size_long(100.0, 2.50, COSTS).contracts == 0


def test_size_naked_call_confidence_and_margin_binding():
    c = call(105)
    kw = dict(contract=c, credit=1.5, stop_premium=3.0, underlying_at_stop=103.0, underlying_price=100.0, costs=COSTS)
    big = size_naked_call(10_000.0, **kw)
    assert big.contracts >= 3 and big.binding == "confidence"
    assert big.risk_usd == pytest.approx(big.detail.max_loss_usd) and big.risk_usd <= 10_000
    cut = size_naked_call(10_000.0, free_margin_usd=16.5 * 100 * 2.5, **kw)
    assert cut.contracts == 2 and cut.binding == "margin"
    assert cut.margin_usd == pytest.approx(16.5 * 100 * 2)
    none = size_naked_call(10.0, **kw)
    assert none.contracts == 0 and none.binding == "none"


def test_size_naked_call_uses_vol_inputs_when_given():
    c = call(105)
    kw = dict(contract=c, credit=1.5, stop_premium=3.0, underlying_at_stop=103.0, underlying_price=100.0, costs=COSTS)
    loose = size_naked_call(5_000.0, **kw)
    tight = size_naked_call(5_000.0, vol=GapVol(0.30, 30 / 365), **kw)
    assert tight.contracts >= loose.contracts and tight.risk_usd <= 5_000
