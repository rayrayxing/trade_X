import datetime as dt
import math

import pandas as pd
import pytest

from tradex.options.contract import OptionContract, OptionPosition, Right
from tradex.options.payoff import (breakeven, early_assignment_risk, expiry_value, max_loss, max_profit,
                                   pnl_at_expiry, pnl_at_premium, pnl_curve, settle_expiry)

EXP = dt.date(2026, 10, 16)
T0 = pd.Timestamp("2026-10-01", tz="UTC")


def pos(right="C", k=100.0, n=1, prem=2.0):
    return OptionPosition(OptionContract("XYZ", EXP, k, Right.parse(right)), n, prem, T0)


def test_long_call_pnl_includes_multiplier():
    p = pos("C", 100, 2, 3.0)
    assert pnl_at_expiry(p, 110) == pytest.approx((10 - 3.0) * 100 * 2)
    assert pnl_at_expiry(p, 90) == pytest.approx(-3.0 * 100 * 2)
    assert pnl_at_expiry(p, 110, fees=10) == pytest.approx(1400 - 10)
    assert pnl_at_premium(p, 5.0) == pytest.approx(400)


def test_long_put_pnl():
    p = pos("P", 100, 1, 4.0)
    assert pnl_at_expiry(p, 80) == pytest.approx((20 - 4.0) * 100)
    assert pnl_at_expiry(p, 120) == pytest.approx(-400)


def test_short_call_pnl_mirrors_long():
    sh, lg = pos("C", 100, -3, 2.0), pos("C", 100, 3, 2.0)
    for s in (80, 100, 102, 130, 200):
        assert pnl_at_expiry(sh, s) == pytest.approx(-pnl_at_expiry(lg, s))
    assert pnl_at_expiry(sh, 90) == pytest.approx(2.0 * 300)           # keep the credit
    assert pnl_at_expiry(sh, 150) == pytest.approx((2.0 - 50) * 300)    # no ceiling on the loss


def test_breakeven_and_extremes():
    assert breakeven(pos("C", 100).contract, 3.0) == 103 and breakeven(pos("P", 100).contract, 3.0) == 97
    lc = pos("C", 100, 2, 3.0)
    assert max_loss(lc) == 600 and math.isinf(max_profit(lc))
    lp = pos("P", 100, 1, 4.0)
    assert max_loss(lp) == 400 and max_profit(lp) == pytest.approx((100 - 4.0) * 100)
    sc = pos("C", 100, -1, 2.5)
    assert math.isinf(max_loss(sc)) and max_profit(sc) == 250
    sp = pos("P", 100, -1, 2.5)
    assert max_loss(sp) == pytest.approx(9750) and max_profit(sp) == 250


def test_pnl_curve_is_indexed_by_spot():
    c = pnl_curve(pos("C", 100, 1, 2.0), [90, 100, 110])
    assert list(c.index) == [90, 100, 110] and list(c.round(6)) == [-200, -200, 800]


def test_expiry_value():
    assert expiry_value(pos("C", 100).contract, 105) == 5


def test_settle_worthless():
    s = settle_expiry(pos("C", 100, 1, 2.0), 99.995)
    assert s.kind == "expired_worthless" and s.fees_usd == 0 and s.option_pnl_usd == pytest.approx(-200)
    assert not s.leaves_stock_position
    sh = settle_expiry(pos("C", 100, -1, 2.0), 95)
    assert sh.kind == "expired_worthless" and sh.option_pnl_usd == pytest.approx(200)


def test_one_cent_itm_is_exercised_threshold():
    assert settle_expiry(pos("C", 100, 1, 2.0), 100.01, sell_fees_usd=1.0).kind == "sold_at_intrinsic"


def test_settle_long_itm_default_sells_at_intrinsic():
    s = settle_expiry(pos("C", 100, 2, 3.0), 110, sell_fees_usd=4.5)
    assert s.kind == "sold_at_intrinsic" and s.shares_delivered == 0 and s.fees_usd == 4.5
    assert s.option_pnl_usd == pytest.approx((10 - 3) * 200) and not s.fee_unknown
    assert settle_expiry(pos("C", 100, 2, 3.0), 110).fee_unknown        # sell fees not supplied


def test_settle_long_call_exercise_delivers_shares():
    s = settle_expiry(pos("C", 100, 2, 3.0), 110, long_itm="exercise", exercise_fee_per_contract=1.0)
    assert s.kind == "exercised" and s.shares_delivered == 200
    assert s.stock_cash_usd == pytest.approx(-20000) and s.stock_pnl_vs_spot_usd == pytest.approx(2000)
    assert s.fees_usd == 2.0 and not s.fee_unknown
    unknown = settle_expiry(pos("C", 100, 2, 3.0), 110, long_itm="exercise")
    assert unknown.fee_unknown and unknown.fees_usd == 0.0


def test_settle_long_put_exercise_delivers_shares_away():
    s = settle_expiry(pos("P", 100, 1, 3.0), 90, long_itm="exercise", exercise_fee_per_contract=0.5)
    assert s.shares_delivered == -100 and s.stock_cash_usd == pytest.approx(10000)
    assert s.stock_pnl_vs_spot_usd == pytest.approx(1000)               # sold at 100 what is worth 90


def test_settle_naked_call_assigned_leaves_short_stock():
    p = pos("C", 100, -2, 1.5)
    s = settle_expiry(p, 112, assignment_fee_per_contract=2.0)
    assert s.kind == "assigned" and s.shares_delivered == -200 and s.leaves_stock_position
    assert s.stock_cash_usd == pytest.approx(20000)                     # shares sold at the strike
    assert s.stock_pnl_vs_spot_usd == pytest.approx(-2400)              # short 200 from 100 to 112
    assert s.option_pnl_usd == pytest.approx((1.5 - 12) * 200)
    assert s.fees_usd == 4.0 and "naked" in s.note
    # the stock leg and the option P&L describe the same loss
    assert s.option_pnl_usd == pytest.approx(s.stock_pnl_vs_spot_usd + 1.5 * 200)


def test_settle_short_put_assigned_receives_shares():
    s = settle_expiry(pos("P", 100, -1, 2.0), 90)
    assert s.kind == "assigned" and s.shares_delivered == 100 and s.stock_cash_usd == pytest.approx(-10000)
    assert s.fee_unknown


def test_settle_rejects_bad_policy():
    with pytest.raises(ValueError):
        settle_expiry(pos(), 110, long_itm="hold")


def test_early_assignment_dividend_rule():
    sc = pos("C", 100, -1, 11.0)
    # ITM by 10 with 1.0 of extrinsic, dividend 1.5 goes ex tomorrow: assignment likely
    assert early_assignment_risk(sc, 110, 11.0, dividend=1.5, ex_dividend_within_days=1)
    assert early_assignment_risk(sc, 110, 11.0, dividend=0.5, ex_dividend_within_days=1) is None
    assert early_assignment_risk(sc, 110, 11.0, dividend=1.5, ex_dividend_within_days=5) is None


def test_early_assignment_no_extrinsic_rule_and_scope():
    sc = pos("C", 100, -1, 10.02)
    assert early_assignment_risk(sc, 110, 10.02)                        # 0.02 of extrinsic
    assert early_assignment_risk(sc, 90, 0.5) is None                   # out of the money
    assert early_assignment_risk(pos("C", 100, 1, 10.0), 110, 10.0, 5.0, 0) is None      # long calls are not assigned
    assert early_assignment_risk(pos("P", 100, -1, 10.0), 90, 10.0, 5.0, 0) is None      # only calls handled here
