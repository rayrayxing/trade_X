import datetime as dt

import pandas as pd
import pytest

from tradex.options.contract import (MULTIPLIER, Greeks, OptionContract, OptionPosition, OptionQuote, Right)

EXP = dt.date(2026, 10, 16)


def call(k=150.0, exp=EXP, u="AAPL"):
    return OptionContract(u, exp, k, Right.CALL)


def test_multiplier_default_is_100():
    assert MULTIPLIER == 100 and call().multiplier == 100


def test_moomoo_code_round_trip():
    c = OptionContract("AAPL", EXP, 150.0, Right.CALL)
    assert c.moomoo_code == "US.AAPL261016C150000"
    assert OptionContract.from_moomoo_code(c.moomoo_code) == c
    assert OptionContract.from_moomoo_code("AAPL261016C150000") == c        # prefix optional
    p = OptionContract("SPY", EXP, 432.5, Right.PUT)
    assert p.moomoo_code == "US.SPY261016P432500"
    assert OptionContract.from_moomoo_code(p.moomoo_code) == p


def test_occ_symbol_round_trip():
    c = OptionContract("AAPL", EXP, 150.0, Right.CALL)
    assert c.occ_symbol == "AAPL  261016C00150000" and len(c.occ_symbol) == 21
    assert OptionContract.from_occ(c.occ_symbol) == c
    odd = OptionContract("F", EXP, 12.5, Right.PUT)
    assert OptionContract.from_occ(odd.occ_symbol) == odd


@pytest.mark.parametrize("bad", ["", "AAPL", "US.AAPL261016X150000", "US.AAPL26101C150000"])
def test_bad_codes_raise(bad):
    with pytest.raises(ValueError):
        OptionContract.from_moomoo_code(bad)


def test_bad_occ_raises():
    with pytest.raises(ValueError):
        OptionContract.from_occ("AAPL261016C150")


def test_right_parse():
    assert Right.parse("call") is Right.CALL and Right.parse("P") is Right.PUT and Right.parse(Right.CALL) is Right.CALL
    with pytest.raises(ValueError):
        Right.parse("straddle")


def test_contract_validation_and_hashing():
    with pytest.raises(ValueError):
        OptionContract("X", EXP, 0.0, Right.CALL)
    with pytest.raises(ValueError):
        OptionContract("X", EXP, 10.0, Right.CALL, multiplier=0)
    assert {call(), call()} == {call()}
    assert OptionContract("X", EXP, 10, "C").right is Right.CALL
    assert OptionContract("X", pd.Timestamp("2026-10-16"), 10, "C").expiry == EXP


def test_intrinsic_and_moneyness():
    c, p = call(100), OptionContract("AAPL", EXP, 100.0, Right.PUT)
    assert c.intrinsic(110) == 10 and c.intrinsic(90) == 0
    assert p.intrinsic(90) == 10 and p.intrinsic(110) == 0
    assert c.is_itm(101) and not c.is_itm(100) and p.is_itm(99)
    assert c.moneyness(80) == 1.25


def test_time_to_expiry():
    c = call()
    asof = pd.Timestamp("2026-10-09 14:00", tz="UTC")
    assert c.days_to_expiry(asof) == 7 and c.days_to_expiry(EXP) == 0
    assert c.expiry_time == pd.Timestamp("2026-10-16 20:00", tz="UTC")      # 16:00 EDT
    assert abs(c.years_to_expiry(asof) - (7 + 6 / 24) / 365) < 1e-9
    assert c.years_to_expiry(pd.Timestamp("2026-11-01", tz="UTC")) == 0.0
    assert c.expired(pd.Timestamp("2026-10-16 20:00", tz="UTC")) and not c.expired(asof)


def test_expiry_time_follows_dst():
    winter = OptionContract("AAPL", dt.date(2026, 12, 18), 100, "C")
    assert winter.expiry_time == pd.Timestamp("2026-12-18 21:00", tz="UTC")   # 16:00 EST


def test_quote_helpers():
    q = OptionQuote(call(), pd.Timestamp("2026-10-01", tz="UTC"), 1.0, 1.2, 1.1, 10, 50, Greeks(delta=0.4, iv=0.3))
    assert q.two_sided and q.mid == pytest.approx(1.1) and q.spread_pct == pytest.approx(0.2 / 1.1)
    assert q.delta == 0.4 and q.iv == 0.3
    one_sided = OptionQuote(call(), q.ts, None, 1.2, 1.1)
    assert not one_sided.two_sided and one_sided.mid == 1.1 and one_sided.spread_pct is None
    crossed = OptionQuote(call(), q.ts, 1.3, 1.2, None)
    assert not crossed.two_sided and crossed.mid is None
    zero_ask = OptionQuote(call(), q.ts, 0.0, 0.0, None)
    assert not zero_ask.two_sided


def test_position_long_and_short():
    t = pd.Timestamp("2026-10-01", tz="UTC")
    lg = OptionPosition(call(), 3, 2.0, t)
    assert lg.is_long and lg.direction == 1 and lg.qty == 3 and lg.shares == 300
    assert lg.market_value(2.5) == 750 and lg.unrealized_pnl(2.5) == pytest.approx(150)
    assert not lg.is_naked_call and lg.delta_shares(0.5) == 150
    sh = OptionPosition(call(), -2, 1.5, t)
    assert not sh.is_long and sh.direction == -1 and sh.qty == 2 and sh.shares == -200
    assert sh.market_value(2.0) == -400 and sh.unrealized_pnl(2.0) == pytest.approx(-100)
    assert sh.is_naked_call and sh.delta_shares(0.3) == pytest.approx(-60)
    short_put = OptionPosition(OptionContract("A", EXP, 5, "P"), -1, 1.0, t)
    assert not short_put.is_naked_call


def test_position_rejects_zero_contracts():
    with pytest.raises(ValueError):
        OptionPosition(call(), 0, 1.0, pd.Timestamp("2026-10-01", tz="UTC"))
