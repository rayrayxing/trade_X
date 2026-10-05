import datetime as dt

import numpy as np
import pandas as pd
import pytest

from opt_fixtures import BSChain, bars_from_closes
from tradex.options.contract import OptionContract, Right
from tradex.options.providers import (OptionDataProvider, RealDataMissing, RecordedChainProvider, SyntheticDataRefused,
                                      UnderlyingRiskProvider, quotes_from_opend_frame, require_real_option_data)

T = pd.Timestamp("2026-10-05", tz="UTC")
EXP = dt.date(2026, 10, 16)


def frame(**over):
    rows = [dict(ts=T, at="open", underlying="XYZ", expiry=EXP, strike=100.0, right="C", bid=1.0, ask=1.2, last=1.1,
                 volume=10, open_interest=500, delta=0.45, gamma=0.05, theta=-0.04, vega=0.1, rho=0.01, iv=0.31,
                 underlying_price=100.0),
            dict(ts=T, at="open", underlying="XYZ", expiry=EXP, strike=105.0, right="C", bid=0.4, ask=0.5, last=0.45,
                 volume=5, open_interest=900, delta=0.25, gamma=0.04, theta=-0.03, vega=0.09, rho=0.01, iv=0.33,
                 underlying_price=100.0),
            dict(ts=T, at="close", underlying="XYZ", expiry=EXP, strike=100.0, right="C", bid=1.4, ask=1.6, last=1.5,
                 volume=40, open_interest=500, delta=0.5, gamma=0.05, theta=-0.04, vega=0.1, rho=0.01, iv=0.30,
                 underlying_price=101.0)]
    return pd.DataFrame(rows).assign(**over) if over else pd.DataFrame(rows)


def test_recorded_provider_implements_the_protocols():
    p = RecordedChainProvider(frame())
    assert isinstance(p, OptionDataProvider)

    class R:
        def next_earnings(self, u, a): ...
        def short_info(self, u, a): ...
    assert isinstance(R(), UnderlyingRiskProvider)


def test_chain_and_quote_lookup_exact_by_bar_and_moment():
    p = RecordedChainProvider(frame())
    opens = p.chain("XYZ", T, "open")
    assert len(opens) == 2 and {q.contract.strike for q in opens} == {100.0, 105.0}
    assert len(p.chain("XYZ", T, "close")) == 1
    c = OptionContract("XYZ", EXP, 100.0, Right.CALL)
    q = p.quote(c, T, "open")
    assert (q.bid, q.ask, q.last, q.volume, q.open_interest) == (1.0, 1.2, 1.1, 10, 500)
    assert q.greeks.delta == 0.45 and q.greeks.iv == 0.31 and q.underlying_price == 100.0
    assert p.quote(c, T, "close").bid == 1.4


def test_missing_snapshot_is_none_never_the_nearest():
    p = RecordedChainProvider(frame())
    c = OptionContract("XYZ", EXP, 100.0, Right.CALL)
    assert p.quote(c, T + pd.Timedelta(days=1), "open") is None
    assert p.quote(c, T - pd.Timedelta(days=1), "close") is None
    assert p.quote(OptionContract("XYZ", EXP, 110.0, Right.CALL), T, "open") is None
    assert p.chain("ABC", T, "open") == [] and p.chain("XYZ", T + pd.Timedelta(days=3), "close") == []


def test_naive_timestamps_are_read_as_utc():
    p = RecordedChainProvider(frame(ts=pd.Timestamp("2026-10-05")))
    assert len(p.chain("XYZ", pd.Timestamp("2026-10-05"), "open")) == 2
    assert len(p.chain("XYZ", T, "open")) == 2


def test_nan_fields_become_none():
    f = frame()
    f.loc[0, "ask"] = np.nan
    f.loc[0, "delta"] = np.nan
    q = RecordedChainProvider(f).quote(OptionContract("XYZ", EXP, 100.0, Right.CALL), T, "open")
    assert q.ask is None and q.greeks.delta is None and not q.two_sided


def test_bad_frames_rejected():
    with pytest.raises(ValueError, match="lacks columns"):
        RecordedChainProvider(frame().drop(columns=["strike"]))
    with pytest.raises(ValueError, match="open or close"):
        RecordedChainProvider(frame(at="midday"))


def test_optional_columns_may_be_absent():
    f = frame()[["ts", "at", "underlying", "expiry", "strike", "right", "bid", "ask"]]
    q = RecordedChainProvider(f).chain("XYZ", T, "open")[0]
    assert q.volume is None and q.greeks.delta is None


# --- OpenD conversion ------------------------------------------------------------------------

def opend_frame():
    return pd.DataFrame([
        {"code": "US.XYZ261016C100000", "bid_price": 1.0, "ask_price": 1.2, "last_price": 1.1, "volume": 12,
         "option_open_interest": 340, "option_implied_volatility": 31.5, "option_delta": 0.45, "option_gamma": 0.05,
         "option_theta": -0.04, "option_vega": 0.11, "option_rho": 0.02},
        {"code": "US.XYZ261016P095000", "bid_price": 0.5, "ask_price": 0.6, "last_price": float("nan"),
         "volume": 3, "option_open_interest": 80, "option_implied_volatility": 40.0, "option_delta": -0.2},
        {"code": "US.XYZ", "bid_price": 100.0, "ask_price": 100.1},            # the stock row in a mixed snapshot
    ])


def test_opend_frame_conversion():
    qs = quotes_from_opend_frame(opend_frame(), T, underlying_price=100.0)
    assert len(qs) == 2
    c, p = qs
    assert c.contract == OptionContract("XYZ", EXP, 100.0, Right.CALL) and c.ts == T
    assert c.greeks.iv == pytest.approx(0.315) and c.greeks.delta == 0.45 and c.open_interest == 340
    assert c.underlying_price == 100.0 and c.two_sided
    assert p.contract.right is Right.PUT and p.contract.strike == 95.0 and p.last is None and p.greeks.iv == 0.40


def test_opend_iv_units_and_column_remap():
    qs = quotes_from_opend_frame(opend_frame(), T, iv_in_percent=False)
    assert qs[0].greeks.iv == pytest.approx(31.5)
    renamed = opend_frame().rename(columns={"option_delta": "delta_x"})
    qs = quotes_from_opend_frame(renamed, T, columns={"delta": "delta_x"})
    assert qs[0].greeks.delta == 0.45


def test_opend_underlying_price_column_wins_over_argument():
    f = opend_frame()
    f["underlying_price"] = 123.0
    assert quotes_from_opend_frame(f, T, underlying_price=100.0)[0].underlying_price == 123.0


def test_opend_frame_feeds_the_recorded_provider_round_trip():
    qs = quotes_from_opend_frame(opend_frame(), T, underlying_price=100.0)
    rows = [dict(ts=T, at="open", underlying=q.contract.underlying, expiry=q.contract.expiry, strike=q.contract.strike,
                 right=q.contract.right.value, bid=q.bid, ask=q.ask, last=q.last, iv=q.greeks.iv, delta=q.greeks.delta)
            for q in qs]
    p = RecordedChainProvider(pd.DataFrame(rows))
    assert p.quote(qs[0].contract, T, "open").greeks.iv == pytest.approx(0.315)


# --- real-data guard ----------------------------------------------------------------------------

def test_paper_and_live_refuse_recorded_and_stand_in_providers():
    rec = RecordedChainProvider(frame())
    for mode in ("paper", "live"):
        with pytest.raises(SyntheticDataRefused):
            require_real_option_data(mode, rec)
        with pytest.raises(SyntheticDataRefused):
            require_real_option_data(mode, "synthetic")
        with pytest.raises(RealDataMissing):
            require_real_option_data(mode, None)
    for mode in ("backtest", "replay"):
        require_real_option_data(mode, rec)
    with pytest.raises(ValueError):
        require_real_option_data("demo", rec)


def test_a_live_provider_passes_in_paper():
    class OpenDChain:                       # stands for the OpenD-backed provider the runtime wires in
        def chain(self, u, ts, at="close"): return []
        def quote(self, c, ts, at="close"): return None
    require_real_option_data("paper", OpenDChain())


def test_bschain_test_helper_is_a_valid_provider():
    data = {"X": bars_from_closes([100.0] * 5)}
    assert isinstance(BSChain(data), OptionDataProvider)
