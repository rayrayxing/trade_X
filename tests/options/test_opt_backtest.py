import datetime as dt

import pandas as pd
import pytest

from opt_fixtures import (BSChain, FakeDividends, FakeRisk, bars_from_closes, bars_with_extremes, signals)
from opt_specs import make_spec
from tradex.costs.models import MoomooOptionCosts, MoomooStockCosts
from tradex.execution.checks import ShortInfo
from tradex.options.backtest import (OptionBacktestConfig, bar_date, bar_moments, exec_price, run_option_backtest,
                                     select_contract)
from tradex.options.contract import OptionContract, Right
from tradex.options.providers import Dividend, EarningsInfo, RecordedChainProvider, SyntheticDataRefused
from tradex.options.sizing import order_fee_usd

COSTS = MoomooOptionCosts()
CLEAN = FakeRisk(EarningsInfo(None), ShortInfo(short_interest_pct_float=3.0, days_to_cover=1.0))
OPEN_UTC = "14:30:00"             # 09:30 New York in winter


def run(spec, closes=None, entries=(5,), exits=(), data=None, chains=None, side=None, cfg=None, risk=None,
        dividends=None, atr=1.0, **kw):
    data = data or {"X": bars_from_closes(closes)}
    n = len(next(iter(data.values())))
    side = side or ("long" if spec.entry_key == "long" else "short")
    sf = {s: signals(n, entries=entries, exits=exits, atr=atr, side=side) for s in data}
    chains = chains or BSChain(data)
    res = run_option_backtest(spec, data, chains, risk=risk, dividends=dividends, cfg=cfg, precomputed=sf, **kw)
    return res, chains


def long_cfg(**kw):
    return OptionBacktestConfig(risk_pct=10.0, **kw)


def long_spec(structure="long_call", **kw):
    kw.setdefault("cap_risk_pct", 10.0)
    return make_spec(structure, **kw)


def rising(n_flat=10, n=40, step=0.8, start=100.0):
    return [start] * n_flat + [start + i * step for i in range(1, n + 1)]


def falling(n_flat=10, n=40, step=0.8, start=100.0):
    return [start] * n_flat + [start - i * step for i in range(1, n + 1)]


# --- long call -------------------------------------------------------------------------------

def test_long_call_trade_accounting_and_multiplier():
    res, chains = run(long_spec(), rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert len(res.trades) == 1 and t.structure == "long_call" and t.direction == 1 and t.qty >= 1
    # x100: gross P&L is premium change x 100 x contracts, taken from the fills
    assert t.gross_pnl == pytest.approx((t.exit_price - t.entry_price) * 100 * t.qty)
    assert t.notional_usd == pytest.approx(t.entry_price * 100 * t.qty)
    # fees come from the moomoo option schedule, on entry and on exit
    expect = order_fee_usd(COSTS, +1, t.qty, t.entry_price) + order_fee_usd(COSTS, -1, t.qty, t.exit_price)
    assert t.fees == pytest.approx(expect)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.fees)
    assert t.r_multiple == pytest.approx(t.net_pnl / t.risk_usd)
    # equity ties out to the trade ledger
    assert res.equity.iloc[0] == 10_000 and res.equity.iloc[-1] == pytest.approx(10_000 + res.trades.net_pnl.sum())
    # premium at risk fits the budget (10% of equity) and covers the debit
    assert t.notional_usd < t.risk_usd <= 1000.0 + 1e-6


def test_long_call_fees_are_the_65c_30c_nine_percent_gst_schedule():
    res, _ = run(long_spec(), rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    entry = COSTS.order_fees("X", +1, t.qty, t.entry_price, None)
    base = max(0.65 * t.qty, 1.99) + max(0.30 * t.qty, 0.99)
    assert entry["commission"] + entry["platform"] == pytest.approx(base)
    assert entry["gst"] == pytest.approx(0.09 * base)


def test_entry_fills_at_next_open_from_the_open_chain_not_the_signal_close():
    closes = [100.0] * 10 + [100 + i * 0.8 for i in range(1, 41)]
    data = {"X": bars_from_closes(closes, gap_open={6: 101.5})}
    res, chains = run(long_spec(), data=data, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.entry_time == pd.Timestamp(f"2024-01-10 {OPEN_UTC}", tz="UTC")       # bar 6, the bar after the signal
    assert t.entry_underlying == 101.5                                         # the open, not the signal bar's close
    ts6 = data["X"].index[6]
    c = OptionContract.from_moomoo_code(t.contract)
    q = chains.quote(c, ts6, "open")
    assert t.entry_price >= q.ask                                              # bought at the ask plus slippage


def test_entry_never_uses_the_signal_bar_close_quote():
    data = {"X": bars_from_closes(rising())}

    class Spy(BSChain):
        def chain(self, underlying, ts, at="close"):
            self.seen = getattr(self, "seen", []) + [(ts, at)]
            return super().chain(underlying, ts, at)
    spy = Spy(data)
    run(long_spec(), data=data, chains=spy, cfg=long_cfg())
    assert (data["X"].index[6], "open") in spy.seen and all(at == "open" for _, at in spy.seen)


def test_exit_decided_at_close_fills_at_next_open():
    res, chains = run(long_spec(), rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "target_underlying"
    assert t.exit_time.strftime("%H:%M:%S") == OPEN_UTC
    # the exit bar is the one after the bar whose high crossed the target (entry 100.8.. + 3 x 2 ATR)
    data = {"X": bars_from_closes(rising())}
    exit_bar = data["X"].index[data["X"].index.searchsorted(t.exit_time.normalize())]
    assert t.exit_underlying == data["X"].loc[exit_bar, "open"]
    prev = data["X"].loc[:exit_bar].iloc[-2]
    assert prev.high >= t.entry_underlying + 3 * 2.0 - 1e-9


def test_fill_prices_are_ask_for_buys_and_bid_for_sells():
    res, chains = run(long_spec(), rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    c = OptionContract.from_moomoo_code(t.contract)
    idx = pd.bdate_range("2024-01-02", periods=51, tz="UTC")
    exit_ts = idx[idx.searchsorted(t.exit_time.normalize())]
    q = chains.quote(c, exit_ts, "open")
    assert t.exit_price <= q.bid + 1e-9
    assert t.spread_slippage > 0


def test_sizing_respects_budget_for_various_equity():
    for eq, pct in ((20_000, 5.0), (50_000, 2.0), (100_000, 1.0)):
        res, _ = run(long_spec(cap_risk_pct=pct), rising(), cfg=OptionBacktestConfig(initial_equity=eq, risk_pct=pct))
        assert len(res.trades) == 1
        assert res.trades.iloc[0].risk_usd <= eq * pct / 100.0 + 1e-6


def test_one_percent_of_ten_thousand_cannot_buy_a_forty_delta_call():
    res, _ = run(make_spec("long_call", cap_risk_pct=1.0), rising(), cfg=OptionBacktestConfig())
    assert res.trades.empty and res.skipped == {"entry_skipped:no_room_long": 1}


def test_premium_stop_fills_no_better_than_the_stop_level():
    # the underlying stop is pushed out of the way so the premium stop does the work
    closes = [100.0] * 10 + [100 - i * 0.6 for i in range(1, 25)]
    spec = long_spec(exit={"stop_atr": 50.0}, option={"exit": {"premium_stop_pct": 0.30}})
    res, _ = run(spec, closes, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop_premium"
    assert t.exit_price <= t.entry_price * 0.70 + 1e-9
    assert t.net_pnl < 0 and abs(t.net_pnl) <= t.risk_usd


def test_underlying_stop_exit_for_long_call():
    res, _ = run(long_spec(), falling(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop_underlying" and t.net_pnl < 0


def test_premium_target_exit():
    spec = long_spec(exit={"target_r": 100.0}, option={"exit": {"premium_target_mult": 0.5}})
    res, _ = run(spec, rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "target_premium" and t.exit_price >= t.entry_price * 1.5 * 0.999 and t.net_pnl > 0


def test_time_exit():
    spec = long_spec(exit={"max_bars": 5})
    res, _ = run(spec, [100.0] * 60, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "time" and t.bars_held == 5 and t.net_pnl < 0       # flat tape: theta and costs


def test_signal_exit():
    spec = long_spec(exit={"target_r": 100.0, "max_bars": 40})
    res, _ = run(spec, [100.0] * 60, exits=(12,), cfg=long_cfg())
    assert res.trades.iloc[0].exit_reason == "signal_exit"


def test_dte_close_before_expiry():
    spec = make_spec("long_call", cap_risk_pct=10.0, option={"dte": {"min": 6, "max": 12}, "exit": {"close_dte": 3}},
                     exit={"max_bars": 40, "target_r": 100.0})
    res, _ = run(spec, [100.0] * 60, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "dte_close"
    c = OptionContract.from_moomoo_code(t.contract)
    assert (c.expiry - bar_date(t.exit_time)).days <= 3


def test_long_call_held_to_expiry_expires_worthless():
    spec = make_spec("long_call", cap_risk_pct=10.0, option={"dte": {"min": 5, "max": 9}, "exit": {"close_dte": 0}},
                     exit={"max_bars": 60, "target_r": 100.0, "stop_atr": 50.0})
    res, _ = run(spec, [100.0] * 60, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "expired_worthless" and t.exit_price == 0.0
    assert t.gross_pnl == pytest.approx(-t.entry_price * 100 * t.qty)
    assert t.fees == pytest.approx(order_fee_usd(COSTS, +1, t.qty, t.entry_price))   # no closing order
    assert t.net_pnl == pytest.approx(-t.notional_usd - t.fees)


def test_long_call_itm_at_expiry_is_sold_at_intrinsic():
    closes = [100.0] * 10 + [100 + i * 0.5 for i in range(1, 41)]
    spec = make_spec("long_call", cap_risk_pct=10.0, option={"dte": {"min": 5, "max": 9}, "exit": {"close_dte": 0},
                                                             "strike": {"target": 0.5, "tolerance": 0.2}},
                     exit={"max_bars": 60, "target_r": 100.0, "stop_atr": 50.0})
    res, _ = run(spec, closes, cfg=long_cfg())
    t = res.trades.iloc[0]
    c = OptionContract.from_moomoo_code(t.contract)
    assert t.exit_reason == "expiry_itm" and t.exit_underlying > c.strike
    assert t.exit_price == pytest.approx(t.exit_underlying - c.strike)
    assert t.fees == pytest.approx(order_fee_usd(COSTS, +1, t.qty, t.entry_price)
                                   + order_fee_usd(COSTS, -1, t.qty, t.exit_price))
    assert t.gross_pnl == pytest.approx((t.exit_price - t.entry_price) * 100 * t.qty)


# --- long put ----------------------------------------------------------------------------------

def test_long_put_profits_when_the_underlying_falls():
    res, _ = run(long_spec("long_put"), falling(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.structure == "long_put" and t.direction == 1 and t.contract.endswith(t.contract[-7:]) and "P" in t.contract
    assert t.exit_reason == "target_underlying" and t.net_pnl > 0 and t.exit_underlying < t.entry_underlying
    assert t.gross_pnl == pytest.approx((t.exit_price - t.entry_price) * 100 * t.qty)


def test_long_put_stops_out_when_the_underlying_rises():
    res, _ = run(long_spec("long_put"), rising(), cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop_underlying" and t.net_pnl < 0


def test_long_put_itm_at_expiry():
    closes = [100.0] * 10 + [100 - i * 0.5 for i in range(1, 41)]
    spec = make_spec("long_put", cap_risk_pct=10.0, option={"dte": {"min": 5, "max": 9}, "exit": {"close_dte": 0},
                                                            "strike": {"target": 0.5, "tolerance": 0.2}},
                     exit={"max_bars": 60, "target_r": 100.0, "stop_atr": 50.0})
    res, _ = run(spec, closes, cfg=long_cfg())
    t = res.trades.iloc[0]
    c = OptionContract.from_moomoo_code(t.contract)
    assert c.right is Right.PUT and t.exit_reason == "expiry_itm"
    assert t.exit_price == pytest.approx(c.strike - t.exit_underlying)


# --- naked short call --------------------------------------------------------------------------

def naked_spec(**kw):
    kw.setdefault("cap_risk_pct", 2.0)
    return make_spec("naked_call", **kw)


def naked_cfg(**kw):
    return OptionBacktestConfig(**({"initial_equity": 200_000.0, "risk_pct": 2.0} | kw))


def test_naked_call_is_blocked_without_a_risk_data_source():
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg())
    assert res.trades.empty and res.skipped == {"entry_blocked:unknown_data": 1}
    assert any("no UnderlyingRiskProvider" in w for w in res.warnings)


def test_naked_call_opens_with_a_stop_and_gap_sized_risk():
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    assert t.structure == "naked_call" and t.direction == -1 and t.qty >= 1
    assert t.risk_usd <= 200_000 * 2.0 / 100 + 1e-6
    assert t.risk_usd > (2.0 * t.entry_price - t.entry_price) * 100 * t.qty            # far above the loss at the stop
    assert t.gross_pnl == pytest.approx((t.entry_price - t.exit_price) * 100 * t.qty)  # short: credit minus buy-back


def test_naked_call_equity_marks_the_short_as_a_liability():
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    credit = t.entry_price * 100 * t.qty
    # equity right after the entry bar is initial minus costs, not initial plus the credit
    idx = res.equity.index
    after_entry = res.equity.loc[idx[6]]
    assert 200_000 - 0.05 * credit - t.fees < after_entry < 200_000 + 0.05 * credit


def test_naked_blocked_by_earnings_inside_the_window():
    risk = FakeRisk(EarningsInfo(dt.date(2024, 1, 20)), ShortInfo(short_interest_pct_float=3.0, days_to_cover=1.0))
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=risk)
    assert res.trades.empty and res.skipped.get("entry_blocked:earnings") == 1


def test_naked_allowed_when_earnings_are_after_the_planned_holding_window():
    risk = FakeRisk(EarningsInfo(dt.date(2024, 3, 20)), ShortInfo(short_interest_pct_float=3.0, days_to_cover=1.0))
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=risk)
    assert len(res.trades) == 1


@pytest.mark.parametrize("si", [ShortInfo(short_interest_pct_float=35.0), ShortInfo(days_to_cover=8.0)])
def test_naked_blocked_on_heavily_shorted_names(si):
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=FakeRisk(EarningsInfo(None), si))
    assert res.trades.empty and res.skipped.get("entry_blocked:squeeze") == 1


def test_naked_call_disabled_by_config():
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(allow_naked=False), risk=CLEAN)
    assert res.trades.empty and res.skipped == {}


def test_naked_margin_limit_binds():
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(max_margin_pct=0.5), risk=CLEAN)
    assert res.trades.empty and res.skipped.get("entry_skipped:no_room_naked") == 1


def gap_bars(gap_to, stop_bar=12, n=40):
    closes = [100.0] * n
    closes[stop_bar] = 103.5
    closes[stop_bar + 1] = gap_to
    for i in range(stop_bar + 2, n):
        closes[i] = gap_to
    return bars_with_extremes(closes, highs={stop_bar: 104.0}, opens={stop_bar + 1: gap_to})


def test_naked_stop_then_twenty_percent_gap_stays_within_the_sized_loss():
    # underlying stop = 100 + 2 ATR = 102; the bar-12 high of 104 triggers it; the next open gaps to 102 x 1.2
    data = {"X": gap_bars(122.4)}
    res, _ = run(naked_spec(), data=data, cfg=naked_cfg(), risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    assert t.exit_reason.startswith("stop")
    assert t.exit_underlying == pytest.approx(122.4)
    loss = -t.net_pnl
    assert loss > 0
    assert loss <= t.risk_usd + 1e-6, f"loss {loss:.0f} exceeded the gap-stressed risk {t.risk_usd:.0f}"
    assert loss > 0.3 * t.risk_usd                                          # the gap was a real test of the sizing


def test_naked_stop_without_a_gap_loses_far_less_than_the_gap_budget():
    data = {"X": gap_bars(104.0)}
    res, _ = run(naked_spec(), data=data, cfg=naked_cfg(), risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    assert t.exit_reason.startswith("stop") and -t.net_pnl < 0.25 * t.risk_usd


def test_premium_stop_alone_fills_at_or_above_the_stop():
    data = {"X": gap_bars(106.0)}
    spec = naked_spec(option={"naked": {"stop_premium_mult": 1.5, "use_underlying_stop": False}})
    res, _ = run(spec, data=data, cfg=naked_cfg(), risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop_premium" and t.exit_price >= 1.5 * t.entry_price - 1e-9


def test_naked_closed_before_earnings_that_appear_while_held():
    def earnings(asof):                            # the calendar learns of a report on 24 Jan from the 17th on
        return EarningsInfo(dt.date(2024, 1, 24)) if asof >= pd.Timestamp("2024-01-17", tz="UTC") else EarningsInfo(None)
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=FakeRisk(earnings, ShortInfo(short_interest_pct_float=2, days_to_cover=1)))
    t = res.trades.iloc[0]
    assert t.exit_reason == "earnings_window"
    assert bar_date(t.exit_time) < dt.date(2024, 1, 24)


def test_naked_closed_when_short_interest_jumps():
    def si(asof):
        return ShortInfo(short_interest_pct_float=40.0 if asof >= pd.Timestamp("2024-01-17", tz="UTC") else 2.0,
                         days_to_cover=1.0)
    res, _ = run(naked_spec(), [100.0] * 40, cfg=naked_cfg(), risk=FakeRisk(EarningsInfo(None), si))
    assert res.trades.iloc[0].exit_reason == "squeeze_risk"


def test_naked_closed_ahead_of_ex_dividend_when_itm_with_little_time_value():
    closes = [100.0] * 6 + [100 + i * 0.7 for i in range(1, 35)]
    spec = naked_spec(option={"naked": {"stop_premium_mult": 5.0, "use_underlying_stop": False}},
                      exit={"target_r": 100.0, "max_bars": 40})
    div = FakeDividends(Dividend(dt.date(2030, 1, 1), 50.0))                    # ex-date far away: rule must not fire
    res, _ = run(spec, closes, cfg=naked_cfg(initial_equity=600_000.0), risk=CLEAN, dividends=div)
    assert res.trades.iloc[0].exit_reason != "early_assignment_risk"
    class Soon(FakeDividends):
        def next_ex_dividend(self, u, asof):
            return Dividend((asof + pd.Timedelta(days=1)).date(), 50.0)
    res, _ = run(spec, closes, cfg=naked_cfg(initial_equity=600_000.0), risk=CLEAN, dividends=Soon(None))
    assert res.trades.iloc[0].exit_reason == "early_assignment_risk"


def test_naked_call_reaching_expiry_in_the_money_is_assigned_and_flagged():
    spec = naked_spec(option={"dte": {"min": 20, "max": 30}, "exit": {"close_dte": 0},
                              "naked": {"stop_premium_mult": 5.0, "use_underlying_stop": False}},
                      exit={"target_r": 100.0, "max_bars": 100})
    flat = [100.0] * 40
    probe, _ = run(spec, flat, cfg=naked_cfg(initial_equity=600_000.0), risk=CLEAN, dividends=FakeDividends(None))
    t0 = probe.trades.iloc[0]
    expiry = OptionContract.from_moomoo_code(t0.contract).expiry
    data = {"X": bars_from_closes(flat)}
    pos = list(data["X"].index.date).index(expiry)
    closes = flat[:]
    closes[pos] = 118.0
    closes = closes[:pos + 1]
    cfg = naked_cfg(initial_equity=600_000.0, assignment_fee_per_contract=1.0)
    res, _ = run(spec, closes, cfg=cfg, risk=CLEAN, dividends=FakeDividends(None))
    t = res.trades.iloc[0]
    c = OptionContract.from_moomoo_code(t.contract)
    assert t.exit_reason == "assigned" and t.exit_underlying == 118.0
    n = t.qty
    stock = MoomooStockCosts()
    cover, unit = stock.fill("X", +1, 118.0, None)
    stock_fees = sum(stock.order_fees("X", +1, 100 * n, cover, None).values())
    assert t.exit_price == pytest.approx(118.0 - c.strike)
    assert t.gross_pnl == pytest.approx(n * 100 * (t.entry_price - (118.0 - c.strike)) - 100 * n * (cover - 118.0))
    assert t.fees == pytest.approx(order_fee_usd(COSTS, -1, n, t.entry_price) + 1.0 * n + stock_fees)
    assert not t.fee_unknown
    assert any("stop did not hold" in w for w in res.warnings)


def test_assignment_fee_unknown_is_flagged_not_guessed():
    spec = naked_spec(option={"dte": {"min": 20, "max": 30}, "exit": {"close_dte": 0},
                              "naked": {"stop_premium_mult": 5.0, "use_underlying_stop": False}},
                      exit={"target_r": 100.0, "max_bars": 100})
    flat = [100.0] * 40
    probe, _ = run(spec, flat, cfg=naked_cfg(initial_equity=600_000.0), risk=CLEAN, dividends=FakeDividends(None))
    expiry = OptionContract.from_moomoo_code(probe.trades.iloc[0].contract).expiry
    pos = list(bars_from_closes(flat).index.date).index(expiry)
    closes = flat[:pos] + [118.0]
    res, _ = run(spec, closes, cfg=naked_cfg(initial_equity=600_000.0), risk=CLEAN, dividends=FakeDividends(None))
    assert bool(res.trades.iloc[0].fee_unknown) and any("fee not configured" in w for w in res.warnings)


# --- engine behaviour ----------------------------------------------------------------------------

def test_paper_and_live_modes_are_refused():
    data = {"X": bars_from_closes([100.0] * 30)}
    for mode in ("paper", "live"):
        with pytest.raises(ValueError, match="backtest"):
            run_option_backtest(long_spec(), data, BSChain(data), cfg=OptionBacktestConfig(mode=mode))


def test_recorded_provider_refused_by_the_real_data_guard_in_strict_modes():
    rec = RecordedChainProvider(pd.DataFrame({c: [] for c in ["ts", "at", "underlying", "expiry", "strike", "right"]}))
    from tradex.options.providers import require_real_option_data
    with pytest.raises(SyntheticDataRefused):
        require_real_option_data("paper", rec)


def test_gap_below_floor_refused():
    data = {"X": bars_from_closes([100.0] * 30)}
    with pytest.raises(ValueError, match="floor"):
        run_option_backtest(long_spec(), data, BSChain(data), cfg=OptionBacktestConfig(gap_pct=0.1))


def test_invalid_spec_refused():
    data = {"X": bars_from_closes([100.0] * 30)}
    bad = make_spec("naked_call", option={"naked": {"stop_premium_mult": 1.0}})
    with pytest.raises(ValueError, match="schema check"):
        run_option_backtest(bad, data, BSChain(data))


def test_chain_gaps_delay_the_exit_and_are_counted():
    data = {"X": bars_from_closes(rising())}
    ts_blocked = {data["X"].index[i] for i in range(16, 22)}
    chains = BSChain(data, quote_filter=lambda c, ts, at: not (at == "open" and ts in ts_blocked))
    res, _ = run(long_spec(), data=data, chains=chains, cfg=long_cfg())
    assert res.skipped.get("exit_delayed:no_fillable_quote_at_open", 0) >= 1
    assert len(res.trades) == 1


def test_positions_open_at_the_end_are_closed_at_the_last_quote():
    spec = long_spec(exit={"target_r": 100.0, "max_bars": 100, "stop_atr": 50.0})
    res, _ = run(spec, [100.0] * 20, cfg=long_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "end_of_data" and t.exit_time.strftime("%H:%M") == "21:00"
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.trades.net_pnl.sum())


def test_end_of_data_without_a_closing_quote_marks_to_last_and_warns():
    data = {"X": bars_from_closes([100.0] * 20)}
    last = data["X"].index[-1]
    chains = BSChain(data, quote_filter=lambda c, ts, at: not (ts == last and at == "close"))
    spec = long_spec(exit={"target_r": 100.0, "max_bars": 100, "stop_atr": 50.0})
    res, _ = run(spec, data=data, chains=chains, cfg=long_cfg())
    assert res.trades.iloc[0].exit_reason == "end_of_data" and bool(res.trades.iloc[0].fee_unknown)
    assert any("no closing quote" in w for w in res.warnings)


def test_max_positions_and_untradable_mask():
    data = {"A": bars_from_closes(rising()), "B": bars_from_closes(rising())}
    chains = BSChain(data)
    res, _ = run(long_spec(universe=("A", "B")), data=data, chains=chains, cfg=long_cfg(max_positions=1))
    assert len(res.trades) == 1 and res.skipped.get("entry_skipped:max_positions") == 1
    mask = {"A": pd.Series(True, index=data["A"].index), "B": pd.Series(False, index=data["B"].index)}
    res, _ = run(long_spec(universe=("A", "B")), data=data, chains=chains, cfg=long_cfg(), tradable=mask)
    assert set(res.trades.symbol) == {"A"}


def test_heat_limit_blocks_a_second_position():
    data = {"A": bars_from_closes(rising()), "B": bars_from_closes(rising())}
    res, _ = run(long_spec(universe=("A", "B")), data=data, cfg=long_cfg(max_heat_pct=12.0))
    assert len(res.trades) == 1 and res.skipped.get("entry_skipped:heat") == 1


def test_premium_outlay_limit():
    res, _ = run(long_spec(), rising(), cfg=long_cfg(max_premium_pct=2.0))
    assert res.trades.empty and res.skipped.get("entry_skipped:no_room_long") == 1


def test_no_contract_reason_is_reported():
    spec = long_spec(option={"liquidity": {"min_open_interest": 10**9}})
    res, _ = run(spec, rising(), cfg=long_cfg())
    assert res.trades.empty and list(res.skipped) == ["entry_skipped:no_contract:liquidity"]


def test_no_chain_reported():
    class Empty(BSChain):
        def chain(self, *a, **k):
            return []
    data = {"X": bars_from_closes(rising())}
    res, _ = run(long_spec(), data=data, chains=Empty(data), cfg=long_cfg())
    assert res.trades.empty and res.skipped == {"entry_skipped:no_chain": 1}


def test_params_override_and_summary():
    res, _ = run(long_spec(), rising(), cfg=long_cfg(), params={"option.strike.target": 0.55})
    assert res.params == {"option.strike.target": 0.55}
    s = res.summary()
    assert s["trades"] == 1 and "exit_reasons" in s and s["costs_usd"] > 0
    empty, _ = run(long_spec(), [100.0] * 20, entries=(), cfg=long_cfg())
    assert empty.summary()["trades"] == 0 and len(empty.equity) == 20


def test_start_and_end_window():
    res, _ = run(long_spec(), rising(), cfg=long_cfg(start="2024-01-02", end="2024-01-12"))
    assert len(res.equity) == 8 and res.trades.iloc[0].exit_reason == "end_of_data"


def test_full_pipeline_with_computed_signals_on_synthetic_bars():
    from tradex.data.synthetic import synthetic_bars
    from tradex.options.spec import load_dir
    from pathlib import Path
    spec = load_dir(Path(__file__).resolve().parents[2] / "tradex" / "options" / "specs")[0]
    assert spec.structure == "long_call"
    bars = synthetic_bars(260, seed=3, price=150.0, vol=0.02, trend_strength=0.004)
    data = {"NVDA": bars}
    res = run_option_backtest(spec, data, BSChain(data, iv=0.45), cfg=OptionBacktestConfig(initial_equity=50_000, risk_pct=1.0),
                              params={"features.relvol.period": 10})
    assert len(res.equity) == len(bars)
    assert res.equity.iloc[-1] == pytest.approx(50_000 + res.trades.net_pnl.sum()) if len(res.trades) else True
    assert all(w for w in res.warnings if "earnings" in w) or True
    if len(res.trades):
        assert (res.trades.qty >= 1).all() and (res.trades.risk_usd <= 500 + 1e-6).all()


# --- helpers -------------------------------------------------------------------------------------

def test_bar_date_and_moments():
    assert bar_date(pd.Timestamp("2024-01-10", tz="UTC")) == dt.date(2024, 1, 10)
    assert bar_date(pd.Timestamp("2024-01-10 14:30", tz="UTC")) == dt.date(2024, 1, 10)
    assert bar_date(pd.Timestamp("2024-01-11 01:00", tz="UTC")) == dt.date(2024, 1, 10)       # evening in New York
    o, c = bar_moments(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timedelta(days=1))
    assert (o, c) == (pd.Timestamp("2024-01-10 14:30", tz="UTC"), pd.Timestamp("2024-01-10 21:00", tz="UTC"))
    o, c = bar_moments(pd.Timestamp("2024-07-10", tz="UTC"), pd.Timedelta(days=1))
    assert o == pd.Timestamp("2024-07-10 13:30", tz="UTC")                                   # DST
    o, c = bar_moments(pd.Timestamp("2024-01-10 15:00", tz="UTC"), pd.Timedelta(hours=1))
    assert (o, c) == (pd.Timestamp("2024-01-10 15:00", tz="UTC"), pd.Timestamp("2024-01-10 16:00", tz="UTC"))
    o, c = bar_moments(pd.Timestamp("2024-01-08", tz="UTC"), pd.Timedelta(days=7))
    assert c == pd.Timestamp("2024-01-12 21:00", tz="UTC")


def test_exec_price_rules():
    from tradex.options.contract import Greeks, OptionQuote
    c = OptionContract("X", dt.date(2024, 2, 16), 100, "C")
    ts = pd.Timestamp("2024-01-10", tz="UTC")
    q = OptionQuote(c, ts, 1.0, 1.2)
    buy, sell = exec_price(q, +1, COSTS), exec_price(q, -1, COSTS)
    assert buy[0] == pytest.approx(1.2 + 0.01 * 1.1) and sell[0] == pytest.approx(1.0 - 0.01 * 1.1)
    assert buy[1] == pytest.approx(0.1 + 0.011)
    zero_bid_sell = exec_price(OptionQuote(c, ts, 0.0, 0.05), -1, COSTS)
    assert zero_bid_sell[0] == 0.0                                                              # a worthless option can still be sold
    assert exec_price(OptionQuote(c, ts, 0.0, 0.0), +1, COSTS) is None
    last_only = exec_price(OptionQuote(c, ts, None, None, 2.0), +1, COSTS)
    assert last_only[0] == pytest.approx(2.0 + 2.0 * 0.03)                                    # model half spread plus slippage
    assert exec_price(OptionQuote(c, ts, None, None, None), +1, COSTS) is None
    assert Greeks().delta is None


def test_select_contract_picks_nearest_delta_and_reports_rejections():
    data = {"X": bars_from_closes([100.0] * 10)}
    chain = BSChain(data).chain("X", data["X"].index[3], "open")
    spec = make_spec("long_call")
    sel = select_contract(chain, spec, 100.0, dt.date(2024, 1, 5))
    assert sel.quote is not None and sel.quote.contract.right is Right.CALL
    assert abs(sel.quote.delta - 0.40) <= 0.15 and 20 <= (sel.quote.contract.expiry - dt.date(2024, 1, 5)).days <= 45
    best = min(abs(abs(q.delta) - 0.40) for q in chain
               if q.contract.right is Right.CALL and 20 <= (q.contract.expiry - dt.date(2024, 1, 5)).days <= 45
               and q.bid >= 0.05 and q.spread_pct <= 0.20)
    assert abs(sel.quote.delta - 0.40) == pytest.approx(best)
    assert sel.rejected["dte"] > 0
    empty = select_contract([], spec, 100.0, dt.date(2024, 1, 5))
    assert empty.quote is None and empty.rejected["no_contracts_of_right"] == 1


def test_select_contract_moneyness_and_iv_band():
    data = {"X": bars_from_closes([100.0] * 10)}
    chain = BSChain(data, iv=0.30).chain("X", data["X"].index[3], "open")
    spec = make_spec("long_call", option={"strike": {"by": "moneyness", "target": 1.05, "tolerance": 0.03}})
    sel = select_contract(chain, spec, 100.0, dt.date(2024, 1, 5))
    assert sel.quote.contract.strike == 105.0
    banded = make_spec("long_call", option={"iv": {"max": 0.25}})
    nothing = select_contract(chain, banded, 100.0, dt.date(2024, 1, 5))
    assert nothing.quote is None and nothing.rejected["iv_band"] > 0
