"""Unit tests for the research feature primitives behind the proposed strategies.

Fixtures are synthetic by construction; nothing here is data a strategy may run on.
"""
import numpy as np
import pandas as pd
import pytest

from tradex.data.synthetic import synthetic_bars
from tradex.research import carry, events, regime, rel_strength, sources

NY = "America/New_York"


def ny_daily(n=300, seed=0, start="2023-01-02", price=100.0):
    """Synthetic daily bars stamped at New York midnight, as OpenD stamps them."""
    b = synthetic_bars(n, seed=seed, price=price)
    b.index = pd.bdate_range(start, periods=n, tz=NY).tz_convert("UTC").astype("datetime64[ns, UTC]")
    return b


def flat_daily(n=200, price=100.0, volume=1e6, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=n, tz=NY).tz_convert("UTC").astype("datetime64[ns, UTC]")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": volume}, index=idx)


def session_date(bars, pos):
    return events.session_dates(bars.index)[pos]


def announce(bars, pos, when="bmo"):
    return pd.DataFrame({"date": [session_date(bars, pos)], "when": [when]})


# --- sources ---------------------------------------------------------------------------------

def test_earnings_calendar_reads_csv_and_reports_missing(tmp_path):
    (tmp_path / "AAA.csv").write_text("date,when\n2024-02-01,AMC\n2024-05-02,\n2024-02-01,amc\n")
    cal = sources.CsvEarningsCalendar(tmp_path)
    a = cal.announcements("AAA")
    assert list(a["when"]) == ["amc", "unknown"] and a["date"].dt.tz is None
    assert cal.has("AAA") and not cal.has("BBB")
    with pytest.raises(sources.DataUnavailable):
        cal.announcements("BBB")
    (tmp_path / "CCC.csv").write_text("date,when\n2024-02-01,evening\n")
    with pytest.raises(ValueError, match="when"):
        cal.announcements("CCC")
    assert isinstance(cal, sources.EarningsCalendar)


def test_event_calendar_reads_csv_and_the_repo_yaml(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("time\n2024-03-20T18:00:00Z\n2024-01-31T19:00:00Z\n")
    t = sources.FileEventCalendar(p).times()
    assert list(t) == [pd.Timestamp("2024-01-31 19:00", tz="UTC"), pd.Timestamp("2024-03-20 18:00", tz="UTC")]
    y = tmp_path / "cb.yaml"
    y.write_text("events:\n- {time: '2026-01-28T19:00:00Z', kind: fomc}\n- {time: '2026-02-05T13:15:00Z', kind: ecb}\n")
    assert list(sources.FileEventCalendar(y, kind="fomc").times()) == [pd.Timestamp("2026-01-28 19:00", tz="UTC")]
    with pytest.raises(sources.DataUnavailable):
        sources.FileEventCalendar(tmp_path / "none.csv").times()
    with pytest.raises(sources.DataUnavailable):
        sources.FileEventCalendar(y, kind="boj").times()


def test_rate_source_is_a_causal_step_function(tmp_path):
    p = tmp_path / "rates.csv"
    p.write_text("# comment\ndate,currency,rate\n2024-01-10,USD,5.0\n2024-03-01,USD,4.5\n2024-01-01,JPY,0.0\n")
    src = sources.CsvRateSource(p)
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-10", "2024-02-15", "2024-03-01", "2024-04-01"], tz="UTC")
    r = src.rates("USD", idx)
    assert np.isnan(r.iloc[0]) and r.iloc[1:].tolist() == [0.05, 0.05, 0.045, 0.045]
    assert src.currencies() == ["JPY", "USD"]
    with pytest.raises(sources.DataUnavailable):
        src.rates("EUR", idx)
    with pytest.raises(sources.DataUnavailable):
        sources.CsvRateSource(tmp_path / "none.csv").rates("USD", idx)
    assert isinstance(src, sources.RateSource)


# --- earnings events -------------------------------------------------------------------------

def test_reaction_session_follows_the_announcement_timing():
    b = flat_daily()
    assert events.reaction_sessions(b, announce(b, 100, "bmo")).iloc[0].tolist() == [100, 100, 100]
    assert events.reaction_sessions(b, announce(b, 100, "amc")).iloc[0].tolist() == [100, 101, 101]
    big_first = b.copy()
    big_first.loc[b.index[100]:, ["open", "high", "low", "close"]] = 110.0
    big_second = b.copy()
    big_second.loc[b.index[101]:, ["open", "high", "low", "close"]] = 110.0
    # unknown timing: the bigger mover is the reaction, but it is only known once both sessions closed
    assert events.reaction_sessions(big_first, announce(b, 100, "unknown")).iloc[0].tolist() == [100, 100, 101]
    assert events.reaction_sessions(big_second, announce(b, 100, "unknown")).iloc[0].tolist() == [100, 101, 101]
    assert events.reaction_sessions(b, announce(b, 199, "unknown")).empty      # next session not in the data yet
    assert events.reaction_sessions(b, announce(b, 0, "bmo")).empty            # no prior close to measure a reaction


def _jump_bars():
    b = flat_daily()
    b.loc[b.index[100]:, ["open", "high", "low", "close"]] = 110.0
    b.iloc[100, b.columns.get_loc("open")] = 108.0   # gaps up 8%, closes at +10%
    b.iloc[100, b.columns.get_loc("volume")] = 4e6
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(0, 0.01, len(b)), index=b.index)
    for c in ("open", "high", "low", "close"):
        b[c] = b[c] * np.exp(noise.cumsum().where(b.index < b.index[100], noise.cumsum().iloc[99]))
    return b


def test_earnings_columns_values():
    b = _jump_bars()
    cols = events.earnings_columns(b, announce(b, 100, "bmo"))
    r = b["close"].pct_change()
    assert cols["ed_age"].iloc[:100].isna().all() and cols["ed_age"].iloc[100] == 0 and cols["ed_age"].iloc[105] == 5
    assert cols["ed_ret"].iloc[100] == pytest.approx(r.iloc[100]) == cols["ed_ret"].iloc[150]
    assert cols["ed_gap"].iloc[100] == pytest.approx(b["open"].iloc[100] / b["close"].iloc[99] - 1)
    assert cols["ed_z"].iloc[100] == pytest.approx(r.iloc[100] / r.iloc[40:100].std())
    assert cols["ed_volx"].iloc[100] == pytest.approx(4e6 / 1e6)
    assert cols["ed_post"].iloc[105] == pytest.approx(b["close"].iloc[105] / b["close"].iloc[100] - 1)
    assert cols["ed_days_to"].iloc[97] == 3 and cols["ed_days_to"].iloc[100] == 0 and np.isnan(cols["ed_days_to"].iloc[101])


def test_earnings_columns_subtract_the_benchmark_and_hold_until_the_next_event():
    b = _jump_bars()
    bench = b.copy()
    bench["close"] = 100.0
    bench.loc[bench.index[100]:, "close"] = 102.0                    # market up 2% on the reaction day
    two = pd.DataFrame({"date": [session_date(b, 100), session_date(b, 160)], "when": ["bmo", "bmo"]})
    cols = events.earnings_columns(b, two, benchmark=bench)
    assert cols["ed_ret"].iloc[100] == pytest.approx(b["close"].pct_change().iloc[100] - 0.02)
    assert cols["ed_age"].iloc[159] == 59 and cols["ed_age"].iloc[160] == 0      # restarts at the next reaction
    assert cols["ed_days_to"].iloc[101] == 59


def test_earnings_columns_are_causal():
    b = _jump_bars()
    ann = pd.DataFrame({"date": [session_date(b, 100), session_date(b, 120)], "when": ["bmo", "amc"]})
    full = events.earnings_columns(b, ann).drop(columns="ed_days_to")      # days_to reads the schedule, by design
    part = events.earnings_columns(b.iloc[:150], ann).drop(columns="ed_days_to")
    pd.testing.assert_frame_equal(full.iloc[:150], part)
    cut = events.earnings_columns(b.iloc[:110], ann).drop(columns="ed_days_to")   # the second event is not yet in the data
    pd.testing.assert_frame_equal(full.iloc[:110], cut)


# --- announcement windows --------------------------------------------------------------------

def h1_sessions(days, start="2024-01-02", seed=0, skip=()):
    d = [x for x in pd.bdate_range(start, periods=days) if x.date().isoformat() not in skip]
    stamps = [pd.Timestamp(f"{x.date()} {h:02d}:30", tz=NY) for x in d for h in range(9, 16)]
    b = synthetic_bars(len(stamps), tf="H1", seed=seed)
    b.index = pd.DatetimeIndex(stamps).tz_convert("UTC").astype("datetime64[ns, UTC]")
    return b


FOMC = pd.DatetimeIndex([pd.Timestamp("2024-01-31 14:00", tz=NY).tz_convert("UTC")])


def test_event_window_marks_the_bars_around_the_24_hours_before():
    b = h1_sessions(40)
    w = events.event_window_columns(b, FOMC, pd.Timedelta(hours=1))
    enter, leave = w.index[w["evt_enter"] > 0], w.index[w["evt_leave"] > 0]
    # the last bars closing at or before 14:00 on the day before and on the day itself; fills at 13:30 next open
    assert [t.tz_convert(NY).strftime("%Y-%m-%d %H:%M") for t in enter] == ["2024-01-30 12:30"]
    assert [t.tz_convert(NY).strftime("%Y-%m-%d %H:%M") for t in leave] == ["2024-01-31 12:30"]


def test_event_window_skips_events_with_a_data_hole():
    b = h1_sessions(40, skip=("2024-01-30",))
    w = events.event_window_columns(b, FOMC, pd.Timedelta(hours=1))
    assert w["evt_enter"].sum() == 0          # no entry without the bars it would fill on (a lone exit flag is harmless)
    outside = pd.DatetimeIndex([pd.Timestamp("2030-01-31 19:00", tz="UTC")])
    assert events.event_window_columns(h1_sessions(40), outside, pd.Timedelta(hours=1)).sum().sum() == 0


def test_event_window_is_causal_in_the_bars():
    b = h1_sessions(40)
    full = events.event_window_columns(b, FOMC, pd.Timedelta(hours=1))
    cut = b.index.get_loc(pd.Timestamp("2024-01-31 10:30", tz=NY).tz_convert("UTC"))
    part = events.event_window_columns(b.iloc[:cut], FOMC, pd.Timedelta(hours=1))
    pd.testing.assert_frame_equal(full.iloc[:cut], part)           # the entry flag is already there before the exit bar exists
    assert part["evt_enter"].sum() == 1 and part["evt_leave"].sum() == 0
    before_target = b.index.get_loc(pd.Timestamp("2024-01-30 11:30", tz=NY).tz_convert("UTC"))
    assert events.event_window_columns(b.iloc[:before_target], FOMC, pd.Timedelta(hours=1)).sum().sum() == 0


# --- regime ----------------------------------------------------------------------------------

def test_expanding_percentile_is_causal_and_ranks_against_history():
    s = pd.Series(np.arange(1.0, 401.0))
    p = regime.expanding_percentile(s, min_periods=252)
    assert p.iloc[:251].isna().all() and (p.iloc[251:] == 1.0).all()
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=600))
    full, part = regime.expanding_percentile(x), regime.expanding_percentile(x.iloc[:400])
    pd.testing.assert_series_equal(full.iloc[:400], part)
    assert full.iloc[-1] == pytest.approx((x <= x.iloc[-1]).mean())


def _basket(common_weight, n=400, k=6, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 1))
    r = common_weight * f + (1 - common_weight) * rng.normal(size=(n, k))
    return pd.DataFrame(r, index=pd.bdate_range("2022-01-03", periods=n), columns=list("ABCDEF")[:k])


def test_absorption_ratio_and_correlation_separate_coupled_from_independent_baskets():
    coupled, free = _basket(0.95), _basket(0.0)
    assert regime.absorption_ratio(coupled).dropna().iloc[-1] > 0.9
    assert regime.absorption_ratio(free).dropna().iloc[-1] < 0.4
    assert regime.avg_pairwise_corr(coupled).dropna().iloc[-1] > 0.8
    assert abs(regime.avg_pairwise_corr(free).dropna().iloc[-1]) < 0.15
    ar = regime.absorption_ratio(coupled)
    pd.testing.assert_series_equal(ar.iloc[:350], regime.absorption_ratio(coupled.iloc[:350]))
    assert ar.iloc[:249].isna().all()


def test_absorption_shift_is_positive_when_the_ratio_jumps():
    ar = pd.Series(np.r_[np.full(300, 0.5) + np.random.default_rng(0).normal(0, 0.01, 300), np.full(15, 0.8)])
    shift = regime.absorption_shift(ar)
    assert shift.iloc[-1] > 2 and abs(shift.iloc[299]) < 3


def test_risk_on_score_counts_the_four_conditions():
    idx = pd.bdate_range("2022-01-03", periods=400)
    up = pd.Series(np.linspace(100, 200, 400), index=idx)
    flat = pd.Series(100.0, index=idx)
    low_vol = pd.Series(0.2, index=idx)
    assert regime.risk_on_score(up, flat, flat, low_vol).iloc[-1] == 4
    assert regime.risk_on_score(up, flat, flat, pd.Series(0.9, index=idx)).iloc[-1] == 3
    down = pd.Series(np.linspace(200, 100, 400), index=idx)
    assert regime.risk_on_score(down, flat, flat, pd.Series(0.9, index=idx)).iloc[-1] == 0
    s = regime.risk_on_score(up, flat, flat, low_vol)
    assert s.iloc[:199].isna().all()                      # needs the 200-day mean
    assert regime.risk_on_score(up, flat).iloc[-1] == 2   # without gold and volatility only two components exist


def test_momentum_crash_state_needs_a_bear_market_and_high_volatility():
    idx = pd.bdate_range("2018-01-01", periods=700)
    falling = pd.Series(np.linspace(200, 100, 700), index=idx)
    rising = pd.Series(np.linspace(100, 200, 700), index=idx)
    hi, lo = pd.Series(0.95, index=idx), pd.Series(0.3, index=idx)
    assert regime.momentum_crash_state(falling, hi).iloc[-1] == 1.0
    assert regime.momentum_crash_state(falling, lo).iloc[-1] == 0.0
    assert regime.momentum_crash_state(rising, hi).iloc[-1] == 0.0
    assert regime.momentum_crash_state(falling, hi).iloc[:500].isna().all()


def _closes(n=900):
    names = ["SPY", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV", "XLY"]
    return {s: synthetic_bars(n, seed=i, start="2018-01-01")["close"] for i, s in enumerate(names)}


def test_regime_columns_shape_and_causality():
    closes = _closes()
    full = regime.regime_columns(closes, basket=["XLK", "XLF", "XLE", "XLV", "XLY"])
    assert list(full.columns) == regime.REGIME_COLUMNS
    assert full["rg_vol_pct"].dropna().between(0, 1).all() and full["rg_score"].dropna().between(0, 4).all()
    assert full["rg_dd"].dropna().le(0).all() and full["rg_absorb"].dropna().between(0, 1).all()
    cut = 700
    part = regime.regime_columns({s: c.iloc[:cut] for s, c in closes.items()}, basket=["XLK", "XLF", "XLE", "XLV", "XLY"])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)
    with pytest.raises(KeyError):
        regime.regime_columns({"SPY": closes["SPY"]})


# --- relative strength -----------------------------------------------------------------------

def test_rs_momentum_matches_the_ratio_formula_and_skips_the_latest_bars():
    stock, bench = synthetic_bars(400, seed=1)["close"], synthetic_bars(400, seed=2)["close"]
    got = rel_strength.rs_momentum(stock, bench, n=231, skip=21)
    rs = stock / bench
    assert got.iloc[-1] == pytest.approx(rs.iloc[-1 - 21] / rs.iloc[-1 - 21 - 231] - 1)
    assert got.iloc[:252].isna().all() and got.iloc[252:].notna().all()
    assert rel_strength.rs_momentum(stock, bench, 63, 0).iloc[-1] == pytest.approx(rs.iloc[-1] / rs.iloc[-64] - 1)
    cut = 300
    pd.testing.assert_series_equal(got.iloc[:cut], rel_strength.rs_momentum(stock.iloc[:cut], bench.iloc[:cut], 231, 21))
    assert rel_strength.rs_trend(stock, bench, 50).iloc[-1] == pytest.approx(rs.iloc[-1] / rs.iloc[-50:].mean() - 1)
    with pytest.raises(ValueError):
        rel_strength.rs_momentum(stock, bench, 0)


def test_currency_strength_signs_base_up_and_quote_down():
    idx = pd.bdate_range("2024-01-01", periods=3)
    closes = {"EUR_USD": pd.Series([1.0, 1.0, 1.1], index=idx), "USD_JPY": pd.Series([100.0, 100.0, 100.0], index=idx)}
    s = rel_strength.currency_strength(closes, n=1)
    up = np.log(1.1)
    assert s["EUR"].iloc[-1] == pytest.approx(up)
    assert s["USD"].iloc[-1] == pytest.approx((-up + 0.0) / 2)
    assert s["JPY"].iloc[-1] == pytest.approx(0.0)
    assert rel_strength.pair_strength_diff("EUR_USD", s).iloc[-1] == pytest.approx(up + up / 2)
    assert s.iloc[0].isna().all()
    cut = rel_strength.currency_strength({k: v.iloc[:2] for k, v in closes.items()}, n=1)
    pd.testing.assert_frame_equal(s.iloc[:2], cut)


# --- carry -----------------------------------------------------------------------------------

class FixtureRates:
    """A RateSource in the test: percent steps by effective date."""

    def __init__(self, steps):
        self.steps = steps

    def rates(self, ccy, index):
        s = pd.Series({pd.Timestamp(d, tz="UTC"): v / 100 for d, v in self.steps[ccy].items()})
        return s.reindex(s.index.union(index)).ffill().reindex(index)


def test_carry_columns_follow_the_rate_differential_and_are_causal():
    bars = ny_daily(400, seed=5, start="2022-01-03", price=1.1)
    src = FixtureRates({"EUR": {"2022-01-01": 0.0, "2022-09-01": 2.0}, "USD": {"2022-01-01": 1.0, "2022-06-01": 3.0}})
    assert isinstance(src, sources.RateSource)
    cols = carry.carry_columns(bars, "EUR_USD", src)
    assert list(cols.columns) == carry.CARRY_COLUMNS
    before, mid, after = (cols["carry"][bars.index >= pd.Timestamp(d, tz="UTC")].iloc[0] for d in ("2022-02-01", "2022-07-01", "2022-10-03"))
    assert (before, mid, after) == pytest.approx((-0.01, -0.03, -0.01))
    t = bars.index[bars.index >= pd.Timestamp("2022-09-01", tz="UTC")][0]
    assert cols["carry"].loc[t] == pytest.approx(-0.01) and cols["carry"].shift(1).loc[t] == pytest.approx(-0.03)
    assert cols["rate_chg"].iloc[-1] == pytest.approx(cols["carry"].iloc[-1] - cols["carry"].iloc[-1 - 63])
    trend = bars["close"].pct_change(100)
    agree = cols["carry_trend"].dropna()
    assert set(agree.unique()) <= {-1.0, 0.0, 1.0}
    assert ((agree == -1.0) == ((cols["carry"].loc[agree.index] < 0) & (trend.loc[agree.index] < 0))).all()
    assert cols["fxvol_pct"].dropna().between(0, 1).all()
    part = carry.carry_columns(bars.iloc[:300], "EUR_USD", src)
    pd.testing.assert_frame_equal(cols.iloc[:300], part)


def test_carry_columns_propagate_missing_rates_as_nan():
    bars = ny_daily(120, seed=6, start="2022-01-03", price=1.1)
    src = FixtureRates({"EUR": {"2022-03-01": 1.0}, "USD": {"2022-01-01": 1.0}})
    cols = carry.carry_columns(bars, "EUR_USD", src)
    assert cols["carry"][bars.index < pd.Timestamp("2022-03-01", tz="UTC")].isna().all()
    assert cols["carry_trend"][bars.index < pd.Timestamp("2022-03-01", tz="UTC")].isna().all()
