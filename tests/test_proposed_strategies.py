"""The proposed strategies: specs validate, builders produce every column a spec reads,
missing inputs report "needs data", and the two event strategies trade where they should.

All bars and calendars here are synthetic test fixtures written to tmp_path; the builders
read them exactly as they read real caches on the machine that runs the gate.
"""
import numpy as np
import pandas as pd
import pytest

from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.backtest.validation import Thresholds, WalkForwardConfig
from tradex.data.providers import CsvProvider
from tradex.data.synthetic import synthetic_bars
from tradex.research import builders, catalog, gate
from tradex.research.sources import DataUnavailable
from tradex.strategy.spec import StrategySpec, compute_signals, load_dir
from test_proposed_features import NY, FOMC, flat_daily, h1_sessions, ny_daily

SPECS = {s.id: s for s in load_dir(gate.PROPOSED)}
PLANNED = [p for p in gate.PLANS if p.spec is not None and p.spec.parent == gate.PROPOSED]


def test_proposed_specs_validate_and_are_never_marked_validated():
    assert len(SPECS) == 11
    for s in SPECS.values():
        assert s.validate() == [], s.id
        assert s.status == "proposed" and s.provenance["author"] == "claude"


def test_every_proposed_spec_has_a_gate_plan_with_a_builder_and_a_catalog_entry():
    entries = {e.id for e in catalog.load()}
    assert {p.spec.stem for p in PLANNED} == set(SPECS)
    for p in PLANNED:
        assert p.catalog_id in entries and p.builder in builders.BUILDERS and p.spec.exists()


# --- fixtures written where the builders look -------------------------------------------------

ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "TLT", "GLD"]


def save_daily(root, syms, n=900, start="2019-01-02"):
    prov = CsvProvider(root)
    for i, s in enumerate(syms):
        b = ny_daily(n, seed=i, start=start)
        prov.save(s, "D1", b)


@pytest.fixture
def us_cache(tmp_path):
    root = tmp_path / "opend"
    stocks = [s for s in SPECS["stk-pead-ear"].universe]
    save_daily(root, sorted(set(ETF + stocks)))
    return root


@pytest.fixture
def oanda_cache(tmp_path, monkeypatch):
    root = tmp_path / "oanda"
    root.mkdir()
    for i, pair in enumerate(SPECS["fx-carry-trend"].universe):
        b = synthetic_bars(1200, seed=40 + i, start="2018-01-01", price=1.1, vol=0.005)
        full = b.assign(**{f"{side}_{c}": b[c] * (1 + sgn * 0.0001) for side, sgn in (("bid", -1), ("ask", 1))
                           for c in ("open", "high", "low", "close")})
        full.to_csv(root / f"{pair}_D1.csv", index_label="ts")
    monkeypatch.setattr(builders, "OANDA_CACHE", root)
    return root


@pytest.fixture
def rates_file(tmp_path, monkeypatch):
    p = tmp_path / "rates.csv"
    rows = []
    for k, ccy in enumerate(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]):
        rows += [f"2017-01-01,{ccy},{k * 0.5}", f"2019-01-01,{ccy},{(k * 7) % 5 * 0.5}", f"2020-06-01,{ccy},{k * 0.25}"]
    p.write_text("date,currency,rate\n" + "\n".join(rows) + "\n")
    monkeypatch.setattr(builders, "RATES_FILE", p)
    return p


def run_signals(spec, data):
    out = {}
    for sym, bars in data.items():
        out[sym] = compute_signals(spec, bars)
    return out


def check_causal(spec, bars):
    full = compute_signals(spec, bars)
    part = compute_signals(spec, bars.iloc[:int(len(bars) * 0.7)])
    n = len(part.long_entry)
    for name in ("long_entry", "short_entry", "long_exit", "short_exit"):
        assert (getattr(full, name).iloc[:n] == getattr(part, name)).all(), f"{spec.id}.{name}"


# --- regime / relative strength builders ------------------------------------------------------

@pytest.mark.parametrize("sid", ["etf-risk-on-trend", "etf-corr-calm-trend", "etf-panic-rebound"])
def test_regime_strategies_run_on_built_columns(sid, us_cache):
    spec = SPECS[sid]
    data, missing = builders.regime_etf(spec, us_cache)
    assert not missing and set(data) == set(spec.universe)
    sig = run_signals(spec, data)
    assert all(len(s.long_entry) == len(data[k]) for k, s in sig.items())
    check_causal(spec, data["SPY"])


def test_regime_builder_needs_the_cross_asset_bars(tmp_path):
    save_daily(tmp_path / "x", ["SPY", "QQQ"])
    with pytest.raises(DataUnavailable, match="TLT"):
        builders.regime_etf(SPECS["etf-risk-on-trend"], tmp_path / "x")


def test_rs_momentum_strategy_runs_on_built_columns(us_cache):
    spec = SPECS["stk-rs-momentum-crash-protected"]
    data, missing = builders.rs_crash(spec, us_cache)
    assert not missing
    b = data["NVDA"]
    assert {"rsm", "rsm_xs", "rg_crash"} <= set(b.columns)
    assert b["rsm_xs"].dropna().between(0, 1).all()
    sig = compute_signals(spec, b)
    assert sig.long_entry.sum() > 0 and not sig.short_entry.any()
    check_causal(spec, b)


# --- earnings --------------------------------------------------------------------------------

def test_earnings_builder_reports_missing_calendars(us_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(builders, "EARNINGS_DIR", tmp_path / "no_earnings")
    with pytest.raises(DataUnavailable, match="earnings"):
        builders.earnings(SPECS["stk-pead-ear"], us_cache)
    cal = tmp_path / "earn"
    cal.mkdir()
    (cal / "NVDA.csv").write_text("date,when\n2019-08-14,amc\n2020-02-12,bmo\n")
    monkeypatch.setattr(builders, "EARNINGS_DIR", cal)
    data, missing = builders.earnings(SPECS["stk-pead-ear"], us_cache)
    assert list(data) == ["NVDA"] and "AMD" in missing
    assert {"ed_z", "ed_age", "ed_post", "ed_volx"} <= set(data["NVDA"].columns)


def test_earnings_jump_strategy_enters_the_day_after_a_confirmed_reaction(tmp_path, monkeypatch):
    b = flat_daily(300)
    rng = np.random.default_rng(2)
    wiggle = np.exp(rng.normal(0, 0.01, len(b)))
    for c in ("open", "high", "low", "close"):
        b[c] = 100.0 * wiggle
    b["high"] = b["high"] * 1.004
    b["low"] = b["low"] * 0.996
    k = 150
    b.loc[b.index[k]:, ["open", "high", "low", "close"]] *= 1.10      # +10% from the reaction session on
    b.iloc[k, b.columns.get_loc("open")] = b["close"].iloc[k - 1] * 1.03
    b.iloc[k, b.columns.get_loc("volume")] = 5e6
    root = tmp_path / "opend"
    CsvProvider(root).save("NVDA", "D1", b)
    cal = tmp_path / "earn"
    cal.mkdir()
    (cal / "NVDA.csv").write_text(f"date,when\n{events_date(b, k)},bmo\n")
    monkeypatch.setattr(builders, "EARNINGS_DIR", cal)
    spec = SPECS["stk-earnings-jump-continuation"]
    data, _ = builders.earnings(spec, root)
    bars = data["NVDA"]
    sig = compute_signals(spec, bars)
    assert list(bars.index[sig.long_entry]) == [bars.index[k]] and not sig.short_entry.any()
    res = run_backtest(spec, data, cfg=EngineConfig(risk_pct=1.0))
    t = res.trades.iloc[0]
    assert len(res.trades) == 1 and t.direction == 1 and t.entry_time == bars.index[k + 1]
    pead = compute_signals(SPECS["stk-pead-ear"], bars)
    assert list(bars.index[pead.long_entry]) == [bars.index[k + 1]]
    check_causal(spec, bars)
    check_causal(SPECS["stk-pead-ear"], bars)


def events_date(bars, pos):
    return pd.Timestamp(bars.index[pos]).tz_convert(NY).date().isoformat()


# --- pre-FOMC --------------------------------------------------------------------------------

def test_pre_fomc_strategy_holds_from_24h_before_to_just_before_the_statement(tmp_path, monkeypatch):
    root = tmp_path / "opend"
    prov = CsvProvider(root)
    for i, s in enumerate(SPECS["etf-pre-fomc-drift-h1"].universe):
        prov.save(s, "H1", h1_sessions(60, seed=i))
    f = tmp_path / "fomc.csv"
    f.write_text("time\n" + FOMC[0].isoformat() + "\n")
    monkeypatch.setattr(builders, "FOMC_FILE", f)
    spec = SPECS["etf-pre-fomc-drift-h1"]
    data, missing = builders.fomc_window(spec, root)
    assert not missing
    res = run_backtest(spec, {"SPY": data["SPY"]}, cfg=EngineConfig(risk_pct=1.0))
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    enter, leave = (pd.Timestamp(x).tz_convert(NY) for x in (t.entry_time, t.exit_time))
    assert enter.strftime("%Y-%m-%d %H:%M") == "2024-01-30 13:30"           # next open after the 12:30 bar closes
    assert leave.strftime("%Y-%m-%d %H:%M") == "2024-01-31 13:30"           # 30 minutes before the 14:00 statement
    assert t.bars_held == 7
    check_causal(spec, data["SPY"])


def test_pre_fomc_needs_the_historical_calendar(tmp_path, monkeypatch):
    root = tmp_path / "opend"
    for i, s in enumerate(["SPY"]):
        CsvProvider(root).save(s, "H1", h1_sessions(10, seed=i))
    monkeypatch.setattr(builders, "FOMC_FILE", tmp_path / "missing.csv")
    with pytest.raises(DataUnavailable, match="calendar"):
        builders.fomc_window(SPECS["etf-pre-fomc-drift-h1"], root)


# --- FX ---------------------------------------------------------------------------------------

def test_fx_strength_strategy_runs_on_built_columns(oanda_cache):
    spec = SPECS["fx-currency-strength-momentum"]
    data, missing = builders.fx_strength(spec)
    assert not missing and {"cs_diff", "cs_xs", "ts_mom"} <= set(data["EUR_USD"].columns)
    sig = compute_signals(spec, data["EUR_USD"])
    assert sig.long_entry.sum() + sig.short_entry.sum() > 0
    check_causal(spec, data["EUR_USD"])


@pytest.mark.parametrize("sid", ["fx-carry-trend", "fx-carry-vol-filter", "fx-rate-diff-trend"])
def test_carry_strategies_run_on_built_columns(sid, oanda_cache, rates_file):
    spec = SPECS[sid]
    data, missing = builders.fx_carry(spec)
    assert not missing
    b = data["EUR_USD"]
    assert {"carry", "carry_xs", "ts_mom", "rate_chg", "fxvol_pct"} <= set(b.columns)
    assert b["carry_xs"].dropna().between(0, 1).all()
    sig = compute_signals(spec, b)
    assert len(sig.long_entry) == len(b)
    check_causal(spec, b)


def test_fx_builders_need_oanda_history_and_rates(tmp_path, monkeypatch, oanda_cache):
    monkeypatch.setattr(builders, "RATES_FILE", tmp_path / "none.csv")
    with pytest.raises(DataUnavailable, match="rate history"):
        builders.fx_carry(SPECS["fx-carry-trend"])
    monkeypatch.setattr(builders, "OANDA_CACHE", tmp_path / "empty")
    with pytest.raises(DataUnavailable, match="Oanda"):
        builders.fx_strength(SPECS["fx-currency-strength-momentum"])


# --- the gate reports missing inputs instead of running on anything else ----------------------

def test_gate_reports_needs_data_for_every_input_it_lacks(tmp_path, monkeypatch):
    for name in ("EARNINGS_DIR", "FOMC_FILE", "RATES_FILE", "OANDA_CACHE"):
        monkeypatch.setattr(builders, name, tmp_path / "absent" / name)
    save_daily(tmp_path / "opend", ["SPY"], n=50)
    entries = {e.id: e for e in catalog.load()}
    seen = {}
    for p in PLANNED:
        row = gate.run_plan(p, entries[p.catalog_id], None, WalkForwardConfig(), Thresholds(), EngineConfig(),
                            cache=tmp_path / "opend")
        seen[p.spec.stem] = row
    assert {r["result"] for r in seen.values()} == {"needs data"}
    assert "earnings" in seen["stk-pead-ear"]["why"].lower() and "Oanda" in seen["fx-carry-trend"]["why"]


def test_single_strategy_runs_select_by_strategy_id_and_write_their_own_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "RESULTS", tmp_path / "results")
    monkeypatch.setenv("TRADEX_TRIALS_DB", str(tmp_path / "trials.db"))
    monkeypatch.setattr(gate, "OANDA_CACHE", tmp_path / "none")
    for name in ("EARNINGS_DIR", "FOMC_FILE", "RATES_FILE", "OANDA_CACHE"):
        monkeypatch.setattr(builders, name, tmp_path / "absent" / name)
    assert gate.main(["fx-carry-trend"]) == 0
    assert "fx-carry-trend: needs data" in capsys.readouterr().out
    assert (tmp_path / "results" / "single" / "fx-carry-trend.json").exists()
    assert not (tmp_path / "results" / "phase1_gate.json").exists()
