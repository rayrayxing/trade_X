import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradex.profit.ruin import (RuinLimits, ShockWindow, exposures_from_legs, hist_es, legs_from_snapshot,
                                load_shock_windows, radar_from_snapshot, ruin_report, stop_risk_usd, window_shocks)
from tradex.risk.exposure import Leg, Scenario, scenarios_from_config

UTC = "UTC"
RATES = {"JPY": 1 / 150.0, "EUR": 1.10, "GBP": 1.30, "AUD": 0.66, "USD": 1.0}      # test inputs only


def rate_fn(ccy, ts):
    return RATES[ccy]


def days(start, n):
    return pd.date_range(start, periods=n, freq="D", tz=UTC)


def returns_with_shock(factors, win_start, win_end, shock_returns, total_days=400, seed=0):
    """Quiet random returns, with the given per-day returns written into the window for each factor."""
    rng = np.random.default_rng(seed)
    idx = days("2014-06-02", total_days)
    df = pd.DataFrame(rng.normal(0, 0.002, (total_days, len(factors))), index=idx, columns=factors)
    inside = (df.index >= pd.Timestamp(win_start, tz=UTC)) & (df.index <= pd.Timestamp(win_end, tz=UTC))
    for f, r in shock_returns.items():
        df.loc[inside, f] = r
    return df


W_YEN = ShockWindow("yen", pd.Timestamp("2014-09-01", tz=UTC), pd.Timestamp("2014-09-03", tz=UTC))
W_FAR = ShockWindow("far_past", pd.Timestamp("2001-01-01", tz=UTC), pd.Timestamp("2001-01-05", tz=UTC))
JPY_LONG = {"FX:JPY": -10_000.0}                                       # long USDJPY, 10,000 USD notional


def test_shipped_windows_load_and_have_dates_only():
    ws = load_shock_windows(Path("config/shock_windows.yaml"))
    assert len(ws) >= 8 and len({w.name for w in ws}) == len(ws)
    assert all(w.start <= w.end for w in ws)
    assert "pct" not in Path("config/shock_windows.yaml").read_text().replace("magnitude", "")


def test_window_shock_is_the_compounded_actual_return():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    shocks, n, blind = window_shocks(r, W_YEN)
    assert n == 3 and blind == [] and shocks["FX:JPY"] == pytest.approx(1.05 ** 3 - 1)


def test_a_factor_without_enough_data_in_the_window_is_blind_not_zero():
    r = returns_with_shock(["FX:JPY", "EQ:AAA"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    r.loc[r.index >= pd.Timestamp("2014-09-02", tz=UTC), "EQ:AAA"] = np.nan
    shocks, _, blind = window_shocks(r, W_YEN)
    assert "EQ:AAA" in blind and "EQ:AAA" not in shocks


def test_replay_applies_the_actual_shock_to_the_book():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = ruin_report(JPY_LONG, RuinLimits(equity=50_000.0), r, [W_YEN, W_FAR])
    yen = next(x for x in rep.results if x.name == "yen")
    assert yen.pnl_usd == pytest.approx(-10_000 * (1.05 ** 3 - 1)) and yen.status == "ok"
    far = next(x for x in rep.results if x.name == "far_past")
    assert far.status == "no_data" and far.pnl_usd == 0.0                  # no history: reported, not assumed safe
    assert rep.worst.name == "yen"
    assert any(f.code == "windows_without_data" for f in rep.flags)


def test_unpriced_exposure_is_reported_as_blind():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = ruin_report({**JPY_LONG, "EQ:ZZZ": 30_000.0}, RuinLimits(equity=100_000.0), r, [W_YEN])
    yen = rep.results[0]
    assert yen.status == "partial" and yen.blind_exposure_usd == 30_000.0 and yen.blind_factors == ["EQ:ZZZ"]
    assert any(f.code == "blind" for f in rep.flags)


def test_policy_scenarios_replay_alongside():
    pol = Path("config/risk/policy.yaml")
    import yaml
    scen = scenarios_from_config(yaml.safe_load(pol.read_text())["scenarios"])
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = ruin_report(JPY_LONG, RuinLimits(equity=50_000.0), r, [W_YEN], scen)
    carry = next(x for x in rep.results if x.name == "yen_carry_unwind_2024")
    assert carry.source == "policy" and carry.pnl_usd == pytest.approx(-10_000 * 0.13)


def test_flags_for_cap_ruin_concentration_and_gaps():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.15})      # ~52% over three days
    lim = RuinLimits(equity=10_000.0, stress_loss_cap_pct=35.0, es_budget_pct=5.0)
    rep = ruin_report(JPY_LONG, lim, r, [W_YEN], stop_risk=400.0)
    codes = {f.code for f in rep.flags}
    assert {"stress_breach", "ruin", "concentration", "gap_through_stops"} <= codes
    assert rep.level == "breach"
    ok = ruin_report({"FX:JPY": -1_000.0}, RuinLimits(equity=100_000.0, stress_loss_cap_pct=35.0), r, [W_YEN])
    assert not ({"stress_breach", "ruin"} & {f.code for f in ok.flags})
    warn = ruin_report({"FX:JPY": -10_000.0}, RuinLimits(equity=25_000.0, stress_loss_cap_pct=35.0), r, [W_YEN])
    assert any(f.code == "stress_warn" for f in warn.flags) and warn.level == "warn"


def test_empty_book_and_no_equity():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = ruin_report({}, RuinLimits(equity=10_000.0), r, [W_YEN])
    assert rep.flags[0].code == "no_book" and "no open positions" in rep.telegram()
    assert ruin_report(JPY_LONG, RuinLimits(equity=0.0), r, [W_YEN]).level == "breach"


def test_historical_expected_shortfall_matches_a_hand_calculation():
    r = returns_with_shock(["FX:JPY"], "2099-01-01", "2099-01-02", {}, total_days=600, seed=7)
    e = {"FX:JPY": -10_000.0}
    es1, _ = hist_es(r, e, 1)
    pnl = np.sort(r["FX:JPY"].tail(600).to_numpy() * -10_000.0)
    k = int(np.ceil(600 * 0.025))
    assert es1 == pytest.approx(-pnl[:k].mean())
    es5, _ = hist_es(r, e, 5)
    assert es5 > es1
    short, missing = hist_es(r.tail(100), e, 1)
    assert short is None
    assert hist_es(r, {"FX:ZZZ": 1.0})[0] is None


def test_es_budget_flag_and_headline():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.01}, total_days=500, seed=3)
    r["FX:JPY"] += np.random.default_rng(1).normal(0, 0.01, len(r))
    tight = ruin_report(JPY_LONG, RuinLimits(equity=10_000.0, es_budget_pct=1.0), r, [W_YEN])
    assert tight.hist_es_1d_usd and any(f.code == "es_breach" for f in tight.flags)
    assert tight.es_usd == max(tight.scenario_es_usd, tight.hist_es_horizon_usd)
    thin = ruin_report(JPY_LONG, RuinLimits(equity=10_000.0, es_budget_pct=1.0), r.tail(50), [W_YEN])
    assert thin.hist_es_1d_usd is None and any(f.code == "es_no_history" for f in thin.flags)


def test_report_serialises_and_reads_as_a_telegram_message():
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = ruin_report(JPY_LONG, RuinLimits(equity=20_000.0, stress_loss_cap_pct=35.0), r, [W_YEN])
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["worst"]["name"] == "yen" and d["scenarios"][0]["top_factors"][0][0] == "FX:JPY"
    msg = rep.telegram()
    assert msg.startswith("Ruin radar [") and "yen" in msg and "Expected shortfall" in msg


def snapshot(positions, equity=10_000.0):
    return {"time": "2014-12-01T21:00:00+00:00", "book": "ensemble", "equity_usd": equity, "positions": positions}


def test_exposures_come_from_the_snapshot_with_supplied_rates_and_the_asset_class_is_read_from_the_symbol():
    snap = snapshot([{"decision_id": "d1", "symbol": "USD_JPY", "direction": 1, "qty": 10_000, "entry": 150.0,
                      "stop": 148.0, "mark": 150.0, "account": "agent"},
                     {"decision_id": "d2", "symbol": "AAPL", "direction": -1, "qty": 10, "entry": 200.0, "stop": 210.0,
                      "mark": 200.0, "account": "ray"}])
    legs = legs_from_snapshot(snap)
    assert [l.asset_class for l in legs] == ["forex", "stocks"] and legs[1].account == "ray"
    ex = exposures_from_legs(legs, rate_fn, pd.Timestamp(snap["time"]))
    assert ex["FX:JPY"] == pytest.approx(-10_000.0) and ex["EQ:AAPL"] == pytest.approx(-2_000.0)
    # loss to the stop: 2 JPY * 10,000 units * (1/150) USD per JPY = 133.33 USD (AAPL: 10 * 10 = 100)
    assert stop_risk_usd(legs, rate_fn, pd.Timestamp(snap["time"])) == pytest.approx(10_000 * 2.0 / 150 + 100.0)


def test_a_missing_rate_stops_the_radar_instead_of_guessing():
    snap = snapshot([{"decision_id": "d1", "symbol": "USD_JPY", "direction": 1, "qty": 1000, "entry": 150.0,
                      "stop": 148.0, "mark": 150.0}])

    def none(ccy, ts):
        raise LookupError(ccy)
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    with pytest.raises(LookupError):
        radar_from_snapshot(snap, r, [W_YEN], none)


def test_radar_from_snapshot_end_to_end_with_the_policy():
    import yaml
    pol = yaml.safe_load(Path("config/risk/policy.yaml").read_text())
    snap = snapshot([{"decision_id": "d1", "symbol": "USD_JPY", "direction": 1, "qty": 10_000, "entry": 150.0,
                      "stop": 148.0, "mark": 150.0}], equity=6_000.0)
    r = returns_with_shock(["FX:JPY"], "2014-09-01", "2014-09-03", {"FX:JPY": 0.05})
    rep = radar_from_snapshot(snap, r, [W_YEN], rate_fn, pol)
    assert rep.worst.pnl_usd < 0 and rep.stop_risk_usd == pytest.approx(10_000 * 2.0 / 150)
    assert {x.source for x in rep.results} == {"history", "policy"}
    assert any(f.code in ("stress_breach", "stress_warn") for f in rep.flags)
