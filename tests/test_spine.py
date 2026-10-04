"""The spine: ledger, simulated broker, four-gate decision path, risk, calendar, replay."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import flat_bars
from tradex.core.counterfactual import CounterfactualTracker, filter_report
from tradex.core.interfaces import BrokerPosition, OrderRequest
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.records import EquitySnapshot, TradePlan, Veto, Vote
from tradex.core.replay import run_replay
from tradex.costs.models import model_for
from tradex.data.synthetic import synthetic_bars
from tradex.decision.ensemble import PlanRules, family_votes, finalise, merge_correlated_families
from tradex.events import EventCalendar, Event
from tradex.execution.checks import ShortInfo, ShortPolicy, sanity_check, SanityPolicy, short_check
from tradex.execution.sim import SimBroker
from tradex.risk.exposure import ESModel, Leg, Scenario, book_exposures, net_open_position, stop_risk_by_currency
from tradex.risk.gate import BookState, RiskGate
from tradex.strategy.spec import StrategySpec

T0 = pd.Timestamp("2026-03-02 21:00", tz="UTC")
ZERO_COST = {"stocks": model_for("stocks", platform_fee=0.0, settlement_fee_per_share=0.0, sec_fee_rate=0.0,
                                  finra_taf_per_share=0.0, default_half_spread_bps=0.0, slippage_bps=0.0),
             "forex": model_for("forex", default_spread_pips=0.0, spread_pips={}, slippage_pips=0.0)}


# --- ledger ------------------------------------------------------------------------------

def test_ledger_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "l.sqlite"
    led = Ledger(path, git_commit="abc")
    led.set_config(T0.isoformat(), "config/risk/policy.yaml", {"heat": 12})
    led.append(Veto("2026-03-02-0001", T0.isoformat(), "calendar", "fomc"))
    led.append(Veto("2026-03-02-0002", T0.isoformat(), "risk", "no room"))
    assert led.verify() == (True, None)
    assert [r["source"] for r in led.why("2026-03-02-0002")] == ["risk"]
    led.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE events SET payload = replace(payload, 'no room', 'fine') WHERE seq = 3")
    db.commit()
    db.close()
    ok, bad = Ledger(path, read_only=True).verify()
    assert not ok and bad == 3


def test_read_only_ledger_refuses_writes(tmp_path):
    path = tmp_path / "l.sqlite"
    Ledger(path).append(Veto("x", T0.isoformat(), "s", "r"))
    ro = Ledger(path, read_only=True)
    with pytest.raises(PermissionError):
        ro.append(Veto("y", T0.isoformat(), "s", "r"))


def test_snapshot_lookback_returns_the_state_of_that_day():
    led = Ledger(":memory:")
    for day, eq in (("2026-03-02", 10_000.0), ("2026-03-03", 10_250.0), ("2026-03-04", 9_900.0)):
        led.append(EquitySnapshot(f"{day}T21:00:00+00:00", "ensemble", eq, eq, 0.0, [], {}, {}, "h"))
    snap = led.snapshot_at("2026-03-03T23:00:00+00:00")
    assert snap["equity_usd"] == 10_250.0


# --- simulated broker -----------------------------------------------------------------------

def _req(did="d1", side=1, qty=10, stop=95.0, tp=110.0, **kw):
    return OrderRequest(f"{did}-entry", did, "X", "stocks", side, qty, stop_loss=stop, take_profit=tp, **kw)


def test_sim_fills_next_open_and_checks_stop_first():
    br = SimBroker(10_000, ZERO_COST, bar=pd.Timedelta(days=1))
    br.place(_req())
    assert br.place(_req()) == "d1-entry" and len(br.orders) == 1          # idempotent client ID
    t1, t2 = T0, T0 + pd.Timedelta(days=1)
    fills = br.on_bar("X", t1, 100, 101, 99, 100)
    assert fills[0].price == 100 and fills[0].reason == "entry"
    fills = br.on_bar("X", t2, 100, 111, 94, 105)                          # both stop and target touched
    assert fills[0].reason == "stop" and fills[0].price == 95
    assert br.closed[0].r_multiple == pytest.approx(-1.0)
    assert br.cash() == pytest.approx(10_000 - 50)


def test_sim_gap_through_stop_fills_at_open():
    br = SimBroker(10_000, ZERO_COST, bar=pd.Timedelta(days=1))
    br.place(_req())
    br.on_bar("X", T0, 100, 101, 99, 100)
    f = br.on_bar("X", T0 + pd.Timedelta(days=1), 90, 92, 89, 91)
    assert f[0].reason == "stop_gap" and f[0].price == 90


def test_sim_never_touches_rays_own_holdings():
    br = SimBroker(10_000, ZERO_COST, bar=pd.Timedelta(days=1))
    br.add_external_position(BrokerPosition("ray-aapl", "X", "stocks", 1, 50, 80.0, T0, 79.0, None))
    br.on_bar("X", T0, 70, 71, 60, 65)                                      # far through its "stop"
    assert [p.decision_id for p in br.positions(account="ray")] == ["ray-aapl"]
    assert br.equity() == 10_000
    with pytest.raises(PermissionError):
        br.place(_req(account="ray"))


# --- exposure and risk gate ---------------------------------------------------------------------

def test_usdjpy_long_and_eurusd_short_are_one_dollar_bet():
    legs = [Leg("USD_JPY", "forex", 1, 10_000, 150.0), Leg("EUR_USD", "forex", -1, 10_000, 1.10)]
    nop = net_open_position(legs)
    assert nop["JPY"] == pytest.approx(-10_000, rel=1e-6)
    assert nop["EUR"] == pytest.approx(-11_000, rel=1e-6)
    assert "USD" not in nop


def test_cross_pair_splits_stop_risk_between_its_currencies():
    risk = stop_risk_by_currency([Leg("EUR_JPY", "forex", 1, 10_000, 160.0, stop=159.0)])
    assert set(risk) == {"EUR", "JPY"} and risk["EUR"] == pytest.approx(risk["JPY"])


def _corr_returns(n=400, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.006, n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(0, 0.006, n)
    return pd.DataFrame({"FX:EUR": a, "FX:JPY": b}, index=pd.bdate_range("2024-01-01", periods=n, tz="UTC"))


def test_marginal_es_charges_repeated_bets_and_credits_offsets():
    es = ESModel(_corr_returns())
    book = {"FX:JPY": -10_000.0}
    same_bet = es.marginal(book, {"FX:EUR": -10_000.0})
    offset = es.marginal(book, {"FX:EUR": 10_000.0})
    standalone = es.es({"FX:EUR": -10_000.0})
    assert same_bet > standalone * 0.8          # correlated: nearly the full charge
    assert offset < 0                           # offsetting trade lowers book risk


def _plan(symbol="EUR_USD", ac="forex", d=1, entry=1.10, stop=1.095, t1=1.11, p=0.5, did="d"):
    rr = abs(t1 - entry) / abs(entry - stop)
    return TradePlan(did, T0.isoformat(), symbol, ac, d, "market", entry, stop, [t1], 20, "", ["trend", "breakout"],
                     ["a", "b"], 0.5, p, "base_rate", rr, p * rr - (1 - p), 0.0)


def test_gate_sizes_only_and_never_moves_prices():
    gate = RiskGate.from_policy()
    plan = _plan()
    v = gate.review(plan, BookState(10_000, []), 1.0)
    assert v.outcome == "accepted" and v.qty > 0
    assert (plan.entry_price, plan.stop, plan.targets) == (1.10, 1.095, [1.11])
    assert v.risk_pct <= gate.policy["sizing"]["per_trade_cap_pct"] + 1e-9


def test_gate_rejects_plans_without_edge():
    v = RiskGate.from_policy().review(_plan(p=0.2), BookState(10_000, []), 1.0)
    assert v.outcome == "rejected" and "no edge" in v.reasons[-1]


def test_gate_currency_cap_binds_on_a_second_yen_trade():
    gate = RiskGate.from_policy()
    held = [Leg("USD_JPY", "forex", 1, 300_000, 150.0, stop=148.5)]       # ~3,000 USD of JPY stop risk
    plan = _plan("EUR_JPY", entry=160.0, stop=158.0, t1=164.0, p=0.55)
    v = gate.review(plan, BookState(10_000, held), 1 / 150)
    assert "currency" in v.checks and v.checks["currency"]["JPY"]["now_usd"] > 2_900
    assert v.outcome == "rejected" or "currency" in " ".join(v.reasons)


def test_gate_stress_scenario_limits_size():
    gate = RiskGate.from_policy()
    gate.scenarios = [Scenario("crash", {}, default_equity=-0.9)]
    plan = _plan("X", "stocks", entry=100.0, stop=99.0, t1=103.0, p=0.6)
    v = gate.review(plan, BookState(10_000, []), 1.0)
    assert v.checks["stress"]["loss_with_trade_usd"] <= 3_500 + 1
    assert "stress" in " ".join(v.reasons)


# --- ensemble ---------------------------------------------------------------------------------

def _vote(sid, fam, d=1, strength=0.5, stop=95.0, target=110.0, sym="X"):
    return Vote(T0.isoformat(), sid, 1, fam, sym, "stocks", d, strength, 100.0, stop, [target], 10, T0.isoformat())


def test_one_family_is_not_enough_and_twins_count_once():
    cost = ZERO_COST["stocks"]
    plan, why = finalise([_vote("a", "trend"), _vote("b", "trend")], "d", PlanRules(), cost)
    assert plan is not None and "only 1 family" in why
    assert family_votes([_vote("a", "trend"), _vote("b", "trend")])["trend"][1] == 0.5


def test_two_families_finalise_a_complete_plan():
    votes = [_vote("a", "trend", stop=95, target=110), _vote("b", "breakout", stop=94, target=115)]
    plan, why = finalise(votes, "d", PlanRules(), ZERO_COST["stocks"])
    assert why == ""
    assert plan.stop == 94 and plan.targets == [110, 115]          # farthest stop, nearest then farthest target
    assert plan.families == ["breakout", "trend"] and plan.p_source == "base_rate"


def test_poor_reward_to_risk_blocks_the_plan():
    votes = [_vote("a", "trend", stop=95, target=106), _vote("b", "breakout", stop=95, target=106)]
    _, why = finalise(votes, "d", PlanRules(), ZERO_COST["stocks"])
    assert "reward to risk" in why


def test_correlated_signals_merge_into_one_family():
    idx = range(200)
    rng = np.random.default_rng(1)
    base = pd.Series(rng.choice([-1, 0, 1], 200), index=idx)
    fam = merge_correlated_families({"a": base, "b": base.copy(), "c": pd.Series(rng.choice([-1, 0, 1], 200), index=idx)},
                                    {"a": "trend", "b": "momentum", "c": "carry"}, 0.7)
    assert fam["a"] == fam["b"] and fam["c"] == "carry"


# --- calendar and checks ----------------------------------------------------------------------

def test_calendar_vetoes_inside_fomc_window_and_halves_over_weekend():
    cal = EventCalendar([Event(pd.Timestamp("2026-03-18 18:00", tz="UTC"), "fomc", "USD")])
    veto, _, _ = cal.check("EUR_USD", "forex", pd.Timestamp("2026-03-18 10:00", tz="UTC"),
                           pd.Timestamp("2026-03-18 14:00", tz="UTC"))
    assert veto and "fomc" in veto
    ok, factor, _ = cal.check("EUR_USD", "forex", pd.Timestamp("2026-03-20 10:00", tz="UTC"),
                              pd.Timestamp("2026-03-23 10:00", tz="UTC"))
    assert ok is None and factor == 0.5
    none, f2, _ = cal.check("AAPL", "stocks", pd.Timestamp("2026-03-10 15:00", tz="UTC"),
                            pd.Timestamp("2026-03-12 15:00", tz="UTC"))
    assert none is None and f2 == 1.0


def test_calendar_blocks_stock_held_into_earnings(tmp_path):
    f = tmp_path / "earn.csv"
    f.write_text("time,kind,scope\n2026-04-30T20:05:00Z,earnings,AAPL\n")
    cal = EventCalendar.load([f])
    veto, _, _ = cal.check("AAPL", "stocks", pd.Timestamp("2026-04-22", tz="UTC"), pd.Timestamp("2026-05-05", tz="UTC"))
    assert veto and "earnings" in veto


def test_short_checks():
    pol = ShortPolicy()
    assert short_check("stocks", 1, None, pol) is None
    assert short_check("stocks", -1, None, pol) == "borrow data unknown"
    assert "borrow" in short_check("stocks", -1, ShortInfo(shortable=False), pol)
    assert "Rule 201" in short_check("stocks", -1, ShortInfo(ssr_active=True), pol)
    assert "squeeze" in short_check("stocks", -1, ShortInfo(short_interest_pct_float=35), pol)
    assert short_check("stocks", -1, ShortInfo(borrow_fee_annual=0.01), pol) is None
    assert sanity_check(1e6, 100.0, 1.0, 10_000, 100.0, SanityPolicy()) is not None


# --- counterfactual -----------------------------------------------------------------------------

def test_counterfactual_follows_blocked_plan_to_its_stop():
    cf = CounterfactualTracker()
    cf.track(_plan("X", "stocks", entry=100.0, stop=95.0, t1=110.0), "plan")
    assert cf.on_bar("X", T0 + pd.Timedelta(days=1), 104, 99, 101) == []
    done = cf.on_bar("X", T0 + pd.Timedelta(days=2), 101, 94, 96)
    assert done[0].exit_reason == "stop" and done[0].r_multiple == pytest.approx(-1.0)
    rep = filter_report([d.to_dict() for d in done])
    assert rep.loc[0, "plans"] == 1


# --- replay ------------------------------------------------------------------------------------

def _two_family_specs():
    base = {"version": 1, "asset_class": "stocks", "universe": ["AAA", "BBB"], "timeframes": {"signal": "D1"},
            "holding": {"expected_hours": 72}, "status": "paper",
            "exit": {"stop_atr": 1.5, "target_r": 2.5, "max_bars": 15}}
    trend = StrategySpec.from_dict(base | {
        "id": "t-trend", "family": "trend",
        "features": {"ema": {"fn": "talib.EMA", "period": 20}},
        "entry": {"long": "close > ema", "short": "close < ema"}})
    mom = StrategySpec.from_dict(base | {
        "id": "t-mom", "family": "momentum",
        "features": {"roc": {"fn": "talib.ROC", "period": 10}},
        "entry": {"long": "roc > 0", "short": "roc < 0"}})
    return [trend, mom]


def _frames():
    return {"AAA": synthetic_bars(500, seed=3, price=60.0), "BBB": synthetic_bars(500, seed=4, price=40.0)}


def test_replay_is_deterministic_and_writes_the_whole_chain():
    frames = _frames()
    start = frames["AAA"].index[260]
    r1 = run_replay(_two_family_specs(), frames, Ledger(":memory:", git_commit="t"), start)
    r2 = run_replay(_two_family_specs(), frames, Ledger(":memory:", git_commit="t"), start)
    assert r1.digest == r2.digest and r1.chain_ok
    led = r1.ledger
    kinds = {r["kind"] for r in led.rows()}
    assert {"plan", "vote", "verdict", "order", "fill", "close", "snapshot", "config_version"} <= kinds
    filled = led.rows(kind="fill")[0]["decision_id"]
    chain = [r["kind"] for r in led.why(filled)]
    assert chain[:4] == ["plan", "vote", "vote", "verdict"]
    assert {"virtual:t-trend", "virtual:t-mom", "ensemble"} <= {r["book"] for r in led.rows(kind="snapshot")}


def test_unqualified_strategies_trade_only_in_virtual_books():
    specs = _two_family_specs()
    for s in specs:
        s.status = "proposed"
    frames = _frames()
    r = run_replay(specs, frames, Ledger(":memory:"), frames["AAA"].index[260])
    assert r.summary["ensemble"]["closed_trades"] == 0
    assert r.summary["virtual:t-trend"]["closed_trades"] > 0


def test_pause_command_stops_new_entries():
    frames = _frames()
    led = Ledger(":memory:")
    led.add_command(T0.isoformat(), "test", "pause")
    r = run_replay(_two_family_specs(), frames, led, frames["AAA"].index[260], cfg=CoreConfig(virtual_books=False))
    assert r.summary["ensemble"]["closed_trades"] == 0 and not led.rows(kind="order")
    assert led.db.execute("SELECT result FROM commands").fetchone()[0] == "no new entries"


def test_protected_paths_check():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cpp", "tools/check_protected_paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prefixes = mod.protected_prefixes()
    assert mod.violations(["config/risk/policy.yaml", "tradex/risk/gate.py", "strategies/x.yaml"], prefixes) == \
        ["config/risk/policy.yaml", "tradex/risk/gate.py"]
    assert mod.main(["--head-ref", "claude/feature"]) == 0
