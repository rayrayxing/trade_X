"""The profit hooks wired into the core (``TradingCore(..., profit=hooks)``).

Replay-level checks that need the patched loop: off means identical decisions, on means the engine's actions appear
in the ledger and the chain stays valid, and nothing the hooks do can size above the gate.
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent / "profit"))

from test_spine import _frames, _two_family_specs
from tradex.core.ledger import Ledger
from tradex.core.replay import run_replay
from tradex.profit.calibration import CalibrationConfig
from tradex.profit.costcal import CostCalConfig
from tradex.profit.exits import ExitPolicy
from tradex.profit.hooks import ProfitHooks
from tradex.profit.suggest import VolTargetPolicy


def replay(profit=None):
    frames = _frames()
    return run_replay(_two_family_specs(), frames, Ledger(":memory:", git_commit="t"), frames["AAA"].index[260],
                      profit=profit)


@pytest.fixture(scope="module")
def baseline():
    return replay()


def test_hooks_with_nothing_switched_on_change_no_decision(baseline):
    assert replay(ProfitHooks()).digest == baseline.digest


def test_exit_engine_acts_through_the_cores_own_orders_and_keeps_the_chain_valid(baseline):
    r = replay(ProfitHooks(exit_policy=ExitPolicy()))
    assert r.chain_ok
    led = r.ledger
    reasons = [x["reason"] for x in led.rows(kind="exit_change")] + [x["purpose"] for x in led.rows(kind="order")]
    assert any(("trail" in s or "breakeven" in s or "volatility shock" in s or "target 1 reached" in s) for s in reasons)
    assert r.digest != baseline.digest
    assert all(not (h["check"] == "profit") for h in led.rows(kind="health"))
    # every entry order still cites a verdict and none is bigger than the verdict that sized it
    verdict_qty = {v["decision_id"]: v["qty"] for v in led.rows(kind="verdict")}
    for o in led.rows(kind="order"):
        if o["purpose"].startswith("entry") and o["decision_id"] in verdict_qty:
            assert o["qty"] <= verdict_qty[o["decision_id"]] + 1e-9


def test_partial_target_attaches_the_last_target_to_the_entry():
    h = ProfitHooks(exit_policy=ExitPolicy())
    from pfx import plan
    assert h.attach_target(plan(targets=(104.0, 108.0))) == 108.0
    assert h.attach_target(plan(targets=(104.0,))) == 104.0
    assert ProfitHooks(exit_policy=ExitPolicy.baseline()).attach_target(plan(targets=(104.0, 108.0))) == 104.0


def test_vol_targeting_only_ever_shrinks_orders(baseline):
    r = replay(ProfitHooks(vol_policy=VolTargetPolicy(target_annual_vol=0.005, min_obs=20)))
    assert r.chain_ok

    def entry_qty(res):
        return {o["decision_id"]: o["qty"] for o in res.ledger.rows(kind="order") if o["purpose"].startswith("entry")}
    base_q, vol_q = entry_qty(baseline), entry_qty(r)
    assert sum(vol_q.values()) < sum(base_q.values())


def test_daily_refit_runs_without_faults(baseline):
    h = ProfitHooks(calibration=CalibrationConfig(min_trades=20), cost_cfg=CostCalConfig(min_fills=20))
    r = replay(h)
    assert r.chain_ok and not [x for x in r.ledger.rows(kind="health") if x["check"] == "profit"]
    assert h.prob_model is not None and h.prob_model.n >= 20
    assert h.last_cost_report["stocks"]["calibrated"] and h.last_cost_report["stocks"]["spread_mult"] == pytest.approx(1.0, abs=0.01)


def test_two_runs_with_the_hooks_are_identical():
    mk = lambda: ProfitHooks(exit_policy=ExitPolicy(), calibration=CalibrationConfig(min_trades=20),  # noqa: E731
                             vol_policy=VolTargetPolicy(min_obs=20), cost_cfg=CostCalConfig(min_fills=20))
    assert replay(mk()).digest == replay(mk()).digest


def test_agent_shrink_and_vol_targeting_each_act_once_and_never_above_the_gate():
    from tradex.core.actions import shrink_qty
    from tradex.core.loop import CoreConfig
    frames = _frames()
    led = Ledger(":memory:", git_commit="t")
    led.add_agent_inbox("2026-10-05T00:00:00+00:00", "scout", "shrink", "AAA", {"factor": 0.5, "until": "2100-01-01"})
    run_replay(_two_family_specs(), frames, led, frames["AAA"].index[260], cfg=CoreConfig(agents_mode="active"),
               profit=ProfitHooks(vol_policy=VolTargetPolicy(target_annual_vol=0.005, min_obs=20)))
    verdicts = {v["decision_id"]: v["qty"] for v in led.rows(kind="verdict")}
    entries = [o for o in led.rows(kind="order") if o["symbol"] == "AAA" and o["purpose"].startswith("entry")
               and o["book"] == "ensemble"]
    assert entries
    for o in entries:
        assert o["qty"] <= shrink_qty(verdicts[o["decision_id"]], 0.5) + 1e-9      # never above the agent's half
