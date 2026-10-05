"""The whole loop on fixtures: stage order, status transitions, holdout discipline, trial counting, resume."""
import pandas as pd
import pytest

from tradex.research.loop.config import LoopConfig
from tradex.research.loop.engine import run_loop
from tradex.research.loop.ports import DataUnavailable
from tradex.research.loop.specstore import content_hash
from tradex.research.loop.stages import STAGES
from tradex.strategy.spec import StrategySpec

from loopkit import (HOLDOUT_START, FakeData, FakeHoldout, ScriptedEvaluator, bad_report, good_report, make_frames,
                     make_rig, spec_dict, write_spec)

SID = "stk-rsi-test"


def go(rig, **kw):
    return rig.run(**kw)


def test_a_good_strategy_walks_the_whole_ladder_and_is_only_recommended_without_apply(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    r = go(rig)
    assert [k for k in r] == list(STAGES)
    assert rig.status(SID) == "validated"                       # holdout passed; paper is only recommended
    assert rig.file_status(SID) == "validated"
    assert r["promote"].items[f"{SID}:v1"]["outcome"] == "recommended"
    ev = {e["kind"]: e for e in rig.state.evidence(SID, 1)}
    assert [e["kind"] for e in rig.state.evidence(SID, 1)] == ["screen", "walk_forward", "holdout", "promotion"]
    assert ev["promotion"]["passed"] is None
    assert rig.holdout.looks and rig.holdout.looks[0][:2] == (SID, 1)


def test_apply_promotes_to_paper_and_writes_the_status(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True))
    rig.add_spec()
    r = go(rig)
    assert r["promote"].items[f"{SID}:v1"]["outcome"] == "promoted"
    assert rig.status(SID) == "paper" and rig.file_status(SID) == "paper"
    assert rig.state.strategy(SID)["promoted_at"]
    assert rig.state.baseline(SID, 1) is not None


def test_screen_failure_rejects_and_stops_there(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.ev.screen_out[SID] = {"trades": 5, "profit_factor": 0.7, "sharpe": -0.2, "max_drawdown": -0.3,
                              "total_return": -0.1, "win_rate": 0.3, "costs_share_of_gross": 0.5, "params": {}, "warnings": []}
    go(rig)
    assert rig.status(SID) == "rejected" and rig.file_status(SID) == "rejected"
    assert ("walk_forward", SID) not in rig.ev.calls and not rig.holdout.looks
    reasons = rig.state.latest_evidence(SID, 1, "screen")["data"]["reasons"]
    assert any("trades" in x for x in reasons) and any("profit factor" in x for x in reasons)


def test_walk_forward_failure_rejects_with_the_failing_rung_and_never_reaches_the_holdout(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.ev.wf_out[SID] = bad_report(StrategySpec.from_dict(spec_dict()))
    go(rig)
    assert rig.status(SID) == "rejected"
    ev = rig.state.latest_evidence(SID, 1, "walk_forward")
    assert ev["passed"] == 0 and ev["data"]["failing"] == ["deflated_sharpe"]
    assert not rig.holdout.looks and rig.state.baseline(SID, 1) is None


def test_the_gate_is_recomputed_not_trusted_from_the_report(tmp_path):
    """A report that says validated but whose numbers miss the protected thresholds is still rejected."""
    rig = make_rig(tmp_path)
    rig.add_spec()
    rep = good_report(StrategySpec.from_dict(spec_dict()), oos={"dsr": 0.5})
    assert rep.status == "validated"
    rig.ev.wf_out[SID] = rep
    go(rig)
    assert rig.status(SID) == "rejected"
    assert rig.state.latest_evidence(SID, 1, "walk_forward")["data"]["failing"] == ["deflated_sharpe"]


def test_research_stages_never_see_holdout_bars_and_the_holdout_is_looked_at_once(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig)
    for stage in ("screen", "walk_forward"):
        assert rig.ev.seen_end[(stage, SID)] + pd.Timedelta(days=1) <= HOLDOUT_START, stage
    assert rig.ev.seen_end[("holdout", SID)] > HOLDOUT_START          # the holdout stage alone gets later bars
    assert len(rig.holdout.looks) == 1
    go(rig)                                                            # another run in the same week: nothing is looked at again
    assert len(rig.holdout.looks) == 1
    rig.run(run_id="2026-W42")                                         # a new week: evidence exists, still one look
    assert len(rig.holdout.looks) == 1


def test_a_data_port_that_returns_holdout_bars_to_the_research_stages_is_caught(tmp_path):
    class LeakyHoldout(FakeHoldout):
        def research_view(self, frames, tf):
            return frames                                              # a broken cut
    rig = make_rig(tmp_path, holdout=LeakyHoldout())
    rig.add_spec()
    r = go(rig)
    assert r["screen"].failed == 1 and "past the holdout start" in r["screen"].items[f"{SID}:v1"]["reason"]
    assert rig.status(SID) == "proposed" and ("screen", SID) not in rig.ev.calls


def test_holdout_failure_rejects_and_consumes_the_look(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.ev.holdout_out[SID] = {"trades": 12, "profit_factor": 0.9, "sharpe": -0.3, "max_drawdown": -0.6,
                               "total_return": -0.2, "win_rate": 0.3, "costs_share_of_gross": 0.2, "params": {}, "days": 200,
                               "warnings": []}
    go(rig)
    assert rig.status(SID) == "rejected"
    reasons = rig.state.latest_evidence(SID, 1, "holdout")["data"]["reasons"]
    assert len(reasons) == 4
    assert len(rig.holdout.looks) == 1


def test_holdout_is_not_looked_at_when_the_bars_do_not_cover_it_yet(tmp_path):
    frames = make_frames(n=1300)                                       # ends before the holdout start + 60 days
    rig = make_rig(tmp_path, data=FakeData(frames), holdout=FakeHoldout(pd.Timestamp("2022-06-01", tz="UTC")))
    rig.add_spec()
    r = go(rig)
    assert r["holdout"].blocked == 1 and not rig.holdout.looks
    assert rig.status(SID) == "validated"
    assert any("look is not used yet" in a["detail"] for a in rig.alerts.items)


def test_a_look_used_without_a_recorded_result_cannot_be_repeated(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.holdout.looks.append((SID, 1, "an earlier crashed run"))
    go(rig)
    assert rig.status(SID) == "rejected"
    ev = rig.state.latest_evidence(SID, 1, "holdout")
    assert ev["passed"] == 0 and ev["data"]["lost"] is True
    assert len(rig.holdout.looks) == 1 and ("holdout", SID) not in rig.ev.calls


def test_an_evaluation_error_after_the_look_is_recorded_as_a_failed_holdout(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.ev.fail[("holdout", SID)] = RuntimeError("engine blew up")
    go(rig)
    assert rig.status(SID) == "rejected"
    assert "engine blew up" in rig.state.latest_evidence(SID, 1, "holdout")["data"]["reasons"][0]
    assert len(rig.holdout.looks) == 1


def test_missing_real_data_blocks_the_item_with_an_alert_and_nothing_is_substituted(tmp_path):
    rig = make_rig(tmp_path, data=FakeData(missing={SID}))
    rig.add_spec()
    r = go(rig)
    assert r["screen"].blocked == 1 and r["screen"].done == 0
    assert rig.status(SID) == "proposed" and not rig.ev.calls
    assert any(a["level"] == "warning" and "blocked" in a["title"] for a in rig.alerts.items)
    assert rig.state.step("2026-W41", "screen", f"{SID}:v1")["status"] == "blocked"


def test_a_missing_holdout_lock_blocks_screening_too(tmp_path):
    from tradex.research.loop.adapters import UnavailableHoldout
    rig = make_rig(tmp_path)
    rig.ports.holdout = UnavailableHoldout("holdout lock not installed")
    rig.add_spec()
    r = go(rig)
    assert r["screen"].blocked == 1 and not rig.ev.calls
    assert rig.status(SID) == "proposed"


def test_a_non_real_data_provider_blocks_every_data_stage(tmp_path):
    class Fake(FakeData):
        real = False
    rig = make_rig(tmp_path, data=Fake())
    rig.add_spec()
    r = go(rig)
    assert {r[s].blocked for s in ("screen", "walk_forward", "holdout")} == {1}
    assert not rig.ev.calls and rig.status(SID) == "proposed"
    assert any(a["level"] == "critical" and "refused" in a["title"] for a in rig.alerts.items)


def test_screen_and_walk_forward_trials_are_counted_in_the_global_ledger(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig)
    assert rig.trials.count(SID) == 4                                   # the screen's set is one of the walk-forward grid points
    wf = rig.state.latest_evidence(SID, 1, "walk_forward")["data"]
    assert wf["n_trials"] >= wf["ledger_count"] == 4


def test_deflated_sharpe_n_below_the_ledger_count_is_an_error(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rep = good_report(StrategySpec.from_dict(spec_dict()), n_trials=1)

    class Under(ScriptedEvaluator):
        def walk_forward(self, spec, frames, trials, data_key):
            self._note("walk_forward", spec, frames)
            trials.record_many([dict(strategy_id=spec.id, version=1, params={"stop_atr": x}, run_id="r") for x in (1, 2, 3)])
            return rep
    rig.ports.evaluator = Under()
    r = go(rig)
    assert r["walk_forward"].failed == 1 and rig.status(SID) == "screened"


def test_editing_a_spec_after_screening_blocks_further_gates(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig, stages=("propose", "screen"))
    assert rig.status(SID) == "screened"
    raw = spec_dict(entry={"long": "close > ema50 and rsi2 < 5"}, status="screened")
    write_spec(rig.root / "strategies" / "proposed", raw)
    r = go(rig, run_id="2026-W42", stages=("propose", "walk_forward"))
    assert r["walk_forward"].blocked == 1 and "spec changed" in r["walk_forward"].items[f"{SID}:v1"]["reason"]
    assert any("content changed" in a["title"] for a in rig.alerts.items)
    assert rig.status(SID) == "screened"


def test_editing_a_spec_after_evidence_makes_it_ineligible_for_paper(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True), evaluator=None)
    rig.add_spec()
    go(rig, stages=("propose", "screen", "walk_forward", "holdout"))
    write_spec(rig.root / "strategies" / "proposed", spec_dict(exit={"stop_atr": 9.0, "target_r": 1.5, "max_bars": 7},
                                                               status="validated"))
    r = go(rig, run_id="2026-W42", stages=("propose", "promote"))
    assert rig.status(SID) == "validated"
    assert "spec changed after" in " ".join(r["promote"].items[f"{SID}:v1"]["problems"])


def test_a_new_version_supersedes_the_old_one_and_starts_again(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig, stages=("propose", "screen"))
    write_spec(rig.root / "strategies" / "proposed", spec_dict(version=2, status="proposed"))
    go(rig, run_id="2026-W42", stages=("propose",))
    assert rig.status(SID, 1) == "rejected" and rig.state.strategy(SID, 1)["reason"] == "superseded by v2"
    assert rig.status(SID, 2) == "proposed"


# --- resume -----------------------------------------------------------------------------------------

def test_a_cut_short_run_resumes_without_redoing_finished_items(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.add_spec(spec_dict("stk-second-test"))
    rig.ev.fail[("walk_forward", "stk-second-test")] = RuntimeError("OpenD went away")
    r1 = go(rig)
    assert r1["walk_forward"].failed == 1
    assert rig.state.run("2026-W41")["status"] == "partial"
    calls_before = list(rig.ev.calls)
    assert calls_before.count(("screen", SID)) == 1
    del rig.ev.fail[("walk_forward", "stk-second-test")]
    r2 = go(rig)                                                        # same ISO week: same run id
    assert r2["screen"].done == 0 and not r2["screen"].items            # both already left the proposed state
    assert r2["walk_forward"].done == 1                                 # only the one that failed is picked up again
    assert rig.ev.calls.count(("screen", SID)) == 1                    # nothing finished was repeated
    assert rig.ev.calls.count(("walk_forward", SID)) == 1
    assert rig.status("stk-second-test") == "validated"
    assert rig.state.run("2026-W41")["status"] == "complete"
    assert rig.state.step("2026-W41", "walk_forward", "stk-second-test:v1")["attempts"] == 2


def test_blocked_items_are_retried_when_the_data_arrives(tmp_path):
    data = FakeData(missing={SID})
    rig = make_rig(tmp_path, data=data)
    rig.add_spec()
    go(rig)
    assert rig.status(SID) == "proposed"
    data.missing.clear()
    go(rig)
    assert rig.status(SID) == "validated"


def test_run_id_defaults_to_the_iso_week_and_the_plan_is_recorded(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig)
    run = rig.state.run("2026-W41")
    assert run["plan"]["run_id"] == "2026-W41"
    assert any(s["stage"] == "screen" for s in run["plan"]["stages"])
    assert run["config"]["max_paper"] == 5


def test_stage_selection_and_unknown_stage(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    r = go(rig, stages=("propose", "screen"))
    assert list(r) == ["propose", "screen"] and rig.status(SID) == "screened"
    with pytest.raises(ValueError):
        run_loop(rig.ports, rig.cfg, rig.state, stages=("nope",))


def test_a_crashing_stage_does_not_stop_the_others(tmp_path, monkeypatch):
    from tradex.research.loop import stages
    rig = make_rig(tmp_path)
    rig.add_spec()
    monkeypatch.setitem(stages.STAGE_FUNCS, "implement", lambda ctx: 1 / 0)
    r = go(rig)
    assert r["implement"].failed == 1
    assert rig.status(SID) == "validated"
    assert any("stage implement crashed" in a["title"] for a in rig.alerts.items)
    assert rig.state.run("2026-W41")["status"] == "partial"


def test_audit_trail_records_every_transition_and_the_chain_verifies(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True))
    rig.add_spec()
    go(rig)
    moves = [(a["detail"]["from"], a["detail"]["to"]) for a in rig.state.audit_rows(strategy_id=SID) if a["event"] == "transition"]
    assert moves == [("proposed", "screened"), ("screened", "validated"), ("validated", "paper")]
    assert rig.state.verify_audit() == (True, None)
    rig.state.db.execute("UPDATE audit SET detail='{}' WHERE id=3")
    rig.state.db.commit()
    assert rig.state.verify_audit() == (False, 3)


def test_content_hash_used_for_evidence_matches_the_spec_file(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    go(rig)
    h = content_hash(rig.specs.get(SID).raw)
    assert {e["spec_hash"] for e in rig.state.evidence(SID, 1) if e["kind"] != "promotion"} == {h}
