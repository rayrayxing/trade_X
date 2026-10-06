"""Stages 6 and 7: who gets a paper slot, and when a paper strategy is demoted."""
import numpy as np
import pandas as pd

from tradex.research.loop.config import LoopConfig

from loopkit import FakeLive, NOW, good_report, oos_returns, spec_dict, make_rig
from tradex.strategy.spec import StrategySpec

APPLY = LoopConfig(screen_min_bars=100, apply=True)


def validated_rig(tmp_path, specs, cfg=APPLY, **kw):
    """A rig where every spec in ``specs`` went through screen, walk-forward and holdout (apply as configured)."""
    rig = make_rig(tmp_path, cfg=LoopConfig(**{**cfg.__dict__, "apply": False}), **kw)
    for raw in specs:
        rig.add_spec(raw)
    rig.run(stages=("propose", "screen", "walk_forward", "holdout"))
    rig.cfg = cfg
    return rig


def test_slots_go_to_the_best_candidates_by_dsr_then_holdout_sharpe(tmp_path):
    ids = ["stk-a-test", "stk-b-test", "stk-c-test"]
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=False, max_paper=2, max_paper_per_family=5))
    for i, sid in enumerate(ids):
        rig.add_spec(spec_dict(sid, family="trend" if i else "momentum"))
        rig.ev.wf_out[sid] = good_report(StrategySpec.from_dict(spec_dict(sid)), oos={"dsr": 0.96 + 0.01 * i})
    rig.run(stages=("propose", "screen", "walk_forward", "holdout"))
    rig.cfg = LoopConfig(screen_min_bars=100, apply=True, max_paper=2, max_paper_per_family=5)
    r = rig.run(run_id="2026-W42", stages=("promote",))
    assert {k: v["outcome"] for k, v in r["promote"].items.items()} == {
        "stk-c-test:v1": "promoted", "stk-b-test:v1": "promoted", "stk-a-test:v1": "waiting"}
    assert rig.status("stk-c-test") == "paper" and rig.status("stk-a-test") == "validated"
    assert "queue full" in r["promote"].items["stk-a-test:v1"]["reason"]


def test_one_family_cannot_take_every_slot(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True, max_paper_per_family=1))
    for sid in ("stk-a-test", "stk-b-test"):
        rig.add_spec(spec_dict(sid, family="trend"))
    r = rig.run()
    outcomes = sorted(v["outcome"] for v in r["promote"].items.values())
    assert outcomes == ["promoted", "waiting"]
    assert "family" in [v for v in r["promote"].items.values() if v["outcome"] == "waiting"][0]["reason"]


def test_existing_paper_strategies_use_up_slots(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True, max_paper=1, max_paper_per_family=5))
    rig.add_spec(spec_dict("stk-a-test"))
    rig.run()
    assert rig.status("stk-a-test") == "paper"
    rig.add_spec(spec_dict("stk-b-test"))
    r = rig.run(run_id="2026-W42")
    assert r["promote"].items["stk-b-test:v1"]["outcome"] == "waiting"


def test_without_apply_nothing_changes_status_into_paper(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.run()
    assert rig.status("stk-rsi-test") == "validated" and rig.file_status("stk-rsi-test") == "validated"
    assert any("recommended for the paper queue" in a["title"] for a in rig.alerts.items)


def test_strategies_without_holdout_evidence_are_not_eligible(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True))
    rig.add_spec()
    rig.run(stages=("propose", "screen", "walk_forward"))
    r = rig.run(run_id="2026-W42", stages=("promote",))
    assert r["promote"].items["stk-rsi-test:v1"]["outcome"] == "not_eligible"
    assert "no holdout evidence" in r["promote"].items["stk-rsi-test:v1"]["problems"]
    assert rig.status("stk-rsi-test") == "validated"


def test_stale_validated_strategies_are_retired(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.run(stages=("propose", "screen", "walk_forward"))
    later = NOW + pd.Timedelta(weeks=13)
    rig.ports.now = lambda: later
    rig.state._now = lambda: later
    r = rig.run(run_id="2026-W54", stages=("health",))
    assert r["health"].items["stale:stk-rsi-test:v1"]["outcome"] == "retired_stale"
    assert rig.status("stk-rsi-test") == "retired" and rig.file_status("stk-rsi-test") == "retired"


# --- health -----------------------------------------------------------------------------------------

def paper_rig(tmp_path, live_series, apply=True, **live_kw):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=apply), live=FakeLive({"stk-rsi-test": live_series}, **live_kw))
    rig.add_spec()
    rig.run(stages=("propose", "screen", "walk_forward", "holdout"))
    rig.cfg = LoopConfig(screen_min_bars=100, apply=True)
    rig.run(run_id="2026-W42", stages=("promote",))
    assert rig.status("stk-rsi-test") == "paper"
    return rig


def test_a_collapsing_paper_strategy_is_demoted_and_the_file_says_so(tmp_path):
    bad = pd.Series(np.random.default_rng(1).normal(-0.006, 0.01, 40), index=pd.bdate_range("2026-08-03", periods=40, tz="UTC"))
    rig = paper_rig(tmp_path, bad)
    r = rig.run(run_id="2026-W43", stages=("health",))
    out = r["health"].items["stk-rsi-test:v1"]
    assert out["outcome"] == "retired" and (out["cusum_alarm"] or out["drawdown_alarm"])
    assert rig.status("stk-rsi-test") == "retired" and rig.file_status("stk-rsi-test") == "retired"
    assert any(a["level"] == "critical" and "demoted" in a["title"] for a in rig.alerts.items)
    ev = rig.state.latest_evidence("stk-rsi-test", 1, "health")
    assert ev["passed"] == 0 and ev["data"]["n"] == 40


def test_the_cusum_alone_can_demote(tmp_path):
    bad = pd.Series(np.full(60, -0.012))
    rig = paper_rig(tmp_path, bad)
    rig.cfg = LoopConfig(screen_min_bars=100, apply=True, health_dd_multiple=1000.0)
    r = rig.run(run_id="2026-W43", stages=("health",))
    out = r["health"].items["stk-rsi-test:v1"]
    assert out["outcome"] == "retired" and out["cusum_alarm"] is True and out["drawdown_alarm"] is False
    assert out["cusum_min"] <= -8 and "CUSUM alarm" in out["reason"]


def test_without_apply_a_demotion_is_only_recommended(tmp_path):
    bad = pd.Series(np.random.default_rng(1).normal(-0.006, 0.01, 40))
    rig = paper_rig(tmp_path, bad)
    rig.cfg = LoopConfig(screen_min_bars=100, apply=False)
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"]["outcome"] == "retire_recommended"
    assert rig.status("stk-rsi-test") == "paper" and rig.file_status("stk-rsi-test") == "paper"
    assert any("would be demoted" in a["title"] for a in rig.alerts.items)


def test_a_healthy_paper_strategy_stays(tmp_path):
    good = oos_returns(60, mean=0.0012, std=0.01, seed=9)
    rig = paper_rig(tmp_path, good)
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"]["outcome"] == "healthy"
    assert rig.status("stk-rsi-test") == "paper"


def test_too_little_paper_history_is_not_judged(tmp_path):
    rig = paper_rig(tmp_path, oos_returns(5, mean=-0.05, std=0.01))
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"] == {"outcome": "insufficient_history", "n": 5, "need": 20}
    assert rig.status("stk-rsi-test") == "paper"


def test_missing_paper_results_are_recorded_not_guessed(tmp_path):
    rig = paper_rig(tmp_path, None, missing={"stk-rsi-test"})
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"]["outcome"] == "no_data"
    assert rig.status("stk-rsi-test") == "paper"


def test_a_paper_strategy_without_a_baseline_raises_an_alert(tmp_path):
    rig = paper_rig(tmp_path, oos_returns(60))
    rig.state.db.execute("DELETE FROM baselines")
    rig.state.db.commit()
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"]["outcome"] == "no_baseline"
    assert any("without a usable baseline" in a["title"] for a in rig.alerts.items)


def test_cusum_warning_alerts_without_demoting(tmp_path):
    live = pd.Series(np.full(30, -0.0130))                                # about -1.3 sigma a day against the reference: S falls ~0.85 a day
    rig = paper_rig(tmp_path, live)
    rig.cfg = LoopConfig(screen_min_bars=100, apply=True, cusum_h=40.0, health_dd_multiple=100.0)
    r = rig.run(run_id="2026-W43", stages=("health",))
    assert r["health"].items["stk-rsi-test:v1"]["outcome"] == "warning", r["health"].items
    assert rig.status("stk-rsi-test") == "paper"
    assert any("CUSUM warning" in a["title"] for a in rig.alerts.items)


def test_health_only_looks_at_paper_returns_since_promotion(tmp_path):
    seen = []

    class SinceLive(FakeLive):
        def daily_returns(self, strategy_id, version, since):
            seen.append(since)
            return oos_returns(30)
    rig = paper_rig(tmp_path, None)
    rig.ports.live = SinceLive()
    rig.run(run_id="2026-W43", stages=("health",))
    assert seen and seen[0] == pd.Timestamp(rig.state.strategy("stk-rsi-test")["promoted_at"])
