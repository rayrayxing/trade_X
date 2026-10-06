import pytest

from tradex.research.loop.ports import Idea
from tradex.research.loop.state import ALLOWED, IllegalTransition, LoopState, NotEligible

H = "abc123"


@pytest.fixture
def st():
    s = LoopState(":memory:")
    s.register("stk-a", 1, "/x/stk-a.yaml", H, family="trend", asset_class="stocks")
    return s


def move(st, to, **kw):
    st.transition("stk-a", 1, to, "test", run_id="r1", stage="test", **kw)


def evidence(st, kinds=("screen", "walk_forward", "holdout"), passed=True, h=H):
    for k in kinds:
        st.add_evidence("stk-a", 1, k, "r1", passed, h, {})


def test_the_state_machine_has_no_way_into_paper_except_from_validated():
    assert [k for k, v in ALLOWED.items() if "paper" in v] == ["validated"]
    assert ALLOWED["rejected"] == set() and ALLOWED["retired"] == set()


def test_illegal_moves_raise(st):
    for to in ("validated", "paper", "retired"):
        with pytest.raises(IllegalTransition):
            move(st, to)
    move(st, "rejected")
    with pytest.raises(IllegalTransition):
        move(st, "screened")


def test_paper_needs_all_three_pieces_of_evidence(st):
    move(st, "screened")
    move(st, "validated")
    for have, missing in (((), "screen"), (("screen",), "walk_forward"), (("walk_forward",), "holdout")):
        evidence(st, have)
        with pytest.raises(NotEligible, match=f"no {missing} evidence"):
            move(st, "paper", spec_hash=H)
    assert st.strategy("stk-a")["status"] == "validated"
    refused = [a for a in st.audit_rows() if a["event"] == "refused_paper"]
    assert len(refused) == 3 and refused[-1]["detail"]["problems"]


def test_paper_refuses_failed_evidence_and_changed_specs(st):
    move(st, "screened")
    move(st, "validated")
    evidence(st, ("screen", "walk_forward"))
    st.add_evidence("stk-a", 1, "holdout", "r1", False, H, {})
    with pytest.raises(NotEligible, match="holdout did not pass"):
        move(st, "paper", spec_hash=H)
    st.add_evidence("stk-a", 1, "holdout", "r2", True, H, {})
    with pytest.raises(NotEligible, match="spec changed after"):
        move(st, "paper", spec_hash="different")
    with pytest.raises(NotEligible, match="no spec hash"):
        move(st, "paper")
    move(st, "paper", spec_hash=H)
    assert st.strategy("stk-a")["promoted_at"]


def test_a_status_taken_from_a_file_is_never_eligible(st):
    st.register("stk-b", 1, "/x/b.yaml", H, status="validated", unverified="spec file says 'validated'")
    for k in ("screen", "walk_forward", "holdout"):
        st.add_evidence("stk-b", 1, k, "r1", True, H, {})
    assert "status came from outside the loop" in st.eligibility("stk-b", 1, H)[0]


def test_versions_are_separate_rows(st):
    assert st.register("stk-a", 2, "/x/a2.yaml", "h2") is True
    assert st.register("stk-a", 2, "/x/a2.yaml", "h2") is False
    assert st.strategy("stk-a")["version"] == 2 and st.strategy("stk-a", 1)["version"] == 1
    assert len(st.strategies()) == 2


def test_steps_track_attempts_and_status(st):
    st.begin_step("r1", "screen", "x")
    st.end_step("r1", "screen", "x", "failed", {"reason": "boom"}, "trace")
    assert st.step("r1", "screen", "x")["status"] == "failed"
    st.begin_step("r1", "screen", "x")
    st.end_step("r1", "screen", "x", "done", {"ok": 1})
    s = st.step("r1", "screen", "x")
    assert (s["status"], s["attempts"], s["result"], s["error"]) == ("done", 2, {"ok": 1}, "")


def test_run_start_is_idempotent(st):
    assert st.start_run("2026-W41", {"a": 1}, {"b": 2}) is True
    st.finish_run("2026-W41", "complete")
    assert st.start_run("2026-W41", {"a": 9}, {}) is False
    r = st.run("2026-W41")
    assert r["status"] == "running" and r["plan"] == {"a": 1}


def test_ideas_are_stored_once_with_their_payload(st):
    i = Idea(id="arxiv:2501.00001", source="arxiv", title="T", arxiv=("2501.00001",), assets=("stocks",))
    assert st.add_idea(i) is True and st.add_idea(i) is False
    row = st.idea(i.id)
    assert row["status"] == "new" and row["payload"]["arxiv"] == ["2501.00001"]
    st.set_idea(i.id, "specced", "spec x", draft_hash="d")
    assert st.ideas("new") == [] and st.ideas("specced")[0]["draft_hash"] == "d"


def test_audit_chain_detects_tampering_and_deletion(st):
    for n in range(4):
        st.audit("r1", "s", "e", "stk-a", {"n": n})
    assert st.verify_audit() == (True, None)
    st.db.execute("DELETE FROM audit WHERE id=2")
    st.db.commit()
    ok, bad = st.verify_audit()
    assert ok is False and bad == 3


def test_read_only_open_does_not_create_or_change_anything(tmp_path):
    p = tmp_path / "none.sqlite"
    with pytest.raises(Exception):
        LoopState(p, read_only=True)
    assert not p.exists()
    LoopState(p).close()
    ro = LoopState(p, read_only=True)
    assert ro.strategies() == []
    with pytest.raises(Exception):
        ro.register("x", 1, "p", "h")


def test_baseline_roundtrip(st):
    st.set_baseline("stk-a", 1, 0.001, 0.01, 250, -0.12)
    assert st.baseline("stk-a", 1) == {"strategy_id": "stk-a", "version": 1, "mean": 0.001, "std": 0.01, "n": 250,
                                       "max_drawdown": -0.12, "created": st.baseline("stk-a", 1)["created"]}
    assert st.baseline("stk-a", 2) is None
