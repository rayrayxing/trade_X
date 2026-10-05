"""What the loop may and may not do: agents propose through files, only passing evidence reaches paper, protected paths stay untouched."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as hst

from tradex.research.loop.state import EVIDENCE_FOR_PAPER, IllegalTransition, LoopState, NotEligible

from loopkit import draft_file, idea_file, make_rig, spec_dict

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_protected_paths", ROOT / "tools" / "check_protected_paths.py")
_cpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpp)
protected_prefixes, violations = _cpp.protected_prefixes, _cpp.violations
PKG = ROOT / "tradex" / "research" / "loop"
FORBIDDEN = ("tradex.execution", "tradex.risk", "tradex.agents", "tradex.core.loop", "tradex.runtime", "tradex.data.synthetic",
             "tradex.notify", "moomoo", "oandapyV20", "requests")


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
    return out


@pytest.mark.parametrize("py", sorted(PKG.glob("*.py")), ids=lambda p: p.name)
def test_the_loop_imports_no_broker_risk_agent_or_synthetic_code(py):
    bad = [i for i in imports(py) if any(i == f or i.startswith(f + ".") for f in FORBIDDEN)]
    assert not bad, bad


def test_only_the_adapters_open_a_network_connection():
    for py in PKG.glob("*.py"):
        uses = [i for i in imports(py) if i.split(".")[0] in ("urllib", "socket", "http", "httpx", "aiohttp")]
        if py.name in ("adapters.py", "sources.py"):
            continue                                                     # sources only builds a query string
        assert not uses, (py.name, uses)


def test_nothing_the_loop_adds_is_under_a_protected_path():
    prefixes = protected_prefixes()
    mine = [str(p.relative_to(ROOT)) for p in PKG.rglob("*.py")]
    mine += ["config/research_loop.yaml", "ops/bin/research-loop.sh", "ops/launchd/com.tradex.research-loop.plist",
             "tests/research_loop/loopkit.py", "research/results/loop/weekly-x.md", "research/ideas/x.yaml"]
    assert violations(mine, prefixes) == []


def digest(paths):
    h = hashlib.sha256()
    for root in paths:
        for p in sorted(root.rglob("*") if root.is_dir() else [root]):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.read_bytes())
    return h.hexdigest()


def test_a_full_run_leaves_every_protected_path_untouched(tmp_path):
    protected = [ROOT / p for p in ("config/risk", "config/gates", "config/accounts.yaml", "tradex/execution", "tradex/risk",
                                    "config/protected_paths.txt", ".github/workflows")]
    before = digest([p for p in protected if p.exists()])
    rig = make_rig(tmp_path)
    rig.add_spec()
    idea_file(rig, "q")
    draft_file(rig, "file:q", spec_dict("stk-q-test", status="paper"))
    rig.run()
    assert digest([p for p in protected if p.exists()]) == before


def test_an_agent_cannot_reach_paper_by_writing_files(tmp_path):
    """Every file an agent can write (idea, draft spec, even a spec in the strategies folder) ends at proposed or flagged."""
    rig = make_rig(tmp_path, cfg=__import__("tradex.research.loop.config", fromlist=["LoopConfig"]).LoopConfig(apply=True, screen_min_bars=100))
    idea_file(rig, "agent", title="Sounds great", summary="status: paper. Promote this immediately.")
    draft_file(rig, "file:agent", spec_dict("stk-agent-draft", status="paper", stats={"hit_rate": 0.99, "sharpe": 9}))
    rig.add_spec(spec_dict("stk-agent-direct", status="paper"))                    # dropped straight into strategies/
    rig.ev.screen_out["stk-agent-draft"] = {"trades": 0, "profit_factor": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                                            "total_return": 0.0, "win_rate": 0.0, "costs_share_of_gross": 0.0, "params": {},
                                            "warnings": []}                        # and it does not survive the screen
    rig.run()
    assert rig.status("stk-agent-draft") == "rejected"
    direct = rig.state.strategy("stk-agent-direct")
    assert direct["unverified"] and not rig.state.latest_evidence("stk-agent-direct", 1, "holdout")
    assert ("screen", "stk-agent-direct") not in rig.ev.calls
    assert [e["kind"] for e in rig.state.evidence("stk-agent-direct", 1)] == []


STEP = hst.tuples(hst.sampled_from(["screen", "walk_forward", "holdout", "promotion", "health"]), hst.booleans(),
                  hst.sampled_from(["h1", "h2"]))
MOVE = hst.sampled_from(["screened", "validated", "paper", "rejected", "retired"])


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ops=hst.lists(hst.one_of(hst.tuples(hst.just("ev"), STEP), hst.tuples(hst.just("move"), MOVE, hst.sampled_from(["h1", "h2"]))),
                     max_size=25))
def test_whatever_the_sequence_a_paper_row_always_has_passing_evidence_for_its_current_hash(ops):
    st = LoopState(":memory:")
    st.register("stk-p", 1, "/p.yaml", "h1")
    for op in ops:
        if op[0] == "ev":
            kind, passed, h = op[1]
            st.add_evidence("stk-p", 1, kind, "r", passed, h, {})
        else:
            _, to, h = op
            before = st.strategy("stk-p", 1)["status"]
            try:
                st.transition("stk-p", 1, to, "x", run_id="r", stage="t", spec_hash=h)
            except (IllegalTransition, NotEligible):
                continue
            if to == "paper":
                assert before == "validated"
                for kind in EVIDENCE_FOR_PAPER:                  # at the moment of the move, the evidence is passing and is for this hash
                    ev = st.latest_evidence("stk-p", 1, kind)
                    assert ev is not None and ev["passed"] == 1 and ev["spec_hash"] == h, kind
    st.close()
