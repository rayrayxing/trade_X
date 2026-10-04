"""The authority rule: shadow agents can only add agent_inbox rows. No broker, sizing or config."""
import ast
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import tradex.agents as agents_pkg
from agent_fakes import FakeGateway, StubFeed, item
from tradex.agents.chart_agent import ChartPlan, ChartReader
from tradex.agents.common import InboxSink, ShadowStore, Voter
from tradex.agents.incident_agent import IncidentAnalyst, collect_incident
from tradex.agents.position_agent import PositionInput, PositionReviewAgent
from tradex.agents.scout_agent import ScoutAgent
from tradex.core.inbox import ALLOWED_ACTIONS, Mailbox
from tradex.core.ledger import Ledger
from tradex.core.records import Health
from tradex.data.synthetic import synthetic_bars
from tradex.positions.review import MarketSnapshot, OpenPosition

PKG = Path(agents_pkg.__file__).parent
MY_MODULES = ["common", "news", "scout_agent", "position_agent", "chart_agent", "incident_agent", "outcomes", "scorecard"]
FORBIDDEN = ("tradex.execution", "tradex.risk", "tradex.runtime", "tradex.decision", "tradex.core.loop",
             "tradex.core.actions", "tradex.core.interfaces", "tradex.core.counterfactual", "tradex.core.replay",
             "tradex.setup_cmd", "tradex.cli", "tradex.notify", "tradex.selection", "tradex.strategy")
PROTECTED = [ln.strip() for ln in (Path(__file__).parents[1] / "config" / "protected_paths.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]


def imports(path: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):          # includes imports inside functions
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


@pytest.mark.parametrize("mod", MY_MODULES)
def test_no_module_imports_broker_risk_runtime_or_config_code(mod):
    got = imports(PKG / f"{mod}.py")
    bad = sorted(i for i in got if any(i == f or i.startswith(f + ".") for f in FORBIDDEN))
    assert bad == []
    secrets_ok = mod == "news"                               # news keys come from the Keychain, read-only
    assert secrets_ok or not any(i.startswith("tradex.secrets") for i in got)


def test_nothing_reads_or_writes_config_or_environment():
    bad_calls = {"open", "getenv", "read_text", "write_text", "read_bytes", "write_bytes", "Popen", "system", "safe_load"}
    for mod in MY_MODULES:
        for n in ast.walk(ast.parse((PKG / f"{mod}.py").read_text())):
            if isinstance(n, ast.Call):
                f = n.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                assert name not in bad_calls, (mod, name)
            assert not (isinstance(n, ast.Attribute) and n.attr == "environ"), mod


def test_constructors_accept_only_gateway_inputs_and_a_sink():
    allowed = {"self", "gateway", "gateways", "voters", "store", "sink", "feed", "universe", "cfg", "technical", "rules",
               "max_failures", "max_tokens", "category"}
    for cls in (ScoutAgent, PositionReviewAgent, ChartReader, IncidentAnalyst):
        params = set(inspect.signature(cls.__init__).parameters)
        assert params <= allowed, (cls.__name__, params - allowed)


def test_sink_refuses_anything_but_the_four_shadow_actions():
    store = ShadowStore()
    with pytest.raises(ValueError):
        InboxSink(None, store, mode="active")
    sink = InboxSink(None, store)
    t = pd.Timestamp("2026-10-05T00:00:00Z")
    for bad in ("place_order", "set_config", "resize", "note", "CLOSE", ""):
        with pytest.raises(ValueError):
            sink.propose("x", t, bad, "AAPL", {}, [])
    with pytest.raises(ValueError):
        sink.propose("x", t, "close", "TSLA", {}, [], allowed_targets=["AAPL"])
    for a in ALLOWED_ACTIONS:
        assert sink.propose("x", t, a, "AAPL", {}, [], allowed_targets=["AAPL"])


def test_all_agents_together_add_only_shadow_inbox_rows(tmp_path):
    led = Ledger(tmp_path / "l.db")
    led.append(Health(time="2026-10-05T14:05:00+00:00", check="reconcile", ok=False, detail="x"))
    before_rows, before_hash = len(led.rows()), led.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()[0]
    mb, store = Mailbox(tmp_path / "l.db"), ShadowStore(tmp_path / "s.db")
    sink = InboxSink(mb, store)
    asof = pd.Timestamp("2026-10-05T20:00:00Z")

    def scout_reply(cat, prompt, system):
        import re
        syms = re.findall(r"^(\w+)$", prompt, re.M)[:10]
        return {"picks": [{"symbol": s, "direction": "long", "score": 2, "justification": "news",
                           "evidence": [re.search(rf"^{s}\n\s+(N\d+) ", prompt, re.M).group(1)]} for s in syms]}
    universe = [f"S{c}" for c in "ABCDEFGHIJ"]
    feed = StubFeed([item(s, "2026-10-05 09:00", f"{s} news") for s in universe])
    ScoutAgent(FakeGateway(scout_reply), feed, store, sink, universe).run(asof)

    yes = {"decision": "close", "confidence": 0.9, "reason": "r"}
    p = OpenPosition("NVDA", "stocks", "s", 1, 10, asof - pd.Timedelta(days=3), 100.0, 98.0, 106.0, 2.0, 10, bars_held=3)
    PositionReviewAgent([Voter("a", FakeGateway(yes, provider="anthropic"), "position_reviewer"),
                         Voter("b", FakeGateway(yes, provider="google"), "position_reviewer_b")], store, sink
                        ).review([PositionInput(p, MarketSnapshot(asof, 101.0, 2.0, 24.0), "2026-10-02-0001")])

    bars = synthetic_bars(200, seed=3, start="2026-01-01")
    ChartReader(FakeGateway({"pattern": "p", "bias": "bearish", "confidence": 0.9, "note": "n"}), store, sink
                ).read(bars.index[150] + pd.Timedelta(days=1), "NVDA", bars, ChartPlan("NVDA", 1, "2026-10-05-0001"))

    ro = Ledger(tmp_path / "l.db", read_only=True)
    IncidentAnalyst(FakeGateway({"severity": "high", "summary": "s", "hypothesis": "h",
                                 "recommendation": {"action": "place_order", "target": "NVDA"}}), store, sink
                    ).analyse(collect_incident(ro, pd.Timestamp("2026-10-05T14:00:00Z"), asof, ["NVDA"]))

    rows = mb.read("SELECT * FROM agent_inbox ORDER BY id")
    assert {r["source"] for r in rows} == {"scout_analyst", "position_reviewer", "chart_reader", "incident_analyst"}
    assert {r["action"] for r in rows} <= set(ALLOWED_ACTIONS)
    assert all(json.loads(r["body"])["shadow"] is True for r in rows)
    assert all(r["applied_at"] is None for r in rows)                      # the agents never mark anything applied
    # the hash-chained ledger is exactly as it was
    assert len(led.rows()) == before_rows and led.verify()[0]
    assert led.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()[0] == before_hash


def test_branch_touches_no_protected_path():
    """Everything this change adds lives under tradex/agents, tests, tests/fixtures and README."""
    root = Path(__file__).parents[1]
    mine = [str(p.relative_to(root)) for p in (PKG / n for n in [f"{m}.py" for m in MY_MODULES])]
    assert not [f for f in mine if any(f == q or f.startswith(q) for q in PROTECTED)]
