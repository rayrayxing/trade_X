"""Agent rows go through the real core in shadow mode: recorded as 'would have', nothing applied."""
import pandas as pd

from agent_fakes import FakeGateway
from test_core_actions import Core
from tradex.agents.chart_agent import ChartPlan, ChartReader
from tradex.agents.common import InboxSink, ShadowStore, Voter
from tradex.agents.position_agent import PositionInput, PositionReviewAgent
from tradex.core.inbox import Mailbox
from tradex.core.ledger import Ledger
from tradex.data.synthetic import synthetic_bars
from tradex.positions.review import MarketSnapshot, OpenPosition


def make_core(tmp_path):
    led = Ledger(tmp_path / "l.db", git_commit="t")
    return Core("shadow", ledger=led), Mailbox(tmp_path / "l.db"), ShadowStore(tmp_path / "s.db")


def test_agent_close_proposal_is_recorded_not_executed(tmp_path):
    c, mb, store = make_core(tmp_path)
    c.until(lambda: c.core.agent_positions())
    p = c.core.agent_positions()[0]
    yes = {"decision": "close", "confidence": 0.9, "reason": "thesis broken"}
    agent = PositionReviewAgent([Voter("a", FakeGateway(yes, provider="anthropic"), "position_reviewer"),
                                 Voter("b", FakeGateway(yes, provider="openai"), "position_reviewer_b")],
                                store, InboxSink(mb, store))
    t = c.core.clock.now()
    op = OpenPosition(p.symbol, "stocks", "s", p.direction, p.qty, t - pd.Timedelta(days=2), 100.0, 98.0, 106.0, 2.0, 10,
                      bars_held=2)
    assert agent.review([PositionInput(op, MarketSnapshot(t, 101.0, 2.0, 24.0), p.decision_id)])[0].status == "proposed"
    c.step()
    agent_orders = [o for o in c.led.rows(kind="order", decision_id=p.decision_id) if "agent" in o["purpose"]]
    assert agent_orders == []
    out = c.outputs()[-1]
    assert out["agent"] == "position_reviewer" and out["action"] == "close" and out["target"] == p.decision_id
    assert out["body"]["applied"] is False and out["body"]["result"].startswith("shadow: would close")
    assert out["body"]["request"]["shadow"] is True
    assert p.decision_id in {q.decision_id for q in c.core.agent_positions()}      # position is still open
    assert c.led.verify()[0]


def test_chart_veto_of_a_pending_entry_is_recorded_not_applied(tmp_path):
    c, mb, store = make_core(tmp_path)
    c.until(lambda: c.core.pending_entries())
    did = c.core.pending_entries()[0]
    bars = synthetic_bars(200, seed=3, start="2026-01-01")
    reader = ChartReader(FakeGateway({"pattern": "double top", "bias": "bearish", "confidence": 0.9, "note": "n"}),
                         store, InboxSink(mb, store))
    r = reader.read(bars.index[150] + pd.Timedelta(days=1), "AAA", bars, ChartPlan("AAA", 1, did))
    assert r.action == "veto"
    c.core.desk.ingest(c.core.clock.now())
    out = c.outputs()[-1]
    assert out["body"]["applied"] is False and out["body"]["result"].startswith(f"shadow: would veto {did}")
    assert did in c.core.pending_entries()                                         # the entry is untouched
    assert not c.led.rows(kind="veto", decision_id=did)
