import json

import pandas as pd
import pytest
import yaml

from agent_fakes import FakeGateway
from tradex.agents.common import InboxSink, ShadowStore
from tradex.agents.outcomes import ChartResolver, IncidentResolver, PositionResolver, ScoutResolver
from tradex.agents.position_agent import PositionInput, PositionReviewAgent
from tradex.agents.common import Voter
from tradex.agents.scorecard import (AGENT_CRITERIA, AGENTS, MIN_UNITS, ShadowScorecard, criteria_yaml, criterion_id,
                                     main)
from tradex.backtest.metrics import wilson_lower
from tradex.core.ledger import Ledger
from tradex.core.records import Health
from tradex.positions.review import MarketSnapshot, OpenPosition
from tradex.readiness import DEFAULT_PATH, evaluate, load_criteria

ASOF = pd.Timestamp("2026-10-06T00:00:00Z")
NOW = pd.Timestamp("2026-10-20T00:00:00Z")


def frame(rows, start="2026-10-05"):
    """rows: (open, high, low, close) per business day from ``start`` (hand-made, tests only)."""
    idx = pd.bdate_range(start, periods=len(rows), tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=1e6).astype(float)


def flat(o, c):
    return (o, max(o, c) + 0.1, min(o, c) - 0.1, c)


# --- scout -----------------------------------------------------------------------------------

def scout_decision(picks):
    return {"asof": str(ASOF), "body": {"picks": [{"symbol": s, "direction": d} for s, d in picks]}}


def test_scout_resolver_scores_picks_against_the_universe_mean():
    bars = {"A": frame([flat(100, 100), flat(100, 104)]), "B": frame([flat(100, 100), flat(100, 102)]),
            "C": frame([flat(100, 100), flat(100, 100)]), "D": frame([flat(100, 100), flat(100, 98)])}
    r = ScoutResolver(bars.get, list(bars))
    out = r(scout_decision([("A", 1), ("D", 1), ("C", -1), ("B", 0)]), NOW)
    assert out.units == 4 and out.hits == 2                       # A and C hit; D misses; B (watch) is not outsized
    by = {p["symbol"]: p for p in out.detail["picks"]}
    assert by["A"]["value"] == pytest.approx(0.03) and by["D"]["value"] == pytest.approx(-0.03)
    assert by["C"]["value"] == pytest.approx(0.01) and out.detail["baseline_ret"] == pytest.approx(0.01)
    assert out.value_sum == pytest.approx(0.03 - 0.03 + 0.01 + (0.01 - 0.02))


def test_scout_resolver_never_looks_at_the_decision_bar_and_waits_for_data():
    bars = {s: frame([flat(100, 150), flat(100, 101)]) for s in "ABC"}   # the 10-05 bar moved 50%: must not count
    out = ScoutResolver(bars.get, list(bars))(scout_decision([("A", 1)]), NOW)
    assert out.detail["baseline_ret"] == pytest.approx(0.01)
    # no forward bars yet -> not knowable
    early = {s: frame([flat(100, 100)]) for s in "ABC"}
    assert ScoutResolver(early.get, list(early))(scout_decision([("A", 1)]), NOW) is None
    # one pick lacks data: wait inside the grace period, score the rest afterwards
    part = {"A": frame([flat(100, 100), flat(100, 103)]), "B": frame([flat(100, 100), flat(100, 101)]),
            "C": frame([flat(100, 100), flat(100, 99)])}
    res = ScoutResolver(part.get, list(part))
    d = scout_decision([("A", 1), ("GONE", 1)])
    assert res(d, ASOF + pd.Timedelta(days=2)) is None
    late = res(d, ASOF + pd.Timedelta(days=11))
    assert late.units == 1 and late.detail["missing"] == ["GONE"]
    assert res(scout_decision([]), NOW) is None


# --- position ----------------------------------------------------------------------------------

def pos_decision(direction=1, r_now=0.5, bars_held=3, max_bars=5, stop=98.0, target=106.0):
    return {"asof": str(ASOF), "body": {"symbol": "X", "direction": direction, "entry_price": 100.0, "stop": stop,
                                        "target": target, "initial_risk": 2.0, "max_bars": max_bars,
                                        "bars_held": bars_held, "r_now": r_now}}


def resolve_pos(rows, **kw):
    return PositionResolver({"X": frame(rows, "2026-10-06")}.get)(pos_decision(**kw), NOW)


def test_position_resolver_counterfactual_exits():
    stop = resolve_pos([(101, 103, 99, 100), (100, 101, 97.5, 98)])
    assert stop.hits == 1 and stop.detail["exit_if_held"] == "stop"
    assert stop.value_sum == pytest.approx(0.5 - (-1.0))          # closing at +0.5R saved 1.5R
    tgt = resolve_pos([(101, 107, 100, 106)])
    assert tgt.hits == 0 and tgt.detail["exit_if_held"] == "target" and tgt.value_sum == pytest.approx(0.5 - 3.0)
    both = resolve_pos([(101, 107, 97, 100)])                      # stop and target in one bar: stop wins
    assert both.detail["exit_if_held"] == "stop"
    gap = resolve_pos([(96, 97, 95, 96)])                          # gapped through the stop: fills at the open
    assert gap.detail["r_if_held"] == pytest.approx(-2.0)
    timeout = resolve_pos([(100, 101, 99, 100.5)] * 2)             # 2 bars left (max 5, held 3), neither level touched
    assert timeout.detail["exit_if_held"] == "time" and timeout.detail["r_if_held"] == pytest.approx(0.25)
    assert timeout.hits == 1                                       # 0.5R at proposal beat 0.25R at the time stop


def test_position_resolver_short_and_waiting():
    d = pos_decision(direction=-1, r_now=0.5, stop=102.0, target=94.0)
    out = PositionResolver({"X": frame([(99, 103, 98, 101)], "2026-10-06")}.get)(d, NOW)
    assert out.detail["exit_if_held"] == "stop" and out.detail["r_if_held"] == pytest.approx(-1.0)
    one_bar = {"X": frame([(100, 101, 99, 100)], "2026-10-06")}.get
    assert PositionResolver(one_bar)(pos_decision(), ASOF + pd.Timedelta(days=3)) is None   # 2 bars still to come
    assert PositionResolver(one_bar)(pos_decision(), ASOF + pd.Timedelta(days=31)) is not None                        # grace over: score what exists
    assert PositionResolver(lambda s: None)(pos_decision(), NOW) is None


# --- chart --------------------------------------------------------------------------------------

def test_chart_resolver_checks_the_expected_direction_in_atr_units():
    rows = [(100, 101, 99, 100)] + [(100, 101, 94, 95)] * 4
    d = {"asof": str(ASOF), "body": {"symbol": "X", "expected_direction": -1, "atr": 2.0}}
    out = ChartResolver({"X": frame(rows, "2026-10-06")}.get, horizon_bars=5)(d, NOW)
    assert out.hits == 1 and out.value_sum == pytest.approx(2.5)
    d["body"]["expected_direction"] = 1
    assert ChartResolver({"X": frame(rows, "2026-10-06")}.get)(d, NOW).hits == 0
    assert ChartResolver({"X": frame(rows[:3], "2026-10-06")}.get)(d, NOW) is None


# --- incident -----------------------------------------------------------------------------------

def test_incident_resolver_recurrence(tmp_path):
    led = Ledger(tmp_path / "l.db")
    led.append(Health(time="2026-10-05T15:30:00+00:00", check="reconcile", ok=False, detail="again"))
    with pytest.raises(PermissionError):
        IncidentResolver(led)
    res = IncidentResolver(Ledger(tmp_path / "l.db", read_only=True), horizon=pd.Timedelta(hours=24))
    body = {"checks": ["reconcile"], "window_end": "2026-10-05T15:00:00+00:00", "severity": "high"}
    d = {"asof": body["window_end"], "body": body}
    assert res(d, pd.Timestamp("2026-10-05T20:00:00Z")) is None            # horizon not over
    out = res(d, pd.Timestamp("2026-10-06T16:00:00Z"))
    assert out.hits == 1 and out.detail["recurred"] is True
    assert res({**d, "body": {**body, "severity": "low"}}, pd.Timestamp("2026-10-06T16:00:00Z")).hits == 0
    other = {**d, "body": {**body, "checks": ["parity"]}}
    out = res(other, pd.Timestamp("2026-10-06T16:00:00Z"))
    assert out.hits == 0 and out.value_sum == -1.0


# --- scorecard ----------------------------------------------------------------------------------

def put(store, agent, units, hits, value=0.0, status="proposed"):
    did = store.add_decision(agent, ASOF, status, "flag", "t", {})
    store.set_outcome(did, {"units": units, "hits": hits, "value_sum": value * units}, NOW)
    return did


def test_replay_and_synthetic_rows_never_become_evidence_that_passes():
    store = ShadowStore(mode="replay", real_data=False)
    put(store, "scout_analyst", 100, 95, 0.02)
    paper_fake = ShadowStore(mode="paper", real_data=False)
    put(paper_fake, "scout_analyst", 100, 95, 0.02)
    for st in (store, paper_fake):
        ev = ShadowScorecard(st).evidence()
        out = {o.id: o for o in evaluate(AGENT_CRITERIA, ev)}
        assert out["shadow_scout_wilson_lb"].status == "unknown" and out["shadow_scout_wilson_lb"].ignored == 1


def test_real_paper_evidence_passes_fails_and_needs_enough_samples():
    def run(units, hits, value):
        st = ShadowStore(mode="paper", real_data=True)
        put(st, "chart_reader", units, hits, value)
        return {o.id: o for o in evaluate(AGENT_CRITERIA, ShadowScorecard(st).evidence())}
    good = run(40, 32, 0.4)
    assert good["shadow_chart_wilson_lb"].status == "pass" and good["shadow_chart_wilson_lb"].value > 0.5
    assert good["shadow_chart_mean_value"].status == "pass"
    bad = run(40, 18, -0.2)
    assert bad["shadow_chart_wilson_lb"].status == "fail" and bad["shadow_chart_mean_value"].status == "fail"
    few = run(MIN_UNITS - 1, MIN_UNITS - 1, 1.0)
    assert few["shadow_chart_wilson_lb"].status == "unknown"                # a small perfect streak is not evidence
    assert all(o.status == "unknown" for k, o in good.items() if not k.startswith("shadow_chart"))


def test_units_pool_across_decisions_and_live_outranks_paper():
    paper = ShadowStore(mode="paper", real_data=True)
    for _ in range(4):
        put(paper, "scout_analyst", 10, 8, 0.01)                             # 4 watchlists of 10 picks each
    ev = ShadowScorecard(paper).evidence()
    (item,) = ev["shadow_scout_wilson_lb"]
    assert item["n"] == 40 and item["mode"] == "paper" and item["real_data"] is True
    assert item["value"] == pytest.approx(wilson_lower(32, 40))
    # one store normally holds one run kind, but a file reused across runs mixes them: non-real, paper, live
    db = paper
    for mode, real in (("live", True), ("replay", False)):
        did = db.add_decision("scout_analyst", ASOF, "proposed", "flag", "t", {})
        db.set_outcome(did, {"units": 40, "hits": 4, "value_sum": -1.0}, NOW)
        db.db.execute("UPDATE shadow_decisions SET mode=?, real_data=? WHERE id=?", (mode, int(real), did))
    items = ShadowScorecard(db).evidence()["shadow_scout_wilson_lb"]
    assert [(i["mode"], i["real_data"]) for i in items] == [("replay", False), ("paper", True), ("live", True)]
    out = evaluate(AGENT_CRITERIA, {"shadow_scout_wilson_lb": items})[0]
    assert out.status == "fail" and out.ignored == 1                         # live (bad) is the latest real item


def test_resolve_fills_outcomes_once_and_isolates_failures():
    store = ShadowStore()
    a = store.add_decision("chart_reader", ASOF, "proposed", "flag", "X", {"symbol": "X", "expected_direction": 1, "atr": 1.0})
    b = store.add_decision("chart_reader", ASOF, "proposed", "flag", "Y", {"symbol": "Y", "expected_direction": 1, "atr": 1.0})
    store.add_decision("chart_reader", ASOF, "neutral", None, "Z", {})
    rows = [(100, 101, 99, 100)] + [(100, 106, 100, 105)] * 4
    def bars_fn(sym):
        if sym == "Y":
            raise RuntimeError("bar store unavailable")
        return frame(rows, "2026-10-06")
    sc = ShadowScorecard(store, {"chart_reader": ChartResolver(bars_fn)})
    assert sc.resolve(NOW) == {"chart_reader": 1}
    d = {x["id"]: x for x in store.decisions("chart_reader")}
    assert d[a]["outcome"]["hits"] == 1 and d[b]["outcome"] is None          # b is retried later, not lost
    assert len(sc.errors) == 1 and "bar store unavailable" in sc.errors[0]
    assert sc.resolve(NOW) == {}                                              # a is untouched; b fails again
    st = sc.stats()["chart_reader"]
    assert st.by_status == {"proposed": 2, "neutral": 1}
    (g,) = st.groups
    assert (g.proposals, g.resolved, g.units, g.hits) == (2, 1, 1, 1)         # the unresolved row is not evidence yet
    assert "chart_reader: 3 decisions" in sc.report() and "not real" in sc.report()


def test_end_to_end_position_agent_to_scorecard(tmp_path):
    store = ShadowStore(tmp_path / "s.db", mode="paper", real_data=True)
    yes = {"decision": "close", "confidence": 0.9, "reason": "r"}
    agent = PositionReviewAgent([Voter("a", FakeGateway(yes, provider="anthropic"), "position_reviewer"),
                                 Voter("b", FakeGateway(yes, provider="openai"), "position_reviewer_b")],
                                store, InboxSink(None, store))
    t = pd.Timestamp("2026-10-06T00:00:00Z")
    p = OpenPosition("X", "stocks", "s", 1, 10, t - pd.Timedelta(days=3), 100.0, 98.0, 106.0, 2.0, 10, bars_held=3)
    assert agent.review([PositionInput(p, MarketSnapshot(t, 101.0, 2.0, 24.0), "2026-10-02-0001")])[0].status == "proposed"
    sc = ShadowScorecard(store, {"position_reviewer": PositionResolver({"X": frame([(101, 103, 97, 99)], "2026-10-06")}.get)})
    sc.resolve(NOW)
    (g,) = sc.stats()["position_reviewer"].groups
    assert (g.mode, g.real, g.units, g.hits) == ("paper", True, 1, 1)
    assert g.value_sum == pytest.approx(0.5 + 1.0)                            # held would have been stopped at -1R


def test_criteria_cover_every_agent_and_the_yaml_block_is_valid_for_the_protected_file(tmp_path):
    assert {c.id for c in AGENT_CRITERIA} == {criterion_id(a, m) for a in AGENTS for m in ("wilson_lb", "mean_value")}
    assert all(c.real_data_only for c in AGENT_CRITERIA)
    base = DEFAULT_PATH.read_text().rstrip("\n") + "\n" + criteria_yaml()
    path = tmp_path / "readiness.yaml"
    path.write_text(base)
    merged = load_criteria(path)                                              # no duplicate IDs, same loader
    assert {c.id for c in merged} >= {c.id for c in AGENT_CRITERIA} | {c.id for c in load_criteria()}
    mine = {c.id: c for c in merged}
    for c in AGENT_CRITERIA:
        assert (mine[c.id].threshold, mine[c.id].real_data_only) == (c.threshold, True)
    assert yaml.safe_load("criteria:\n" + criteria_yaml())["criteria"][0]["id"] == AGENT_CRITERIA[0].id


def test_cli_prints_report_and_yaml(tmp_path, capsys):
    ShadowStore(tmp_path / "s.db")
    assert main(["--store", str(tmp_path / "s.db")]) == 0
    assert "Shadow scorecard" in capsys.readouterr().out
    assert main(["--store", str(tmp_path / "s.db"), "--yaml"]) == 0
    assert "shadow_scout_wilson_lb" in capsys.readouterr().out
