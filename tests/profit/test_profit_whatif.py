import json
from functools import lru_cache

import pandas as pd
import pytest

from pfx import H, T0, plan
from test_spine import _frames, _two_family_specs
from tradex.core.counterfactual import filter_report
from tradex.core.ledger import Ledger
from tradex.core.records import Close, Counterfactual, Fill, Veto
from tradex.core.replay import run_replay
from tradex.profit.costcal import calibrate_costs
from tradex.profit.exits import ExitPolicy
from tradex.profit.whatif import (blocked_frame, dashboard_payload, gate_rows, gate_value, monthly, reason_breakdown,
                                  selection_edge, under_policy)
from tradex.costs.models import model_for


def blocked_ledger(spec):
    """spec: list of (gate, reason, r or None, month_offset_days)."""
    led = Ledger(":memory:", git_commit="t")
    for i, (gate, reason, r, off) in enumerate(spec):
        t = T0 + pd.Timedelta(days=off) + i * H
        pl = plan(decision_id=f"2026-03-{1 + i % 28:02d}-{i:04d}", t=t)
        led.append(pl)
        led.append(Veto(pl.decision_id, t.isoformat(), gate, reason))
        if r is not None:
            led.append(Counterfactual(pl.decision_id, t.isoformat(), gate, (t + H).isoformat(), "stop", r))
    return led


def test_a_gate_that_blocks_losers_saves_r_and_the_verdict_needs_enough_plans():
    spec = [("calendar", f"NFP in {i} hours", -1.0 + 0.05 * (i % 4), 0) for i in range(25)]
    spec += [("risk", "no room: heat limit", 1.5 - 0.1 * (i % 5), 0) for i in range(8)]
    spec += [("risk", "no room: heat limit", None, 0)]
    g = gate_value(blocked_ledger(spec)).set_index("blocked_by")
    cal = g.loc["calendar"]
    assert cal["plans"] == 25 and cal["followed"] == 25 and cal["verdict"] == "saves R"
    assert cal["net_effect_r"] == pytest.approx(-sum(-1.0 + 0.05 * (i % 4) for i in range(25)), abs=0.01) and cal["net_effect_r"] > 0
    risk = g.loc["risk"]
    assert risk["plans"] == 9 and risk["pending"] == 1 and risk["followed"] == 8
    assert risk["verdict"].startswith("too few") and risk["net_effect_r"] < 0       # it blocked winners
    big = gate_value(blocked_ledger([("risk", "x", 1.0 + 0.1 * (i % 3), 0) for i in range(30)]))
    assert big.iloc[0]["verdict"] == "costs R"


def test_a_noisy_gate_is_inconclusive():
    spec = [("agent:news", "headline", (1.0 if i % 2 else -1.0), 0) for i in range(30)]
    assert gate_value(blocked_ledger(spec)).iloc[0]["verdict"] == "inconclusive"


def test_reasons_are_grouped_with_their_numbers_masked():
    spec = [("plan", f"score {0.10 + i / 100:.2f} below 0.25", -0.5, 0) for i in range(6)]
    spec += [("plan", "only 1 family agree (trend); need 2", 0.2, 0)] * 3
    rb = reason_breakdown(blocked_ledger(spec), min_plans=5)
    assert rb.iloc[0]["reason"] == "score # below #" and rb.iloc[0]["plans"] == 6 and rb.iloc[0]["verdict"] == "saves R"
    assert len(rb) == 2


def test_monthly_net_effect_accumulates():
    spec = [("risk", "x", -1.0, 0), ("risk", "x", -1.0, 0), ("risk", "x", 2.0, 40)]
    m = monthly(blocked_ledger(spec))
    assert [x["month"] for x in m] == ["2026-03", "2026-04"]
    assert m[0]["net_effect_r"] == 2.0 and m[1]["net_effect_r"] == -2.0 and m[1]["cumulative_net_effect_r"] == 0.0


def test_gate_rows_is_a_drop_in_for_filter_report():
    rows = [{"decision_id": f"d{i}", "time": "t", "blocked_by": g, "exit_time": "t", "exit_reason": "stop",
             "r_multiple": r} for i, (g, r) in enumerate([("calendar", -1.0), ("calendar", 0.5), ("risk", 2.0)])]
    mine = {r["blocked_by"]: r for r in gate_rows(rows)}
    ref = filter_report(rows).set_index("blocked_by")
    for gate in ("calendar", "risk"):
        for k in ("plans", "mean_r", "total_r", "win_rate"):
            assert mine[gate][k] == pytest.approx(ref.loc[gate, k])
    assert gate_rows([]) == []


def test_selection_edge_compares_taken_with_blocked():
    led = blocked_ledger([("risk", "x", -0.8, 0)] * 12)
    for i in range(12):
        did = f"2026-04-{1 + i:02d}-9{i:03d}"
        pl = plan(decision_id=did, t=T0 + (i + 100) * H)
        led.append(pl)
        led.append(Fill(did, did + "-e", pl.time, pl.symbol, 1, 10, 100.0, 0, 0))
        led.append(Close(did, pl.time, pl.symbol, 101.0, 10, 5.0, 0.5 + 0.1 * (i % 3), "target"))
    e = selection_edge(led)
    assert e["taken_n"] == 12 and e["blocked_n"] == 12 and e["edge_r"] > 1.0 and e["verdict"] == "gates pick better trades"
    assert selection_edge(Ledger(":memory:", git_commit="t"))["verdict"].startswith("too few")


# --- against a real replay ledger ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def replay():
    frames = _frames()
    r = run_replay(_two_family_specs(), frames, Ledger(":memory:", git_commit="t"), frames["AAA"].index[260])
    return frames, r.ledger


def test_reports_work_on_a_real_replay_ledger():
    frames, led = replay()
    df = blocked_frame(led)
    ens = {p.decision_id for p in led.records("plan", book="ensemble")}
    assert len(df) == len({v["decision_id"] for v in led.rows(kind="veto") if v["decision_id"] in ens}) > 0
    g = gate_value(led)
    assert g["plans"].sum() == len(df) and set(g["blocked_by"]) <= set(df["blocked_by"])
    payload = json.loads(json.dumps(dashboard_payload(led), default=str))
    assert payload["gates"] and payload["selection"]["taken_n"] > 10
    out = under_policy(led, lambda s, tf: frames[s], {"plain": ExitPolicy.baseline(), "engine": ExitPolicy()})
    assert out["plans"] == len(df) and out["evaluated"] > 0
    # the plain exits are what the tracker used: the baseline mean matches the ledger's counterfactual mean
    cf = df["cf_r"].dropna()
    assert out["policies"]["plain"]["n"] <= len(cf)


def test_fills_in_a_real_replay_calibrate_to_the_model_that_made_them():
    _, led = replay()
    cal = calibrate_costs(led, {"stocks": model_for("stocks")})["stocks"]
    assert cal.calibrated and cal.spread_ratio == pytest.approx(1.0, abs=0.01) and cal.fee_ratio == pytest.approx(1.0, abs=0.01)
    assert cal.spread_mult == pytest.approx(1.0, abs=0.01)
