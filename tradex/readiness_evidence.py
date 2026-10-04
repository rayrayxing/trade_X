"""Evidence for the readiness scorecard, read from the ledger and the research results file.

``tradex.readiness.evaluate`` takes ``{criterion_id: [item, ...]}``; this module builds those items.
A ledger row counts as real only when its ``run_id`` says paper or live (``paper``, ``paper-oanda``,
``live:2026-11-02`` ...): replay, backtest and parity runs are measured too, but flagged non-real so
``evaluate`` drops them. Where the ledger does not record what a criterion needs, the item is left out
and the reason goes into ``reasons[id]`` so the scorecard shows ``unknown`` and why.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tradex.core.ledger import Ledger

GATE_RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "phase1_gate.json"
GO_COMMAND = "go_live_approved"
GO_SOURCES = ("telegram",)               # Ray, 4 Oct 2026: Telegram only; never cli, dashboard or an agent
GO_USER = "ray"
VENUES = {"forex": "oanda", "stocks": "moomoo"}
REAL_RUNS = ("paper", "live")
Reasons = dict[str, str]


def _mode(run_id: str) -> str:
    head = re.split(r"[^a-z]", (run_id or "").lower(), maxsplit=1)[0]
    return head if head in REAL_RUNS else "replay"


def _day(t: str) -> str:
    return str(t)[:10]


class _View:
    """Ledger rows of one data class (real or not), payloads decoded."""

    def __init__(self, led: Ledger, real: bool):
        self.rows: dict[str, list[dict[str, Any]]] = {}
        for r in led.db.execute("SELECT kind, run_id, payload FROM events ORDER BY seq"):
            if (_mode(r["run_id"]) in REAL_RUNS) == real:
                self.rows.setdefault(r["kind"], []).append(json.loads(r["payload"]))

    def of(self, kind: str) -> list[dict[str, Any]]:
        return self.rows.get(kind, [])


Metric = Callable[[_View], "tuple[float | None, str, dict[str, Any]]"]   # value, reason when None, detail


def clean_days(v: _View):
    per: dict[str, list[bool]] = {}        # a clean day has both checks, all ok
    seen: dict[str, set[str]] = {}
    for h in v.of("health"):
        if h.get("check") in ("reconcile", "parity"):
            per.setdefault(_day(h["time"]), []).append(bool(h.get("ok")))
            seen.setdefault(_day(h["time"]), set()).add(h["check"])
    per = {d: ok + ([] if len(seen[d]) == 2 else [False]) for d, ok in per.items()}
    if not per:
        return None, "ledger has no health rows with check=reconcile or check=parity", {}
    days = sorted(per)
    streak, prev = 0, None
    for d in reversed(days):
        if not all(per[d]):
            break
        cur = datetime.fromisoformat(d).date()
        if prev is not None and (prev - cur).days != 1:
            break
        streak, prev = streak + 1, cur
    return float(streak), "", {"days_with_checks": len(days), "latest": days[-1]}


def trades_per_venue(v: _View):
    closes = v.of("close")
    if not closes:
        return None, "no closed trades", {}
    cls = {p["decision_id"]: p.get("asset_class") for p in v.of("plan")}
    n = {name: 0 for name in VENUES.values()}
    for c in closes:
        ven = VENUES.get(cls.get(c["decision_id"]))
        if ven:
            n[ven] += 1
    return float(min(n.values())), "", {"closed_by_venue": n}


def cost_error(v: _View):
    plans = {p["decision_id"]: p for p in v.of("plan")}
    risk = {x["decision_id"]: x for x in v.of("verdict") if x.get("outcome") == "accepted"}
    closed = {c["decision_id"] for c in v.of("close")}
    meas: dict[str, float] = {}
    for f in v.of("fill"):
        meas[f["decision_id"]] = meas.get(f["decision_id"], 0.0) + float(f["fees_usd"]) + float(f["spread_slippage_usd"])
    ids = [d for d in meas if d in closed and d in plans and d in risk]
    model = sum(float(plans[d]["cost_r"]) * float(risk[d]["risk_usd"]) for d in ids)
    if not ids or model <= 0:
        return None, "no closed trade with fills, a plan cost_r and an accepted verdict risk_usd", {}
    got = sum(meas[d] for d in ids)
    return abs(got - model) / model * 100, "", {"trades": len(ids), "measured_usd": round(got, 2),
                                                  "modelled_usd": round(model, 2)}


def unstopped(v: _View):
    snaps = v.of("snapshot")
    if not snaps:
        return None, "no snapshot rows", {}
    bad = sum(1 for s in snaps for p in s.get("positions") or [] if not p.get("stop"))
    guard = sum(1 for h in v.of("health") if h.get("check") == "stop_guard" and not h.get("ok"))
    return float(bad + guard), "", {"snapshots": len(snaps), "positions_without_stop": bad, "stop_guard_failures": guard}


def tier_changes(v: _View):
    snaps = v.of("snapshot")
    if not snaps:
        return None, "no snapshot rows (limits.tier is recorded there)", {}
    last: dict[str, Any] = {}
    n = 0
    for s in snaps:
        t = (s.get("limits") or {}).get("tier")
        if s["book"] in last and t != last[s["book"]]:
            n += 1
        last[s["book"]] = t
    return float(n), "", {"note": "observed in paper/live snapshots; unit tests are not counted"}


def forward_vs_band(v: _View):
    return None, ("the ledger records no walk-forward bootstrap band for strategies and no decay_alarm health row, "
                  "so forward results cannot be compared to the backtest range yet"), {}


METRICS: dict[str, Metric] = {
    "clean_reconcile_parity": clean_days, "paper_trades_per_venue": trades_per_venue,
    "costs_vs_model": cost_error, "stops_on_every_position": unstopped, "governor_tiers_tested": tier_changes,
    "forward_inside_backtest_range": forward_vs_band,
}


def strategies_passing(path: str | Path = GATE_RESULTS):
    p = Path(path)
    if not p.exists():
        return None, f"{p.name} not found; run the phase 1 gate first", {}
    rows = json.loads(p.read_text()).get("rows", [])
    ok = [r["strategy_id"] for r in rows
          if r.get("result") == "pass" and r.get("data_source", "real") == "real" and r.get("bars_used", 0) > 0]
    return float(len(ok)), "", {"passing": ok, "rows": len(rows), "source": p.name}


def go_ahead(led: Ledger):
    n, who = 0, []
    for r in led.db.execute("SELECT time, source, args FROM commands WHERE command=? ORDER BY id", (GO_COMMAND,)):
        try:
            by = str(json.loads(r["args"]).get("by", "")).lower()
        except ValueError:
            by = ""
        if r["source"] in GO_SOURCES and by == GO_USER:
            n += 1
            who.append(f"{r['source']} {r['time']}")
    return float(n), {"approvals": who}


def _items(view_real: _View, view_other: _View, fn: Metric, reasons: Reasons, cid: str) -> list[dict[str, Any]]:
    out = []
    for view, real in ((view_other, False), (view_real, True)):
        val, why, det = fn(view)
        if val is None:
            if real:
                reasons[cid] = why
            continue
        out.append({"value": val, "mode": "paper" if real else "replay", "real_data": real, **det})
    return out


def collect(led: Ledger, gate_results: str | Path = GATE_RESULTS) -> tuple[dict[str, list[dict[str, Any]]], Reasons]:
    """``(evidence, reasons)``: evidence for ``evaluate``; reasons says why a criterion has none."""
    real, other = _View(led, True), _View(led, False)
    reasons: Reasons = {}
    ev = {cid: _items(real, other, fn, reasons, cid) for cid, fn in METRICS.items()}
    val, why, det = strategies_passing(gate_results)
    ev["strategies_passing_real_data"] = [] if val is None else [
        {"value": val, "mode": "real_history", "real_data": True, **det}]
    if val is None:
        reasons["strategies_passing_real_data"] = why
    n, det = go_ahead(led)
    ev["ray_go_ahead"] = [{"value": n, "mode": "live", "real_data": True, **det}]
    return ev, reasons


def scorecard(led: Ledger, gate_results: str | Path = GATE_RESULTS, criteria_path: str | Path | None = None):
    from tradex.readiness import DEFAULT_PATH, evaluate, load_criteria
    ev, reasons = collect(led, gate_results)
    outs = evaluate(load_criteria(criteria_path or DEFAULT_PATH), ev)
    for o in outs:
        if o.status == "unknown" and reasons.get(o.id):
            o.detail = reasons[o.id]
    return outs


def render(outcomes) -> str:
    from tradex.readiness import ready
    lines = []
    for o in outcomes:
        th = ", ".join(f"{k} {v:g}" for k, v in o.threshold.items())
        val = "-" if o.value is None else f"{o.value:g}"
        extra = {k: v for k, v in (o.evidence[-1] if o.evidence else {}).items()
                 if k not in ("value", "mode", "real_data")}
        note = o.detail or (json.dumps(extra, default=str) if extra else "")
        if o.ignored:
            note += f" [{o.ignored} non-real item(s) ignored]"
        lines.append(f"{o.status.upper():8} {o.id:32} value {val:>6} (need {th})  {note}".rstrip())
    lines.append("READY for Ray's go-ahead" if ready(outcomes) else "NOT READY")
    return "\n".join(lines)
