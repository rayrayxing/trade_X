"""Incident analyst agent: reads failed health checks and proposes a flag (or a narrow cut).

Input is an ``IncidentBundle`` built from the ledger's ``health`` rows (read only; the
analyst never gets a writable ledger handle). The model returns a severity, a short root-cause
hypothesis and one recommendation. Authority is clamped after the reply:

- ``flag`` is always allowed (target ``incident:<check>``);
- ``veto`` / ``shrink`` / ``close`` need a target from the bundle's ``symbols`` or
  ``decision_ids`` (what the caller says is affected), and ``close`` also needs severity
  ``high``; anything else is downgraded to a ``flag`` and the downgrade is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from tradex.agents.common import (INJECTION_GUARD, InboxSink, ModelGateway, ShadowStore, ask, clean, data_block, num,
                                  parse_json)
from tradex.core.ledger import Ledger

AGENT = "incident_analyst"
SEVERITIES = ("low", "medium", "high")

SYSTEM = ("You are the incident analyst for a trading system. You get failed health checks from the system's "
          "own log. Say how severe it is, what most likely went wrong, and what single action to recommend. "
          "Recommend flag (tell a human) unless a specific listed symbol or decision is clearly affected. "
          + INJECTION_GUARD)

SCHEMA_HINT = ('{"severity":"low|medium|high","summary":"one sentence","hypothesis":"likely cause",'
               '"evidence":[0,2],"recommendation":{"action":"flag|veto|shrink|close","target":"id from the lists",'
               '"factor":0.5,"reason":"one sentence"}}')


@dataclass
class IncidentBundle:
    start: pd.Timestamp
    end: pd.Timestamp
    failures: list[dict[str, Any]]                      # {time, check, detail}
    ok_counts: dict[str, int] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)

    @property
    def checks(self) -> list[str]:
        return sorted({f["check"] for f in self.failures})

    @property
    def key(self) -> str:
        return "+".join(self.checks)


def collect_incident(ledger: Ledger, since: pd.Timestamp, until: pd.Timestamp, symbols: Sequence[str] = (),
                     decision_ids: Sequence[str] = ()) -> IncidentBundle:
    """Failed health rows in [since, until] plus how many checks passed in the same window.
    Needs a read-only ledger handle: the analyst never holds a handle that can write the chain."""
    if not ledger.read_only:
        raise PermissionError("open the ledger with read_only=True for the incident analyst")
    fails, oks = [], {}
    for r in ledger.rows(kind="health"):
        t = pd.Timestamp(r["time"])
        if not (since <= t <= until):
            continue
        if r.get("ok"):
            oks[r["check"]] = oks.get(r["check"], 0) + 1
        else:
            fails.append({"time": r["time"], "check": r["check"], "detail": r.get("detail", "")})
    return IncidentBundle(since, until, fails, oks, list(symbols), list(decision_ids))


@dataclass
class IncidentResult:
    status: str                         # proposed | skipped | invalid | none | duplicate
    severity: str = ""
    action: str | None = None
    target: str | None = None
    summary: str = ""
    hypothesis: str = ""
    downgraded: str = ""
    decision_id: int | None = None


class IncidentAnalyst:
    def __init__(self, gateway: ModelGateway, store: ShadowStore, sink: InboxSink, max_failures: int = 40,
                 max_tokens: int = 800, category: str = AGENT):
        self.gw, self.store, self.sink = gateway, store, sink
        self.max_failures, self.max_tokens, self.category = max_failures, max_tokens, category

    def build_prompt(self, b: IncidentBundle) -> str:
        lines = [f"[{i}] {f['time']} check={clean(f['check'], 60)} detail={clean(f['detail'], 240)}"
                 for i, f in enumerate(b.failures[: self.max_failures])]
        ctx = [f"window {b.start.isoformat()} to {b.end.isoformat()}",
               f"{len(b.failures)} failed checks; passes in window: {dict(sorted(b.ok_counts.items())) or 'none'}",
               f"affected symbols you may target: {b.symbols or 'none'}",
               f"decision IDs you may target: {b.decision_ids or 'none'}"]
        return (f"Analyse this incident.\nReply as JSON: {SCHEMA_HINT}\n\nCONTEXT (from the trading system):\n"
                + "\n".join(f"- {x}" for x in ctx) + "\n\n" + data_block("FAILED_HEALTH_CHECKS", lines))

    def analyse(self, b: IncidentBundle) -> IncidentResult:
        if not b.failures:
            did = self.store.add_decision(AGENT, b.end, "none", note="no failed checks in the window")
            return IncidentResult("none", decision_id=did)
        flag_target = f"incident:{b.key}"
        if any(d["body"].get("check_key") == b.key
               for d in self.store.decisions(AGENT, "proposed", unresolved=True)):
            return IncidentResult("duplicate")
        prompt = self.build_prompt(b)
        res, cid = ask(self.store, self.gw, AGENT, self.category, prompt, SYSTEM, b.end, self.max_tokens)
        if not res.ok:
            did = self.store.add_decision(AGENT, b.end, "skipped", target=flag_target, call_ids=[cid],
                                          note=f"model call failed: {res.error}")
            return IncidentResult("skipped", decision_id=did)
        reply = parse_json(res.text) or {}
        sev = str(reply.get("severity") or "").lower()
        summary, hyp = clean(reply.get("summary"), 300), clean(reply.get("hypothesis"), 300)
        rec = reply.get("recommendation") if isinstance(reply.get("recommendation"), dict) else {}
        if sev not in SEVERITIES or not summary:
            did = self.store.add_decision(AGENT, b.end, "invalid", target=flag_target, call_ids=[cid],
                                          note="reply had no valid severity and summary")
            return IncidentResult("invalid", decision_id=did)
        action, target, extra, downgraded = self._clamp(b, sev, rec)
        first = min(pd.Timestamp(f["time"]) for f in b.failures)
        body = {"kind": "incident", "checks": b.checks, "check_key": b.key, "severity": sev, "summary": summary,
                "hypothesis": hyp, "n_failures": len(b.failures), "first_failure": first.isoformat(),
                "window_end": b.end.isoformat(), "downgraded": downgraded,
                "reason": clean(rec.get("reason"), 300) or summary, **extra}
        did = self.sink.propose(AGENT, b.end, action, target, body, [cid],
                                allowed_targets=[flag_target, *b.symbols, *b.decision_ids],
                                providers=[res.provider], models=[res.model])
        return IncidentResult("proposed", sev, action, target, summary, hyp, downgraded, did)

    def _clamp(self, b: IncidentBundle, sev: str, rec: dict) -> tuple[str, str, dict, str]:
        flag = f"incident:{b.key}"
        action = str(rec.get("action") or "flag").lower()
        target = str(rec.get("target") or "").strip()
        if action == "flag":
            return "flag", flag, {}, ""
        if action not in ("veto", "shrink", "close"):
            return "flag", flag, {}, f"unknown action {action!r}"
        if target not in set(b.symbols) | set(b.decision_ids):
            return "flag", flag, {}, f"{action} target {target!r} not in the affected lists"
        if action == "close" and sev != "high":
            return "flag", flag, {}, "close needs severity high"
        extra: dict[str, Any] = {}
        if action == "shrink":
            f = num(rec.get("factor"))
            if f is None or not 0.0 <= f < 1.0:
                return "flag", flag, {}, "shrink needs a factor below 1"
            extra["factor"] = f
        if action == "close":
            extra["fraction"] = 1.0
        return action, target, extra, ""
