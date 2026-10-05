"""Shadow scorecard: agent proposals against what happened next, as readiness evidence.

``ShadowScorecard.resolve`` asks each agent's resolver (``tradex.agents.outcomes``) for the
outcome of every unresolved proposal and stores it. ``stats`` then summarises per agent and
per run kind, and ``evidence`` converts the numbers into the ``{criterion_id: [items]}``
shape ``tradex.readiness.evaluate`` takes.

Honesty rules, same as the readiness scorecard:
- every shadow row carries the run's ``mode`` and ``real_data`` flag (``ShadowStore``
  defaults to replay / False); evidence items carry them unchanged, so replay, test or
  synthetic rows can never turn a criterion green;
- no evidence is emitted for a group with fewer than ``min_units`` scored items, so a small
  lucky streak reads as "unknown", not "pass";
- the hit-rate figure is the Wilson lower bound (one-sided 95%), not the raw rate.

``AGENT_CRITERIA`` are the criteria these numbers feed. They live in code because
``config/gates/readiness.yaml`` is a protected path; ``criteria_yaml()`` prints the block to
append there (see the pull request), after which the default scorecard picks them up.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from tradex.agents.common import REAL_MODES, ShadowStore
from tradex.agents.outcomes import Resolver
from tradex.backtest.metrics import wilson_lower
from tradex.readiness import Criterion

# agent name -> short key used in criterion IDs, and what "value" means for it
AGENTS: dict[str, tuple[str, str]] = {
    "scout_analyst": ("scout", "mean return of a pick over the universe mean, in the pick's direction"),
    "position_reviewer": ("position_close", "mean R saved by the close versus holding to the plan's exit"),
    "chart_reader": ("chart", "mean move in the read's direction over the horizon, in ATR"),
    "incident_analyst": ("incident", "mean of +1 when a flag was warranted by what followed, else -1"),
}
MIN_UNITS = 30


@dataclass
class GroupStats:
    mode: str
    real_data: bool
    proposals: int = 0
    resolved: int = 0
    units: int = 0
    hits: int = 0
    value_sum: float = 0.0
    last_resolved: str = ""

    @property
    def hit_rate(self) -> float:
        return self.hits / self.units if self.units else 0.0

    @property
    def wilson_lb(self) -> float:
        return wilson_lower(self.hits, self.units)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.units if self.units else 0.0

    @property
    def real(self) -> bool:
        return self.real_data and self.mode in REAL_MODES


@dataclass
class AgentStats:
    agent: str
    by_status: dict[str, int] = field(default_factory=dict)
    groups: list[GroupStats] = field(default_factory=list)
    calls: int = 0
    calls_ok: int = 0

    @property
    def decisions(self) -> int:
        return sum(self.by_status.values())

    @property
    def call_success_rate(self) -> float:
        return self.calls_ok / self.calls if self.calls else 0.0


def criterion_id(agent: str, metric: str) -> str:
    return f"shadow_{AGENTS[agent][0]}_{metric}"


def _build_criteria() -> list[Criterion]:
    out = []
    for agent, (key, meaning) in AGENTS.items():
        out.append(Criterion(f"shadow_{key}_wilson_lb",
                             f"Shadow {agent}: lower bound of the hit rate of its proposals against later outcomes",
                             {"source": f"{agent} shadow proposals vs outcomes (tradex.agents.scorecard)",
                              "metric": "wilson_lower_bound_hit_rate"}, {"min": 0.5}, True))
        out.append(Criterion(f"shadow_{key}_mean_value",
                             f"Shadow {agent}: {meaning}",
                             {"source": f"{agent} shadow proposals vs outcomes (tradex.agents.scorecard)",
                              "metric": "mean_value_per_scored_item"}, {"min": 0.0}, True))
    return out


AGENT_CRITERIA: list[Criterion] = _build_criteria()


def criteria_yaml() -> str:
    """The block to append under ``criteria:`` in config/gates/readiness.yaml (a protected path)."""
    lines = []
    for c in AGENT_CRITERIA:
        lines += [f"  - id: {c.id}", f"    description: {json.dumps(c.description)}",
                  f"    evidence: {{source: {json.dumps(c.evidence['source'])}, metric: {c.evidence['metric']}}}",
                  f"    threshold: {{min: {c.threshold['min']}}}", "    real_data_only: true"]
    return "\n".join(lines) + "\n"


class ShadowScorecard:
    def __init__(self, store: ShadowStore, resolvers: dict[str, Resolver] | None = None):
        self.store, self.resolvers = store, resolvers or {}
        self.errors: list[str] = []

    def resolve(self, now: pd.Timestamp) -> dict[str, int]:
        """Fill outcomes for proposals whose result is knowable at ``now``. Returns count per agent."""
        now = pd.Timestamp(now)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        done: dict[str, int] = {}
        self.errors = []
        for agent, resolver in self.resolvers.items():
            for d in self.store.decisions(agent, "proposed", unresolved=True):
                try:
                    out = resolver(d, now)
                except Exception as exc:  # noqa: BLE001 - one bad row must not hide the others; it is retried next time
                    self.errors.append(f"{agent} decision {d['id']}: {type(exc).__name__}: {exc}")
                    continue
                if out is not None:
                    self.store.set_outcome(d["id"], asdict(out), now)
                    done[agent] = done.get(agent, 0) + 1
        return done

    def stats(self) -> dict[str, AgentStats]:
        res: dict[str, AgentStats] = {}
        for agent in AGENTS:
            st = AgentStats(agent)
            for r in self.store.db.execute("SELECT ok FROM shadow_calls WHERE agent=?", (agent,)):
                st.calls += 1
                st.calls_ok += r["ok"]
            groups: dict[tuple[str, bool], GroupStats] = {}
            for d in self.store.decisions(agent):
                st.by_status[d["status"]] = st.by_status.get(d["status"], 0) + 1
                if d["status"] != "proposed":
                    continue
                g = groups.setdefault((d["mode"], bool(d["real_data"])), GroupStats(d["mode"], bool(d["real_data"])))
                g.proposals += 1
                o = d["outcome"]
                if o:
                    g.resolved += 1
                    g.units += int(o["units"])
                    g.hits += int(o["hits"])
                    g.value_sum += float(o["value_sum"])
                    g.last_resolved = max(g.last_resolved, d["resolved_at"] or "")
            st.groups = sorted(groups.values(), key=lambda g: (g.real, g.mode == "live", g.mode))
            res[agent] = st
        return res

    def evidence(self, min_units: int = MIN_UNITS) -> dict[str, list[dict[str, Any]]]:
        """Input for ``tradex.readiness.evaluate(AGENT_CRITERIA, ...)``. Non-real groups are included
        (flagged as such) so the report can show they were ignored; they never count."""
        ev: dict[str, list[dict[str, Any]]] = {}
        for agent, st in self.stats().items():
            for g in st.groups:
                if g.units < min_units:
                    continue
                base = {"mode": g.mode, "real_data": g.real_data, "n": g.units, "agent": agent}
                ev.setdefault(criterion_id(agent, "wilson_lb"), []).append({**base, "value": g.wilson_lb})
                ev.setdefault(criterion_id(agent, "mean_value"), []).append({**base, "value": g.mean_value})
        return ev

    def report(self) -> str:
        lines = ["Shadow scorecard (agents propose, nothing is applied)", ""]
        for agent, st in self.stats().items():
            lines.append(f"{agent}: {st.decisions} decisions {dict(sorted(st.by_status.items()))}, "
                         f"model calls ok {st.calls_ok}/{st.calls}")
            for g in st.groups:
                tag = "REAL" if g.real else "not real (ignored by readiness)"
                lines.append(f"  {g.mode:7s} {tag}: {g.proposals} proposals, {g.resolved} resolved, {g.units} items, "
                             f"hit {g.hit_rate:.0%} (lower bound {g.wilson_lb:.0%}), mean value {g.mean_value:+.4f}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tradex.agents.scorecard", description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="path to the shadow store SQLite file")
    ap.add_argument("--yaml", action="store_true", help="print the readiness.yaml block instead")
    a = ap.parse_args(argv)
    if a.yaml:
        print(criteria_yaml(), end="")
        return 0
    print(ShadowScorecard(ShadowStore(a.store, mode="replay")).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
