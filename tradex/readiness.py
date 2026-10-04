"""Readiness scorecard: loads config/gates/readiness.yaml and scores each criterion.

Result per criterion is pass, fail or unknown (no evidence yet). Evidence flagged
``real_data_only`` is ignored unless its ``source`` field says it is real and live
(paper or live run): synthetic or replay numbers can never turn a criterion green.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "gates" / "readiness.yaml"
REAL_MODES = ("paper", "live", "real_history")   # real_history: real vendor bars (research gate), never synthetic


@dataclass
class Criterion:
    id: str
    description: str
    evidence: dict[str, Any]
    threshold: dict[str, float]
    real_data_only: bool = True


@dataclass
class Outcome:
    id: str
    status: str                        # pass | fail | unknown
    value: float | None
    threshold: dict[str, float]
    detail: str = ""
    ignored: int = 0                   # evidence items dropped for not being real
    evidence: list[dict[str, Any]] = field(default_factory=list)


def load_criteria(path: str | Path = DEFAULT_PATH) -> list[Criterion]:
    doc = yaml.safe_load(Path(path).read_text())
    out = [Criterion(c["id"], c["description"], c["evidence"], c["threshold"], bool(c.get("real_data_only", True)))
           for c in doc["criteria"]]
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate criterion id")
    return out


def _is_real(item: dict[str, Any]) -> bool:
    return item.get("mode") in REAL_MODES and item.get("real_data", False) is True


def evaluate(criteria: list[Criterion], evidence: dict[str, list[dict[str, Any]]]) -> list[Outcome]:
    """``evidence[id]`` is a list of ``{"value": float, "mode": "paper"|"live"|..., "real_data": bool, ...}``;
    the latest real item decides. A real-data-only criterion with only non-real items is unknown."""
    out = []
    for c in criteria:
        items = list(evidence.get(c.id, []))
        usable = [i for i in items if _is_real(i)] if c.real_data_only else items
        dropped = len(items) - len(usable)
        if not usable:
            out.append(Outcome(c.id, "unknown", None, c.threshold, "no real-data evidence yet", dropped, items))
            continue
        v = float(usable[-1]["value"])
        lo, hi = c.threshold.get("min"), c.threshold.get("max")
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        out.append(Outcome(c.id, "pass" if ok else "fail", v, c.threshold, "", dropped, usable))
    return out


def ready(outcomes: list[Outcome]) -> bool:
    return bool(outcomes) and all(o.status == "pass" for o in outcomes)
