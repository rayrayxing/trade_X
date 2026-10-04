"""The locked holdout: recent bars research never sees, and one look per strategy version.

``config/gates/holdout.yaml`` (protected) sets ``start``. ``research_view`` cuts every
frame so that only bars that have closed before ``start`` remain; the CLI applies it to
everything ``backtest`` and ``validate`` load, so walk-forward tuning and testing happen
on pre-holdout data only.

``holdout_look`` hands over the holdout bars for one strategy version exactly once and
records the look (who, when, why) in an append-only JSONL ledger. A second look at the
same version raises ``HoldoutLocked``: a strategy changed after seeing the holdout is a
new version, and the trial ledger already counts it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from tradex.timeframes import duration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "gates" / "holdout.yaml"


class HoldoutLocked(PermissionError):
    pass


@dataclass(frozen=True)
class HoldoutPolicy:
    start: pd.Timestamp
    looks_ledger: Path

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "HoldoutPolicy":
        doc = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()) or {}
        start = pd.Timestamp(doc["start"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        ledger = Path(doc.get("looks_ledger", "data/holdout/looks.jsonl"))
        return cls(start, ledger if ledger.is_absolute() else ROOT / ledger)


def research_view(frames: dict[str, pd.DataFrame], tf: str, policy: HoldoutPolicy | None = None
                  ) -> dict[str, pd.DataFrame]:
    """Each frame cut to the bars of ``tf`` that closed at or before the holdout start."""
    policy = policy or HoldoutPolicy.from_config()
    last_open = policy.start - duration(tf)
    return {s: df[df.index <= last_open] for s, df in frames.items()}


def looks(policy: HoldoutPolicy | None = None) -> list[dict]:
    policy = policy or HoldoutPolicy.from_config()
    if not policy.looks_ledger.exists():
        return []
    return [json.loads(ln) for ln in policy.looks_ledger.read_text().splitlines() if ln.strip()]


def holdout_look(frames: dict[str, pd.DataFrame], strategy_id: str, version: str | int, reason: str,
                 policy: HoldoutPolicy | None = None, now: pd.Timestamp | None = None
                 ) -> dict[str, pd.DataFrame]:
    """The holdout bars (opening at or after ``start``) for one look by ``strategy_id`` at
    ``version``. Records the look first; refuses a second look at the same version."""
    policy = policy or HoldoutPolicy.from_config()
    key = (strategy_id, str(version))
    for lk in looks(policy):
        if (lk["strategy_id"], str(lk["version"])) == key:
            raise HoldoutLocked(f"{strategy_id} v{version} already looked at the holdout on {lk['time']}")
    rec = {"time": (now or pd.Timestamp.now(tz="UTC")).isoformat(), "strategy_id": strategy_id,
           "version": str(version), "reason": reason, "holdout_start": policy.start.isoformat()}
    policy.looks_ledger.parent.mkdir(parents=True, exist_ok=True)
    with policy.looks_ledger.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return {s: df[df.index >= policy.start] for s, df in frames.items()}
