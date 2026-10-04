"""Runtime switches from config/runtime.yaml (agents mode, feed and scheduler limits)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tradex.core.actions import AGENT_MODES

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "runtime.yaml"


@dataclass
class RuntimeConfig:
    agents_mode: str = "shadow"
    stale_after_s: float = 120.0
    quote_max_age_s: float = 30.0
    grace_s: float = 5.0
    overrun_s: float = 60.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RuntimeConfig":
        p = Path(path) if path else DEFAULT_PATH
        doc = yaml.safe_load(p.read_text()) if p.exists() else {}
        doc = doc or {}
        a, f, s = doc.get("agents") or {}, doc.get("feed") or {}, doc.get("scheduler") or {}
        d = cls()
        out = cls(agents_mode=str(a.get("mode", d.agents_mode)),
                  stale_after_s=float(f.get("stale_after_s", d.stale_after_s)),
                  quote_max_age_s=float(f.get("quote_max_age_s", d.quote_max_age_s)),
                  grace_s=float(s.get("grace_s", d.grace_s)), overrun_s=float(s.get("overrun_s", d.overrun_s)))
        if out.agents_mode not in AGENT_MODES:
            raise ValueError(f"{p}: agents.mode must be one of {AGENT_MODES}, not {out.agents_mode!r}")
        return out
