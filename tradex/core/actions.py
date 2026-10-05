"""What agents may do to the ensemble book, applied by the core at each bar close.

Agents ask through ``agent_inbox`` rows; the core ingests them before it processes the
bars (``AgentDesk.ingest``), so a request written during a bar acts at that bar's close.
Only the ensemble book is touched; virtual books are research and stay agent-free.

- veto: a decision ID cancels that plan's entry while it is still unfilled. A symbol also
  cancels its unfilled entries and blocks new plans on it at gate 3 until ``body.until``
  (ISO time) or for ``body.bars`` base bars (default 1: plans at this close).
- shrink: ``body.factor`` in (0, 1), or ``body.qty`` for a decision, cuts the size of an
  unfilled entry (cancel, then re-place under the same verdict) or of plans on a symbol.
  It never increases a size.
- close: a reducing exit for the decision's (or the symbol's) open position, through the
  same broker and order guard as every other exit; ``body.fraction`` closes part of it.
- flag: recorded only.

Agents start in shadow (``agents.mode`` in config/runtime.yaml, default shadow): for their
first 60 trading days every handler records what it would have done and applies nothing,
so their rows can be scored against what the trades actually did before they get power.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradex.core.inbox import IngestResult, ingest_inbox
from tradex.core.records import AgentOutput, TradePlan

if TYPE_CHECKING:
    from tradex.core.loop import TradingCore

AGENT_MODES = ("shadow", "active")
DECISION_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}$")


@dataclass
class StandingRule:
    """A symbol-wide veto or shrink that gate 3 applies to new ensemble plans until ``until``."""
    inbox_id: int
    source: str
    action: str                       # veto | shrink
    symbol: str
    until: pd.Timestamp
    factor: float = 1.0
    reason: str = ""
    shadow: bool = True


def shrink_qty(qty: float, factor: float) -> float:
    """``qty * factor`` rounded down; whole units stay whole (shares, Oanda units)."""
    q = qty * factor
    return float(math.floor(q + 1e-9)) if float(qty).is_integer() else q


class AgentDesk:
    def __init__(self, core: "TradingCore", mode: str = "shadow"):
        if mode not in AGENT_MODES:
            raise ValueError(f"agents.mode must be one of {AGENT_MODES}, not {mode!r}")
        self.core, self.mode = core, mode
        self.rules: list[StandingRule] = []
        self.t: pd.Timestamp | None = None

    @property
    def shadow(self) -> bool:
        return self.mode != "active"

    def ingest(self, t: pd.Timestamp) -> list[IngestResult]:
        self.t = t
        self.rules = [r for r in self.rules if r.until > t]
        return ingest_inbox(self.core.ledger, t.isoformat(),
                            {"veto": self.veto, "shrink": self.shrink, "close": self.close, "flag": self.flag})

    # --- handlers: each returns (result text, applied) ---------------------------------

    def _done(self, text: str) -> tuple[str, bool]:
        return (f"shadow: would {text}", False) if self.shadow else (text, True)

    def _until(self, body: dict[str, Any]) -> pd.Timestamp:
        if body.get("until"):
            u = pd.Timestamp(body["until"])
            return u.tz_localize("UTC") if u.tzinfo is None else u.tz_convert("UTC")
        return self.t + self.core.bar * max(1, int(body.get("bars", 1)))

    def veto(self, row: dict[str, Any]) -> tuple[str, bool]:
        target, src, body = row["target"], row["source"], row["body"]
        reason = str(body.get("reason") or "agent veto")
        if DECISION_ID.match(target):
            if target not in self.core.pending_entries():
                return f"no unfilled entry for {target}", False
            if not self.shadow and not self.core.cancel_entry(target, f"agent:{src}", reason, self.t):
                return f"could not cancel the entry of {target}", False
            return self._done(f"veto {target}: entry cancelled")
        pend = self.core.pending_entries(target)
        if not self.shadow:
            pend = [d for d in pend if self.core.cancel_entry(d, f"agent:{src}", reason, self.t)]
        until = self._until(body)
        self.rules.append(StandingRule(row["id"], src, "veto", target, until, reason=reason, shadow=self.shadow))
        return self._done(f"veto {target} until {until.isoformat()}; cancelled entries: {pend or 'none'}")

    def shrink(self, row: dict[str, Any]) -> tuple[str, bool]:
        target, src, body = row["target"], row["source"], row["body"]
        factor = float(body["factor"]) if body.get("factor") is not None else None
        if factor is not None and not 0.0 <= factor < 1.0:
            return "ignored: shrink factor must be in [0, 1); a shrink never increases a size", False
        why = f"agent {src}: {body.get('reason') or 'shrink'}"
        if DECISION_ID.match(target):
            if target not in self.core.pending_entries():
                return f"no unfilled entry for {target}", False
            cur = self.core.entry_qty(target)
            if factor is not None:
                new = shrink_qty(cur, factor)
            elif body.get("qty") is not None:
                new = float(body["qty"])
            else:
                return "ignored: shrink needs body.factor or body.qty", False
            if new >= cur:
                return f"ignored: {new:g} is not smaller than {cur:g}; a shrink never increases a size", False
            if not self.shadow and not self.core.resize_entry(target, new, self.t, why):
                return f"could not resize the entry of {target}", False
            return self._done(f"shrink {target} from {cur:g} to {new:g}")
        if factor is None:
            return "ignored: a symbol shrink needs body.factor", False
        done = []
        for did in self.core.pending_entries(target):
            cur = self.core.entry_qty(did)
            if self.shadow or self.core.resize_entry(did, shrink_qty(cur, factor), self.t, why):
                done.append(did)
        until = self._until(body)
        self.rules.append(StandingRule(row["id"], src, "shrink", target, until, factor, why, self.shadow))
        return self._done(f"shrink plans on {target} by {factor:g} until {until.isoformat()}; "
                          f"resized entries: {done or 'none'}")

    def close(self, row: dict[str, Any]) -> tuple[str, bool]:
        target, src, body = row["target"], row["source"], row["body"]
        frac = float(body.get("fraction", 1.0))
        if not 0.0 < frac <= 1.0:
            return "ignored: close fraction must be in (0, 1]", False
        pos = self.core.agent_positions(target)
        if not pos:
            return f"no open agent position for {target}", False
        why = f"agent {src}: {body.get('reason') or 'close request'}"
        sent = []
        for p in pos:
            qty = p.qty if frac >= 1.0 else shrink_qty(p.qty, frac)
            if qty <= 0:
                continue
            if self.shadow or self.core.exit_position(p, qty, self.t, why, f"{p.decision_id}-agent-{row['id']}"):
                sent.append(f"{p.decision_id} {qty:g}")
        if not sent:
            return f"no exit sent for {target}", False
        return self._done(f"close {', '.join(sent)}")

    def flag(self, row: dict[str, Any]) -> tuple[str, bool]:
        return "flagged for review", True

    # --- gate 3 ------------------------------------------------------------------------

    def at_gate(self, plan: TradePlan, t: pd.Timestamp) -> tuple[StandingRule | None, float]:
        """(veto rule, size factor) for a new ensemble plan. Shadow rules are recorded only."""
        veto, factor = None, 1.0
        for r in self.rules:
            if r.symbol != plan.symbol or r.until <= t:
                continue
            if r.shadow:
                self.core.ledger.append(AgentOutput(
                    time=t.isoformat(), agent=r.source, provider="", model="", action=r.action,
                    target=plan.decision_id, body={"inbox_id": r.inbox_id, "applied": False, "shadow": True,
                                                   "would": r.action, "factor": r.factor, "reason": r.reason}))
            elif r.action == "veto" and veto is None:
                veto = r
            elif r.action == "shrink":
                factor *= r.factor
        return veto, factor
