"""Record types every process reads and writes through the ledger.

One decision ID follows a trade from the strategy votes to the Telegram alert:
vote -> plan -> veto/verdict -> order -> fill -> position -> exit changes -> close.
Rejected and vetoed plans keep their ID too, so the counterfactual ledger can follow
them to the exit they would have had.

Records are plain dataclasses so the backtester, replay, paper and live loops all
write exactly the same rows.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

import pandas as pd

FAMILIES = (
    "trend", "breakout", "mean_reversion", "momentum", "carry",
    "chart_pattern", "candlestick", "event_drift", "seasonality", "other",
)


class Stage(str, Enum):
    """The four gates a trade passes, in order (Ray, 4 Oct 2026)."""
    QUALIFIED = "qualified"      # strategy passed walk-forward, deflated Sharpe, cost stress, holdout
    FINALISED = "finalised"      # ensemble turned votes into one complete trade plan
    CONTEXT = "context"          # scout, calendar and chart reader may veto (subtract only)
    RISK = "risk"                # risk gate sets the size; never moves entry, stop or targets


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    VETOED = "vetoed"


class DecisionIds:
    """Decision IDs like 2026-11-02-0147: date of the decision plus a per-day counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: str | None = None
        self._n = itertools.count(1)

    def next(self, ts: pd.Timestamp) -> str:
        day = pd.Timestamp(ts).strftime("%Y-%m-%d")
        with self._lock:
            if day != self._day:
                self._day, self._n = day, itertools.count(1)
            return f"{day}-{next(self._n):04d}"


@dataclass
class Record:
    """Base: every record knows its kind and can flatten itself to a JSON-able dict."""

    kind: str = field(init=False, default="record")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: _jsonable(v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        names = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in names})


def _jsonable(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item"):  # numpy scalars
        return v.item()
    return v


@dataclass
class Vote(Record):
    """One strategy's opinion on one symbol at one closed bar."""
    time: str
    strategy_id: str
    strategy_version: int
    family: str
    symbol: str
    asset_class: str
    direction: int                    # +1 long, -1 short, 0 none
    strength: float                   # 0..1, calibrated hit rate when known
    entry_ref: float                  # close of the signal bar
    stop: float
    targets: list[float]
    max_bars: int
    knowable_at: str                  # when the signal became knowable (bar close or later confirmation)
    decision_id: str = ""             # set when the vote is part of a plan
    book: str = "ensemble"
    kind: str = field(init=False, default="vote")


@dataclass
class TradePlan(Record):
    """A finalised plan: everything about the trade except its size."""
    decision_id: str
    time: str
    symbol: str
    asset_class: str
    direction: int
    entry_type: str                   # market | limit
    entry_price: float                # expected fill (next open estimate, or limit price)
    stop: float
    targets: list[float]
    max_bars: int
    invalidation: str
    families: list[str]
    strategies: list[str]
    score: float                      # weighted family score, -1..1
    p_target: float                   # P(T1 before stop): base rate until the size model is calibrated
    p_source: str                     # "base_rate" or "model"
    reward_risk: float                # (T1 - entry) / (entry - stop), after costs
    ev_r: float                       # expected value in R after costs
    cost_r: float                     # round-trip cost in R
    book: str = "ensemble"            # "ensemble" or "virtual:<strategy_id>"
    kind: str = field(init=False, default="plan")

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop)


@dataclass
class Veto(Record):
    decision_id: str
    time: str
    source: str                       # calendar | scout | chart_reader | short_check | ...
    reason: str
    kind: str = field(init=False, default="veto")


@dataclass
class Verdict(Record):
    """The risk gate's answer. Size only; the plan's prices are untouched."""
    decision_id: str
    time: str
    outcome: str                      # accepted | rejected
    qty: float
    risk_usd: float
    risk_pct: float
    reasons: list[str]
    checks: dict[str, Any]            # each check's value and limit, for the dashboard drawer
    verdict_id: str = ""              # set by the core; every entry order cites it and the order guard checks it
    kind: str = field(init=False, default="verdict")


@dataclass
class Order(Record):
    decision_id: str
    client_order_id: str
    time: str
    symbol: str
    side: int                         # +1 buy, -1 sell
    qty: float
    order_type: str                   # market | limit | stop
    price: float | None
    purpose: str                      # entry | stop | target | exit | hedge
    book: str = "ensemble"
    account: str = "agent"
    status: str = "new"
    kind: str = field(init=False, default="order")


@dataclass
class Fill(Record):
    decision_id: str
    client_order_id: str
    time: str
    symbol: str
    side: int
    qty: float
    price: float
    fees_usd: float
    spread_slippage_usd: float
    book: str = "ensemble"
    kind: str = field(init=False, default="fill")


@dataclass
class ExitChange(Record):
    decision_id: str
    time: str
    field_name: str                   # stop | target | time_stop
    old: float | None
    new: float | None
    reason: str
    kind: str = field(init=False, default="exit_change")


@dataclass
class Close(Record):
    decision_id: str
    time: str
    symbol: str
    exit_price: float
    qty: float
    net_pnl_usd: float
    r_multiple: float
    reason: str
    book: str = "ensemble"
    kind: str = field(init=False, default="close")


@dataclass
class Counterfactual(Record):
    """What a rejected or vetoed plan would have done, followed with the same exit rules."""
    decision_id: str
    time: str
    blocked_by: str                   # the gate or source that stopped it
    exit_time: str
    exit_reason: str
    r_multiple: float
    kind: str = field(init=False, default="counterfactual")


@dataclass
class EquitySnapshot(Record):
    """End-of-day (or any-time) picture of the book, so the dashboard can look back."""
    time: str
    book: str
    equity_usd: float
    cash_usd: float
    open_risk_usd: float
    positions: list[dict[str, Any]]
    exposure_by_currency: dict[str, float]
    limits: dict[str, Any]
    config_hash: str
    kind: str = field(init=False, default="snapshot")


@dataclass
class ConfigVersion(Record):
    time: str
    config_hash: str
    path: str
    content: dict[str, Any]
    git_commit: str = ""
    kind: str = field(init=False, default="config_version")


@dataclass
class AgentOutput(Record):
    """A row an LLM agent writes for the core to apply: a card, a veto or a close request."""
    time: str
    agent: str
    provider: str
    model: str
    action: str                       # card | veto | close_request | note
    target: str                       # symbol or decision ID
    body: dict[str, Any]
    kind: str = field(init=False, default="agent_output")


@dataclass
class Health(Record):
    time: str
    check: str
    ok: bool
    detail: str = ""
    kind: str = field(init=False, default="health")


RECORD_TYPES: dict[str, type[Record]] = {
    c.__dataclass_fields__["kind"].default: c  # type: ignore[attr-defined]
    for c in (Vote, TradePlan, Veto, Verdict, Order, Fill, ExitChange, Close, Counterfactual,
              EquitySnapshot, ConfigVersion, AgentOutput, Health)
}
