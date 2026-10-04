"""The order guard: the last check between the core and a real venue (protected path).

Every venue adapter submits through ``OrderGuard.check`` (``VenueAdapter.place`` is the
only way in, and ``GuardedBroker`` wraps anything else). An order is refused unless:

- it is for an agent account: ``account="agent"`` and the venue account ID is listed as
  agent-owned in config/accounts.yaml (an unresolved ID owns nothing);
- an entry or hedge cites a risk-gate verdict ID that exists, was accepted, belongs to the
  same decision, and its quantity is no larger than the verdict's size;
- an exit only reduces an open agent position of the same decision (exits close risk, so
  they need no verdict: a flatten must never be blocked by a missing one).

Refusals raise ``OrderRefused``; nothing reaches the venue.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import pandas as pd

from tradex.core.interfaces import AccountInfo, Broker, BrokerFill, BrokerPosition, OrderRequest, OrderStatus

QTY_TOL = 1e-9

VerdictLookup = Callable[[str, str], Any]     # (decision_id, verdict_id) -> verdict dict/record or None


class OrderRefused(PermissionError):
    pass


def ledger_verdicts(ledger) -> VerdictLookup:
    """Look verdicts up in the ledger, where the core writes them before it places the order."""
    def find(decision_id: str, verdict_id: str):
        for r in ledger.rows(kind="verdict", decision_id=decision_id):
            if r.get("verdict_id") == verdict_id:
                return r
        return None
    return find


def _get(v: Any, k: str):
    return v.get(k) if isinstance(v, dict) else getattr(v, k, None)


class OrderGuard:
    def __init__(self, verdicts: VerdictLookup, agent_accounts: Iterable[str]):
        self.verdicts = verdicts
        self.agent_accounts = {a for a in agent_accounts if a}

    def check(self, req: OrderRequest, account_id: str, positions: list[BrokerPosition]) -> None:
        if req.account != "agent":
            raise OrderRefused(f"{req.client_order_id}: the agent never trades Ray's own account")
        if not account_id or account_id not in self.agent_accounts:
            raise OrderRefused(f"{req.client_order_id}: account is not listed as agent-owned in config/accounts.yaml")
        if req.qty <= 0:
            raise OrderRefused(f"{req.client_order_id}: non-positive quantity")
        if req.purpose == "exit":
            pos = [p for p in positions if p.decision_id == req.decision_id and p.account == "agent"]
            if not pos or pos[0].symbol != req.symbol or req.side != -pos[0].direction:
                raise OrderRefused(f"{req.client_order_id}: exit does not reduce an open agent position")
            if req.qty > pos[0].qty + QTY_TOL:
                raise OrderRefused(f"{req.client_order_id}: exit qty {req.qty} exceeds position {pos[0].qty}")
            return
        if not req.verdict_id:
            raise OrderRefused(f"{req.client_order_id}: no risk-gate verdict cited")
        v = self.verdicts(req.decision_id, req.verdict_id)
        if v is None:
            raise OrderRefused(f"{req.client_order_id}: verdict {req.verdict_id} does not exist")
        if _get(v, "decision_id") != req.decision_id:
            raise OrderRefused(f"{req.client_order_id}: verdict {req.verdict_id} is for another decision")
        if _get(v, "outcome") != "accepted":
            raise OrderRefused(f"{req.client_order_id}: verdict {req.verdict_id} was not accepted")
        if req.qty > float(_get(v, "qty") or 0.0) + QTY_TOL:
            raise OrderRefused(f"{req.client_order_id}: qty {req.qty} exceeds verdict size {_get(v, 'qty')}")


class VenueAdapter(ABC):
    """Base class for real-venue adapters (Oanda, moomoo). ``place`` is fixed: it runs the
    guard, then ``_submit``. Adapters implement ``_submit`` and the read methods."""

    venue: str = ""

    def __init__(self, account_id: str, guard: OrderGuard):
        if not isinstance(guard, OrderGuard):
            raise TypeError("a venue adapter needs an OrderGuard")
        self.account_id = account_id
        self.guard = guard

    def place(self, req: OrderRequest) -> str:
        self.guard.check(req, self.account_id, self.positions(account=None))
        return self._submit(req)

    @abstractmethod
    def _submit(self, req: OrderRequest) -> str: ...

    @abstractmethod
    def cancel(self, client_order_id: str) -> bool: ...

    @abstractmethod
    def amend_stop(self, decision_id: str, stop: float) -> None: ...

    @abstractmethod
    def positions(self, account: str | None = "agent") -> list[BrokerPosition]: ...

    @abstractmethod
    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]: ...

    @abstractmethod
    def order_status(self, client_order_id: str) -> OrderStatus: ...

    @abstractmethod
    def account(self) -> AccountInfo: ...


class GuardedBroker:
    """Any Broker behind the guard (used for a simulated stand-in of a venue in tests)."""

    def __init__(self, inner: Broker, guard: OrderGuard, account_id: str | None = None):
        self.inner, self.guard = inner, guard
        self.account_id = account_id if account_id is not None else inner.account().account_id

    def place(self, req: OrderRequest) -> str:
        self.guard.check(req, self.account_id, self.inner.positions(account=None))
        return self.inner.place(req)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)
