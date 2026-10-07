"""Mailbox for everything that is not the core, and the core's ingest step.

Telegram, the dashboard and the agents write only ``commands`` and ``agent_inbox`` rows.
``Mailbox`` enforces that with a SQLite authorizer: its connection is denied any write
to ``events`` (the hash-chained ledger), so a bug or a prompt-injected agent cannot
forge or alter a decision row. The core calls ``ingest_inbox`` each bar: it applies the
whitelisted actions through handlers it supplies, marks each row with what happened, and
appends an ``agent_output`` record, so the chain shows what was applied.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tradex.core.ledger import BUSY_MS, SCHEMA, Ledger, migrate
from tradex.core.records import AgentOutput

ALLOWED_ACTIONS = ("veto", "shrink", "close", "flag")
WRITABLE = {"commands", "agent_inbox", "agent_calls"}


SAFE_PRAGMAS = {"busy_timeout", "table_info", "table_xinfo", "index_list", "foreign_key_list", "database_list"}
SCHEMA_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema"}


def _make_authorizer(writable: set[str]):
    """Deny everything that could make the core, which has no authorizer, write to ``events`` on the
    mailbox's behalf: writes outside the mailbox tables (and the schema tables), schema changes,
    triggers, ATTACH and PRAGMAs such as writable_schema."""
    writes = (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE)
    schema_ops = {getattr(sqlite3, n) for n in (
        "SQLITE_DROP_TABLE", "SQLITE_ALTER_TABLE", "SQLITE_DROP_INDEX", "SQLITE_DROP_TRIGGER", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_ATTACH", "SQLITE_DETACH",
        "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_DROP_VIEW", "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE") if hasattr(sqlite3, n)}

    def auth(op: int, a1: str | None, a2: str | None, db: str | None, src: str | None) -> int:
        if op in writes and (a1 not in writable or a1 in SCHEMA_TABLES):
            return sqlite3.SQLITE_DENY
        if op in schema_ops:
            return sqlite3.SQLITE_DENY
        if op == sqlite3.SQLITE_PRAGMA and (a1 or "").lower() not in SAFE_PRAGMAS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return auth


class Mailbox:
    """The only write handle Telegram, the dashboard and agents get on the ledger file."""

    def __init__(self, path: str | Path, init_schema: bool = True, extra_schema: str = "",
                 extra_writable: tuple[str, ...] = ()):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=BUSY_MS / 1000)
        self.db.execute(f"PRAGMA busy_timeout={BUSY_MS}")
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        if init_schema:  # IF NOT EXISTS only: safe to run next to the core
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA + extra_schema)
            migrate(self.db)
        self.db.set_authorizer(_make_authorizer(WRITABLE | set(extra_writable)))

    def add_command(self, time: str, source: str, command: str, args: dict | None = None) -> int:
        with self._lock, self.db:
            cur = self.db.execute("INSERT INTO commands (time, source, command, args) VALUES (?,?,?,?)",
                                  (time, source, command, json.dumps(args or {})))
            return int(cur.lastrowid)

    def add_inbox(self, time: str, source: str, action: str, target: str, body: dict | None = None) -> int:
        with self._lock, self.db:
            cur = self.db.execute("INSERT INTO agent_inbox (time, source, action, target, body) VALUES (?,?,?,?,?)",
                                  (time, source, action, target, json.dumps(body or {}, default=str)))
            return int(cur.lastrowid)

    def read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return list(self.db.execute(sql, args))

    def close(self) -> None:
        self.db.close()


@dataclass
class IngestResult:
    id: int
    action: str
    target: str
    applied: bool
    result: str


Handler = Callable[[dict[str, Any]], "str | tuple[str, bool]"]   # result text, or (text, applied)


def ingest_inbox(ledger: Ledger, time: str, handlers: dict[str, Handler] | None = None) -> list[IngestResult]:
    """Core-side: apply pending inbox rows. ``handlers[action](row) -> result text`` does the work
    (veto a plan, shrink a size, close a position, flag for review). A handler that only records
    what it would have done (agents in shadow mode) returns ``(text, False)``. Unknown actions, actions
    without a handler and handlers that raise are logged and ignored; nothing here can crash the bar.
    Each row is marked once, so re-running ingest never applies it twice."""
    handlers = handlers or {}
    out: list[IngestResult] = []
    rows = list(ledger.db.execute("SELECT * FROM agent_inbox WHERE applied_at IS NULL ORDER BY id"))
    for r in rows:
        action, target = r["action"], r["target"]
        try:
            body = json.loads(r["body"])
        except ValueError:
            body = {}
        row = {"id": r["id"], "time": r["time"], "source": r["source"], "action": action, "target": target,
               "body": body}
        applied = False
        if action not in ALLOWED_ACTIONS:
            res = f"ignored: action '{action}' is not allowed"
        elif action not in handlers:
            res = "ignored: no handler"
        else:
            try:
                got = handlers[action](row)
                res, applied = got if isinstance(got, tuple) else (got, True)
            except Exception as exc:  # noqa: BLE001 - a bad agent row must not stop trading
                res = f"ignored: handler failed ({type(exc).__name__})"
        # Record in the chain first, then mark the row done: a crash in between re-runs the handler (every
        # handler is idempotent: cancels, never-larger shrinks, exits keyed by inbox id) and records it
        # again, instead of marking a request done that the ledger never saw.
        ledger.append(AgentOutput(time=time, agent=r["source"], provider="", model="",
                                  action=action if action in ALLOWED_ACTIONS else "note", target=target,
                                  body={"inbox_id": r["id"], "applied": applied, "result": res, "request": body}))
        with ledger._lock, ledger.db:
            ledger.db.execute("UPDATE agent_inbox SET applied_at=?, result=? WHERE id=?", (time, res, r["id"]))
        out.append(IngestResult(r["id"], action, target, applied, res))
    return out
