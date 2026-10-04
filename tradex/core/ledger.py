"""The ledger: one SQLite file every process reads and writes.

- WAL mode, so the dashboard and agents can read while the core writes.
- Every record goes into one append-only ``events`` table. Each row stores the hash of
  the row before it, so a decision cannot be quietly rewritten: ``verify()`` recomputes
  the chain and names the first row that does not match.
- Rows carry the decision ID (when the record has one), the book, the git commit and
  the config hash, so any trade can be printed in full by decision ID (``why``) and read
  against the rules of its day.
- Agents and Telegram never talk to the core directly: they write ``commands`` and
  ``agent_output`` rows, and the core applies them on its next cycle.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator

from tradex.core.records import RECORD_TYPES, ConfigVersion, Record

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    time         TEXT NOT NULL,
    decision_id  TEXT,
    book         TEXT,
    symbol       TEXT,
    payload      TEXT NOT NULL,
    git_commit   TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_decision ON events(decision_id);
CREATE INDEX IF NOT EXISTS events_kind_time ON events(kind, time);
CREATE INDEX IF NOT EXISTS events_book ON events(book, kind);

CREATE TABLE IF NOT EXISTS commands (
    id          INTEGER PRIMARY KEY,
    time        TEXT NOT NULL,
    source      TEXT NOT NULL,          -- telegram | dashboard
    command     TEXT NOT NULL,          -- pause | resume | flatten
    args        TEXT NOT NULL DEFAULT '{}',
    applied_at  TEXT,
    result      TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    time        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    result      TEXT
);
CREATE INDEX IF NOT EXISTS jobs_agent_payload ON jobs(agent, payload);
"""


def row_hash(prev_hash: str, kind: str, time: str, decision_id: str | None, payload: str,
             git_commit: str, config_hash: str) -> str:
    h = hashlib.sha256()
    for part in (prev_hash, kind, time, decision_id or "", payload, git_commit, config_hash):
        h.update(part.encode())
        h.update(b"\x1f")
    return h.hexdigest()


def config_hash(content: Any) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:16]


def current_git_commit(repo: str | Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - outside a checkout the commit is simply unknown
        return "unknown"


class Ledger:
    """Writer and reader for the ledger file. Use ``read_only=True`` for the dashboard and agents."""

    def __init__(self, path: str | Path = ":memory:", read_only: bool = False, run_id: str = "",
                 git_commit: str | None = None):
        self.path = str(path)
        self.read_only = read_only
        self.run_id = run_id or "run"
        self.git_commit = git_commit if git_commit is not None else current_git_commit()
        self.config_hash = "none"
        self._lock = threading.Lock()
        if read_only:
            uri = f"file:{self.path}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row

    # --- writing ---------------------------------------------------------------------

    def set_config(self, time: str, path: str, content: dict[str, Any]) -> str:
        """Record the active config; every later row carries its hash."""
        h = config_hash(content)
        if h != self.config_hash:
            self.config_hash = h
            self.append(ConfigVersion(time=time, config_hash=h, path=path, content=content,
                                      git_commit=self.git_commit))
        return h

    def append(self, rec: Record) -> int:
        if self.read_only:
            raise PermissionError("ledger opened read-only")
        d = rec.to_dict()
        payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
        kind, time = d["kind"], str(d.get("time", ""))
        decision_id = d.get("decision_id")
        with self._lock, self.db:
            last = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else GENESIS
            h = row_hash(prev, kind, time, decision_id, payload, self.git_commit, self.config_hash)
            cur = self.db.execute(
                "INSERT INTO events (kind, time, decision_id, book, symbol, payload, git_commit, config_hash,"
                " run_id, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (kind, time, decision_id, d.get("book"), d.get("symbol"), payload, self.git_commit,
                 self.config_hash, self.run_id, prev, h),
            )
            return int(cur.lastrowid)

    def add_command(self, time: str, source: str, command: str, args: dict | None = None) -> int:
        with self._lock, self.db:
            cur = self.db.execute("INSERT INTO commands (time, source, command, args) VALUES (?,?,?,?)",
                                  (time, source, command, json.dumps(args or {})))
            return int(cur.lastrowid)

    def pending_commands(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM commands WHERE applied_at IS NULL ORDER BY id"))

    def mark_command(self, cid: int, time: str, result: str) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE commands SET applied_at=?, result=? WHERE id=?", (time, result, cid))

    def claim_job(self, agent: str, payload: dict[str, Any], time: str) -> int | None:
        """Claim a job exactly once: insert it as running unless a job with the same agent and
        payload already exists (in any status). Returns the new job ID, or None if taken.
        BEGIN IMMEDIATE makes the check-and-insert atomic across processes."""
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                hit = self.db.execute("SELECT id FROM jobs WHERE agent=? AND payload=?", (agent, body)).fetchone()
                jid = None
                if hit is None:
                    jid = int(self.db.execute("INSERT INTO jobs (time, agent, payload, status) VALUES (?,?,?,'running')",
                                              (time, agent, body)).lastrowid)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
        return jid

    def finish_job(self, job_id: int, status: str, result: str = "") -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE jobs SET status=?, result=? WHERE id=?", (status, result, job_id))

    def jobs(self, agent: str) -> list[dict[str, Any]]:
        return [dict(r) | {"payload": json.loads(r["payload"])}
                for r in self.db.execute("SELECT * FROM jobs WHERE agent=? ORDER BY id", (agent,))]

    # --- reading ---------------------------------------------------------------------

    def rows(self, kind: str | None = None, book: str | None = None, decision_id: str | None = None,
             since: str | None = None, until: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        for col, val in (("kind", kind), ("book", book), ("decision_id", decision_id)):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        if since is not None:
            q += " AND time>=?"
            args.append(since)
        if until is not None:
            q += " AND time<=?"
            args.append(until)
        q += " ORDER BY seq"
        out = []
        for r in self.db.execute(q, args):
            d = json.loads(r["payload"])
            d["_seq"], d["_git_commit"], d["_config_hash"] = r["seq"], r["git_commit"], r["config_hash"]
            out.append(d)
        return out

    def records(self, kind: str, **kw) -> list[Record]:
        cls = RECORD_TYPES[kind]
        return [cls.from_dict(d) for d in self.rows(kind=kind, **kw)]

    def why(self, decision_id: str) -> list[dict[str, Any]]:
        """Every row about one decision, in order: the full record behind an alert or dashboard row."""
        return self.rows(decision_id=decision_id)

    def snapshot_at(self, time: str, book: str = "ensemble") -> dict[str, Any] | None:
        """The last book snapshot at or before ``time``: what the look-back slider shows."""
        r = self.db.execute(
            "SELECT payload FROM events WHERE kind='snapshot' AND book=? AND time<=? ORDER BY time DESC, seq DESC LIMIT 1",
            (book, time)).fetchone()
        return json.loads(r["payload"]) if r else None

    def iter_events(self) -> Iterator[sqlite3.Row]:
        yield from self.db.execute("SELECT * FROM events ORDER BY seq")

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the hash chain. Returns (ok, first bad seq)."""
        prev = GENESIS
        for r in self.iter_events():
            h = row_hash(prev, r["kind"], r["time"], r["decision_id"], r["payload"], r["git_commit"], r["config_hash"])
            if r["prev_hash"] != prev or r["hash"] != h:
                return False, int(r["seq"])
            prev = r["hash"]
        return True, None

    def digest(self, kinds: tuple[str, ...] = ("plan", "veto", "verdict", "order", "fill", "close")) -> str:
        """Hash of the decision rows only (no commit or run metadata): two runs of the same
        day through the same code must produce the same digest. This is the parity check."""
        h = hashlib.sha256()
        marks = ",".join("?" * len(kinds))
        for r in self.db.execute(f"SELECT payload FROM events WHERE kind IN ({marks}) ORDER BY seq", kinds):
            h.update(r["payload"].encode())
        return h.hexdigest()

    def close(self) -> None:
        self.db.close()
