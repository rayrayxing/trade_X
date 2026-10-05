"""Resumable run state for the research loop, in one SQLite file (``data/research/loop.sqlite``).

Tables
    runs        one row per loop run (ISO week by default), with the plan it started from
    steps       one row per (run, stage, item): done / blocked / failed, with the result. A run that is
                started again with the same id skips the items that are done and retries the rest
    ideas       every idea ever seen, with where it stands
    strategies  one row per (strategy id, version): proposed -> screened -> validated -> paper, or rejected / retired
    evidence    append-only measurements (screen, walk_forward, holdout, health, promotion), each tied to the
                hash of the spec it was measured on
    baselines   the walk-forward out-of-sample return statistics the health monitor compares paper results with
    audit       append-only, hash-chained log of every decision (``verify_audit`` recomputes the chain)

The state layer enforces the gate itself: ``transition`` only allows the moves in ``ALLOWED``, and a move
to ``paper`` is refused unless passing screen, walk-forward and holdout evidence exists for the spec's
current content hash. Nothing above this module can produce a paper strategy by another route.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pandas as pd

STATUSES = ("proposed", "screened", "validated", "paper", "rejected", "retired")
ALLOWED: dict[str, set[str]] = {
    "proposed": {"screened", "rejected"},
    "screened": {"validated", "rejected"},
    "validated": {"paper", "rejected", "retired"},
    "paper": {"retired"},
    "rejected": set(),
    "retired": set(),
}
EVIDENCE_FOR_PAPER = ("screen", "walk_forward", "holdout")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id   TEXT PRIMARY KEY,
    started  TEXT NOT NULL,
    finished TEXT,
    status   TEXT NOT NULL,
    plan     TEXT NOT NULL DEFAULT '{}',
    config   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS steps (
    run_id   TEXT NOT NULL,
    stage    TEXT NOT NULL,
    item     TEXT NOT NULL,
    status   TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    started  TEXT,
    finished TEXT,
    result   TEXT NOT NULL DEFAULT '{}',
    error    TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, stage, item)
);
CREATE TABLE IF NOT EXISTS ideas (
    idea_id    TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    title      TEXT NOT NULL,
    status     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    catalog_id TEXT NOT NULL DEFAULT '',
    draft_hash TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    first_seen TEXT NOT NULL,
    updated    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    status      TEXT NOT NULL,
    spec_path   TEXT NOT NULL,
    spec_hash   TEXT NOT NULL,
    idea_id     TEXT NOT NULL DEFAULT '',
    family      TEXT NOT NULL DEFAULT '',
    asset_class TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    unverified  TEXT NOT NULL DEFAULT '',
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    promoted_at TEXT,
    PRIMARY KEY (strategy_id, version)
);
CREATE TABLE IF NOT EXISTS evidence (
    id          INTEGER PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    time        TEXT NOT NULL,
    passed      INTEGER,
    spec_hash   TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS evidence_key ON evidence (strategy_id, version, kind);
CREATE TABLE IF NOT EXISTS baselines (
    strategy_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    mean         REAL NOT NULL,
    std          REAL NOT NULL,
    n            INTEGER NOT NULL,
    max_drawdown REAL NOT NULL,
    created      TEXT NOT NULL,
    PRIMARY KEY (strategy_id, version)
);
CREATE TABLE IF NOT EXISTS audit (
    id          INTEGER PRIMARY KEY,
    time        TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    strategy_id TEXT NOT NULL DEFAULT '',
    event       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}',
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);
"""


class IllegalTransition(RuntimeError):
    pass


class NotEligible(RuntimeError):
    """A move to paper without the evidence for it."""


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class LoopState:
    def __init__(self, path: str | Path = ":memory:", now: Callable[[], pd.Timestamp] | None = None,
                 read_only: bool = False):
        self.path = str(path)
        self.read_only = read_only
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        if read_only:       # `--dry-plan` looks at the state but never creates or changes the file
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
            self.db.row_factory = sqlite3.Row
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def now(self) -> str:
        return self._now().isoformat()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return list(self.db.execute(sql, args))

    def _x(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        cur = self.db.execute(sql, args)
        self.db.commit()
        return cur

    # --- runs ---------------------------------------------------------------------------------

    def start_run(self, run_id: str, plan: dict, config: dict) -> bool:
        """True for a new run, False when the id exists (the run resumes; its steps stay)."""
        if self._q("SELECT 1 FROM runs WHERE run_id=?", (run_id,)):
            self._x("UPDATE runs SET status='running', finished=NULL WHERE run_id=?", (run_id,))
            return False
        self._x("INSERT INTO runs (run_id, started, status, plan, config) VALUES (?,?,?,?,?)",
                (run_id, self.now(), "running", _dump(plan), _dump(config)))
        return True

    def finish_run(self, run_id: str, status: str) -> None:
        self._x("UPDATE runs SET status=?, finished=? WHERE run_id=?", (status, self.now(), run_id))

    def run(self, run_id: str) -> dict | None:
        r = self._q("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(r[0]) | {"plan": json.loads(r[0]["plan"]), "config": json.loads(r[0]["config"])} if r else None

    def runs(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT run_id, started, finished, status FROM runs ORDER BY started")]

    # --- steps --------------------------------------------------------------------------------

    def step(self, run_id: str, stage: str, item: str) -> dict | None:
        r = self._q("SELECT * FROM steps WHERE run_id=? AND stage=? AND item=?", (run_id, stage, item))
        return dict(r[0]) | {"result": json.loads(r[0]["result"])} if r else None

    def begin_step(self, run_id: str, stage: str, item: str) -> None:
        self._x("INSERT INTO steps (run_id, stage, item, status, attempts, started) VALUES (?,?,?,?,1,?) "
                "ON CONFLICT(run_id, stage, item) DO UPDATE SET status='running', attempts=attempts+1, started=?, "
                "finished=NULL, error=''", (run_id, stage, item, "running", self.now(), self.now()))

    def end_step(self, run_id: str, stage: str, item: str, status: str, result: dict | None = None,
                 error: str = "") -> None:
        self._x("UPDATE steps SET status=?, finished=?, result=?, error=? WHERE run_id=? AND stage=? AND item=?",
                (status, self.now(), _dump(result or {}), error, run_id, stage, item))

    def steps(self, run_id: str, stage: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM steps WHERE run_id=?", [run_id]
        if stage:
            sql, args = sql + " AND stage=?", args + [stage]
        return [dict(r) | {"result": json.loads(r["result"])} for r in self._q(sql + " ORDER BY started, item", tuple(args))]

    # --- ideas --------------------------------------------------------------------------------

    def add_idea(self, idea, status: str = "new") -> bool:
        if self._q("SELECT 1 FROM ideas WHERE idea_id=?", (idea.id,)):
            return False
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in idea.__dict__.items()}
        self._x("INSERT INTO ideas (idea_id, source, title, status, payload, catalog_id, first_seen, updated) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (idea.id, idea.source, idea.title, status, _dump(payload), idea.catalog_id, self.now(), self.now()))
        return True

    def idea(self, idea_id: str) -> dict | None:
        r = self._q("SELECT * FROM ideas WHERE idea_id=?", (idea_id,))
        return dict(r[0]) | {"payload": json.loads(r[0]["payload"])} if r else None

    def ideas(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM ideas", ()
        if status:
            sql, args = sql + " WHERE status=?", (status,)
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in self._q(sql + " ORDER BY first_seen, idea_id", args)]

    def set_idea(self, idea_id: str, status: str, note: str = "", draft_hash: str | None = None) -> None:
        if draft_hash is None:
            self._x("UPDATE ideas SET status=?, note=?, updated=? WHERE idea_id=?", (status, note, self.now(), idea_id))
        else:
            self._x("UPDATE ideas SET status=?, note=?, draft_hash=?, updated=? WHERE idea_id=?",
                    (status, note, draft_hash, self.now(), idea_id))

    # --- strategies ---------------------------------------------------------------------------

    def register(self, strategy_id: str, version: int, spec_path: str, spec_hash: str, *, family: str = "",
                 asset_class: str = "", idea_id: str = "", status: str = "proposed", unverified: str = "") -> bool:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if self._q("SELECT 1 FROM strategies WHERE strategy_id=? AND version=?", (strategy_id, version)):
            return False
        t = self.now()
        self._x("INSERT INTO strategies (strategy_id, version, status, spec_path, spec_hash, idea_id, family, "
                "asset_class, unverified, created, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (strategy_id, version, status, spec_path, spec_hash, idea_id, family, asset_class, unverified, t, t))
        return True

    def strategy(self, strategy_id: str, version: int | None = None) -> dict | None:
        if version is None:
            r = self._q("SELECT * FROM strategies WHERE strategy_id=? ORDER BY version DESC LIMIT 1", (strategy_id,))
        else:
            r = self._q("SELECT * FROM strategies WHERE strategy_id=? AND version=?", (strategy_id, version))
        return dict(r[0]) if r else None

    def strategies(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM strategies", ()
        if status:
            sql, args = sql + " WHERE status=?", (status,)
        return [dict(r) for r in self._q(sql + " ORDER BY created, strategy_id, version", args)]

    def update_spec(self, strategy_id: str, version: int, spec_path: str, spec_hash: str) -> None:
        self._x("UPDATE strategies SET spec_path=?, spec_hash=?, updated=? WHERE strategy_id=? AND version=?",
                (spec_path, spec_hash, self.now(), strategy_id, version))

    def transition(self, strategy_id: str, version: int, to: str, reason: str, *, run_id: str, stage: str,
                   spec_hash: str | None = None) -> None:
        cur = self.strategy(strategy_id, version)
        if cur is None:
            raise KeyError(f"unknown strategy {strategy_id} v{version}")
        if to not in ALLOWED.get(cur["status"], set()):
            raise IllegalTransition(f"{strategy_id} v{version}: {cur['status']} -> {to} is not allowed")
        if to == "paper":
            problems = self.eligibility(strategy_id, version, spec_hash)
            if problems:
                self.audit(run_id, stage, "refused_paper", strategy_id, {"problems": problems})
                raise NotEligible(f"{strategy_id} v{version} cannot go to paper: " + "; ".join(problems))
        promoted = ", promoted_at=?" if to == "paper" else ""
        args: tuple = (to, reason, self.now()) + ((self.now(),) if promoted else ()) + (strategy_id, version)
        self._x(f"UPDATE strategies SET status=?, reason=?, updated=?{promoted} WHERE strategy_id=? AND version=?", args)
        self.audit(run_id, stage, "transition", strategy_id, {"version": version, "from": cur["status"], "to": to,
                                                                "reason": reason})

    # --- evidence -----------------------------------------------------------------------------

    def add_evidence(self, strategy_id: str, version: int, kind: str, run_id: str, passed: bool | None,
                     spec_hash: str, data: dict) -> int:
        cur = self._x("INSERT INTO evidence (strategy_id, version, kind, run_id, time, passed, spec_hash, data) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (strategy_id, version, kind, run_id, self.now(), None if passed is None else int(passed),
                       spec_hash, _dump(data)))
        return int(cur.lastrowid)

    def evidence(self, strategy_id: str, version: int, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM evidence WHERE strategy_id=? AND version=?", [strategy_id, version]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [dict(r) | {"data": json.loads(r["data"])} for r in self._q(sql + " ORDER BY id", tuple(args))]

    def latest_evidence(self, strategy_id: str, version: int, kind: str) -> dict | None:
        rows = self.evidence(strategy_id, version, kind)
        return rows[-1] if rows else None

    def eligibility(self, strategy_id: str, version: int, spec_hash: str | None) -> list[str]:
        """Why this strategy version may not be in the paper queue; empty means it may."""
        cur = self.strategy(strategy_id, version)
        if cur is None:
            return ["unknown strategy"]
        problems = []
        if cur["unverified"]:
            problems.append(f"status came from outside the loop ({cur['unverified']})")
        if spec_hash is None:
            problems.append("no spec hash supplied")
        for kind in EVIDENCE_FOR_PAPER:
            ev = self.latest_evidence(strategy_id, version, kind)
            if ev is None:
                problems.append(f"no {kind} evidence")
            elif not ev["passed"]:
                problems.append(f"{kind} did not pass")
            elif spec_hash is not None and ev["spec_hash"] != spec_hash:
                problems.append(f"spec changed after {kind} (evidence is for another content hash)")
        return problems

    # --- baselines ----------------------------------------------------------------------------

    def set_baseline(self, strategy_id: str, version: int, mean: float, std: float, n: int, max_drawdown: float) -> None:
        self._x("INSERT OR REPLACE INTO baselines (strategy_id, version, mean, std, n, max_drawdown, created) "
                "VALUES (?,?,?,?,?,?,?)", (strategy_id, version, mean, std, n, max_drawdown, self.now()))

    def baseline(self, strategy_id: str, version: int) -> dict | None:
        r = self._q("SELECT * FROM baselines WHERE strategy_id=? AND version=?", (strategy_id, version))
        return dict(r[0]) if r else None

    # --- audit --------------------------------------------------------------------------------

    def audit(self, run_id: str, stage: str, event: str, strategy_id: str = "", detail: dict | None = None) -> int:
        last = self._q("SELECT hash FROM audit ORDER BY id DESC LIMIT 1")
        prev = last[0]["hash"] if last else ""
        t, d = self.now(), _dump(detail or {})
        h = hashlib.sha256("|".join((prev, t, run_id, stage, strategy_id, event, d)).encode()).hexdigest()
        cur = self._x("INSERT INTO audit (time, run_id, stage, strategy_id, event, detail, prev_hash, hash) "
                      "VALUES (?,?,?,?,?,?,?,?)", (t, run_id, stage, strategy_id, event, d, prev, h))
        return int(cur.lastrowid)

    def audit_rows(self, run_id: str | None = None, strategy_id: str | None = None) -> list[dict]:
        sql, args, where = "SELECT * FROM audit", [], []
        if run_id:
            where.append("run_id=?")
            args.append(run_id)
        if strategy_id:
            where.append("strategy_id=?")
            args.append(strategy_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in self._q(sql + " ORDER BY id", tuple(args))]

    def verify_audit(self) -> tuple[bool, int | None]:
        prev = ""
        for r in self._q("SELECT * FROM audit ORDER BY id"):
            h = hashlib.sha256("|".join((prev, r["time"], r["run_id"], r["stage"], r["strategy_id"], r["event"],
                                         r["detail"])).encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return False, int(r["id"])
            prev = r["hash"]
        return True, None
