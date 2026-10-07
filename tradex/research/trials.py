"""Global trial ledger: every parameter set ever evaluated for a strategy.

The deflated Sharpe ratio is only honest if N counts every configuration that was
tried, including the ones tried last month and thrown away. Each walk-forward run
appends one row per (parameter set, fold) here, and the DSR uses the number of distinct
parameter sets ever recorded for the strategy id (all versions: a new version of an idea
is another trial of it). The file is local research state, not the trading ledger, so
it is a plain SQLite table outside git (data/research/, gitignored).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "research" / "trials.sqlite"
ENV_VAR = "TRADEX_TRIALS_DB"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id           INTEGER PRIMARY KEY,
    time         TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    strategy_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    params_hash  TEXT NOT NULL,
    params       TEXT NOT NULL,
    source       TEXT NOT NULL,
    fold         INTEGER,
    window_start TEXT,
    window_end   TEXT,
    data_key     TEXT,
    sharpe       REAL,
    n_obs        INTEGER,
    trades       INTEGER
);
CREATE INDEX IF NOT EXISTS trials_strategy ON trials (strategy_id, params_hash);
"""


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


class TrialLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]

    def record(self, strategy_id: str, version: int, params: dict, *, run_id: str, source: str = "walk_forward",
               fold: int | None = None, window: tuple[object, object] | None = None, data_key: str | None = None,
               sharpe: float | None = None, n_obs: int | None = None, trades: int | None = None) -> None:
        self.record_many([dict(strategy_id=strategy_id, version=version, params=params, run_id=run_id, source=source,
                               fold=fold, window=window, data_key=data_key, sharpe=sharpe, n_obs=n_obs, trades=trades)])

    def record_many(self, rows: list[dict]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        vals = []
        for r in rows:
            w = r.get("window") or (None, None)
            vals.append((now, r["run_id"], r["strategy_id"], int(r["version"]), params_hash(r["params"]),
                         json.dumps(r["params"], sort_keys=True, default=str), r.get("source", "walk_forward"),
                         r.get("fold"), None if w[0] is None else str(w[0]), None if w[1] is None else str(w[1]),
                         r.get("data_key"), r.get("sharpe"), r.get("n_obs"), r.get("trades")))
        with self._conn() as c:
            c.executemany("INSERT INTO trials (time, run_id, strategy_id, version, params_hash, params, source, fold, "
                          "window_start, window_end, data_key, sharpe, n_obs, trades) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)

    def count(self, strategy_id: str) -> int:
        """Distinct (parameter set, data) pairs ever evaluated for this strategy (the DSR's N): the same parameters
        on other data is another trial; a re-run on the same data is not."""
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(DISTINCT params_hash || '|' || COALESCE(data_key, '')) FROM trials "
                                 "WHERE strategy_id = ?", (strategy_id,)).fetchone()[0])

    def sharpe_variance(self, strategy_id: str) -> float:
        """Variance of the trial Sharpes over everything ever recorded for the strategy: within each window of each
        data set the spread across its parameter sets (latest value per set), averaged over windows. Zero when no
        window holds two parameter sets. Re-running one parameter set cannot shrink it."""
        with self._conn() as c:
            rows = c.execute("SELECT COALESCE(data_key, ''), COALESCE(fold, -1), COALESCE(window_start, ''), "
                             "COALESCE(window_end, ''), params_hash, sharpe FROM trials "
                             "WHERE strategy_id = ? AND sharpe IS NOT NULL ORDER BY id", (strategy_id,)).fetchall()
        groups: dict[tuple, dict[str, float]] = {}
        for *key, ph, sr in rows:
            groups.setdefault(tuple(key), {})[ph] = float(sr)       # later rows replace earlier ones
        vs = []
        for g in groups.values():
            if len(g) > 1:
                m = sum(g.values()) / len(g)
                vs.append(sum((x - m) ** 2 for x in g.values()) / (len(g) - 1))
        return float(sum(vs) / len(vs)) if vs else 0.0

    def runs(self, strategy_id: str) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(DISTINCT run_id) FROM trials WHERE strategy_id = ?",
                                 (strategy_id,)).fetchone()[0])

    def summary(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT strategy_id, COUNT(DISTINCT params_hash), COUNT(DISTINCT run_id), COUNT(*), "
                             "MAX(time) FROM trials GROUP BY strategy_id ORDER BY strategy_id").fetchall()
        return [dict(strategy_id=r[0], param_sets=r[1], runs=r[2], rows=r[3], last=r[4]) for r in rows]
