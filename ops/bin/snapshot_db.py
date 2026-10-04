#!/usr/bin/env python3
"""Consistent copy of a live SQLite file, for backups.

Copying a WAL-mode database file while the core writes it can give a torn copy. This uses SQLite's
online backup API on a read-only connection, checks the copy, and (with --verify-chain) recomputes
the ledger's hash chain on the copy so a backup of a tampered ledger fails loudly.

    snapshot_db.py SRC DST [--verify-chain]

Exit codes: 0 ok, 1 copy failed or integrity check failed, 2 hash chain broken, 3 source missing.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


def snapshot(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    tmp.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
    try:
        target = sqlite3.connect(tmp)
        try:
            source.backup(target)
            ok = target.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            target.close()
    finally:
        source.close()
    if ok != "ok":
        tmp.unlink(missing_ok=True)
        raise sqlite3.DatabaseError(f"integrity_check on the copy said: {ok}")
    os.replace(tmp, dst)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 64
    src, dst = Path(args[0]), Path(args[1])
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 3
    try:
        snapshot(src, dst)
    except sqlite3.Error as exc:
        print(f"snapshot of {src} failed: {exc}", file=sys.stderr)
        return 1
    if "--verify-chain" in argv:
        sys.path.append(str(Path(__file__).resolve().parents[2]))      # running from a checkout without pip install -e
        from tradex.core.ledger import Ledger
        led = Ledger(dst, read_only=True, git_commit="")
        ok, bad = led.verify()
        led.close()
        if not ok:
            print(f"ledger hash chain broken at row {bad}; the copy is kept at {dst} for inspection", file=sys.stderr)
            return 2
    print(f"snapshot ok: {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
