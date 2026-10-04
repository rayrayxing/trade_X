#!/usr/bin/env python3
"""Merge check: agent pull requests may not change protected paths (Ray, 4 Oct 2026).

Protected: the risk gate and risk policy, the validation gate thresholds, the execution
layer and the holdout data (kept outside the repo). The list lives in
config/protected_paths.txt, which is itself protected.

A pull request counts as an agent's when its head branch starts with ``agent/`` (the
researcher and strategy designer push there). Changes to protected paths need a commit
pushed by Ray on a non-agent branch.

Usage: check_protected_paths.py --base origin/main --head-ref "$GITHUB_HEAD_REF"
Exit 1 and list the offending files when an agent branch touches a protected path.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_PREFIXES = ("agent/",)


def protected_prefixes(path: Path = ROOT / "config" / "protected_paths.txt") -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def changed_files(base: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return [ln for ln in out.stdout.splitlines() if ln]


def violations(files: list[str], prefixes: list[str]) -> list[str]:
    return [f for f in files if any(f == p or f.startswith(p) for p in prefixes)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head-ref", default="")
    a = ap.parse_args(argv)
    if not a.head_ref.startswith(AGENT_PREFIXES):
        print(f"{a.head_ref or 'this branch'} is not an agent branch: protected-path check not applied")
        return 0
    bad = violations(changed_files(a.base), protected_prefixes())
    if bad:
        print("Agent pull request changes protected paths, which only Ray may change:")
        for f in bad:
            print(f"  - {f}")
        return 1
    print("no protected paths changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
