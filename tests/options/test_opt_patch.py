"""patches/options-order-guard.patch: scope, applicability, and a full run of the patched tree.

The patch changes protected files, so this PR does not apply it; Ray does. These tests make sure
what he applies is exactly what was reviewed and that it works: they apply it to a scratch copy
and run the patch's own tests (gate, guard, gap-formula parity with tradex.options.sizing) plus the
existing gate, guard and runtime suites against the patched code.

Once the patch is applied to main this file and the patch can be deleted: the forward check then
fails, the reverse check passes, and the scratch-copy test skips itself.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches" / "options-order-guard.patch"
EXPECTED_FILES = {
    "config/accounts.yaml", "config/risk/policy.yaml", "tradex/execution/checks.py", "tradex/execution/guard.py",
    "tradex/risk/exposure.py", "tradex/risk/gate.py", "tradex/risk/options.py", "tests/test_options_risk.py"}
PROTECTED = [ln.strip() for ln in (ROOT / "config" / "protected_paths.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _has_git():
    try:
        return _git("--version").returncode == 0
    except FileNotFoundError:
        return False


needs_git = pytest.mark.skipif(not _has_git(), reason="git not available")


def _state(cwd=ROOT) -> str:
    if _git("apply", "--check", str(PATCH), cwd=cwd).returncode == 0:
        return "applicable"
    if _git("apply", "--check", "-R", str(PATCH), cwd=cwd).returncode == 0:
        return "applied"
    return "drifted"


def _files() -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", PATCH.read_text(), flags=re.M))


def test_patch_exists_and_touches_exactly_the_reviewed_files():
    assert PATCH.exists()
    assert _files() == EXPECTED_FILES


def test_every_protected_file_in_the_patch_is_really_protected_and_the_rest_is_new_tests():
    for f in _files():
        is_protected = any(f == p or f.startswith(p) for p in PROTECTED)
        assert is_protected or f == "tests/test_options_risk.py", f


def test_patch_text_has_no_trailing_junk():
    text = PATCH.read_text()
    assert text.endswith("\n") and "\r" not in text


@needs_git
def test_patch_applies_to_this_tree_or_is_already_applied():
    state = _state()
    if state == "drifted":
        pytest.skip("patches/options-order-guard.patch no longer applies cleanly: regenerate it from main")
    assert state in ("applicable", "applied")


@needs_git
def test_patched_tree_passes_its_own_tests_and_the_existing_risk_and_guard_suites(tmp_path):
    if _state() != "applicable":
        pytest.skip("patch already applied or drifted; nothing to verify on a scratch copy")
    work = tmp_path / "tree"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "options", "*.egg-info")
    for d in ("tradex", "config", "tests", "strategies", "research", "data", "tools", "patches"):
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, work / d, ignore=ignore if d == "tests" else shutil.ignore_patterns(
                "__pycache__", "*.pyc"))
    shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
    r = _git("apply", str(PATCH), cwd=work)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, PYTHONPATH=str(work))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/test_options_risk.py",
         "tests/test_execution.py", "tests/test_runtime_venues.py", "tests/test_spine.py"],
        cwd=work, env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-1500:]
    assert "passed" in run.stdout
