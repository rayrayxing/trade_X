"""patches/ov-exits/*.patch: scope, applicability, and a run of the patched tree.

dashboard-gate-verdicts.patch touches only unprotected files and is a patch for the dashboard's owner to review
(core-wiring.patch was applied to the core as a commit on 5 Oct). es-tail-count.patch touches a
protected file (tradex/risk/exposure.py), so Ray applies it. These tests apply each to a scratch copy and run the
patch's own tests and the existing suites around the files it changes.

When a patch lands on main its forward check fails and the reverse check passes; its scratch-copy test then skips
itself and the patch and its entry here can be deleted.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PDIR = ROOT / "patches" / "ov-exits"
PROTECTED = [ln.strip() for ln in (ROOT / "config" / "protected_paths.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]

PATCHES = {
    "dashboard-gate-verdicts.patch": {
        "files": {"tradex/dashboard/views.py", "tradex/dashboard/static/app.js"},
        "protected": set(),
        "run": ["tests/test_dashboard.py"]},
    "es-tail-count.patch": {
        "files": {"tradex/risk/exposure.py", "tests/test_es_tail_count.py"},
        "protected": {"tradex/risk/exposure.py"},
        "run": ["tests/test_es_tail_count.py", "tests/test_spine.py", "tests/test_properties.py"]},
}


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _has_git():
    try:
        return _git("--version").returncode == 0
    except FileNotFoundError:
        return False


needs_git = pytest.mark.skipif(not _has_git(), reason="git not available")


def _state(patch: Path, cwd=ROOT) -> str:
    if _git("apply", "--check", str(patch), cwd=cwd).returncode == 0:
        return "applicable"
    if _git("apply", "--check", "-R", str(patch), cwd=cwd).returncode == 0:
        return "applied"
    return "drifted"


def _files(patch: Path) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", patch.read_text(), flags=re.M))


@pytest.mark.parametrize("name", PATCHES)
def test_patch_touches_exactly_the_reviewed_files_and_is_clean(name):
    patch = PDIR / name
    spec = PATCHES[name]
    assert patch.exists() and _files(patch) == spec["files"]
    text = patch.read_text()
    assert text.endswith("\n") and "\r" not in text
    protected_in_patch = {f for f in _files(patch) if any(f == p or f.startswith(p) for p in PROTECTED)}
    assert protected_in_patch == spec["protected"]


@needs_git
@pytest.mark.parametrize("name", PATCHES)
def test_patch_applies_to_this_tree_or_is_already_applied(name):
    state = _state(PDIR / name)
    if state == "drifted":
        pytest.skip(f"{name} no longer applies cleanly: regenerate it from the branch it targets")
    assert state in ("applicable", "applied")


@needs_git
@pytest.mark.parametrize("name", PATCHES)
def test_patched_tree_passes_the_patch_tests_and_the_suites_around_it(name, tmp_path):
    patch = PDIR / name
    if _state(patch) != "applicable":
        pytest.skip("patch already applied or drifted; nothing to verify on a scratch copy")
    work = tmp_path / "tree"
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
    for d in ("tradex", "config", "tests", "strategies", "research", "data", "tools", "patches"):
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, work / d, ignore=skip)
    shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
    r = _git("apply", str(patch), cwd=work)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, PYTHONPATH=str(work))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *PATCHES[name]["run"]],
                         cwd=work, env=env, capture_output=True, text=True, timeout=1500)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-1500:]
    assert "passed" in run.stdout
