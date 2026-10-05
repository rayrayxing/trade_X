"""The weekly report, `tradex research loop` on the command line, and the launchd entry point."""
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tradex import cli
from tradex.research.loop import adapters
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.report import md, render
from tradex.research.loop.state import LoopState

from loopkit import CATALOG_ROWS, idea_file, make_rig, spec_dict, write_spec

ROOT = Path(__file__).resolve().parents[2]
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


# --- report ------------------------------------------------------------------------------------------

def test_report_lists_every_stage_with_the_numbers_and_the_audit_status(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(screen_min_bars=100, apply=True))
    rig.add_spec()
    rig.add_spec(spec_dict("stk-weak-test"))
    rig.ev.screen_out["stk-weak-test"] = {"trades": 3, "profit_factor": 0.5, "sharpe": -1.0, "max_drawdown": -0.2,
                                          "total_return": -0.1, "win_rate": 0.2, "costs_share_of_gross": 0.4,
                                          "params": {}, "warnings": []}
    rig.run()
    text = (rig.out_dir / "weekly-2026-W41.md").read_text()
    for heading in ("# Research loop, 2026-W41", "## Where the strategies stand", "## Screen", "## Walk-forward (stage-3 gate)",
                    "## Holdout (one recorded look per version)", "## Paper queue decisions", "## Trial ledger (DSR's N)"):
        assert heading in text, heading
    assert "Audit trail: intact." in text and "**applied**" in text
    assert "`stk-rsi-test:v1`" in text and "`stk-weak-test:v1`" in text and "**rejected**" in text
    assert "| paper | 1 |" in text and "| rejected | 1 |" in text
    assert "stk-rsi-test`: 4 parameter sets" in text
    data = json.loads((rig.out_dir / "weekly-2026-W41.json").read_text())
    assert data["funnel"] == {"paper": 1, "rejected": 1} and data["audit_ok"] is True


def test_report_shows_blocked_items_and_status_claims(tmp_path):
    from loopkit import FakeData
    rig = make_rig(tmp_path, data=FakeData(missing={"stk-rsi-test"}))
    rig.add_spec()
    rig.add_spec(spec_dict("stk-claims-test", status="paper"))
    rig.run()
    text = (rig.out_dir / "weekly-2026-W41.md").read_text()
    assert "## Blocked or failed (retried next run)" in text and "no bars for stk-rsi-test" in text
    assert "## Status claims without evidence" in text and "stk-claims-test" in text
    assert "## Alerts this run" in text


def test_untrusted_text_cannot_break_out_of_the_report(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "evil", title="Nice | idea\n# injected heading [click](http://evil.example) <script>x</script> `code`")
    rig.run()
    text = (rig.out_dir / "weekly-2026-W41.md").read_text()
    assert "<script>" not in text and "[click](" not in text and "\n# injected heading" not in text
    row = [ln for ln in text.splitlines() if "file:evil" in ln][0]
    assert row.count("|") - row.count("\\|") == 5                                 # still one table row of four cells
    assert md("a|b") == "a\\|b" and md("x" * 500).endswith("...") and len(md("x" * 500)) == 163


def test_rerunning_a_finished_week_rewrites_the_report_without_changing_state(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.run()
    n_looks, n_ev = len(rig.holdout.looks), len(rig.state.evidence("stk-rsi-test", 1))
    first = (rig.out_dir / "weekly-2026-W41.md").read_text()
    (rig.out_dir / "weekly-2026-W41.md").unlink()
    rig.run()
    assert (rig.out_dir / "weekly-2026-W41.md").exists()
    assert len(rig.holdout.looks) == n_looks and len(rig.state.evidence("stk-rsi-test", 1)) == n_ev
    assert first.splitlines()[0] == (rig.out_dir / "weekly-2026-W41.md").read_text().splitlines()[0]


def test_render_handles_an_empty_week():
    from types import SimpleNamespace
    s = {"run_id": "r", "generated": "now", "data_source": "x", "apply": False, "audit_ok": True, "audit_bad_row": None,
         "funnel": {}, "by_stage": {}, "steps": [], "awaiting_spec": [], "unverified": [], "alerts": [], "trial_counts": {}}
    text = render(s)
    assert text.startswith("# Research loop, r") and "recommendations only" in text


# --- command line ------------------------------------------------------------------------------------

def tree(path: Path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else []


def cli_args(tmp_path, *extra):
    strategies = tmp_path / "strategies"
    (strategies / "proposed").mkdir(parents=True, exist_ok=True)
    return ["research", "loop", "--strategies", str(strategies), "--extra-specs", "--ideas-dir", str(tmp_path / "ideas"),
            "--db", str(tmp_path / "state" / "loop.sqlite"), "--results-dir", str(tmp_path / "out"),
            "--ledger", str(tmp_path / "none.sqlite"), *extra]


def test_dry_plan_prints_the_plan_loads_no_data_and_writes_nothing(tmp_path, capsys, monkeypatch):
    write_spec(tmp_path / "strategies" / "proposed", spec_dict("stk-plan-test"))
    idea_file(type("R", (), {"ideas_dir": tmp_path / "ideas"}), "newidea")
    drafts = tmp_path / "ideas" / "specs"
    drafts.mkdir()
    write_spec(drafts, spec_dict("stk-plan-draft"))
    (drafts / "file__newidea.yaml").write_text((drafts / "stk-plan-draft.yaml").read_text())

    def boom(*a, **k):
        raise AssertionError("dry plan touched data")
    monkeypatch.setattr(adapters.GateResearchData, "frames", boom)
    monkeypatch.setattr(adapters.LockedHoldout, "__init__", boom)
    monkeypatch.setattr("tradex.core.ledger.Ledger.__init__", boom)
    before = tree(tmp_path)
    assert cli.main(cli_args(tmp_path, "--dry-plan")) == 0
    out = capsys.readouterr().out
    assert "research loop plan for run" in out and "no data loaded, nothing written" in out
    assert "state file: absent (first run)" in out
    assert "stk-plan-test:v1" in out and "file:newidea" in out and "draft_available: 1" in out
    assert "[propose]" in out and "[health]" in out and "[report]" in out
    assert tree(tmp_path) == before                                   # no state file, no results folder, nothing


def test_dry_plan_json_and_existing_state(tmp_path, capsys):
    rig = make_rig(tmp_path / "rig")
    rig.add_spec()
    rig.run(stages=("propose", "screen"))
    args = ["research", "loop", "--strategies", str(rig.root / "strategies"), "--extra-specs", "--ideas-dir", str(rig.ideas_dir),
            "--db", str(rig.state.path), "--results-dir", str(rig.out_dir), "--dry-plan", "--json"]
    before = rig.state.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
    assert cli.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    stages = {s["stage"]: s for s in plan["stages"]}
    assert plan["state_file"] == "present" and list(stages["walk_forward"]["strategies"]) == ["stk-rsi-test:v1"]
    assert stages["walk_forward"]["strategies"]["stk-rsi-test:v1"]["timeframe"] == "D1"
    assert rig.state.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == before


def test_a_real_run_with_no_data_blocks_every_data_item_and_invents_nothing(tmp_path, capsys, monkeypatch):
    """The production wiring on a machine with no bar cache: nothing runs on made-up data, the report says why."""
    from tradex.research import gate
    monkeypatch.setattr(gate, "OPEND_CACHE", tmp_path / "no-cache")
    monkeypatch.setattr("tradex.research.loop.adapters.GateResearchData.__init__",
                        lambda self, cache=None: setattr(self, "cache", tmp_path / "no-cache"))
    write_spec(tmp_path / "strategies" / "proposed", spec_dict("stk-cli-test"))
    rc = cli.main(cli_args(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    summary = json.loads(out[:out.rindex("}") + 1])
    assert summary["screen"]["blocked"] == 1 and summary["walk_forward"]["done"] == 0 and summary["holdout"]["done"] == 0
    assert "report: " in out
    text = next((tmp_path / "out").glob("weekly-*.md")).read_text()
    assert "Blocked or failed" in text and "stk-cli-test" in text
    st = LoopState(tmp_path / "state" / "loop.sqlite")
    assert st.strategy("stk-cli-test")["status"] == "proposed" and st.verify_audit()[0]
    assert "status: proposed" in (tmp_path / "strategies" / "proposed" / "stk-cli-test.yaml").read_text()


def test_stage_names_are_checked_by_argparse(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(cli_args(tmp_path, "--stages", "nonsense"))


def test_python_dash_m_entry_point_matches_the_cli():
    r = subprocess.run([sys.executable, "-m", "tradex.research.loop", "--dry-plan", "--strategies", str(ROOT / "strategies")],
                       capture_output=True, text=True, cwd=ROOT, timeout=120,
                       env={**os.environ, "TRADEX_LOOP_DB": str(ROOT / "data" / "research" / "never-created.sqlite")})
    assert r.returncode == 0 and "research loop plan for run" in r.stdout
    assert not (ROOT / "data" / "research" / "never-created.sqlite").exists()


# --- ops entry point ---------------------------------------------------------------------------------

SCRIPT = ROOT / "ops" / "bin" / "research-loop.sh"
PLIST = ROOT / "ops" / "launchd" / "com.tradex.research-loop.plist"


def test_script_is_executable_and_parses():
    assert SCRIPT.read_text().startswith("#!/bin/bash") and os.access(SCRIPT, os.X_OK)
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    mode = subprocess.run(["git", "ls-files", "-s", str(SCRIPT)], capture_output=True, text=True, cwd=ROOT).stdout
    assert not mode or mode.startswith("100755")


def test_plist_is_a_weekly_timer_that_is_not_loaded_by_anything():
    d = plistlib.loads(PLIST.read_bytes())
    assert d["Label"] == "com.tradex.research-loop" and d["RunAtLoad"] is False and "KeepAlive" not in d
    assert d["StartCalendarInterval"] == {"Weekday": 6, "Hour": 10, "Minute": 0}
    assert d["ProgramArguments"] == ["/bin/bash", "__TRADEX_HOME__/ops/bin/research-loop.sh"]
    assert "launchctl" not in SCRIPT.read_text().lower().replace("launchd", "")


def fake_python(tmp_path, code=0):
    p = tmp_path / "fakepy"
    p.write_text(f'#!/bin/bash\necho "$@" > "{tmp_path}/args.txt"\nexit {code}\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def script_env(tmp_path, py, **extra):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRADEX")}
    env.update(HOME=str(home), TRADEX_OPS_ENV=str(tmp_path / "no.env"), TRADEX_HOME=str(ROOT), TRADEX_PYTHON=str(py),
               TRADEX_STATE=str(tmp_path / "state"), TRADEX_LOG_DIR=str(tmp_path / "logs"),
               TRADEX_LEDGER=str(tmp_path / "ledger.sqlite"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_script(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


@needs_bash
def test_script_runs_the_loop_without_apply_by_default_and_leaves_a_stamp(tmp_path):
    py = fake_python(tmp_path)
    r = run_script(script_env(tmp_path, py))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "args.txt").read_text().split() == ["-m", "tradex", "research", "loop", "--ledger", str(tmp_path / "ledger.sqlite")]
    assert (tmp_path / "state" / "research_loop_ok").exists()
    assert not (tmp_path / "state" / "research-loop.lock").exists()


@needs_bash
def test_script_passes_apply_and_the_arxiv_query_only_when_asked(tmp_path):
    py = fake_python(tmp_path)
    r = run_script(script_env(tmp_path, py, TRADEX_LOOP_APPLY="1", TRADEX_LOOP_ARXIV_QUERY="cat:q-fin.TR"), "--stages", "report")
    assert r.returncode == 0
    args = (tmp_path / "args.txt").read_text().split()
    assert "--apply" in args and args[args.index("--arxiv-query") + 1] == "cat:q-fin.TR" and args[-2:] == ["--stages", "report"]


@needs_bash
def test_script_propagates_failure_and_leaves_no_stamp(tmp_path):
    r = run_script(script_env(tmp_path, fake_python(tmp_path, code=3)))
    assert r.returncode == 3 and not (tmp_path / "state" / "research_loop_ok").exists()
    assert not (tmp_path / "state" / "research-loop.lock").exists()


@needs_bash
def test_script_refuses_anything_but_paper(tmp_path):
    r = run_script(script_env(tmp_path, fake_python(tmp_path), TRADEX_MODE="live"))
    assert r.returncode == 78 and "paper only" in r.stderr and not (tmp_path / "args.txt").exists()


@needs_bash
def test_script_takes_a_lock_and_clears_a_stale_one(tmp_path):
    env = script_env(tmp_path, fake_python(tmp_path))
    lock = tmp_path / "state" / "research-loop.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))                      # a live process holds it
    r = run_script(env)
    assert r.returncode == 75 and "in progress" in r.stderr and lock.exists()
    (lock / "pid").write_text("999999")                              # no such process
    r = run_script(env)
    assert r.returncode == 0 and "stale lock" in r.stderr and not lock.exists()


@needs_bash
def test_script_dry_plan_goes_straight_to_the_plan(tmp_path):
    py = fake_python(tmp_path)
    r = run_script(script_env(tmp_path, py), "--dry-plan")
    assert r.returncode == 0
    assert (tmp_path / "args.txt").read_text().split() == ["-m", "tradex", "research", "loop", "--dry-plan"]
    assert not (tmp_path / "state" / "research_loop_ok").exists()


@needs_bash
def test_script_refuses_without_a_python(tmp_path):
    r = run_script(script_env(tmp_path, tmp_path / "missing-python"))
    assert r.returncode == 78 and "no Python" in r.stderr


@needs_bash
def test_the_nightly_backup_snapshots_the_loops_run_state_too(tmp_path):
    import sqlite3
    sys.path.insert(0, str(ROOT / "tests"))
    from test_ops import BIN, backup_env, run as run_cmd
    LoopState(tmp_path / "repo-loop.sqlite").register("stk-a", 1, "p", "h")
    env, calls = backup_env(tmp_path, TRADEX_LOOP_DB=tmp_path / "repo-loop.sqlite")
    r = run_cmd(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    snap = tmp_path / "state" / "backup-staging" / "research-loop.sqlite"
    assert snap.exists() and sqlite3.connect(snap).execute("SELECT strategy_id FROM strategies").fetchall() == [("stk-a",)]
