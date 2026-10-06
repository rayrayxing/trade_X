"""ops/: launchd plists parse and are safe, scripts behave with fake restic/curl/launchctl, the installer copies but never loads."""
import json
import os
import plistlib
import re
import shutil
import site
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tradex.core.ledger import Ledger
from tradex.core.records import Health

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
BIN = OPS / "bin"
PLISTS = sorted((OPS / "launchd").glob("*.plist"))
SERVICES = {"core", "agent-worker", "telegram", "gateway-proxy", "dashboard", "backup"}
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash needed")


def load(p: Path) -> dict:
    return plistlib.loads(p.read_bytes())          # same job plutil -lint does: the file must be a valid XML property list


# --- plists ---------------------------------------------------------------------------------------

def test_six_services_and_a_heartbeat_timer_exist():
    labels = {load(p)["Label"] for p in PLISTS}
    # timers, not services: the heartbeat and the weekly research loop
    assert labels == {f"com.tradex.{s}" for s in SERVICES} | {"com.tradex.heartbeat", "com.tradex.research-loop"}
    assert len(labels) == len(PLISTS)


@pytest.mark.parametrize("path", PLISTS, ids=lambda p: p.stem)
def test_plist_is_valid_and_complete(path):
    d = load(path)
    assert path.name == d["Label"] + ".plist"
    args = d["ProgramArguments"]
    assert all(isinstance(a, str) and a for a in args) and args[0].startswith("/")        # launchd needs an absolute program
    assert d["WorkingDirectory"] == "__TRADEX_HOME__"
    assert d["StandardOutPath"].startswith("__HOME__/Library/Logs/tradex/") and d["StandardErrorPath"].startswith("__HOME__/Library/Logs/tradex/")
    # every script a plist runs exists in the repo
    for a in args:
        if a.startswith("__TRADEX_HOME__/"):
            assert (ROOT / a.removeprefix("__TRADEX_HOME__/")).is_file(), a


@pytest.mark.parametrize("path", PLISTS, ids=lambda p: p.stem)
def test_plist_holds_no_secrets_and_no_account_ids(path):
    text = path.read_text()
    env = load(path)["EnvironmentVariables"]
    assert set(env) <= {"PATH", "TRADEX_HOME"}
    assert not re.search(r"(?i)api[_-]?key|token|password|secret|sk-[A-Za-z0-9]{10}", text)


def test_schedules_and_restart_policy():
    by = {load(p)["Label"].split(".")[-1]: load(p) for p in PLISTS}
    assert by["backup"]["StartCalendarInterval"] == {"Hour": 6, "Minute": 30}
    assert by["backup"]["RunAtLoad"] is False and "KeepAlive" not in by["backup"]
    assert by["heartbeat"]["StartInterval"] == 300
    for s in ("core", "agent-worker", "telegram", "gateway-proxy", "dashboard"):
        assert by[s]["KeepAlive"] == {"SuccessfulExit": False}, s          # restart after a crash, not after a clean stop
        assert by[s]["ThrottleInterval"] >= 30, s                             # no hot restart loop
    assert by["core"]["ProgramArguments"][0] == "/usr/bin/caffeinate"
    assert by["dashboard"]["ProgramArguments"][-1] == "dashboard"


def test_only_the_heartbeat_talks_to_launchctl_and_only_to_read():
    for sh in BIN.glob("*.sh"):
        lines = [ln for ln in sh.read_text().splitlines() if "launchctl" in ln.lower()]
        if sh.name == "healthcheck.sh":
            assert any('LAUNCHCTL_BIN:-launchctl}" list ' in ln for ln in lines)
            assert not any(re.search(r"\b(bootstrap|load|kickstart|enable|submit)\b", ln) for ln in lines)
        elif sh.name == "install.sh":
            # only in the instructions it prints (the behavioural test below proves it never runs launchctl)
            assert all(ln.lstrip().startswith(("launchctl", "To stop", "#", "1.", "Suggested")) or "launchctl" in ln.split("#")[0] and "bootstrap" in ln or "bootout" in ln for ln in lines)
        else:
            assert not lines, sh.name


# --- shell syntax -------------------------------------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("sh", sorted(BIN.glob("*.sh")), ids=lambda p: p.name)
def test_scripts_parse_and_are_executable_in_git(sh):
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    assert sh.read_text().startswith(("#!/bin/bash", "# shellcheck shell=bash"))
    if sh.name != "common.sh":
        assert os.access(sh, os.X_OK), f"{sh.name} must be executable (chmod +x)"


# --- helpers for fake binaries ------------------------------------------------------------------------------

def fake(dirpath: Path, name: str, body: str) -> Path:
    p = dirpath / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def env_for(tmp: Path, **extra) -> dict:
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRADEX", "HC_", "RESTIC"))}
    env.update(HOME=str(home), TRADEX_OPS_ENV=str(tmp / "no-such-ops.env"), TRADEX_HOME=str(ROOT),
               TRADEX_PYTHON=sys.executable, TRADEX_STATE=str(tmp / "state"), TRADEX_LOG_DIR=str(tmp / "logs"),
               TRADEX_LEDGER=str(tmp / "ledger.sqlite"), SECURITY_BIN="/bin/false",
               PYTHONUSERBASE=site.getuserbase(), PYTHONPATH=str(ROOT))            # a fake HOME must not hide user-installed packages
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(cmd, env, **kw):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, **kw)


def make_ledger(path: Path, run_id="paper-1") -> None:
    led = Ledger(path, run_id=run_id, git_commit="x")
    led.append(Health("2026-10-06T00:00:00+00:00", "core", True))
    led.close()


# --- installer ---------------------------------------------------------------------------------------------------

@needs_bash
def test_install_copies_renders_and_never_loads(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    calls = tmp_path / "launchctl.calls"
    fake(bins, "launchctl", f'echo "$@" >> "{calls}"\n')
    home, dest = tmp_path / "ray home", tmp_path / "LaunchAgents"            # a space in the path on purpose
    env = env_for(tmp_path, PATH=f"{bins}:{os.environ['PATH']}")
    r = run(["bash", str(BIN / "install.sh"), "--dest", str(dest), "--home", str(home)], env)
    assert r.returncode == 0, r.stderr
    assert not calls.exists(), "installer must not call launchctl"
    written = sorted(p.name for p in dest.glob("*.plist"))
    assert written == sorted(p.name for p in PLISTS)
    for p in dest.glob("*.plist"):
        d = load(p)
        blob = p.read_text()
        assert "__TRADEX_HOME__" not in blob and "__HOME__" not in blob
        assert d["WorkingDirectory"] == str(ROOT)
        assert d["StandardOutPath"].startswith(str(home / "Library/Logs/tradex"))
    envfile = home / ".config/tradex/ops.env"
    assert envfile.exists() and stat.S_IMODE(envfile.stat().st_mode) == 0o600
    assert not re.search(r"(?im)^[A-Z_]*(KEY|TOKEN|PASSWORD|SECRET)[A-Z_]*=", envfile.read_text())
    assert "Nothing has been started" in r.stdout
    again = run(["bash", str(BIN / "install.sh"), "--dest", str(dest), "--home", str(home)], env)       # idempotent
    assert again.returncode == 0 and "kept" in again.stdout and not calls.exists()


@needs_bash
def test_install_dry_run_and_only(tmp_path):
    dest, home = tmp_path / "la", tmp_path / "h"
    env = env_for(tmp_path)
    r = run(["bash", str(BIN / "install.sh"), "--dry-run", "--dest", str(dest), "--home", str(home)], env)
    assert r.returncode == 0 and "[dry run]" in r.stdout and not dest.exists() and not (home / ".config").exists()
    r = run(["bash", str(BIN / "install.sh"), "--only", "dashboard,backup", "--dest", str(dest), "--home", str(home)], env)
    assert r.returncode == 0
    assert sorted(p.name for p in dest.glob("*.plist")) == ["com.tradex.backup.plist", "com.tradex.dashboard.plist"]


@needs_bash
def test_install_does_not_clobber_a_different_plist_without_force(tmp_path):
    dest, home = tmp_path / "la", tmp_path / "h"
    dest.mkdir()
    (dest / "com.tradex.dashboard.plist").write_text("hand edited")
    env = env_for(tmp_path)
    r = run(["bash", str(BIN / "install.sh"), "--only", "dashboard", "--dest", str(dest), "--home", str(home)], env)
    assert "skip" in r.stdout and (dest / "com.tradex.dashboard.plist").read_text() == "hand edited"
    run(["bash", str(BIN / "install.sh"), "--only", "dashboard", "--force", "--dest", str(dest), "--home", str(home)], env)
    assert load(dest / "com.tradex.dashboard.plist")["Label"] == "com.tradex.dashboard"


# --- run-service / proxy ---------------------------------------------------------------------------------------------

@needs_bash
def test_run_service_refuses_non_paper_modes_and_missing_commands(tmp_path):
    env = env_for(tmp_path, TRADEX_MODE="live")
    r = run(["bash", str(BIN / "run-service.sh"), "core"], env)
    assert r.returncode == 78 and "paper accounts only" in r.stderr
    env = env_for(tmp_path)
    r = run(["bash", str(BIN / "run-service.sh"), "agent-worker"], env)
    assert r.returncode == 78 and "does not exist" in r.stderr
    assert run(["bash", str(BIN / "run-service.sh"), "nope"], env).returncode == 64


@needs_bash
def test_run_proxy_needs_an_executable(tmp_path):
    assert run(["bash", str(BIN / "run-proxy.sh")], env_for(tmp_path)).returncode == 78
    assert run(["bash", str(BIN / "run-proxy.sh")], env_for(tmp_path, TRADEX_PROXY_BIN=tmp_path / "missing")).returncode == 78
    out = tmp_path / "args.txt"
    prox = fake(tmp_path, "proxy", f'echo "$@" > "{out}"\n')
    r = run(["bash", str(BIN / "run-proxy.sh")], env_for(tmp_path, TRADEX_PROXY_BIN=prox, TRADEX_PROXY_ARGS="--port 8317"))
    assert r.returncode == 0 and out.read_text().strip() == "--port 8317"


# --- snapshot helper -----------------------------------------------------------------------------------------------------

def test_snapshot_copies_a_live_wal_ledger_and_verifies_the_chain(tmp_path):
    src, dst = tmp_path / "live.sqlite", tmp_path / "stage" / "ledger.sqlite"
    make_ledger(src)
    holder = Ledger(src, run_id="paper-1", git_commit="x")                        # an open writer, WAL mode
    holder.append(Health("2026-10-06T00:01:00+00:00", "core", True))
    r = subprocess.run([sys.executable, str(BIN / "snapshot_db.py"), str(src), str(dst), "--verify-chain"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    copy = Ledger(dst, read_only=True, git_commit="")
    assert len(copy.rows(kind="health")) == 2 and copy.verify() == (True, None)
    copy.close()
    holder.close()


def test_snapshot_flags_a_tampered_ledger_and_a_missing_source(tmp_path):
    src, dst = tmp_path / "live.sqlite", tmp_path / "copy.sqlite"
    make_ledger(src)
    db = sqlite3.connect(src)
    db.execute("UPDATE events SET payload=replace(payload,'core','evil')")
    db.commit()
    db.close()
    r = subprocess.run([sys.executable, str(BIN / "snapshot_db.py"), str(src), str(dst), "--verify-chain"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "hash chain broken" in r.stderr
    assert subprocess.run([sys.executable, str(BIN / "snapshot_db.py"), str(tmp_path / "none"), str(dst)], capture_output=True, cwd=ROOT).returncode == 3


# --- backup ---------------------------------------------------------------------------------------------------------------------

def backup_env(tmp: Path, mounted=True, restic_exit=0, **extra):
    bins = tmp / "fakebin"
    bins.mkdir(exist_ok=True)
    calls = tmp / "calls.log"
    fake(bins, "restic", f'echo "restic $*" >> "{calls}"\n[ "$1" = backup ] && exit {restic_exit}\nexit 0\n')
    fake(bins, "curl", f'echo "curl $*" >> "{calls}"\n')
    vol = "/Volumes/TradexBackup"
    fake(bins, "mount", f'echo "{"/dev/disk9s1 on " + vol + " (apfs, local, nodev)" if mounted else "/dev/disk1s1 on / (apfs, local)"}"\n')
    ledger = tmp / "ledger.sqlite"
    make_ledger(ledger)
    (tmp / "repo" / "config").mkdir(parents=True, exist_ok=True)                  # a stand-in checkout for TRADEX_HOME
    (tmp / "repo" / "ops" / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy(BIN / "snapshot_db.py", tmp / "repo" / "ops" / "bin" / "snapshot_db.py")
    return env_for(tmp, RESTIC_BIN=bins / "restic", CURL_BIN=bins / "curl", MOUNT_BIN=bins / "mount",
                   TRADEX_BACKUP_VOLUME=vol, HC_BACKUP_URL="https://hc.example/ping/abc", TRADEX_HOME=tmp / "repo", **extra), calls


@needs_bash
def test_backup_snapshots_the_ledger_backs_up_prunes_checks_and_pings(tmp_path):
    env, calls = backup_env(tmp_path)
    r = run(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    lines = calls.read_text().splitlines()
    verbs = [ln.split()[1] for ln in lines if ln.startswith("restic")]
    assert verbs == ["cat", "backup", "forget", "check"]
    backup_line = next(ln for ln in lines if ln.startswith("restic backup"))
    assert str(tmp_path / "state" / "backup-staging") in backup_line and str(tmp_path / "repo" / "config") in backup_line
    assert "data/cache" not in backup_line                                       # bars are re-downloadable
    assert "--keep-daily 14" in next(ln for ln in lines if ln.startswith("restic forget"))
    assert (tmp_path / "state" / "backup-staging" / "ledger.sqlite").exists()
    assert lines[-1] == "curl -fsS -m 10 --retry 3 -o /dev/null https://hc.example/ping/abc"
    assert (tmp_path / "state" / "backup_ok").read_text().strip().endswith("+00:00")


@needs_bash
def test_backup_refuses_when_the_drive_is_not_mounted(tmp_path):
    env, calls = backup_env(tmp_path, mounted=False)
    r = run(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    assert r.returncode == 75 and "not mounted" in r.stderr
    log = calls.read_text()
    assert "restic" not in log and "/fail" in log                                 # nothing written anywhere; the failure was reported


@needs_bash
def test_backup_failure_sends_fail_ping_and_leaves_no_success_stamp(tmp_path):
    env, calls = backup_env(tmp_path, restic_exit=1)
    r = run(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    assert r.returncode == 1 and "restic backup failed" in r.stderr
    assert "/fail" in calls.read_text() and "forget" not in calls.read_text()
    assert not (tmp_path / "state" / "backup_ok").exists()


@needs_bash
def test_backup_repository_must_be_on_the_backup_volume(tmp_path):
    env, _ = backup_env(tmp_path, RESTIC_REPOSITORY=tmp_path / "internal-disk-repo")
    r = run(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    assert r.returncode == 78 and "not on the backup volume" in r.stderr


@needs_bash
def test_backup_password_comes_from_the_keychain_command_not_a_file(tmp_path):
    bins = tmp_path / "fakebin"
    env, calls = backup_env(tmp_path)
    fake(bins, "restic", f'echo "pwcmd=$RESTIC_PASSWORD_COMMAND" >> "{calls}"\nexit 0\n')
    fake(bins, "security", "echo hunter2\n")
    env["SECURITY_BIN"] = str(bins / "security")
    run(["bash", str(BIN / "backup.sh")], env, cwd=ROOT)
    text = calls.read_text()
    assert "find-generic-password -s trade-x-backup -a restic -w" in text and "hunter2" not in text


# --- heartbeat --------------------------------------------------------------------------------------------------------------------

def heartbeat_env(tmp: Path, services_running=("core", "telegram", "dashboard"), url="https://hc.example/ping/xyz", **extra):
    bins = tmp / "fakebin"
    bins.mkdir(exist_ok=True)
    calls = tmp / "calls.log"
    up = " ".join(services_running)
    fake(bins, "launchctl", f'for s in {up}; do [ "$2" = "com.tradex.$s" ] && {{ echo \'"PID" = 123;\'; exit 0; }}; done\nexit 113\n')
    fake(bins, "curl", f'echo "curl $*" >> "{calls}"\n')
    make_ledger(tmp / "ledger.sqlite")
    e = env_for(tmp, **{"LAUNCHCTL_BIN": bins / "launchctl", "CURL_BIN": bins / "curl", "TRADEX_MIN_FREE_GB": 0, **extra})
    if url:
        e["HC_PING_URL"] = url
    return e, calls


@needs_bash
def test_heartbeat_pings_when_everything_runs_and_leaves_a_stamp(tmp_path):
    env, calls = heartbeat_env(tmp_path)
    r = run(["bash", str(BIN / "healthcheck.sh")], env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == "curl -fsS -m 10 --retry 3 -o /dev/null https://hc.example/ping/xyz"
    from tradex.dashboard.views import parse_ts
    assert parse_ts((tmp_path / "state" / "healthcheck_ok").read_text().strip()) is not None      # the dashboard can read the stamp


@needs_bash
def test_heartbeat_reports_a_stopped_service_instead_of_pinging_ok(tmp_path):
    env, calls = heartbeat_env(tmp_path, services_running=("telegram", "dashboard"))
    r = run(["bash", str(BIN / "healthcheck.sh")], env, cwd=ROOT)
    assert r.returncode == 1 and "service core is not running" in r.stderr
    log = calls.read_text()
    assert "/fail" in log and "service core is not running" in log and log.count("curl") == 1
    assert not (tmp_path / "state" / "healthcheck_ok").exists()


@needs_bash
def test_heartbeat_flags_unreadable_or_missing_ledger_and_low_disk(tmp_path):
    env, calls = heartbeat_env(tmp_path, TRADEX_MIN_FREE_GB=10_000_000)
    (tmp_path / "ledger.sqlite").unlink()
    r = run(["bash", str(BIN / "healthcheck.sh")], env, cwd=ROOT)
    assert r.returncode == 1 and "no ledger file" in r.stderr and "free disk" in r.stderr


@needs_bash
def test_heartbeat_without_a_url_does_nothing_and_says_so(tmp_path):
    env, calls = heartbeat_env(tmp_path, url=None)
    r = run(["bash", str(BIN / "healthcheck.sh")], env, cwd=ROOT)
    assert r.returncode == 2 and "no healthchecks.io URL" in r.stderr and not calls.exists()


# --- power ---------------------------------------------------------------------------------------------------------------------------

@needs_bash
def test_power_check_reports_and_never_changes_anything(tmp_path):
    changed = tmp_path / "changed"
    good = " sleep 0\n disksleep 0\n womp 1\n autorestart 1\n powernap 0\n"
    pm = fake(tmp_path, "pmset", f'[ "$1" = "-g" ] || {{ touch "{changed}"; exit 9; }}\nprintf \'{good}\'\n')
    r = run(["bash", str(BIN / "power-check.sh")], env_for(tmp_path, PMSET_BIN=pm))
    assert r.returncode == 0 and "look right" in r.stdout
    bad = good.replace("sleep 0\n disksleep", "sleep 1\n disksleep")
    pm = fake(tmp_path, "pmset", f'[ "$1" = "-g" ] || {{ touch "{changed}"; exit 9; }}\nprintf \'{bad}\'\n')
    r = run(["bash", str(BIN / "power-check.sh")], env_for(tmp_path, PMSET_BIN=pm))
    assert r.returncode == 1 and "FIX   sleep" in r.stdout and "sudo pmset" in r.stdout
    assert not changed.exists()


def test_gitignore_keeps_runtime_state_out_of_git():
    ignore = (ROOT / ".gitignore").read_text().split()
    assert "data/state/" in ignore and "data/ledger/" in ignore


def test_readme_exists_and_names_every_service():
    text = (OPS / "README.md").read_text()
    for s in SERVICES | {"heartbeat"}:
        assert f"com.tradex.{s}" in text, s
    for needle in ("Time Machine", "restic", "healthchecks.io", "pmset", "Amphetamine", "trade-x setup"):
        assert needle in text, needle
