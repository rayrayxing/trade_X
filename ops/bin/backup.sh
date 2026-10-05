#!/bin/bash
# Nightly encrypted backup with restic to an external drive. launchd runs it at 06:30; run it by hand to test.
#
#   backup.sh           back up, prune old snapshots, spot-check the repository
#   backup.sh --init    create the restic repository on the drive (once)
#
# Settings in ~/.config/tradex/ops.env:
#   TRADEX_BACKUP_VOLUME=/Volumes/TradexBackup       the drive's mount point (required)
#   RESTIC_REPOSITORY=/Volumes/TradexBackup/tradex-restic   (default: that volume + /tradex-restic)
# Secrets (Keychain, never files):
#   restic password   security add-generic-password -s trade-x-backup -a restic -w
#   healthchecks URL  security add-generic-password -s trade-x-ops -a hc_backup_url -w   (optional)
#
# What it never does: write to the repository path when the drive is not mounted (that would silently
# fill the Mac's own disk), or copy a live SQLite file (it uses SQLite's backup API instead).
set -euo pipefail
# shellcheck source=ops/bin/common.sh
. "$(dirname "$0")/common.sh"

RESTIC="${RESTIC_BIN:-restic}"
mode="backup"
[ "${1:-}" = "--init" ] && mode="init"

HC_URL="$(secret_get HC_BACKUP_URL trade-x-ops hc_backup_url || true)"

fail() { # fail CODE message
    local code="$1"
    shift
    log "BACKUP FAILED: $*"
    [ -n "$HC_URL" ] && ping_hc "$HC_URL" "/fail" "$*" || true
    exit "$code"
}

[ -n "${TRADEX_BACKUP_VOLUME:-}" ] || fail 78 "TRADEX_BACKUP_VOLUME is not set in $OPS_ENV"
command -v "$RESTIC" >/dev/null 2>&1 || fail 78 "restic is not installed (brew install restic)"
[ -x "$TRADEX_PYTHON" ] || fail 78 "no Python at $TRADEX_PYTHON"

# The drive must really be mounted, not just a leftover empty folder under /Volumes.
if ! "${MOUNT_BIN:-mount}" | grep -F -q " on $TRADEX_BACKUP_VOLUME ("; then
    fail 75 "backup drive is not mounted at $TRADEX_BACKUP_VOLUME"
fi

export RESTIC_REPOSITORY="${RESTIC_REPOSITORY:-$TRADEX_BACKUP_VOLUME/tradex-restic}"
case "$RESTIC_REPOSITORY" in
    "$TRADEX_BACKUP_VOLUME"/*) ;;
    *) fail 78 "RESTIC_REPOSITORY ($RESTIC_REPOSITORY) is not on the backup volume" ;;
esac
if [ -z "${RESTIC_PASSWORD:-}" ] && [ -z "${RESTIC_PASSWORD_FILE:-}" ] && [ -z "${RESTIC_PASSWORD_COMMAND:-}" ]; then
    export RESTIC_PASSWORD_COMMAND="${SECURITY_BIN:-security} find-generic-password -s trade-x-backup -a restic -w"
fi

if [ "$mode" = "init" ]; then
    "$RESTIC" init || fail 1 "restic init failed"
    log "repository created at $RESTIC_REPOSITORY. Store the restic password somewhere that is NOT this Mac."
    exit 0
fi

"$RESTIC" cat config >/dev/null 2>&1 || fail 1 "no restic repository at $RESTIC_REPOSITORY (run: backup.sh --init) or the password is wrong"

# 1. Consistent copies of the SQLite files, in a fixed staging folder so restic dedups them night to night.
stage="$TRADEX_STATE/backup-staging"
mkdir -p "$stage"
rm -f "$stage"/*.sqlite
if [ -f "$TRADEX_LEDGER" ]; then
    "$TRADEX_PYTHON" "$TRADEX_HOME/ops/bin/snapshot_db.py" "$TRADEX_LEDGER" "$stage/ledger.sqlite" --verify-chain \
        || fail 1 "ledger snapshot failed or its hash chain is broken (see log)"
else
    log "no ledger at $TRADEX_LEDGER yet; backing up everything else"
fi
trials="${TRADEX_TRIALS_DB:-$TRADEX_HOME/data/research/trials.sqlite}"
if [ -f "$trials" ]; then
    "$TRADEX_PYTHON" "$TRADEX_HOME/ops/bin/snapshot_db.py" "$trials" "$stage/trials.sqlite" || fail 1 "trials snapshot failed"
fi
loopdb="${TRADEX_LOOP_DB:-$TRADEX_HOME/data/research/loop.sqlite}"      # the weekly research loop's run state and audit trail
if [ -f "$loopdb" ]; then
    "$TRADEX_PYTHON" "$TRADEX_HOME/ops/bin/snapshot_db.py" "$loopdb" "$stage/research-loop.sqlite" || fail 1 "research-loop snapshot failed"
fi

# 2. Back up only what cannot be rebuilt. Bars (data/cache) and the venv are re-downloadable and excluded.
paths="$stage $TRADEX_HOME/config $TRADEX_HOME/strategies $TRADEX_HOME/reports $TRADEX_HOME/research"
[ -d "$TRADEX_HOME/data/calendar" ] && paths="$paths $TRADEX_HOME/data/calendar"
[ -f "$OPS_ENV" ] && paths="$paths $OPS_ENV"
existing=""
for p in $paths; do
    [ -e "$p" ] && existing="$existing $p"
done
# shellcheck disable=SC2086  # $existing is a list of paths without spaces
"$RESTIC" backup --tag nightly --host tradex-mac --exclude-caches --exclude '*.pyc' --exclude '__pycache__' $existing \
    || fail 1 "restic backup failed"

# 3. Keep 14 daily, 8 weekly and 12 monthly snapshots; then spot-check 2% of the stored data.
"$RESTIC" forget --tag nightly --host tradex-mac --keep-daily 14 --keep-weekly 8 --keep-monthly 12 --prune \
    || fail 1 "restic forget/prune failed"
"$RESTIC" check --read-data-subset=2% || fail 1 "restic check found a problem: do not trust this repository"

stamp backup_ok
[ -n "$HC_URL" ] && ping_hc "$HC_URL" || true
log "backup ok -> $RESTIC_REPOSITORY"
