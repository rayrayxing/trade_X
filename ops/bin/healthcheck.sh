#!/bin/bash
# Dead-man switch: every 5 minutes (com.tradex.heartbeat) check that trading is really running, then ping
# healthchecks.io. If the Mac dies, sleeps, loses power or the network, or a service stops, the ping stops
# and healthchecks.io alerts Ray (set its grace period to 15 minutes, notifications to email/Telegram/SMS).
# A failed local check sends /fail with the reasons, so the alert says what is wrong.
#
#   URL (Keychain):  security add-generic-password -s trade-x-ops -a hc_ping_url -w
#   settings (ops.env): TRADEX_REQUIRED_SERVICES="core telegram dashboard"   (default)
#                       TRADEX_MIN_FREE_GB=5
#
# Exit codes: 0 pinged ok, 1 a check failed (fail ping sent), 2 no URL configured.
set -euo pipefail
# shellcheck source=ops/bin/common.sh
. "$(dirname "$0")/common.sh"

HC_URL="$(secret_get HC_PING_URL trade-x-ops hc_ping_url || true)"
[ -n "$HC_URL" ] || die 2 "no healthchecks.io URL (Keychain item trade-x-ops / hc_ping_url); nothing to ping"

problems=""
add() { problems="${problems}${problems:+; }$1"; }

# 1. Each required service has a running process (launchctl lists a PID only while it runs).
for s in ${TRADEX_REQUIRED_SERVICES:-core telegram dashboard}; do
    out="$("${LAUNCHCTL_BIN:-launchctl}" list "com.tradex.$s" 2>/dev/null || true)"
    case "$out" in
        *'"PID" ='*) ;;
        *) add "service $s is not running" ;;
    esac
done

# 2. The ledger exists and opens read-only.
if [ -f "$TRADEX_LEDGER" ]; then
    "$TRADEX_PYTHON" - "$TRADEX_LEDGER" <<'PY' >/dev/null 2>&1 || add "ledger cannot be opened"
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=5)
db.execute("SELECT 1 FROM events LIMIT 1")
PY
else
    add "no ledger file"
fi

# 3. Optional: the ledger is being written (set for a venue that trades every bar; quiet markets can be quiet for hours).
if [ -n "${TRADEX_LEDGER_MAX_AGE_S:-}" ] && [ -f "$TRADEX_LEDGER" ]; then
    newest=0
    for f in "$TRADEX_LEDGER" "$TRADEX_LEDGER-wal"; do
        [ -f "$f" ] || continue
        m="$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)"
        [ "$m" -gt "$newest" ] && newest="$m"
    done
    age=$(( $(date +%s) - newest ))
    [ "$age" -le "$TRADEX_LEDGER_MAX_AGE_S" ] || add "ledger not written for ${age}s"
fi

# 4. Disk space for the ledger and logs.
free_kb="$(df -k "$TRADEX_HOME" | awk 'NR==2 {print $4}')"
min_kb=$(( ${TRADEX_MIN_FREE_GB:-5} * 1024 * 1024 ))
[ "${free_kb:-0}" -ge "$min_kb" ] || add "less than ${TRADEX_MIN_FREE_GB:-5} GB free disk"

if [ -n "$problems" ]; then
    log "UNHEALTHY: $problems"
    ping_hc "$HC_URL" "/fail" "$problems" || log "could not reach healthchecks.io"
    exit 1
fi
ping_hc "$HC_URL" || { log "could not reach healthchecks.io"; exit 1; }
stamp healthcheck_ok
