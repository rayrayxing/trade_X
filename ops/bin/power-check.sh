#!/bin/bash
# Read-only report on the Mac's sleep settings. Changes nothing; prints the commands to run yourself.
# Usage: ops/bin/power-check.sh
set -uo pipefail

PMSET="${PMSET_BIN:-pmset}"
out="$($PMSET -g 2>/dev/null || true)"
[ -n "$out" ] || { echo "pmset not available (this script is for the Mac)"; exit 2; }

val() { printf '%s\n' "$out" | awk -v k="$1" '$1 == k {print $2; exit}'; }
bad=0
check() { # name want hint
    local got
    got="$(val "$1")"
    if [ "$got" = "$2" ]; then
        printf 'ok    %-14s %s\n' "$1" "$got"
    else
        printf 'FIX   %-14s is %s, want %s  (%s)\n' "$1" "${got:-unset}" "$2" "$3"
        bad=1
    fi
}

check sleep 0 "Mac must not sleep while plugged in"
check disksleep 0 "disk must not spin down mid-write"
check womp 1 "wake for network access"
check autorestart 1 "start again after a power cut"
check powernap 0 "optional; avoids odd wake cycles"

if [ "$bad" -eq 1 ]; then
    cat <<'MSG'

To apply (asks for your password; takes effect immediately, survives reboots):
  sudo pmset -c sleep 0 disksleep 0 womp 1 autorestart 1 powernap 0
  (-c = while on the charger. On a MacBook also keep the lid open or use clamshell mode with power + display.)
MSG
    exit 1
fi
echo "power settings look right"
