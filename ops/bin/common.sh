# shellcheck shell=bash
# Shared by the ops scripts. Source it; do not run it. Works with macOS's stock bash 3.2.
#
# Settings come from ~/.config/tradex/ops.env (paths and switches only). Secrets never go in
# that file: the healthchecks.io URLs and the restic password are read from the macOS Keychain.

OPS_ENV="${TRADEX_OPS_ENV:-$HOME/.config/tradex/ops.env}"
if [ -f "$OPS_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$OPS_ENV"
    set +a
fi

TRADEX_HOME="${TRADEX_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRADEX_PYTHON="${TRADEX_PYTHON:-$TRADEX_HOME/.venv/bin/python}"
TRADEX_LEDGER="${TRADEX_LEDGER:-$TRADEX_HOME/data/ledger/live.sqlite}"
TRADEX_STATE="${TRADEX_STATE:-$TRADEX_HOME/data/state}"
TRADEX_LOG_DIR="${TRADEX_LOG_DIR:-$HOME/Library/Logs/tradex}"
export TRADEX_HOME TRADEX_PYTHON TRADEX_LEDGER TRADEX_STATE TRADEX_LOG_DIR

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

die() { # die CODE message...
    local code="$1"
    shift
    log "ERROR: $*"
    exit "$code"
}

# secret_get ENV_VAR KEYCHAIN_SERVICE KEYCHAIN_ACCOUNT: the env var wins (handy in tests), else the Keychain.
# Prints nothing and returns 1 when the secret is not there. Never prints the secret anywhere but stdout.
secret_get() {
    local var="$1" service="$2" account="$3" val=""
    val="${!var:-}"
    if [ -z "$val" ]; then
        val="$("${SECURITY_BIN:-security}" find-generic-password -s "$service" -a "$account" -w 2>/dev/null || true)"
    fi
    [ -n "$val" ] || return 1
    printf '%s' "$val"
}

# stamp NAME: leave a UTC timestamp in $TRADEX_STATE/NAME; the dashboard reads it to show the age of the last success.
stamp() {
    mkdir -p "$TRADEX_STATE"
    date -u +%Y-%m-%dT%H:%M:%S+00:00 >"$TRADEX_STATE/$1.tmp.$$"
    mv "$TRADEX_STATE/$1.tmp.$$" "$TRADEX_STATE/$1"
}

# ping URL [suffix]: healthchecks.io ping; failures here must never break the caller.
ping_hc() {
    local url="$1" suffix="${2:-}" body="${3:-}"
    if [ -n "$body" ]; then
        "${CURL_BIN:-curl}" -fsS -m 10 --retry 3 -o /dev/null --data-raw "$body" "$url$suffix" || return 1
    else
        "${CURL_BIN:-curl}" -fsS -m 10 --retry 3 -o /dev/null "$url$suffix" || return 1
    fi
}
