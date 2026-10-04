#!/bin/bash
# Start one tradex service in the foreground (launchd runs this). Usage: run-service.sh core|agent-worker|telegram|dashboard
#
# It only starts a service whose command exists in this checkout. If the command or a flag is missing
# (for example the venue adapters have not merged yet) it exits 78 with a plain message instead of guessing.
# Phase 1 is paper only: any other TRADEX_MODE is refused here.
set -euo pipefail
# shellcheck source=ops/bin/common.sh
. "$(dirname "$0")/common.sh"

svc="${1:-}"
[ -n "$svc" ] || die 64 "usage: run-service.sh core|agent-worker|telegram|dashboard"
[ "${TRADEX_MODE:-paper}" = "paper" ] || die 78 "TRADEX_MODE=${TRADEX_MODE} refused: Phase 1 runs paper accounts only"
[ -x "$TRADEX_PYTHON" ] || die 78 "no Python at $TRADEX_PYTHON (create the venv: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]')"

cd "$TRADEX_HOME"
mkdir -p "$(dirname "$TRADEX_LEDGER")" "$TRADEX_STATE" "$TRADEX_LOG_DIR"

help_of() { "$TRADEX_PYTHON" -m tradex "$@" --help 2>&1 || true; }

case "$svc" in
    core)
        if help_of run | grep -q -- '--ledger'; then
            exec "$TRADEX_PYTHON" -m tradex run --mode paper --ledger "$TRADEX_LEDGER"
        fi
        exec "$TRADEX_PYTHON" -m tradex run --mode paper
        ;;
    agent-worker)
        help_of agents | grep -q 'worker' || die 78 "'tradex agents worker' does not exist in this checkout yet; do not load this service"
        exec "$TRADEX_PYTHON" -m tradex agents worker --ledger "$TRADEX_LEDGER"
        ;;
    telegram)
        exec "$TRADEX_PYTHON" -m tradex telegram run --ledger "$TRADEX_LEDGER"
        ;;
    dashboard)
        help_of dashboard | grep -q -- '--ledger' || die 78 "'tradex dashboard' is missing (pip install -e '.[dashboard]')"
        exec "$TRADEX_PYTHON" -m tradex dashboard --ledger "$TRADEX_LEDGER" --host 127.0.0.1 \
            --port "${TRADEX_DASH_PORT:-8765}" --state "$TRADEX_STATE" --reports "$TRADEX_HOME/reports"
        ;;
    *)
        die 64 "unknown service '$svc'"
        ;;
esac
