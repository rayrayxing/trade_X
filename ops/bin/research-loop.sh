#!/bin/bash
# The weekly research loop (launchd runs this on Saturdays; run it by hand to test).
#
#   research-loop.sh --dry-plan        print what the next run would do (no data, writes nothing)
#   research-loop.sh                   run the eight stages; the same ISO week resumes a cut-short run
#
# Settings in ~/.config/tradex/ops.env (all optional):
#   TRADEX_LOOP_APPLY=1                let the loop change status into and out of paper (default: recommend only)
#   TRADEX_LOOP_ARXIV_QUERY='cat:q-fin.TR AND all:momentum'   also pull ideas from arXiv search results
#   TRADEX_LOOP_DB=...                 run-state file (default data/research/loop.sqlite)
#
# Paper only: any other TRADEX_MODE is refused. The loop places no orders and touches no broker; it reads bar
# caches, the locked holdout and the paper ledger, and writes spec status lines, its own SQLite file and a
# weekly markdown report in research/results/loop/. A missing input blocks that item with an alert; it never
# falls back to anything else. Only one run at a time (a lock folder in the state directory).
set -euo pipefail
# shellcheck source=ops/bin/common.sh
. "$(dirname "$0")/common.sh"

[ "${TRADEX_MODE:-paper}" = "paper" ] || die 78 "TRADEX_MODE=${TRADEX_MODE} refused: the research loop is paper only"
[ -x "$TRADEX_PYTHON" ] || die 78 "no Python at $TRADEX_PYTHON (create the venv: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]')"

cd "$TRADEX_HOME"
mkdir -p "$TRADEX_STATE" "$TRADEX_LOG_DIR"

if [ "${1:-}" = "--dry-plan" ]; then
    exec "$TRADEX_PYTHON" -m tradex research loop --dry-plan
fi

lock="$TRADEX_STATE/research-loop.lock"
if ! mkdir "$lock" 2>/dev/null; then
    other="$(cat "$lock/pid" 2>/dev/null || true)"
    if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
        die 75 "another research loop run is in progress (pid $other)"
    fi
    log "removing a stale lock left by pid ${other:-unknown}"
    rm -rf "$lock"
    mkdir "$lock" || die 75 "could not take the lock at $lock"
fi
echo "$$" >"$lock/pid"
trap 'rm -rf "$lock"' EXIT

args=(--ledger "$TRADEX_LEDGER")
[ "${TRADEX_LOOP_APPLY:-0}" = "1" ] && args+=(--apply)
[ -n "${TRADEX_LOOP_ARXIV_QUERY:-}" ] && args+=(--arxiv-query "$TRADEX_LOOP_ARXIV_QUERY")
[ "$#" -gt 0 ] && args+=("$@")

log "research loop starting (apply=${TRADEX_LOOP_APPLY:-0})"
rc=0
"$TRADEX_PYTHON" -m tradex research loop "${args[@]}" || rc=$?
if [ "$rc" -eq 0 ]; then
    stamp research_loop_ok
    log "research loop finished"
else
    log "research loop exited with code $rc (blocked items are retried on the next run; see the weekly report)"
fi
exit "$rc"
