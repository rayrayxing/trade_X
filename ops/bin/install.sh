#!/bin/bash
# Copy the launchd plists to ~/Library/LaunchAgents with this Mac's paths filled in.
# It COPIES ONLY. It never runs launchctl, so nothing starts until you load a service yourself.
#
#   ops/bin/install.sh [--dry-run] [--force] [--only core,telegram] [--dest DIR] [--home DIR]
#
# Also creates the log and state folders and a starter ~/.config/tradex/ops.env (mode 600) if none exists.
# Compatible with macOS's stock bash 3.2.
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true   # bash 5.2: keep '&' in paths literal

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
home_dir="$HOME"
dest=""
dry=0
force=0
only=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1 ;;
        --force) force=1 ;;
        --only) only="${2:?--only needs a comma list}"; shift ;;
        --dest) dest="${2:?--dest needs a folder}"; shift ;;
        --home) home_dir="${2:?--home needs a folder}"; shift ;;
        -h | --help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 64 ;;
    esac
    shift
done
dest="${dest:-$home_dir/Library/LaunchAgents}"
tradex_home="${TRADEX_HOME:-$repo}"

say() { printf '%s\n' "$*"; }
do_it() { if [ "$dry" -eq 1 ]; then say "[dry run] $*"; else "$@"; fi; }

wanted() { # wanted SERVICE
    [ -z "$only" ] && return 0
    case ",$only," in *",$1,"*) return 0 ;; esac
    return 1
}

do_it mkdir -p "$dest" "$home_dir/Library/Logs/tradex" "$tradex_home/data/state" "$tradex_home/data/ledger"

copied=""
for src in "$repo"/ops/launchd/com.tradex.*.plist; do
    base="$(basename "$src")"
    svc="${base#com.tradex.}"
    svc="${svc%.plist}"
    wanted "$svc" || continue
    out="$dest/$base"
    content="$(cat "$src")"
    content="${content//__TRADEX_HOME__/$tradex_home}"
    content="${content//__HOME__/$home_dir}"
    if [ -e "$out" ] && [ "$force" -ne 1 ] && [ "$(cat "$out")" != "$content" ]; then
        say "skip   $base: a different file is already there (use --force to replace; unload it first with launchctl bootout)"
        continue
    fi
    if [ "$dry" -eq 1 ]; then
        say "[dry run] would write $out"
    else
        printf '%s\n' "$content" >"$out.tmp"
        mv "$out.tmp" "$out"
        chmod 644 "$out"
        say "copied $out"
    fi
    copied="$copied $svc"
done

env_file="$home_dir/.config/tradex/ops.env"
if [ ! -e "$env_file" ]; then
    if [ "$dry" -eq 1 ]; then
        say "[dry run] would create $env_file"
    else
        mkdir -p "$(dirname "$env_file")"
        umask 077
        cat >"$env_file" <<ENV
# Settings for the tradex ops scripts. Paths and switches only: NEVER put keys, passwords or URLs with tokens here.
TRADEX_HOME=$tradex_home
TRADEX_PYTHON=$tradex_home/.venv/bin/python
TRADEX_LEDGER=$tradex_home/data/ledger/live.sqlite
TRADEX_MODE=paper
# Backup drive (see ops/README.md): its mount point, exactly as 'mount' shows it.
TRADEX_BACKUP_VOLUME=/Volumes/TradexBackup
# Model gateway proxy (EasyCLIProxyAPI): full path to its binary, and optional arguments.
#TRADEX_PROXY_BIN=/opt/homebrew/bin/cli-proxy-api
#TRADEX_PROXY_ARGS=--config $home_dir/.config/tradex/proxy.yaml
# Services the heartbeat insists on (add agent-worker and gateway-proxy once they are loaded).
TRADEX_REQUIRED_SERVICES="core telegram dashboard"
ENV
        chmod 600 "$env_file"
        say "created $env_file (mode 600)"
    fi
else
    say "kept    $env_file (already exists)"
fi

chmod +x "$here"/*.sh "$here"/snapshot_db.py 2>/dev/null || true

cat <<MSG

Nothing has been started. launchd services are loaded only when you run these yourself (one per service):

  launchctl bootstrap gui/\$(id -u) $dest/com.tradex.<name>.plist

Suggested order, checking the log in $home_dir/Library/Logs/tradex after each:
  1. dashboard      2. gateway-proxy (needs TRADEX_PROXY_BIN)   3. telegram   4. backup   5. heartbeat
  6. core           (needs the venue adapters merged)          7. agent-worker (needs 'tradex agents worker')
To stop one:  launchctl bootout gui/\$(id -u)/com.tradex.<name>
MSG
