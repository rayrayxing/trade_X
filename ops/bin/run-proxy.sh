#!/bin/bash
# Start the model gateway proxy (EasyCLIProxyAPI) in the foreground. launchd runs this.
#
# Set in ~/.config/tradex/ops.env:
#   TRADEX_PROXY_BIN=/full/path/to/the/proxy/binary
#   TRADEX_PROXY_ARGS="--config /Users/ray/.config/tradex/proxy.yaml"   (optional)
# Make the proxy listen on 127.0.0.1 only. Its own upstream logins live in its config file
# (chmod 600), not in this repo. Then store proxy_base_url (http://127.0.0.1:PORT/v1) and
# proxy_api_key with `trade-x setup`.
set -euo pipefail
# shellcheck source=ops/bin/common.sh
. "$(dirname "$0")/common.sh"

[ -n "${TRADEX_PROXY_BIN:-}" ] || die 78 "TRADEX_PROXY_BIN is not set in $OPS_ENV; do not load this service yet"
[ -x "$TRADEX_PROXY_BIN" ] || die 78 "TRADEX_PROXY_BIN=$TRADEX_PROXY_BIN is not an executable file"
mkdir -p "$TRADEX_LOG_DIR"
# shellcheck disable=SC2086  # the args are a space-separated list on purpose
exec "$TRADEX_PROXY_BIN" ${TRADEX_PROXY_ARGS:-}
