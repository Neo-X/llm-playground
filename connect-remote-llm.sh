#!/bin/bash
# Forward remote:8001 (llama-server) to localhost:8001
# Run this on your laptop; then point OpenCode at http://localhost:8001
# Port 8001 is used so it doesn't clash with a local llama-server on 8000.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

kinit -r 28d "$KERB_PRINCIPAL" 2>/dev/null || true

echo "Forwarding ${REMOTE_HOST}:8001 → localhost:8001 (llama-server)"
echo "Point OpenCode at: http://localhost:8001/v1"
echo ""

SSH_ARGS=()
SSH_ARGS+=(-N)
SSH_ARGS+=(-L 8001:localhost:8001)
SSH_ARGS+=("$REMOTE_HOST")

echo "Connecting to ${REMOTE_HOST}..."
ssh "${SSH_ARGS[@]}"
