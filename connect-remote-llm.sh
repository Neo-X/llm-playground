#!/bin/bash
# Forward remote:8001 (llama-server) to localhost:8001 and launch llama-server
# Run this on your laptop; then point OpenCode at http://localhost:8001

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

kinit -r 28d "$KERB_PRINCIPAL" 2>/dev/null || true

# Kill any stale SSH tunnel on port 8001
if pkill -f "ssh.*L 8001" 2>/dev/null || pkill -f "ssh.*8001:localhost:8001" 2>/dev/null; then
  echo "Killing stale tunnel on localhost:8001..."
  sleep 1
fi

# Port forwarding in background
echo "Setting up port forwarding: ${REMOTE_HOST}:8001 → localhost:8001"
ssh -f -N -L 8001:localhost:8001 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes \
  "$REMOTE_HOST"

# Interactive shell that launches the model in the background
echo "Opening interactive shell on ${REMOTE_HOST}..."
ssh -t "$REMOTE_HOST" "source ~/.bashrc 2>/dev/null || true; echo Launching llama-server on remote...; bash \"$SCRIPT_DIR/launch_local_llm_remote.sh\" $@ & disown; echo 'Model launching in background'; exec bash"
