#!/bin/bash
# Forward remote ports (llama-server, Ollama) to localhost and launch
# llama-server on the remote machine via launch_local_llm.sh.
# Run this on your laptop; then point OpenCode at http://localhost:8001
# (llama-server) or the Ollama-on-remote provider (localhost:11435).
#
# Usage: ./connect-remote-llm.sh [model] [quant]
# See launch_local_llm.sh for the list of supported models/quants.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=${1:-qwen3.6-35b-a3b}
QUANT=${2:-}
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

kinit -r 28d "$KERB_PRINCIPAL" 2>/dev/null || true

# Kill any stale SSH tunnels on port 8001/11435
if pkill -f "ssh.*L 8001" 2>/dev/null || pkill -f "ssh.*8001:localhost" 2>/dev/null; then
  echo "Killing stale tunnel on localhost:8001..."
  sleep 1
fi
if pkill -f "ssh.*L 11435" 2>/dev/null || pkill -f "ssh.*11435:localhost" 2>/dev/null; then
  echo "Killing stale tunnel on localhost:11435..."
  sleep 1
fi

# Port forwarding in background (uses ~/.ssh/config aliases: onyx-llamacpp,
# onyx-ollama, which define the LocalForward ports)
echo "Setting up port forwarding: ${REMOTE_HOST}-llamacpp (llama-server -> localhost:8001)"
ssh -f -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes \
  "${REMOTE_HOST}-llamacpp"

echo "Setting up port forwarding: ${REMOTE_HOST}-ollama (Ollama -> localhost:11435)"
ssh -f -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes \
  "${REMOTE_HOST}-ollama"

# Interactive shell that launches the model in the background
echo "Model: $MODEL_NAME${QUANT:+ ($QUANT)}"
echo "Opening interactive shell on ${REMOTE_HOST}..."
ssh -t "$REMOTE_HOST" "source ~/.bashrc 2>/dev/null || true; echo Launching llama-server on remote...; bash \"$SCRIPT_DIR/launch_local_llm.sh\" $(printf '%q' "$MODEL_NAME") $(printf '%q' "$QUANT") & disown; echo 'Model launching in background'; exec bash"
