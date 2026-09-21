#!/bin/bash
# Forward remote ports (llama-server, Ollama, vLLM) to localhost and launch
# the model on the remote machine via launch_local_llm.sh.
# Run this on your laptop; then point OpenCode at http://localhost:8001
# (llama-server), the Ollama-on-remote provider (localhost:11435), or
# http://localhost:8020 (vLLM, when BACKEND=vllm).
#
# Usage: ./connect-remote-llm.sh [model] [quant]
#        BACKEND=vllm ./connect-remote-llm.sh qwen2.5-3b
# See launch_local_llm.sh for the list of supported models/quants, and for
# which models/quants support BACKEND=vllm.
#
# BACKEND=vllm requires port 8020 forwarded too. No separate ssh alias
# needed -- just add a second "LocalForward 8020 localhost:8020" line to
# your existing "${REMOTE_HOST}-llamacpp" ~/.ssh/config entry (one ssh
# connection can carry multiple LocalForwards).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=${1:-qwen3.8-27b}
QUANT=${2:-}
BACKEND=${BACKEND:-}
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

kinit -r 28d "$KERB_PRINCIPAL" 2>/dev/null || true

# Kill any stale SSH tunnels on port 8001/11435. The forwarding is defined in
# ~/.ssh/config (LocalForward), not on the command line, so match on the
# ssh config alias instead of a -L/port pattern.
if pkill -f "ssh.*onyx-llamacpp" 2>/dev/null; then
  echo "Killing stale tunnel on localhost:8001..."
  sleep 1
fi
if pkill -f "ssh.*onyx-ollama" 2>/dev/null; then
  echo "Killing stale tunnel on localhost:11435..."
  sleep 1
fi

# Port forwarding in background (uses ~/.ssh/config aliases: onyx-llamacpp,
# onyx-ollama, which define the LocalForward ports -- port 8020 for vLLM
# rides the onyx-llamacpp connection, see the header comment)
echo "Setting up port forwarding: ${REMOTE_HOST}-llamacpp (llama-server -> localhost:8001, vLLM -> localhost:8020)"
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
echo "Model: $MODEL_NAME${QUANT:+ ($QUANT)}${BACKEND:+ [BACKEND=$BACKEND]}"
echo "Opening interactive shell on ${REMOTE_HOST}..."
ssh -t "$REMOTE_HOST" "source ~/.bashrc 2>/dev/null || true; echo Launching model on remote...; $(printf '%q=%q ' BACKEND "$BACKEND")bash \"$SCRIPT_DIR/launch_local_llm.sh\" $(printf '%q' "$MODEL_NAME") $(printf '%q' "$QUANT") & disown; echo 'Model launching in background'; exec bash"
