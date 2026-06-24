#!/usr/bin/env bash
# Launch llama-server inside the llama-cuda distrobox on remote.
# Run this script ON the remote server.
# Requires setup_llm_distrobox_remote.sh to have been run first.
#
# Usage: bash launch_local_llm_remote.sh [model] [quant]
#
# Models:
#   qwen3.6-35b-a3b   (default) — Qwen3.6-35B-A3B MoE, single-file GGUF
#   minimax-m3              — MiniMax-M3, sharded GGUF (~265 GB)
#
# Quant (qwen3.6-35b-a3b only, default UD-Q4_K_XL):
#   UD-Q4_K_XL  UD-Q4_K_M  UD-Q5_K_M  UD-Q6_K  Q8_0  etc.

set -e

MODELS=/home/gberseth/playground/llama.cpp/models
MODEL_NAME=${1:-qwen3.6-35b-a3b}
LLAMA_SERVER=/home/gberseth/playground/llama.cpp/build/bin/llama-server

case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-${QUANT}.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS="-b 256 -ub 256"
    ;;
  minimax-m3)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_DIR="$MODELS/minimax-m3/$QUANT"
    MODEL_FILE=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | sort | head -1)
    ALIAS="minimax-m3"
    CTX=131072
    EXTRA_FLAGS="--parallel 4"
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|minimax-m3] [quant]"
    exit 1
    ;;
esac

if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
  echo "Model file not found: $MODEL_FILE"
  exit 1
fi

# Kill any existing server on port 8000
EXISTING=$(lsof -ti :8001 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
  echo "Killing existing process on port 8001 (PID $EXISTING)..."
  kill -9 $EXISTING
  sleep 1
fi

echo "Starting llama-server: $MODEL_NAME ($QUANT)"
echo "Model: $MODEL_FILE"
echo "Backend: CUDA (4× RTX PRO 6000 Blackwell, 96 GB each)"
echo "Endpoint: http://0.0.0.0:8001 (OpenAI-compatible)"
echo ""

export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:/projects/autodata/.venv/lib/python3.11/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH:-}"
exec "$LLAMA_SERVER" \
  --model "$MODEL_FILE" \
  --alias "$ALIAS" \
  -ngl 999 \
  --ctx-size $CTX \
  --host 0.0.0.0 \
  --port 8001 \
  --jinja \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  $EXTRA_FLAGS 2>&1
