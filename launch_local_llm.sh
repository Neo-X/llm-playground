#!/bin/bash
## Launch a local LLM server via llama-server inside the llama-vulkan-radv distrobox.
##
## Usage: ./launch_local_llm.sh [model] [quant]
##
## Models:
##   qwen3.6-35b-a3b  (default) — Qwen3.6-35B-A3B MoE
##   qwen3.6-27b                — Qwen3.6-27B dense

MODELS=/home/gberseth/playground/llama.cpp/models
MODEL_NAME=${1:-qwen3.6-35b-a3b}

case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-${QUANT}.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS="-b 128 -ub 128"
    ;;
  qwen3.6-27b)
    QUANT=${2:-Q8_0}
    MODEL_FILE="$MODELS/qwen3.6-27b/Qwen3.6-27B-${QUANT}.gguf"
    ALIAS="qwen3.6-27b"
    CTX=256000
    EXTRA_FLAGS="-b 128 -ub 128"
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|qwen3.6-27b] [quant]"
    exit 1
    ;;
esac

distrobox enter llama-vulkan-radv -- bash -c "llama-server -m '$MODEL_FILE' --alias '$ALIAS' -ngl 999 --no-mmap --ctx-size $CTX --host 0.0.0.0 --port 8000 --jinja --cache-type-k q8_0 --cache-type-v q8_0 $EXTRA_FLAGS"