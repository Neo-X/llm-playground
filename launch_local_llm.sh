#!/bin/bash
## Launch a local LLM server via llama-server, using the official
## ghcr.io/ggml-org/llama.cpp Vulkan image (works on both AMD and NVIDIA GPUs
## via their respective Vulkan drivers - no vendor-specific image needed).
##
## Use tests/update_and_test_llama_image.py to pull/test new image builds; it
## promotes tested images to the "vulkan-known-good" tag used here.
##
## Usage: ./launch_local_llm.sh [model] [quant]
##
## Models:
##   qwen3.6-35b-a3b       (default) — Qwen3.6-35B-A3B MoE
##   qwen3.6-27b                     — Qwen3.6-27B dense
##   deepseek-v4-flash-q8             — DeepSeek-V4-Flash MoE, Q8_0

set -euo pipefail

MODELS=/home/gberseth/playground/llama.cpp/models
IMAGE=${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan-known-good}
CONTAINER_NAME=llama-vulkan-server
MODEL_NAME=${1:-qwen3.6-35b-a3b}

case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-35b-a3b/mmproj-F16.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS="-b 128 -ub 128"
    ;;
  qwen3.6-27b)
    QUANT=${2:-Q8_0}
    MODEL_FILE="$MODELS/qwen3.6-27b/Qwen3.6-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-27b/mmproj-F16.gguf"
    ALIAS="qwen3.6-27b"
    CTX=256000
    EXTRA_FLAGS="-b 128 -ub 128"
    ;;
  deepseek-v4-flash-q8)
    MODEL_FILE="$MODELS/DeepSeek-V4-Flash-Q8/Q8_0/DeepSeek-V4-Flash-Q8_0-00001-of-00007.gguf"
    MMPROJ=""
    ALIAS="deepseek-v4-flash-q8"
    CTX=256000
    EXTRA_FLAGS="-b 128 -ub 128"
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|qwen3.6-27b|deepseek-v4-flash-q8] [quant]"
    exit 1
    ;;
esac

MMPROJ_FLAGS=()
[[ -n "$MMPROJ" && -f "$MMPROJ" ]] && MMPROJ_FLAGS=(--mmproj "$MMPROJ")

GPU_FLAGS=(--device /dev/dri)
for group in render video; do
  gid=$(getent group "$group" | cut -d: -f3) || true
  [[ -n "${gid:-}" ]] && GPU_FLAGS+=(--group-add "$gid")
done
command -v nvidia-smi >/dev/null 2>&1 && GPU_FLAGS+=(--gpus all)

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --rm --name "$CONTAINER_NAME" \
  "${GPU_FLAGS[@]}" \
  -v "$MODELS:$MODELS" \
  -p 8000:8000 \
  "$IMAGE" \
  -m "$MODEL_FILE" --alias "$ALIAS" "${MMPROJ_FLAGS[@]}" \
  -ngl 999 --no-mmap --ctx-size "$CTX" --host 0.0.0.0 --port 8000 --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0 "$EXTRA_FLAGS"