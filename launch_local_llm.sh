#!/bin/bash
## Launch a local LLM server via llama-server, using official ggml-org
## llama.cpp images. Vulkan is used by default (works on both AMD and
## NVIDIA GPUs); models needing CUDA-only fused kernels (e.g. DeepSeek V4's
## Lightning Indexer / HC ops, which silently fall back to CPU on Vulkan and
## tank decode speed) are pinned to the CUDA image instead.
##
## Two backends, auto-selected by GPU hardware:
##   - Onyx (4x Nvidia GPUs): docker run using the official
##     ghcr.io/ggml-org/llama.cpp Vulkan or CUDA image (see above).
##     Use tests/update_and_test_llama_image.py to pull/test new image builds;
##     it promotes tested images to the "vulkan-known-good" tag used here.
##   - AMD laptop (Strix Halo iGPU, no Nvidia): the existing
##     "llama-vulkan-radv" distrobox container (see setup_llm_distrobox.sh),
##     entered directly with distrobox enter.
##
## Set BACKEND=docker or BACKEND=distrobox to override auto-detection.
##
## Usage: ./launch_local_llm.sh [model] [quant]
##
## Models:
##   qwen3.6-35b-a3b       (default) — Qwen3.6-35B-A3B MoE, Vulkan
##   qwen3.6-27b                     — Qwen3.6-27B dense, Vulkan
##   deepseek-v4-flash-q8             — DeepSeek-V4-Flash MoE, Q8_0, CUDA (needs fused ops)

set -euo pipefail

MODELS=/home/gberseth/playground/llm-playground/models
VULKAN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-vulkan-known-good
CUDA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
IMAGE=${LLAMA_IMAGE:-$VULKAN_IMAGE}
CONTAINER_NAME=llama-vulkan-server
DISTROBOX_CONTAINER=llama-vulkan-radv
MODEL_NAME=${1:-qwen3.6-35b-a3b}

case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-35B-A3B/mmproj-F16.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    ;;
  qwen3.6-27b)
    QUANT=${2:-Q8_0}
    MODEL_FILE="$MODELS/qwen3.6-27b/Qwen3.6-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-27b/mmproj-F16.gguf"
    ALIAS="qwen3.6-27b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    ;;
  deepseek-v4-flash-q8)
    MODEL_FILE="$MODELS/DeepSeek-V4-Flash-Q8/Q8_0/DeepSeek-V4-Flash-Q8_0-00001-of-00007.gguf"
    MMPROJ=""
    ALIAS="deepseek-v4-flash-q8"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    IMAGE=${LLAMA_IMAGE:-$CUDA_IMAGE}
    CONTAINER_NAME=llama-cuda-server
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|qwen3.6-27b|deepseek-v4-flash-q8] [quant]"
    exit 1
    ;;
esac

# Auto-detect backend: onyx has 4 Nvidia GPUs and uses the docker server;
# the AMD laptop (Strix Halo iGPU, no nvidia-smi) uses the old distrobox
# container instead.
if [[ -z "${BACKEND:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L | wc -l)" -eq 4 ]]; then
    BACKEND=docker
  else
    BACKEND=distrobox
  fi
fi

MMPROJ_FLAGS=()
[[ -n "$MMPROJ" && -f "$MMPROJ" ]] && MMPROJ_FLAGS=(--mmproj "$MMPROJ")

if [[ "$BACKEND" == "distrobox" ]]; then
  echo "Using distrobox container '$DISTROBOX_CONTAINER' (AMD laptop backend)"
  CMD="llama-server -m $MODEL_FILE --alias $ALIAS"
  [[ -n "$MMPROJ" && -f "$MMPROJ" ]] && CMD="$CMD --mmproj $MMPROJ"
  CMD="$CMD -ngl 999 --no-mmap --ctx-size $CTX --host 0.0.0.0 --port 8000 --jinja"
  CMD="$CMD --cache-type-k q8_0 --cache-type-v q8_0 ${EXTRA_FLAGS[*]}"
  distrobox enter "$DISTROBOX_CONTAINER" -- bash -c "$CMD"
  exit 0
fi

echo "Using docker image '$IMAGE' (onyx 4x Nvidia backend)"

GPU_FLAGS=(--device /dev/dri)
for group in render video; do
  gid=$(getent group "$group" | cut -d: -f3) || true
  [[ -n "${gid:-}" ]] && GPU_FLAGS+=(--group-add "$gid")
done
command -v nvidia-smi >/dev/null 2>&1 && GPU_FLAGS+=(--gpus all)

docker rm -f llama-vulkan-server llama-cuda-server >/dev/null 2>&1 || true

docker run --rm --name "$CONTAINER_NAME" \
  "${GPU_FLAGS[@]}" \
  -v "$MODELS:$MODELS" \
  -p 8000:8000 \
  "$IMAGE" \
  -m "$MODEL_FILE" --alias "$ALIAS" "${MMPROJ_FLAGS[@]}" \
  -ngl 999 --load-mode none --ctx-size "$CTX" --host 0.0.0.0 --port 8000 --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0 "${EXTRA_FLAGS[@]}"
