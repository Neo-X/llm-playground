#!/bin/bash
## Launch a local LLM server via llama-server, using official ggml-org
## llama.cpp images.
##
## Two backends, auto-selected by GPU hardware:
##   - Onyx (4x Nvidia GPUs): docker run using the official
##     ghcr.io/ggml-org/llama.cpp CUDA image by default (this machine's dev
##     container is CUDA-based, so native CUDA kernels are used rather than
##     Vulkan). Use tests/update_and_test_llama_image.py to pull/test new
##     Vulkan image builds; it promotes tested images to the
##     "vulkan-known-good" tag, which LLAMA_IMAGE can still select.
##   - AMD laptop (Strix Halo iGPU, no Nvidia): the existing
##     "llama-vulkan-radv" distrobox container (see setup_llm_distrobox.sh),
##     entered directly with distrobox enter. Vulkan there since there's no
##     CUDA-capable GPU.
##
## Set LLAMA_IMAGE to override the docker image on either backend.
##
## Set BACKEND=docker or BACKEND=distrobox to override auto-detection.
##
## Usage: ./launch_local_llm.sh [model] [quant]
##
## Models (image is CUDA on onyx / Vulkan on the AMD laptop, per backend above,
## except deepseek-v4-flash-q8 which always forces CUDA):
##   qwen3.6-35b-a3b       (default) — Qwen3.6-35B-A3B MoE
##   qwen3.6-27b                     — Qwen3.6-27B dense
##   qwen3.8-27b                     — Qwen3.8-27B dense
##   qwen2.5-3b                      — Qwen2.5-3B-Instruct, Q4_K_M
##   llama3.2-3b                     — Llama-3.2-3B-Instruct, Q4_K_M
##   deepseek-v4-flash-q8             — DeepSeek-V4-Flash MoE, Q8_0, CUDA (needs fused ops)

set -euo pipefail

MODELS=/home/gberseth/playground/llm-playground/models
VULKAN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-vulkan-known-good
CUDA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
DISTROBOX_CONTAINER=llama-vulkan-radv
MODEL_NAME=${1:-qwen3.6-35b-a3b}

# Auto-detect backend: onyx has 4 Nvidia GPUs and uses the docker server
# (CUDA image, this machine's dev container); the AMD laptop (Strix Halo
# iGPU, no nvidia-smi) uses the old distrobox container (Vulkan) instead.
if [[ -z "${BACKEND:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L | wc -l)" -eq 4 ]]; then
    BACKEND=docker
  else
    BACKEND=distrobox
  fi
fi

if [[ "$BACKEND" == "docker" ]]; then
  IMAGE=${LLAMA_IMAGE:-$CUDA_IMAGE}
  CONTAINER_NAME=llama-cuda-server
else
  IMAGE=${LLAMA_IMAGE:-$VULKAN_IMAGE}
  CONTAINER_NAME=llama-vulkan-server
fi

case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-35B-A3B/mmproj-F16.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
    ;;
  qwen3.6-27b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-27b/Qwen3.6-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-27b/mmproj-F16.gguf"
    ALIAS="qwen3.6-27b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.6-27B-GGUF"
    ;;
  qwen3.8-27b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.8-27b/Qwen3.8-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.8-27b/mmproj-F16.gguf"
    ALIAS="qwen3.8-27b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.8-27B-GGUF"
    ;;
  qwen2.5-3b)
    MODEL_FILE="$MODELS/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"
    MMPROJ=""
    ALIAS="qwen2.5-3b"
    CTX=32768
    EXTRA_FLAGS=(-b 512 -ub 512)
    HF_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
    ;;
  llama3.2-3b)
    MODEL_FILE="$MODELS/llama3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    MMPROJ=""
    ALIAS="llama3.2-3b"
    CTX=32768
    EXTRA_FLAGS=(-b 512 -ub 512)
    HF_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
    ;;
  deepseek-v4-flash-q8)
    MODEL_FILE="$MODELS/DeepSeek-V4-Flash-Q8/Q8_0/DeepSeek-V4-Flash-Q8_0-00001-of-00007.gguf"
    MMPROJ=""
    ALIAS="deepseek-v4-flash-q8"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    IMAGE=${LLAMA_IMAGE:-$CUDA_IMAGE}
    # No HF_REPO: this file is one of 7 shards, too large/complex to
    # single-file auto-download -- run `hf download` for it manually.
    HF_REPO=""
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|qwen3.6-27b|qwen3.8-27b|qwen2.5-3b|llama3.2-3b|deepseek-v4-flash-q8] [quant]"
    exit 1
    ;;
esac

maybe_download() {
  local file_path="$1"
  [[ -f "$file_path" ]] && return 0

  if [[ -z "$HF_REPO" ]]; then
    echo "Model file not found and no known HuggingFace repo configured to fetch it from: $file_path" >&2
    exit 1
  fi
  if [[ ! -t 0 ]]; then
    echo "Model file not found: $file_path (repo: $HF_REPO). Not prompting -- stdin isn't a terminal." >&2
    exit 1
  fi

  local answer
  read -r -p "Model file not found: $file_path. Download $(basename "$file_path") from $HF_REPO now? [y/N] " answer || answer="n"
  if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Not downloading. Exiting." >&2
    exit 1
  fi

  echo "Downloading $(basename "$file_path") from $HF_REPO..."
  HF_XET_HIGH_PERFORMANCE=1 hf download "$HF_REPO" "$(basename "$file_path")" --local-dir "$(dirname "$file_path")"
}

maybe_download "$MODEL_FILE"
[[ -n "$MMPROJ" ]] && maybe_download "$MMPROJ"

MMPROJ_FLAGS=()
[[ -n "$MMPROJ" && -f "$MMPROJ" ]] && MMPROJ_FLAGS=(--mmproj "$MMPROJ" --image-min-tokens 1024)

if [[ "$BACKEND" == "distrobox" ]]; then
  echo "Using distrobox container '$DISTROBOX_CONTAINER' (AMD laptop backend)"
  CMD="llama-server -m $MODEL_FILE --alias $ALIAS"
  [[ -n "$MMPROJ" && -f "$MMPROJ" ]] && CMD="$CMD --mmproj $MMPROJ --image-min-tokens 1024"
  CMD="$CMD -ngl 999 --no-mmap --ctx-size $CTX --host 0.0.0.0 --port 8000 --jinja"
  CMD="$CMD --cache-type-k q8_0 --cache-type-v q8_0 ${EXTRA_FLAGS[*]}"
  distrobox enter "$DISTROBOX_CONTAINER" -- bash -c "$CMD"
  exit 0
fi

echo "Using docker image '$IMAGE' as container '$CONTAINER_NAME' (onyx 4x Nvidia backend)"

GPU_FLAGS=(--device /dev/dri)
for group in render video; do
  gid=$(getent group "$group" | cut -d: -f3) || true
  [[ -n "${gid:-}" ]] && GPU_FLAGS+=(--group-add "$gid")
done
command -v nvidia-smi >/dev/null 2>&1 && GPU_FLAGS+=(--gpus all)

docker rm -f llama-vulkan-server llama-cuda-server >/dev/null 2>&1 || true

# Runs attached (no -d), so all llama-server output stays in this terminal and
# the shell blocks here until the container exits. If it exits early (crash,
# OOM, etc.) that's easy to miss as just "the prompt came back" -- make it loud.
set +e
docker run --rm --name "$CONTAINER_NAME" \
  "${GPU_FLAGS[@]}" \
  -v "$MODELS:$MODELS" \
  -p 8000:8000 \
  "$IMAGE" \
  -m "$MODEL_FILE" --alias "$ALIAS" "${MMPROJ_FLAGS[@]}" \
  -ngl 999 --load-mode none --ctx-size "$CTX" --host 0.0.0.0 --port 8000 --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0 "${EXTRA_FLAGS[@]}"
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "llama-server container '$CONTAINER_NAME' exited with code $status -- see output above for the crash reason." >&2
else
  echo "llama-server container '$CONTAINER_NAME' exited normally (code 0)."
fi
exit "$status"
