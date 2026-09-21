#!/bin/bash
## Launch a local LLM server, using either llama-server (llama.cpp) or
## vLLM's OpenAI-compatible server, both via docker.
##
## Three backends:
##   - BACKEND=docker (auto-selected on Onyx, 4x Nvidia GPUs): llama-server
##     via the official ghcr.io/ggml-org/llama.cpp CUDA image (this
##     machine's dev container is CUDA-based, so native CUDA kernels are
##     used rather than Vulkan). Use tests/update_and_test_llama_image.py to
##     pull/test new Vulkan image builds; it promotes tested images to the
##     "vulkan-known-good" tag, which LLAMA_IMAGE can still select.
##   - BACKEND=distrobox (auto-selected on the AMD laptop, Strix Halo iGPU,
##     no Nvidia): llama-server in the existing "llama-vulkan-radv"
##     distrobox container (see setup_llm_distrobox.sh), entered directly
##     with distrobox enter. Vulkan there since there's no CUDA-capable GPU.
##   - BACKEND=vllm (opt-in only, never auto-selected): vLLM's
##     OpenAI-compatible server via the official vllm/vllm-openai docker
##     image (https://docs.vllm.ai/en/stable/deployment/docker/), loading
##     the same local GGUF file as the llama.cpp backends via vLLM's
##     experimental GGUF loader (vllm-gguf-plugin). Use this when you want
##     continuous batching / PagedAttention for many concurrent users or
##     agents hitting the server at once -- llama-server serves requests
##     with much less request-level concurrency. Onyx (Nvidia) only; not
##     every model alias supports it (see VLLM_SUPPORTED below -- sharded
##     GGUFs aren't supported by vLLM's GGUF loader).
##     qwen3.6-35b-a3b/qwen3.6-27b/qwen3.8-27b need vllm-gguf-plugin's
##     qwen3_5 weight-name-mapping fix (PR #98, merged 2026-08-18), which
##     landed on GitHub main *after* the last PyPI release (0.0.5,
##     2026-08-10) -- so the image installs the plugin pinned to that PR's
##     merge commit (VLLM_GGUF_PLUGIN_REF below) instead of the PyPI wheel.
##     Bump VLLM_GGUF_PLUGIN_REF to a PyPI version once one ships past 0.0.5.
##     Verified working end-to-end (2026-09-21): qwen3.8-27b, qwen2.5-3b.
##
## Set LLAMA_IMAGE to override the docker image for the llama.cpp backends.
## Set VLLM_IMAGE to override the vLLM docker image (default vllm/vllm-openai:latest).
## Set TP to override vLLM's --tensor-parallel-size (default 1 -- GGUF+TP>1
## is undocumented/untested upstream, so this only opts in explicitly).
## Set HF_TOKEN for gated tokenizer repos (e.g. Llama) with the vllm backend.
##
## Set BACKEND=docker, BACKEND=distrobox, or BACKEND=vllm to override auto-detection.
##
## Usage: ./launch_local_llm.sh [model] [quant]
##
## Models (image is CUDA on onyx / Vulkan on the AMD laptop, per backend above,
## except deepseek-v4-flash-q8 which always forces CUDA):
##   qwen3.6-35b-a3b       (default) — Qwen3.6-35B-A3B MoE
##   qwen3.6-27b                     — Qwen3.6-27B dense
##   qwen3.8-27b                     — Qwen3.8-27B dense
##   qwen2.5-3b                      — Qwen2.5-3B-Instruct, Q4_K_M
##   qwen2.5-1.5b                    — Qwen2.5-1.5B-Instruct, Q4_K_M
##   llama3.2-3b                     — Llama-3.2-3B-Instruct, Q4_K_M
##   deepseek-v4-flash-q8             — DeepSeek-V4-Flash MoE, Q8_0, CUDA (needs fused ops)
##   qwen3.8-flash-next               — Qwen3.8-Flash-Next MoE, UD-Q4_K_XL (4 shards, ~111G)

set -euo pipefail

MODELS=/home/gberseth/playground/llm-playground/models
VULKAN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-vulkan-known-good
CUDA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:latest}
# Pinned to the vllm-gguf-plugin commit that merged qwen3_5 GGUF support
# (PR #98) -- not yet on a PyPI release (latest is 0.0.5). The image tag
# bakes in the ref so bumping it below forces a rebuild automatically.
VLLM_GGUF_PLUGIN_REF=${VLLM_GGUF_PLUGIN_REF:-4ec8d61565cb21a380a7532bb20675883c6734d9}
VLLM_GGUF_IMAGE="vllm-openai-gguf:${VLLM_GGUF_PLUGIN_REF:0:12}"
DISTROBOX_CONTAINER=llama-vulkan-radv
MODEL_NAME=${1:-qwen3.6-35b-a3b}

# Auto-detect backend: onyx has 4 Nvidia GPUs and uses the docker server
# (CUDA image, this machine's dev container); the AMD laptop (Strix Halo
# iGPU, no nvidia-smi) uses the old distrobox container (Vulkan) instead.
# BACKEND=vllm is never auto-selected -- opt in explicitly.
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
elif [[ "$BACKEND" == "vllm" ]]; then
  CONTAINER_NAME=vllm-server
else
  IMAGE=${LLAMA_IMAGE:-$VULKAN_IMAGE}
  CONTAINER_NAME=llama-vulkan-server
fi

VLLM_SUPPORTED=0
# Every model here fits comfortably on a single 96GB GPU except the two
# sharded MoE models below (~100GB+), so default to pinning the docker
# backend to one GPU (GPU_DEVICE) rather than exposing --gpus all. Exposing
# all 4 GPUs to a model that fits on one made llama.cpp auto-split its
# layers across all of them by default -- e.g. qwen2.5-1.5b (<2GB) was
# observed spread ~1-1.5GB across all 4 GPUs, and the resulting cross-GPU
# hop on every layer boundary tanked decode throughput (measured ~460 tok/s
# for a 1.5B model, well below what a single Blackwell GPU should give it).
MULTI_GPU=0
# Quantized KV cache (q8_0) trades decode speed for VRAM -- worth it for the
# 256k+-context models below where the cache would otherwise be huge, but
# for the short-context (32768) dense models it saves VRAM nobody needs (a
# 96GB GPU vs. a model using single-digit GB) at a real decode cost:
# measured ~12% slower decode (472 -> 528.6 tok/s on qwen2.5-1.5b) with it
# forced on vs. off. So it defaults on, and the short-context models below
# turn it off.
KV_QUANT=1
case "$MODEL_NAME" in
  qwen3.6-35b-a3b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-35B-A3B/mmproj-F16.gguf"
    ALIAS="qwen3.6-35b-a3b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
    TOKENIZER_REPO="Qwen/Qwen3.6-35B-A3B"
    TOOL_PARSER="hermes"
    VLLM_SUPPORTED=1
    ;;
  qwen3.6-27b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.6-27b/Qwen3.6-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.6-27b/mmproj-F16.gguf"
    ALIAS="qwen3.6-27b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.6-27B-GGUF"
    TOKENIZER_REPO="Qwen/Qwen3.6-27B"
    TOOL_PARSER="hermes"
    VLLM_SUPPORTED=1
    ;;
  qwen3.8-27b)
    QUANT=${2:-UD-Q4_K_XL}
    MODEL_FILE="$MODELS/qwen3.8-27b/Qwen3.8-27B-${QUANT}.gguf"
    MMPROJ="$MODELS/qwen3.8-27b/mmproj-F16.gguf"
    ALIAS="qwen3.8-27b"
    CTX=256000
    EXTRA_FLAGS=(-b 128 -ub 128)
    HF_REPO="unsloth/Qwen3.8-27B-GGUF"
    TOKENIZER_REPO="Qwen/Qwen3.8-27B"
    TOOL_PARSER="hermes"
    VLLM_SUPPORTED=1
    ;;
  qwen2.5-3b)
    MODEL_FILE="$MODELS/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"
    MMPROJ=""
    ALIAS="qwen2.5-3b"
    CTX=32768
    EXTRA_FLAGS=(-b 512 -ub 512)
    HF_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
    TOKENIZER_REPO="Qwen/Qwen2.5-3B-Instruct"
    TOOL_PARSER="hermes"
    VLLM_SUPPORTED=1
    KV_QUANT=0
    ;;
  qwen2.5-1.5b)
    MODEL_FILE="$MODELS/qwen2.5-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    MMPROJ=""
    ALIAS="qwen2.5-1.5b"
    CTX=32768
    EXTRA_FLAGS=(-b 512 -ub 512)
    HF_REPO="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    TOKENIZER_REPO="Qwen/Qwen2.5-1.5B-Instruct"
    TOOL_PARSER="hermes"
    VLLM_SUPPORTED=1
    KV_QUANT=0
    ;;
  llama3.2-3b)
    MODEL_FILE="$MODELS/llama3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    MMPROJ=""
    ALIAS="llama3.2-3b"
    CTX=32768
    EXTRA_FLAGS=(-b 512 -ub 512)
    HF_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
    TOKENIZER_REPO="meta-llama/Llama-3.2-3B-Instruct"
    TOOL_PARSER="llama3_json"
    VLLM_SUPPORTED=1
    KV_QUANT=0
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
    MULTI_GPU=1  # too large for a single 96GB GPU
    ;;
  qwen3.8-flash-next)
    QUANT="UD-Q4_K_XL"
    MODEL_DIR="$MODELS/qwen3.8-flash-next"
    MODEL_FILE="$MODEL_DIR/$QUANT/Qwen3.8-Flash-Next-${QUANT}-00001-of-00004.gguf"
    MMPROJ="$MODEL_DIR/mmproj-F16.gguf"
    ALIAS="qwen3.8-flash-next"
    CTX=512000
    # Native training context is 262144 -- CTX exceeds that, so YaRN rope
    # scaling is needed to extend it correctly (per the model card).
    EXTRA_FLAGS=(-b 128 -ub 128 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144)
    HF_REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
    HF_INCLUDE="$QUANT/*"
    MULTI_GPU=1  # ~111GB, too large for a single 96GB GPU
    ;;
  *)
    echo "Unknown model: $MODEL_NAME"
    echo "Usage: $0 [qwen3.6-35b-a3b|qwen3.6-27b|qwen3.8-27b|qwen2.5-3b|qwen2.5-1.5b|llama3.2-3b|deepseek-v4-flash-q8|qwen3.8-flash-next] [quant]"
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

maybe_download_sharded() {
  local check_file="$1" include_pattern="$2" local_dir="$3"
  [[ -f "$check_file" ]] && return 0

  if [[ ! -t 0 ]]; then
    echo "Model shards not found: $check_file (repo: $HF_REPO, pattern: $include_pattern). Not prompting -- stdin isn't a terminal." >&2
    exit 1
  fi

  local answer
  read -r -p "Model shards not found: $check_file. Download '$include_pattern' from $HF_REPO now? [y/N] " answer || answer="n"
  if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Not downloading. Exiting." >&2
    exit 1
  fi

  echo "Downloading '$include_pattern' from $HF_REPO..."
  HF_XET_HIGH_PERFORMANCE=1 hf download "$HF_REPO" --include "$include_pattern" --local-dir "$local_dir"
}

if [[ -n "${HF_INCLUDE:-}" ]]; then
  maybe_download_sharded "$MODEL_FILE" "$HF_INCLUDE" "$MODEL_DIR"
else
  maybe_download "$MODEL_FILE"
fi
[[ -n "$MMPROJ" ]] && maybe_download "$MMPROJ"

MMPROJ_FLAGS=()
[[ -n "$MMPROJ" && -f "$MMPROJ" ]] && MMPROJ_FLAGS=(--mmproj "$MMPROJ" --image-min-tokens 1024)

if [[ "$BACKEND" == "distrobox" ]]; then
  echo "Using distrobox container '$DISTROBOX_CONTAINER' (AMD laptop backend)"
  CMD="llama-server -m $MODEL_FILE --alias $ALIAS"
  [[ -n "$MMPROJ" && -f "$MMPROJ" ]] && CMD="$CMD --mmproj $MMPROJ --image-min-tokens 1024"
  CMD="$CMD -ngl 999 --no-mmap --ctx-size $CTX --host 0.0.0.0 --port 8000 --jinja"
  [[ "$KV_QUANT" -eq 1 ]] && CMD="$CMD --cache-type-k q8_0 --cache-type-v q8_0"
  CMD="$CMD ${EXTRA_FLAGS[*]}"
  distrobox enter "$DISTROBOX_CONTAINER" -- bash -c "$CMD"
  exit 0
fi

if [[ "$BACKEND" == "vllm" ]]; then
  if [[ "$VLLM_SUPPORTED" -ne 1 ]]; then
    echo "Model '$MODEL_NAME' isn't supported on BACKEND=vllm -- it's a sharded/multi-file GGUF, and vLLM's GGUF loader only supports single-file checkpoints." >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BACKEND=vllm requires an Nvidia GPU (vLLM's CUDA image) -- none detected." >&2
    exit 1
  fi

  # vLLM's official image doesn't ship GGUF support -- build a thin local
  # layer adding vllm-gguf-plugin on top of it (see
  # https://docs.vllm.ai/en/stable/features/quantization/gguf.html), cached
  # by image tag so this only runs once per (VLLM_IMAGE, VLLM_GGUF_PLUGIN_REF)
  # pair. Installed from git pinned to VLLM_GGUF_PLUGIN_REF (not the PyPI
  # wheel, which is stuck at 0.0.5 -- see the header comment) so qwen3.6/3.8
  # GGUFs work; --no-build-isolation builds its CUDA bits against the image's
  # already-installed torch instead of pip fetching a fresh one.
  if ! docker image inspect "$VLLM_GGUF_IMAGE" >/dev/null 2>&1; then
    echo "Building '$VLLM_GGUF_IMAGE' (one-time: $VLLM_IMAGE + vllm-gguf-plugin@${VLLM_GGUF_PLUGIN_REF:0:12})..."
    printf '%s\n' \
      "FROM $VLLM_IMAGE" \
      "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*" \
      "RUN pip install --no-cache-dir --no-build-isolation \"vllm-gguf-plugin @ git+https://github.com/vllm-project/vllm-gguf-plugin.git@${VLLM_GGUF_PLUGIN_REF}\"" \
      | docker build -t "$VLLM_GGUF_IMAGE" -
  fi

  TP=${TP:-1}
  HF_TOKEN_FLAGS=()
  [[ -n "${HF_TOKEN:-}" ]] && HF_TOKEN_FLAGS=(--env "HF_TOKEN=$HF_TOKEN")

  echo "Using docker image '$VLLM_GGUF_IMAGE' as container '$CONTAINER_NAME' (onyx Nvidia backend)"
  echo "Model file: $MODEL_FILE | tokenizer: $TOKENIZER_REPO | tensor-parallel-size: $TP | max-model-len: $CTX"

  docker rm -f vllm-server llama-vulkan-server llama-cuda-server >/dev/null 2>&1 || true

  set +e
  docker run --rm --name "$CONTAINER_NAME" \
    --runtime nvidia --gpus all --ipc=host \
    -v "$MODELS:$MODELS" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${HF_TOKEN_FLAGS[@]}" \
    -p 8020:8000 \
    "$VLLM_GGUF_IMAGE" \
    --model "$MODEL_FILE" --tokenizer "$TOKENIZER_REPO" --served-model-name "$ALIAS" \
    --tensor-parallel-size "$TP" --max-model-len "$CTX" --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "vLLM container '$CONTAINER_NAME' exited with code $status -- see output above for the crash reason." >&2
  else
    echo "vLLM container '$CONTAINER_NAME' exited normally (code 0)."
  fi
  exit "$status"
fi

echo "Using docker image '$IMAGE' as container '$CONTAINER_NAME' (onyx 4x Nvidia backend)"

GPU_FLAGS=(--device /dev/dri)
for group in render video; do
  gid=$(getent group "$group" | cut -d: -f3) || true
  [[ -n "${gid:-}" ]] && GPU_FLAGS+=(--group-add "$gid")
done
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$MULTI_GPU" -eq 1 ]]; then
    GPU_FLAGS+=(--gpus all)
  else
    # Pin to a single GPU (default 0, override with GPU_DEVICE) so
    # llama.cpp doesn't auto-split a model that fits on one GPU across all
    # of them -- see the MULTI_GPU comment above the model case statement.
    GPU_FLAGS+=(--gpus "device=${GPU_DEVICE:-0}")
  fi
fi

docker rm -f llama-vulkan-server llama-cuda-server vllm-server >/dev/null 2>&1 || true

KV_CACHE_FLAGS=()
[[ "$KV_QUANT" -eq 1 ]] && KV_CACHE_FLAGS=(--cache-type-k q8_0 --cache-type-v q8_0)

# Runs attached (no -d), so all llama-server output stays in this terminal and
# the shell blocks here until the container exits. If it exits early (crash,
# OOM, etc.) that's easy to miss as just "the prompt came back" -- make it loud.
set +e
docker run --rm --name "$CONTAINER_NAME" \
  "${GPU_FLAGS[@]}" \
  -v "$MODELS:$MODELS" \
  -p 8010:8010 \
  "$IMAGE" \
  -m "$MODEL_FILE" --alias "$ALIAS" "${MMPROJ_FLAGS[@]}" \
  -ngl 999 --load-mode none --ctx-size "$CTX" --host 0.0.0.0 --port 8010 --jinja \
  "${KV_CACHE_FLAGS[@]}" "${EXTRA_FLAGS[@]}"
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "llama-server container '$CONTAINER_NAME' exited with code $status -- see output above for the crash reason." >&2
else
  echo "llama-server container '$CONTAINER_NAME' exited normally (code 0)."
fi
exit "$status"
