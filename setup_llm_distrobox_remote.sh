#!/usr/bin/env bash
# Setup llama.cpp + MiniMax-M3 via distrobox on remote server (NVIDIA CUDA)
# Run this script on the remote server after SSH-ing in.
# Requires: Docker + NVIDIA Container Toolkit
#
# Usage: bash setup_llm_distrobox_remote.sh

set -e

PLAYGROUND=~/playground
LLAMA_DIR="$PLAYGROUND/llama.cpp"
MODELS_DIR="$LLAMA_DIR/models"
CONTAINER_NAME="llama-cuda"
# nvidia/cuda:12.x images work with the CUDA 13.2 driver (forward-compatible)
CUDA_IMAGE="nvidia/cuda:12.8.1-devel-ubuntu24.04"
HF_REPO="unsloth/MiniMax-M3-GGUF"
MODEL_QUANT="UD-Q4_K_XL"   # ~265 GB — fits in 4×96 GB VRAM with room for KV cache

# ── Step 1: Install distrobox ────────────────────────────────────────────────
echo "=== Step 1: Install distrobox ==="
if ! command -v distrobox &>/dev/null; then
  curl -s https://raw.githubusercontent.com/89luca89/distrobox/main/install \
    | sh -s -- --prefix ~/.local
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "distrobox: $(distrobox --version)"

# ── Step 2: Clone llama.cpp and check out MiniMax-M3 PR ─────────────────────
# PR #24523 adds preliminary M3 support (not yet merged to master as of 2026-06-19)
echo "=== Step 2: Clone llama.cpp and fetch PR #24523 (MiniMax-M3 support) ==="
mkdir -p "$PLAYGROUND"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"

# Fetch the PR branch; fall back to master if the PR ref no longer exists
if git fetch origin pull/24523/head:minimax-m3 2>/dev/null; then
  echo "Checked out PR #24523 (minimax-m3 branch)"
  git checkout minimax-m3
else
  echo "WARNING: PR #24523 ref not found — staying on current branch ($(git rev-parse --abbrev-ref HEAD))"
  echo "MiniMax-M3 support may already be in master, or the PR was rebased."
fi
mkdir -p "$MODELS_DIR"

# ── Step 3: Create CUDA distrobox container ──────────────────────────────────
echo "=== Step 3: Create distrobox container: $CONTAINER_NAME ==="
if ! distrobox list 2>/dev/null | grep -q "$CONTAINER_NAME"; then
  distrobox create \
    --name "$CONTAINER_NAME" \
    --image "$CUDA_IMAGE" \
    --home "$HOME" \
    --additional-flags "--gpus all --security-opt seccomp=unconfined"
fi

# Initialize the container on first run (sets up user env non-interactively)
distrobox enter "$CONTAINER_NAME" -- true

# ── Step 4: Build llama.cpp with CUDA inside the container ──────────────────
echo "=== Step 4: Build llama.cpp with CUDA (-DGGML_CUDA=ON) ==="

# Install build deps and compile in a single distrobox session.
# Inside distrobox, users have passwordless sudo — no docker exec needed.
distrobox enter "$CONTAINER_NAME" -- bash -c "
  set -e
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo apt-get install -y cmake ninja-build build-essential git

  export PATH='/usr/local/cuda/bin:\$PATH'
  cd '$LLAMA_DIR'
  cmake -B '$LLAMA_DIR/build' -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
  cmake --build '$LLAMA_DIR/build' --parallel \$(nproc)
  echo 'Build complete: '\$(ls '$LLAMA_DIR/build/bin/llama-server')
"

# ── Step 5: Download MiniMax-M3 GGUF ────────────────────────────────────────
echo "=== Step 5: Download $HF_REPO / $MODEL_QUANT (~265 GB) ==="
MODEL_DIR="$MODELS_DIR/minimax-m3"
mkdir -p "$MODEL_DIR"

if command -v uv &>/dev/null; then
  # uv preferred — no venv needed
  uv run --with huggingface_hub \
    huggingface-cli download "$HF_REPO" \
    --include "${MODEL_QUANT}/*.gguf" \
    --local-dir "$MODEL_DIR"
else
  pip install -q huggingface-hub
  huggingface-cli download "$HF_REPO" \
    --include "${MODEL_QUANT}/*.gguf" \
    --local-dir "$MODEL_DIR"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Model files:"
ls -lh "$MODEL_DIR/$MODEL_QUANT/"
echo ""
echo "Next steps:"
echo "  1. On remote: bash ~/playground/llm-playground/launch_local_llm_remote.sh"
echo "  2. On local:   bash ~/playground/llm-playground/connect-remote-llm.sh"
echo "  3. Start LiteLLM proxy locally:"
echo "       litellm --config ~/playground/llama.cpp/litellm-config-remote.yaml --port 4000"
