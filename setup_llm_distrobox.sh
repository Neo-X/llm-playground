#!/usr/bin/env bash
# Setup Llama.cpp LLM inference via distrobox on Ubuntu (Strix Halo / AMD iGPU)
# Based on: https://github.com/kyuz0/amd-strix-halo-toolboxes
# and:      https://strix-halo-toolboxes.com/#config
set -e

MODELS_DIR="/home/gberseth/playground/llama.cpp/models"
CONTAINER_NAME="llama-vulkan-radv"

echo "=== Step 1: GPU permissions ==="
sudo usermod -aG video,render "$USER"

echo "=== Step 2: udev rules for /dev/kfd and /dev/renderD* ==="
echo -e 'SUBSYSTEM=="kfd", KERNEL=="kfd", MODE="0666"\nSUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/70-kfd.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "=== Step 3: Install tuned for performance profile ==="
sudo apt-get update -qq && sudo apt-get install -y tuned
sudo systemctl enable --now tuned
sudo tuned-adm profile accelerator-performance
echo "Active tuned profile: $(sudo tuned-adm active)"

echo "=== Step 4: Create distrobox container (Vulkan RADV — most stable) ==="
# Replace CONTAINER_NAME below with llama-rocm-7.2.1 if you want ROCm instead
# ROCm image: docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.1
distrobox create \
  --name "$CONTAINER_NAME" \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  --home "$MODELS_DIR" \
  --additional-flags "--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (or run: newgrp video) for group membership to take effect"
echo "  2. Download a model if you haven't already:"
echo "       ./download_model.sh"
echo "  3. Run the model:"
echo "       ./run_qwen3.sh"
echo ""
echo "To enter the container manually:"
echo "  distrobox enter $CONTAINER_NAME"
echo "  llama-cli --list-devices   # verify GPU is detected"
