#!/bin/bash
## This script launches a local LLM server using llama-server with the specified model and configuration.
## Designed to work will with llama.cpp and the Qwen3.6-35B-A3B model.

MODELS=/home/gberseth/playground/llama.cpp/models

distrobox enter llama-vulkan-radv -- bash -c "llama-server -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --alias qwen3.6-moe -ngl 999 --no-mmap --ctx-size 256000 --host 0.0.0.0 --port 8000 --jinja --cache-type-k q8_0 --cache-type-v q8_0 -b 128 -ub 128"