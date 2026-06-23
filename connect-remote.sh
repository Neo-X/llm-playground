#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

kinit -r 28d "$KERB_PRINCIPAL"

# Forward port 11435 -> $REMOTE_HOST:11434 (Ollama) so OpenCode can use "Ollama on remote" provider
ssh -N "${REMOTE_HOST}-ollama"
