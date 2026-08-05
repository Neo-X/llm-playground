# Fast Setup — LLM Server (Local + Remote) + OpenCode

Run a vision-capable LLM locally or on a remote GPU server and connect OpenCode to it.

---

## Prerequisites

- SSH access to a remote machine with a GPU (for remote setup)
- Local machine has `ssh` and [opencode.ai](https://opencode.ai) installed
- [distrobox](https://github.com/89luca89/distrobox) + `llama-vulkan-radv` container (for local setup)
- [`hf` CLI](https://github.com/huggingface/huggingface_hub): `uv tool install huggingface_hub`

---

## Option A — Local (AMD iGPU / Strix Halo)

### 1. Download models

**Qwen3.6-35B-A3B (MoE, default):**
```bash
HF_XET_HIGH_PERFORMANCE=1 hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mmproj-F16.gguf \
  --local-dir ~/playground/llama.cpp/models/qwen3.6-35B-A3B
```

**Qwen3.6-27B (dense, higher quality):**
```bash
HF_XET_HIGH_PERFORMANCE=1 hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-Q8_0.gguf mmproj-F16.gguf \
  --local-dir ~/playground/llama.cpp/models/qwen3.6-27b
```

Both models include `mmproj-F16.gguf` for vision/image support.

### 2. Launch the server

```bash
./launch_local_llm.sh                  # 35B-A3B (default)
./launch_local_llm.sh qwen3.6-27b     # 27B dense
```

Server listens on `http://localhost:8000` (OpenAI-compatible).

### 3. Configure OpenCode

Add to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp-local": {
      "name": "Local llama.cpp",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen3.6 35B-A3B (local)",
          "limit": { "context": 256000, "output": 8192 },
          "modalities": { "input": ["text", "image"], "output": ["text"] }
        },
        "qwen3.6-27b": {
          "name": "Qwen3.6 27B (local)",
          "limit": { "context": 256000, "output": 8192 },
          "modalities": { "input": ["text", "image"], "output": ["text"] }
        }
      }
    }
  }
}
```

---

## Option B — Remote GPU Server (llama.cpp via SSH tunnel)

### 1. Clone this repo and set up `.env`

```bash
git clone <your-repo-url>
cd llm-playground
cp .env.example .env
```

Edit `.env`:

```env
REMOTE_HOST=your-remote-server.edu
KERB_PRINCIPAL=youruser@YOUR.INSTITUTION.EDU
```

### 2. Download the model and mmproj on the remote

SSH into the remote server and run:

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mmproj-F16.gguf \
  --local-dir ~/playground/llama.cpp/models/qwen3.6-35b-a3b
```

### 3. Open the tunnel and launch the server

```bash
./connect-remote-llm.sh
```

This sets up an SSH port-forward (`localhost:8001` → `remote:8001`) and launches `llama-server` on the remote in the background. The interactive shell stays open so you can monitor output.

### 4. Configure OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp-remote": {
      "name": "Remote llama.cpp",
      "options": {
        "baseURL": "http://127.0.0.1:8001/v1"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen3.6 35B-A3B (remote)",
          "limit": { "context": 256000, "output": 8192 },
          "modalities": { "input": ["text", "image"], "output": ["text"] }
        }
      }
    }
  }
}
```

---

## Option C — Remote Ollama (simpler, slower)

### 1. Install Ollama on the remote

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.6:35b-a3b
```

### 2. Add SSH config for Ollama tunnel

```ssh-config
Host <REMOTE_HOST>-ollama
    HostName <your-remote-server-ip-or-hostname>
    User <your-username>
    LocalForward 11435 localhost:11434
```

### 3. Open the tunnel

```bash
./connect-remote.sh
```

### 4. Configure OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "name": "Ollama on remote",
      "options": {
        "baseURL": "http://127.0.0.1:11435/v1"
      },
      "models": {
        "qwen3.6:35b-a3b": {
          "name": "Qwen3.6 35B",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

---

## Complete OpenCode Config (all options)

A ready-to-use config covering all providers is at [`opencode.json`](opencode.json) in this repo. Copy it to a new machine with:

```bash
cp opencode.json ~/.config/opencode/opencode.json
```

| Provider key | Backend | Port | Launch command |
|---|---|---|---|
| `llama-cpp` | llama.cpp local | 8000 | `./launch_local_llm.sh` |
| `llama-cpp-onyx` | llama.cpp on onyx (SSH tunnel) | 8001 | `./connect-remote-llm.sh` |
| `ollama` | Ollama local | 11434 | `ollama serve` |
| `onyx` | Ollama on onyx (SSH tunnel) | 11435 | `./connect-remote.sh` |

---

## Quick reference

| Command | Purpose |
|---|---|
| `./launch_local_llm.sh` | Start local llama-server (35B-A3B, port 8000) |
| `./launch_local_llm.sh qwen3.6-27b` | Start local llama-server (27B, port 8000) |
| `./connect-remote-llm.sh` | Open SSH tunnel + launch llama-server on remote (port 8001) |
| `./connect-remote.sh` | Open SSH tunnel to remote Ollama (port 11435) |
| `ollama-remote list` | List models on remote Ollama |
| `ollama-remote pull qwen3.6:35b-a3b` | Pull a model to remote Ollama |
| `ollama-remote-stop` | Close the Ollama SSH tunnel |

> **Vision/image support:** Both local models ship with `mmproj-F16.gguf`. The launch scripts pick it up automatically when the file is present alongside the model — no extra flags needed. Add `"modalities": { "input": ["text", "image"], "output": ["text"] }` to each model entry in your OpenCode config to enable image uploads.

> **Performance:** Local Vulkan (AMD Strix Halo) ≈ remote CUDA for these model sizes. Use remote for the largest quants or multi-user scenarios.
