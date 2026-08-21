# LLM Inference Speed Benchmark (Prompt vs Generation)

This project benchmarks:
- **Prompt (prefill) token speed** in tokens/sec
- **Generation (decode) token speed** in tokens/sec

Default model set (see `sweep_models.py`):
- `qwen2.5:3b`
- `llama3.2:3b`
- `qwen3.6:27b`
- `qwen3.6:35b-a3b`
- `gpt-oss:20b`

## 0) Quick start

New to the project? See [`FAST_SETUP.md`](./FAST_SETUP.md) for a 5-minute guide to get Ollama on a remote server and connect OpenCode to it.

## 1) Configure `.env`

Copy the template and fill in your values:

```bash
cp .env.example .env   # or create from scratch
```

`.env` contents:

```env
REMOTE_HOST=your-remote-server.edu
KERB_PRINCIPAL=youruser@YOUR.INSTITUTION.EDU
```

- **`REMOTE_HOST`** — hostname of the remote GPU server (used in SSH tunnels)
- **`KERB_PRINCIPAL`** — your Kerberos principal for `kinit` (e.g. `jdoe@MIT.EDU`)

> Note: `.env` contains credentials — it is ignored by git. Never commit it.

## 2) Create and activate environment (uv)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

For reproducible setup on another computer (using lockfile):

```bash
uv sync
```

This uses `pyproject.toml` + `uv.lock` to recreate the same environment.

## 3) Benchmark a set of models (Ollama + llama.cpp)

One command benchmarks a list of models — pulling/unloading each Ollama model
and, for models with a local GGUF (see `LLAMACPP_ALIASES` in `sweep_models.py`),
launching `llama-server` via `launch_local_llm.sh` — then writes a combined CSV
and a grouped bar-chart PNG comparing prefill/decode tok/s by backend:

```bash
uv run python sweep_models.py \
  --models qwen2.5:3b llama3.2:3b qwen3.6:27b qwen3.6:35b-a3b gpt-oss:20b \
  --runs 3 \
  --out-csv logs/model_sweep.csv \
  --out-png logs/model_sweep.png
```

Models without a known llama.cpp alias are benchmarked on Ollama only (a note
is printed, the run isn't interrupted). Add new GGUF-backed models by adding
them to `LLAMACPP_ALIASES` and to `launch_local_llm.sh`.

For one-off, single-model, single-backend runs use `benchmark_llm_speed.py`
directly — it supports `--backend transformers|ollama|llamacpp`:

```bash
# Transformers (HF) backend
python benchmark_llm_speed.py --backend transformers --model Qwen/Qwen2.5-3B-Instruct --device auto

# Ollama backend
python benchmark_llm_speed.py --backend ollama --model qwen2.5:3b --ollama-pull

# llama.cpp backend (expects launch_local_llm.sh already running)
python benchmark_llm_speed.py --backend llamacpp --llamacpp-host http://localhost:8000 --model qwen3.6-27b
```

## 4) Logs

Per-run metrics from `benchmark_llm_speed.py` are appended to:
- `logs/benchmark_metrics.csv`
- `logs/benchmark_metrics.jsonl`

Columns include `backend`, `prompt_tokens`, `generated_tokens`,
`prefill_time_s`, `decode_time_s`, `prefill_tps`, `decode_tps`.

`sweep_models.py` and `rank_ollama_models.py` write averaged, ranked results
to `logs/model_sweep.csv` / `logs/ollama_model_rankings.csv` instead
(columns: `backend`, `model`, `avg_prefill_tps`, `avg_decode_tps`, ...).

## Notes

- First run is usually slower due to model loading and kernel warmup.
- If CUDA or ROCm is unavailable, benchmark runs on CPU automatically when `--device auto` is used.
- AMD GPU benchmarking with Transformers requires a ROCm-enabled PyTorch install. AMD GPU benchmarking with Ollama requires an Ollama build with ROCm support.
- For strict repeatability, run with deterministic prompts and disable sampling (default).

---

## 5) llama.cpp benchmarking (Strix Halo / AMD iGPU)

For llama.cpp models, use **`llama-cpp-bencher.py`** from
[lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing/tree/main/llm-bench).
It wraps `llama-bench` (the built-in llama.cpp binary) and sweeps multiple backends/token
counts automatically, producing `results.jsonl`, summary tables, and plots.

### Setup

```bash
# Download the bencher script
curl -O https://raw.githubusercontent.com/lhl/strix-halo-testing/main/llm-bench/llama-cpp-bencher.py

# Create the directory structure the script expects:
#   <build-root>/llama.cpp-<name>/build/bin/llama-bench
ln -s ~/playground/llama.cpp ~/playground/llama.cpp-vulkan-radv
```

The distrobox container (`llama-vulkan-radv`) ships `llama-bench` alongside `llama-server`,
so no separate build is needed.

### Downloading models

Use the `hf` CLI from this environment. Do **not** use `huggingface-cli` — it is not installed.

```bash
# General pattern (run from this directory so uv picks up the correct venv)
cd /home/gberseth/playground/llm-playground

uv run hf download <repo_id> \
  --include "<filename>.gguf" \
  --local-dir /home/gberseth/playground/llama.cpp/models/<model-dir>
```

Example — Qwen3.6 35B-A3B MoE (UD-Q4_K_XL):

```bash
mkdir -p /home/gberseth/playground/llama.cpp/models/qwen3.6-35B-A3B

cd /home/gberseth/playground/llm-playground && uv run hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" --local-dir /home/gberseth/playground/llama.cpp/models/qwen3.6-35B-A3B
```

### Run a benchmark

```bash
MODELS=/home/gberseth/playground/llama.cpp/models
BENCH=/home/gberseth/playground/llm-playground

distrobox enter llama-vulkan-radv -- bash -c "cd $BENCH && uv run python llama-cpp-bencher.py  --port 8000 \
  --jinja \
  --ctx-size 64000 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --moe \
  --build-root /home/gberseth/playground \
  -m $MODELS/qwen3-coder-30B-A3B/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf"
```

Note: line continuations (`\`) do not work inside `bash -c "..."` — keep the command on one line.

`--moe` is required for Qwen3-30B-A3B — it enables `-b 256` batching, which is what
produces the ~78 t/s decode result on this hardware.

### Launch server for Claude Code

```bash
MODELS=/home/gberseth/playground/llama.cpp/models

distrobox enter llama-vulkan-radv -- bash -c "llama-server -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --alias qwen3.6-moe -ngl 999 --no-mmap --ctx-size 65536 --host 0.0.0.0 --port 8000 --jinja --cache-type-k q8_0 --cache-type-v q8_0 -b 128 -ub 128"
```

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="sk-no-key-required"
export ANTHROPIC_MODEL="private-model"

claude --model private-model[100k]
```

### Key flags

| Flag | Description |
|---|---|
| `--moe` | Enable `-b 256` batching for MoE models (Qwen3, etc.) |
| `--build-root` | Directory containing `llama.cpp-*/build/bin/llama-bench` |
| `-p` | Prompt token counts to sweep (default: powers of 2 up to 4096) |
| `-n` | Generation token counts to sweep |
| `--rerun` | Force re-run even if results already exist |
| `--resummarize` | Regenerate README/plots from existing `results.jsonl` without re-running |

### Output

Results are written to a directory named after the model stem:
- `results.jsonl` — raw timing data per run
- `README.md` — summary table of pp/tg t/s across backends
- `pp_tokens_per_sec.png`, `tg_tokens_per_sec.png` — performance curves
- `system_info.json` — hardware/driver snapshot

---

## 6) Sweep llama-server settings for prefill speed

`bench_server_settings.py` starts llama-server with different `-b`/`-ub` batch
sizes, sends fixed prompts of various lengths, and records prefill tok/s from
the server's own timing data. Use this to find the optimal batch size for your
model before committing it to the main server launch command.

### Run

```bash
MODELS=/home/gberseth/playground/llama.cpp/models

uv run python bench_server_settings.py \
  -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --container llama-vulkan-radv \
  --port 8001
```

Also test flash attention (`-fa 1`) alongside each batch size:

```bash
uv run python bench_server_settings.py \
  -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --container llama-vulkan-radv \
  --port 8001 \
  --flash-attn
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--batch-sizes` | `128 256 512 1024 2048` | `-b`/`-ub` values to sweep |
| `--prompt-sizes` | `512 1024 2048 4096 8192 16384` | Prompt token counts to test (sized for real coding sessions: system prompt + context + user message) |
| `--ctx-size` | `32768` | Server KV context (must exceed largest prompt size) |
| `--flash-attn` | off | Also test each batch size with `-fa 1` |
| `--runs` | `2` | Timed runs per (config, prompt size) |
| `--port` | `8001` | Use a different port than the main server (8000) |
| `--resummarize` | — | Regenerate plots from existing `results.jsonl` |

### Output

Results are written to `<model-stem>-server-settings/`:
- `results.jsonl` / `results.csv` — raw records per run
- `pp_tps.png` — prefill tok/s vs prompt length, one line per config
- `tg_tps.png` — decode tok/s vs prompt length
- `README.md` — summary table with best value per column in **bold**

---

## 7) Using llama.cpp with OpenCode (opencode.ai)

[OpenCode](https://opencode.ai/) is an open-source terminal/IDE coding agent that supports any OpenAI-compatible endpoint via `opencode.json`.

### Start the llama-server

Same command as above — llama-server's `/v1` endpoint is already OpenAI-compatible:

```bash
MODELS=/home/gberseth/playground/llama.cpp/models

distrobox enter llama-vulkan-radv -- bash -c "llama-server -m $MODELS/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --alias qwen3.6-moe -ngl 999 --no-mmap --ctx-size 65536 --host 0.0.0.0 --port 8000 --jinja --cache-type-k q8_0 --cache-type-v q8_0 -b 128 -ub 128"
```

### Configure opencode.json

Place an `opencode.json` in your project directory (or `~/.config/opencode/opencode.json` for a global config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local)",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3.6-moe": {
          "name": "Qwen3.6 35B-A3B MoE",
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    }
  }
}
```

The `model` key (`qwen3.6-moe`) must match the `--alias` you passed to `llama-server`.

### Run OpenCode

```bash
opencode
```

Select **llama.cpp (local) › Qwen3.6 35B-A3B MoE** from the model picker.

---

### Using OpenCode with Ollama on remote (SSH tunnel)

Ollama runs on the remote server but is not reachable directly. The `<REMOTE_HOST>-ollama` SSH host (in `~/.ssh/config`) forwards `localhost:11435` → `<REMOTE_HOST>:11434`. The `opencode.json` provider `"remote"` points to `http://127.0.0.1:11435/v1`.

**Step 1 — open the tunnel** (keep this terminal running):

```bash
./connect-remote-llm.sh
```

This renews your Kerberos ticket and starts the SSH port-forwards (llama-server and Ollama).

**Step 2 — run OpenCode** in another terminal:

```bash
opencode
```

Select **Ollama on remote › \<model\>** from the model picker.

> **Note:** Ollama on the remote server is available but noticeably slower than llama.cpp. Use llama.cpp for performance-critical work and Ollama for convenience or models not yet available in GGUF format.

### Using the `ollama-remote` helper

A convenience shell function `ollama-remote` is defined in `.bashrc` that automatically manages the SSH tunnel and routes `ollama` commands to the remote server:

```bash
# List available models on remote
ollama-remote list

# Pull a model
ollama-remote pull qwen3.6:35b-a3b

# Run an interactive chat
ollama-remote run qwen3.6:35b-a3b

# Stop the tunnel
ollama-remote-stop
```

The function checks if a tunnel on port 11435 is already active before opening a new one. It reads `REMOTE_HOST` from `.env` to determine which server to connect to.

---

## 8) Choosing models to benchmark

Before downloading and running models locally, use [Artificial Analysis](https://artificialanalysis.ai/models/) to compare models across quality, speed, and context length. It covers both hosted APIs and open-weight models, making it a useful starting point for deciding which models are worth pulling for local inference.

Key things to check there:
- **Quality index** — overall capability ranking relative to model size
- **Tokens/s** — typical inference speed on hosted hardware (gives a ceiling estimate for local runs)
- **Context length** — maximum supported context window
- **Open weights** — filter to models you can actually run locally

Use this to narrow down candidates before spending time downloading multi-GB GGUF files.

---

## 9) Local model evaluations (text + vision regression tests)

`tests/update_and_test_llama_image.py` pulls the llama.cpp Vulkan Docker image, starts it locally against each configured model, and runs opencode-based regression checks (README summarization + image transcription for vision-capable models). On success it promotes the image to the `known-good` tag used by `launch_local_llm.sh`.

```bash
uv run tests/update_and_test_llama_image.py
```

Useful options:

```bash
uv run tests/update_and_test_llama_image.py --skip-pull        # test the image already on disk
uv run tests/update_and_test_llama_image.py --image <tag>      # test a specific image tag
```

For the remote (onyx) server instead, see `tests/update_and_test_remote_llm.py`:

```bash
uv run tests/update_and_test_remote_llm.py
```
