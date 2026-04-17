# LLM Inference Speed Benchmark (Prompt vs Generation)

This project benchmarks:
- **Prompt (prefill) token speed** in tokens/sec
- **Generation (decode) token speed** in tokens/sec

Models configured for small GPU:
- `qwen2.5:3b`
- `llama3.2:3b`

## 1) Create and activate environment (uv)

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

## 2) Run benchmark (Transformers backend)

```bash
python benchmark_llm_speed.py
```

Useful options:

```bash
python benchmark_llm_speed.py \
  --backend transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --runs 5 \
  --warmup 1 \
  --max-new-tokens 128 \
  --device auto
```

Force GPU:

```bash
python benchmark_llm_speed.py --device cuda
```

Force AMD GPU (ROCm-enabled PyTorch build):

```bash
python benchmark_llm_speed.py --device amd
```

Force CPU:

```bash
python benchmark_llm_speed.py --device cpu
```

Run on CPU then GPU with separate output files:

```bash
python benchmark_llm_speed.py \
  --backend transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device cpu \
  --no-load-in-4bit \
  --runs 5 \
  --warmup 1 \
  --csv logs/benchmark_metrics_cpu.csv \
  --jsonl logs/benchmark_metrics_cpu.jsonl \
  --reset-output

python benchmark_llm_speed.py \
  --backend transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device cuda \
  --runs 5 \
  --warmup 1 \
  --csv logs/benchmark_metrics_gpu.csv \
  --jsonl logs/benchmark_metrics_gpu.jsonl \
  --reset-output

python benchmark_llm_speed.py \
  --backend transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device amd \
  --runs 5 \
  --warmup 1 \
  --csv logs/benchmark_metrics_amd.csv \
  --jsonl logs/benchmark_metrics_amd.jsonl \
  --reset-output
```

## 3) Run benchmark (Ollama backend)

Start Ollama server if it is not running:

```bash
ollama serve
```

Benchmark using Ollama model:

```bash
python benchmark_llm_speed.py \
  --backend ollama \
  --model qwen2.5:3b \
  --runs 5 \
  --warmup 1 \
  --max-new-tokens 128
```

Auto-pull model before running:

```bash
python benchmark_llm_speed.py \
  --backend ollama \
  --model qwen2.5:3b \
  --ollama-pull
```

Use a non-default host:

```bash
python benchmark_llm_speed.py --backend ollama --ollama-host http://localhost:11434
```

## 4) Logs

Metrics are appended to:
- `logs/benchmark_metrics.csv`
- `logs/benchmark_metrics.jsonl`

Columns include:
- `backend`
- `prompt_tokens`, `generated_tokens`
- `prefill_time_s`, `decode_time_s`
- `prefill_tps`, `decode_tps`

## 5) Rank Ollama models (2+)

Use the helper to benchmark 2 or more Ollama models and print a ranked table:

```bash
uv run python rank_ollama_models.py \
  --models qwen2.5:3b llama3.2:3b glm-4.7-flash gpt-oss:20b gpt-oss:120b qwen3.5:latest mdq100/qwen3.5:27b-96g \
  --runs 3 \
  --warmup 1 \
  --max-new-tokens 256 \
  --rank-by decode \
  --device rocm \
  --ollama-pull \
  --csv logs/benchmark_metrics_rocm.csv
```

Auto-pull models before benchmarking:

```bash
python rank_ollama_models.py \
  --models qwen2.5:3b llama3.2:3b \
  --ollama-pull \
  --device cpu
```

Run the same ranking flow against an AMD GPU with Ollama's ROCm build:

```bash
python rank_ollama_models.py \
  --models qwen2.5:3b llama3.2:3b \
  --ollama-pull \
  --device amd
```

Ranking outputs are saved to:
- `logs/ollama_model_rankings.csv`

## 6) Plot CPU vs accelerator benchmark results

After producing `logs/benchmark_metrics_cpu.csv` and `logs/benchmark_metrics_cuda.csv`:

```bash
python plot_cpu_gpu_comparison.py \
  --cpu-csv logs/benchmark_metrics_cpu.csv \
  --cuda-csv logs/benchmark_metrics_cuda.csv \
  --out-png logs/cpu_vs_cuda_speed.png
```

Add AMD GPU results to the same chart:

```bash
python plot_cpu_gpu_comparison.py \
  --cpu-csv logs/benchmark_metrics_cpu.csv \
  --cuda-csv logs/benchmark_metrics_cuda.csv \
  --amd-csv logs/benchmark_metrics_amd.csv \
  --out-png logs/cpu_cuda_amd_speed.png
```

`plot_cpu_gpu_comparison.py` supports either:
- per-run benchmark CSV columns: `prefill_tps`, `decode_tps`
- ranked-model CSV columns: `avg_prefill_tps`, `avg_decode_tps`

## Notes

- First run is usually slower due to model loading and kernel warmup.
- If CUDA or ROCm is unavailable, benchmark runs on CPU automatically when `--device auto` is used.
- AMD GPU benchmarking with Transformers requires a ROCm-enabled PyTorch install. AMD GPU benchmarking with Ollama requires an Ollama build with ROCm support.
- For strict repeatability, run with deterministic prompts and disable sampling (default).

---

## 7) llama.cpp benchmarking (Strix Halo / AMD iGPU)

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

Launch server for CLAUDE

distrobox enter llama-vulkan-radv -- bash -c "llama-server -m $MODELS/qwen3-30B-A3B/Qwen3-30B-A3B-UD-Q4_K_XL.gguf -ngl 999 --no-mmap --ctx-size 100000 --host 0.0.0.0 --port 8000 --jinja --cache-type-k q8_0 --cache-type-v q8_0"

export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="sk-no-key-required"
export ANTHROPIC_MODEL="private-model"

claude --model private-model[100k]
 


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
