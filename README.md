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

Use the helper to benchmark 2 or more models and print a ranked table:

```bash
python rank_ollama_models.py \
  --models qwen2.5:3b llama3.2:3b glm-4.7-flash qwen3-coder-next qwen3.5:latest \
  --runs 3 \
  --warmup 1 \
  --max-new-tokens 256 \
  --rank-by decode \
  --device cuda \
  --ollama-pull \
  --csv logs/benchmark_metrics_cuda.csv
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
