
uv run python rank_ollama_models.py \
  --models qwen2.5:3b glm-4.7-flash qwen3:30b-a3b gpt-oss:20b gpt-oss:120b \
  --runs 3 \
  --warmup 1 \
  --max-new-tokens 256 \
  --rank-by decode \
  --device amd \
  --ollama-pull \
  --csv logs/benchmark_metrics_amd_$(date "+%Y-%m-%d").csv
