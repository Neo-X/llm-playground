#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ollama model ranking metrics.")
    parser.add_argument("--in-csv", type=str, default="logs/ollama_model_rankings.csv")
    parser.add_argument("--out-png", type=str, default="logs/ollama_model_rankings.png")
    return parser.parse_args()


# Approximate GPU memory (GB) for Ollama default quantizations
GPU_MEMORY_GB = {
    "llama3.2:3b": 2.0,
    "llama3.2:1b": 1.3,
    "llama3.1:8b": 4.7,
    "llama3.1:70b": 40.0,
    "llama3.3:70b": 40.0,
    "qwen2.5:3b": 1.9,
    "qwen2.5:7b": 4.7,
    "qwen2.5:14b": 9.0,
    "qwen2.5:32b": 20.0,
    "qwen2.5:72b": 47.0,
    "mistral:7b": 4.1,
    "gemma2:2b": 1.6,
    "gemma2:9b": 5.5,
    "gemma2:27b": 16.0,
    "phi4:14b": 9.1,
    "phi3:3.8b": 2.2,
    "deepseek-r1:7b": 4.7,
    "deepseek-r1:14b": 9.0,
    "deepseek-r1:32b": 20.0,
    "glm4:9b": 5.5,
    "glm4:27b": 16.0,
    "glm-4v:9b": 6.0,
    "glm-z1-air:32b": 20.0,
    "glm-z1-rumination:32b": 20.0,
    "glm-4.7-flash": 2.8,
}


def main() -> None:
    args = parse_args()
    dataframe = pd.read_csv(args.in_csv)
    dataframe = dataframe.sort_values("rank")

    models = dataframe["model"].tolist()
    decode = dataframe["avg_decode_tps"].tolist()
    prefill = dataframe["avg_prefill_tps"].tolist()

    x = range(len(models))
    width = 0.38

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar([value - width / 2 for value in x], decode, width=width, label="Decode tok/s")
    axis.bar([value + width / 2 for value in x], prefill, width=width, label="Prefill tok/s")

    axis.set_title("Ollama Model Speed Comparison")
    axis.set_ylabel("Tokens per second")
    axis.set_xticks(list(x))
    model_labels = [
        f"{m}\n({GPU_MEMORY_GB[m]:.1f} GB)" if m in GPU_MEMORY_GB else m
        for m in models
    ]
    axis.set_xticklabels(model_labels, rotation=15, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(decode):
        axis.text(index - width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(prefill):
        axis.text(index + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    figure.tight_layout()

    parent = os.path.dirname(args.out_png)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.savefig(args.out_png, dpi=180)
    print(f"Saved plot to: {args.out_png}")


if __name__ == "__main__":
    main()
