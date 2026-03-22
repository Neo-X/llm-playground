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
    axis.set_xticklabels(models, rotation=15, ha="right")
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
