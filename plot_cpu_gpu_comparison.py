#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU vs accelerator LLM benchmark speed from benchmark CSV files."
    )
    parser.add_argument("--cpu-csv", type=str, default="logs/benchmark_metrics_cpu.csv")
    parser.add_argument("--cuda-csv", type=str, default="logs/benchmark_metrics_cuda.csv")
    parser.add_argument(
        "--amd-csv",
        type=str,
        default=None,
        help="Optional CSV for AMD GPU / ROCm benchmark results.",
    )
    parser.add_argument("--out-png", type=str, default="logs/cpu_vs_cuda_speed.png")
    return parser.parse_args()


def load_per_model(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"No rows found in: {csv_path}")

    if {"prefill_tps", "decode_tps"}.issubset(dataframe.columns):
        prefill_col, decode_col = "prefill_tps", "decode_tps"
    elif {"avg_prefill_tps", "avg_decode_tps"}.issubset(dataframe.columns):
        prefill_col, decode_col = "avg_prefill_tps", "avg_decode_tps"
    else:
        raise ValueError(
            f"Missing required columns in {csv_path}. Expected either "
            "(prefill_tps, decode_tps) or (avg_prefill_tps, avg_decode_tps)."
        )

    return (
        dataframe.groupby("model")[[prefill_col, decode_col]]
        .mean()
        .rename(columns={prefill_col: "prefill_tps", decode_col: "decode_tps"})
    )


def annotate_bars(ax: plt.Axes, bars: list, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def load_device_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    device_frames = {
        "CPU": load_per_model(args.cpu_csv),
        "CUDA GPU": load_per_model(args.cuda_csv),
    }
    if args.amd_csv:
        device_frames["AMD GPU"] = load_per_model(args.amd_csv)
    return device_frames


def main() -> None:
    args = parse_args()

    device_frames = load_device_frames(args)

    models = pd.Index([])
    for dataframe in device_frames.values():
        models = models.union(dataframe.index)

    aligned_frames = {
        device_name: dataframe.reindex(models, fill_value=0)
        for device_name, dataframe in device_frames.items()
    }

    x = np.arange(len(models))
    width = 0.8 / max(len(aligned_frames), 1)

    figure, (ax_prefill, ax_decode) = plt.subplots(1, 2, figsize=(max(10, len(models) * 3), 5))

    for ax, metric, title in [
        (ax_prefill, "prefill_tps", "Prefill tokens/s"),
        (ax_decode, "decode_tps", "Decode tokens/s"),
    ]:
        for offset_index, (device_name, dataframe) in enumerate(aligned_frames.items()):
            vals = dataframe[metric].values
            bar_positions = x - 0.4 + width / 2 + offset_index * width
            bars = ax.bar(bar_positions, vals, width=width, label=device_name)
            annotate_bars(ax, bars)

        cpu_vals = aligned_frames["CPU"][metric].values
        for model_index in range(len(models)):
            comparisons = []
            for device_name, dataframe in aligned_frames.items():
                if device_name == "CPU":
                    continue
                device_val = dataframe[metric].iloc[model_index]
                cpu_val = cpu_vals[model_index]
                if cpu_val > 0 and device_val > 0:
                    comparisons.append(f"{device_name.split()[0]} {device_val / cpu_val:.1f}x")

            if comparisons:
                group_top = max(frame[metric].iloc[model_index] for frame in aligned_frames.values())
                ax.text(
                    x[model_index],
                    group_top * 1.04,
                    " | ".join(comparisons),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(title)
        ax.set_ylabel("Tokens per second")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    figure.suptitle("CPU vs Accelerator LLM Speed by Model", fontsize=13)
    figure.tight_layout()

    parent = os.path.dirname(args.out_png)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.savefig(args.out_png, dpi=180)
    print(f"Saved plot to: {args.out_png}")


if __name__ == "__main__":
    main()
