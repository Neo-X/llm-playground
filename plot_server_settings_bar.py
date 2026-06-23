#!/usr/bin/env python3
"""plot_server_settings_bar.py

Bar chart of prefill (and optionally decode) tok/s from bench_server_settings.py
results, grouped by batch size with one bar per prompt size (or vice versa).

Usage:
  uv run python plot_server_settings_bar.py --results <outdir>/results.jsonl
  uv run python plot_server_settings_bar.py --results <outdir>/results.jsonl --metric tg_tps
  uv run python plot_server_settings_bar.py --results <outdir>/results.jsonl --group-by prompt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    # average over repeated runs
    return (
        df.groupby(["label", "b", "flash_attn", "target_prompt_tokens"])
        .agg(pp_tps=("pp_tps", "mean"), tg_tps=("tg_tps", "mean"))
        .reset_index()
    )


def bar_plot(
    df: pd.DataFrame,
    metric: str,
    group_by: str,
    outpath: Path,
) -> None:
    """
    group_by='batch'  → x-axis = batch size, grouped bars = prompt sizes
    group_by='prompt' → x-axis = prompt size, grouped bars = batch sizes
    """
    ylabel = "Prefill speed (tok/s)" if metric == "pp_tps" else "Decode speed (tok/s)"
    title  = ylabel

    if group_by == "batch":
        x_col    = "b"
        grp_col  = "target_prompt_tokens"
        xlabel   = "Batch size (-b / -ub)"
        grp_label = "prompt tokens"
    else:
        x_col    = "target_prompt_tokens"
        grp_col  = "b"
        xlabel   = "Prompt tokens"
        grp_label = "batch size"

    x_vals  = sorted(df[x_col].unique())
    grp_vals = sorted(df[grp_col].unique())

    n_grp   = len(grp_vals)
    n_x     = len(x_vals)
    width   = 0.8 / n_grp
    x_idx   = np.arange(n_x)

    cmap    = plt.get_cmap("tab10")
    colors  = [cmap(i / max(n_grp - 1, 1)) for i in range(n_grp)]

    fig, ax = plt.subplots(figsize=(max(8, n_x * n_grp * 0.4 + 2), 5))

    for i, gv in enumerate(grp_vals):
        sub = df[df[grp_col] == gv].set_index(x_col)[metric]
        heights = [sub.get(xv, np.nan) for xv in x_vals]
        offset  = (i - n_grp / 2 + 0.5) * width
        bars = ax.bar(
            x_idx + offset, heights, width=width * 0.9,
            label=f"{grp_label}={gv}", color=colors[i], alpha=0.85,
        )
        # value labels on top of each bar
        for bar, h in zip(bars, heights):
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bar chart of bench_server_settings results")
    ap.add_argument(
        "--results", required=True, type=Path,
        help="Path to results.jsonl from bench_server_settings.py",
    )
    ap.add_argument(
        "--metric", choices=["pp_tps", "tg_tps"], default="pp_tps",
        help="Metric to plot (default: pp_tps = prefill speed)",
    )
    ap.add_argument(
        "--group-by", choices=["batch", "prompt"], default="batch",
        help="batch: x-axis=batch size, bars=prompt sizes  |  prompt: x-axis=prompt size, bars=batch sizes",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: next to results.jsonl)",
    )
    args = ap.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        raise SystemExit(1)

    df = load(args.results)
    if df.empty or df[args.metric].isna().all():
        print(f"No valid {args.metric} data in {args.results}")
        raise SystemExit(1)

    stem = f"bar_{args.metric}_by_{args.group_by}"
    outpath = args.out or args.results.parent / f"{stem}.png"

    bar_plot(df, args.metric, args.group_by, outpath)

    # also print a quick summary table
    pivot = df.pivot_table(
        index="b", columns="target_prompt_tokens",
        values=args.metric, aggfunc="max",
    ).round(1)
    pivot.index.name = "batch_size"
    pivot.columns.name = "prompt_tokens"
    print("\nPrefill tok/s (rows=batch size, cols=prompt tokens):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
