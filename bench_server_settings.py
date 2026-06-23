#!/usr/bin/env python3
"""bench_server_settings.py

Sweeps llama-server launch flags (batch size, uber-batch, flash-attn) for a
single model and measures prefill (prompt-processing) speed by reading the
timing data returned by the /completion endpoint.

Each configuration:
  1. Starts llama-server via distrobox with the given flags
  2. Sends a warmup request, then N timed runs per prompt size
  3. Records prompt tok/s from the server's own timings block
  4. Stops the server before the next configuration

Output written to <outdir>/:
  results.jsonl  — one record per (config, prompt_size, run)
  results.csv
  pp_tps.png     — prefill tok/s vs prompt tokens, one line per config
  tg_tps.png     — decode tok/s vs prompt tokens
  README.md      — summary table

Usage:
  uv run python bench_server_settings.py \\
    -m /home/gberseth/playground/llama.cpp/models/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \\
    --container llama-vulkan-radv \\
    --port 8001
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_server")


# ── prompt generation ──────────────────────────────────────────────────────

def make_prompt(approx_tokens: int) -> str:
    """Return a plain-text prompt of roughly `approx_tokens` tokens.

    Uses short common words (~1 token each) so the count is close to the target
    without needing a tokenizer. The server reports the exact count in timings.
    """
    words = "the quick brown fox jumps over the lazy dog "
    repeats = max(1, approx_tokens * len("the ") // len(words))
    return (words * repeats).strip() + " Summarise the above in one sentence."


# ── server lifecycle ───────────────────────────────────────────────────────

def _server_cmd(model: str, port: int, ctx_size: int, base_flags: str, extra_flags: str) -> str:
    flags = " ".join(f for f in (base_flags, extra_flags) if f).strip()
    return (
        f"llama-server -m {model} -ngl 999 --no-mmap "
        f"--ctx-size {ctx_size} "
        f"--host 0.0.0.0 --port {port} "
        f"--jinja --cache-type-k q8_0 --cache-type-v q8_0 "
        f"{flags}"
    )


def start_server(container: str, cmd: str) -> subprocess.Popen:
    full_cmd = ["distrobox", "enter", container, "--", "bash", "-c", cmd]
    log.debug("Launching: %s", " ".join(full_cmd))
    return subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_server(port: int, timeout: float = 120.0) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── single completion request ──────────────────────────────────────────────

def run_completion(port: int, prompt: str, max_tokens: int = 1) -> dict[str, Any] | None:
    """POST to /completion and return the parsed JSON response."""
    url = f"http://localhost:{port}/completion"
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.warning("Request failed: %s", exc)
        return None


# ── config sweep definition ────────────────────────────────────────────────

DEFAULT_BATCH_SIZES = [128, 256, 512, 1024, 2048]
# Realistic sizes for a coding assistant session:
#   ~2-4K system prompt + user message + tool results/file context
DEFAULT_PROMPT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]


def build_configs(batch_sizes: list[int], test_flash_attn: bool) -> list[dict]:
    configs = []
    fa_options = [False, True] if test_flash_attn else [False]
    for b in batch_sizes:
        for fa in fa_options:
            label_parts = [f"b={b}"]
            flag_parts = [f"-b {b} -ub {b}"]
            if fa:
                label_parts.append("fa")
                flag_parts.append("-fa 1")
            configs.append({
                "label": " ".join(label_parts),
                "flags": " ".join(flag_parts),
                "b": b,
                "flash_attn": fa,
            })
    return configs


# ── summary / plots ────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, metric: str, ylabel: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, grp in df.groupby("label"):
        grp = grp.sort_values("target_prompt_tokens")
        ax.plot(grp["target_prompt_tokens"], grp[metric], marker="o", label=label)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", outpath)


def _md_table(headers: list[str], rows: list[list]) -> str:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    head = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return "\n".join([head, sep, body])


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    agg = (
        df.groupby(["label", "b", "flash_attn", "target_prompt_tokens"])
        .agg(pp_tps=("pp_tps", "mean"), tg_tps=("tg_tps", "mean"))
        .reset_index()
    )

    _plot(agg, "pp_tps", "Prefill speed (tok/s)", outdir / "pp_tps.png")
    _plot(agg, "tg_tps", "Decode speed (tok/s)", outdir / "tg_tps.png")

    # pivot: rows = config label, cols = prompt size
    pivot = agg.pivot_table(
        index="label", columns="target_prompt_tokens", values="pp_tps", aggfunc="max"
    )
    prompt_cols = sorted(pivot.columns.tolist())
    best = {c: pivot[c].max() for c in prompt_cols}

    headers = ["config"] + [f"{c}t" for c in prompt_cols]
    rows = []
    for label in pivot.index:
        row = [label]
        for c in prompt_cols:
            val = pivot.loc[label, c]
            if pd.isna(val):
                row.append("-")
            else:
                s = f"{val:.1f}"
                row.append(f"**{s}**" if val == best[c] else s)
        rows.append(row)

    table = _md_table(headers, rows)
    md = (
        "# Server Settings Benchmark — Prefill Speed\n\n"
        "Prefill tok/s by batch size and prompt length. "
        "Best value per column in **bold**.\n\n"
        + table
        + "\n\n## Charts\n\n"
        "![Prefill speed](pp_tps.png)\n\n"
        "![Decode speed](tg_tps.png)\n"
    )
    (outdir / "README.md").write_text(md)
    log.info("Summary written to %s/README.md", outdir)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep llama-server batch settings and measure prefill speed"
    )
    ap.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    ap.add_argument(
        "--container", default="llama-vulkan-radv",
        help="Distrobox container with llama-server (default: llama-vulkan-radv)",
    )
    ap.add_argument(
        "--port", type=int, default=8001,
        help="Port for llama-server (default: 8001; use a free port)",
    )
    ap.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="B",
        help="Batch sizes (-b / -ub) to sweep",
    )
    ap.add_argument(
        "--prompt-sizes", nargs="+", type=int, default=DEFAULT_PROMPT_SIZES, metavar="N",
        help="Approximate prompt token counts to test",
    )
    ap.add_argument(
        "--ctx-size", type=int, default=32768,
        help="KV context size for the server (must be > max prompt size; default covers up to 16K prompt + reply)",
    )
    ap.add_argument(
        "--base-flags", default="",
        help="Extra flags passed to every llama-server invocation",
    )
    ap.add_argument(
        "--flash-attn", action="store_true",
        help="Also test each batch size with -fa 1 (flash attention)",
    )
    ap.add_argument(
        "--warmup-tokens", type=int, default=64,
        help="Prompt size for the warmup run (result discarded)",
    )
    ap.add_argument(
        "--runs", type=int, default=2,
        help="Timed runs per (config, prompt_size)",
    )
    ap.add_argument("-o", "--outdir", help="Output directory (default: <model-stem>-server-settings)")
    ap.add_argument(
        "--resummarize", action="store_true",
        help="Regenerate plots and README from existing results.jsonl without re-running",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(Path(args.model).stem + "-server-settings")
    outdir.mkdir(exist_ok=True)

    fh = logging.FileHandler(outdir / "run.log")
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    results_file = outdir / "results.jsonl"

    if args.resummarize:
        if not results_file.exists():
            log.error("No results.jsonl in %s", outdir)
            sys.exit(1)
        df = pd.read_json(results_file, orient="records", lines=True)
        write_summary(df, outdir)
        return

    configs = build_configs(args.batch_sizes, args.flash_attn)
    log.info(
        "Sweeping %d configs × %d prompt sizes × %d runs",
        len(configs), len(args.prompt_sizes), args.runs,
    )

    records: list[dict] = []

    for cfg in configs:
        server_cmd = _server_cmd(
            args.model, args.port, args.ctx_size, args.base_flags, cfg["flags"]
        )
        log.info("=== Config: %s ===", cfg["label"])

        proc = start_server(args.container, server_cmd)
        ready = wait_for_server(args.port, timeout=120)
        if not ready:
            log.error("Server failed to start for config: %s", cfg["label"])
            stop_server(proc)
            continue

        # warmup — discarded
        log.info("Warmup (%dt)...", args.warmup_tokens)
        run_completion(args.port, make_prompt(args.warmup_tokens), max_tokens=1)

        for target_tokens in args.prompt_sizes:
            prompt = make_prompt(target_tokens)
            for run_idx in range(args.runs):
                result = run_completion(args.port, prompt, max_tokens=1)
                if result is None:
                    continue

                timings = result.get("timings", {})
                prompt_n   = timings.get("prompt_n", 0)
                prompt_ms  = timings.get("prompt_ms", 0)
                pred_n     = timings.get("predicted_n", 0)
                pred_ms    = timings.get("predicted_ms", 0)

                pp_tps = prompt_n / (prompt_ms / 1000) if prompt_ms > 0 else None
                tg_tps = pred_n   / (pred_ms   / 1000) if pred_ms  > 0 and pred_n > 0 else None

                log.info(
                    "  prompt=%dt  run=%d  pp=%.1f tok/s",
                    prompt_n, run_idx, pp_tps or 0,
                )

                rec = {
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "label": cfg["label"],
                    "b": cfg["b"],
                    "flash_attn": cfg["flash_attn"],
                    "target_prompt_tokens": target_tokens,
                    "prompt_n": prompt_n,
                    "prompt_ms": prompt_ms,
                    "pp_tps": pp_tps,
                    "predicted_n": pred_n,
                    "predicted_ms": pred_ms,
                    "tg_tps": tg_tps,
                    "run": run_idx,
                }
                records.append(rec)
                with results_file.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

        stop_server(proc)
        log.info("Server stopped.")

    if not records:
        log.warning("No results collected.")
        return

    df = pd.DataFrame(records)
    df.to_csv(outdir / "results.csv", index=False)
    write_summary(df, outdir)
    log.info("Done — results in %s", outdir)


if __name__ == "__main__":
    main()
