#!/usr/bin/env python3
"""Benchmark a list of models across the Ollama and llama.cpp backends in one run,
writing a combined CSV and a grouped bar-chart PNG."""
import argparse
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_llm_speed import (
    ollama_pull_model,
    ollama_unload_model,
    resolve_ollama_device,
    run_benchmark_llamacpp_server,
    run_benchmark_ollama,
)

# Ollama tag -> launch_local_llm.sh alias, for models that also have a local GGUF.
# Models not listed here are skipped for the llamacpp backend.
LLAMACPP_ALIASES = {
    "qwen3.6:27b": "qwen3.6-27b",
    "qwen3.6:35b-a3b": "qwen3.6-35b-a3b",
    "qwen2.5:3b": "qwen2.5-3b",
    "llama3.2:3b": "llama3.2-3b",
}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def unique_prompt(prompt: str, nonce: str) -> str:
    """Prepend a unique nonce so each call gets a distinct prefix, defeating
    server-side prompt/context caching so prefill timing reflects real work."""
    return f"[{nonce}] {prompt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a list of models on Ollama and llama.cpp and plot the results."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:3b", "llama3.2:3b", "qwen3.6:27b", "qwen3.6:35b-a3b", "gpt-oss:20b"],
        help="Ollama model tags to benchmark. Tags also present in LLAMACPP_ALIASES are "
        "additionally benchmarked via llama.cpp (launch_local_llm.sh).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["ollama", "llamacpp"],
        default=["ollama", "llamacpp"],
        help="Which backends to run for each model (llamacpp only applies to models with a known alias).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompts/prompt_2048.txt",
        help="File to load the benchmark prompt from.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "amd", "cpu"])
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--llamacpp-host", type=str, default="http://localhost:8000")
    parser.add_argument("--llamacpp-ready-timeout", type=int, default=900, help="Seconds to wait for llama-server to come up.")
    parser.add_argument("--out-csv", type=str, default="logs/model_sweep.csv")
    parser.add_argument("--out-png", type=str, default="logs/model_sweep.png")
    return parser.parse_args()


def wait_for_llamacpp_server(host: str, timeout: int, process: subprocess.Popen, log_path: str) -> None:
    deadline = time.monotonic() + timeout
    url = f"{host.rstrip('/')}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                    tail = "".join(handle.readlines()[-10:])
            except OSError:
                pass
            raise RuntimeError(
                f"launch_local_llm.sh exited (code {process.returncode}) before the server became "
                f"healthy. Tail of {log_path}:\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(3)
    raise TimeoutError(f"llama.cpp server at {host} did not become ready within {timeout}s")


def start_llamacpp_server(alias: str, log_path: str) -> subprocess.Popen:
    log_file = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        ["./launch_local_llm.sh", alias],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def stop_llamacpp_server(process: subprocess.Popen) -> None:
    subprocess.run(["docker", "rm", "-f", "llama-vulkan-server", "llama-cuda-server"], capture_output=True)
    subprocess.run(["distrobox", "enter", "llama-vulkan-radv", "--", "pkill", "-f", "llama-server"], capture_output=True)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_backend_for_model(
    *,
    backend: str,
    model: str,
    args: argparse.Namespace,
) -> dict | None:
    if backend == "ollama":
        ollama_pull_model(args.ollama_host, model)
        device = resolve_ollama_device(args.device)
        for warmup_idx in range(args.warmup):
            run_benchmark_ollama(
                host=args.ollama_host, model_name=model, prompt=unique_prompt(args.prompt, f"warmup-{warmup_idx}"),
                max_new_tokens=min(16, args.max_new_tokens), do_sample=False,
                temperature=0.7, top_p=0.9, device=device,
            )
        runs = [
            run_benchmark_ollama(
                host=args.ollama_host, model_name=model, prompt=unique_prompt(args.prompt, f"run-{run_idx}"),
                max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=0.7, top_p=0.9, device=device,
            )
            for run_idx in range(args.runs)
        ]
        ollama_unload_model(args.ollama_host, model)
    else:
        alias = LLAMACPP_ALIASES.get(model)
        if alias is None:
            print(f"  Skipping llamacpp backend for '{model}': no local GGUF/alias configured.")
            return None

        log_path = f"logs/llama_server_{alias}.log"
        os.makedirs("logs", exist_ok=True)
        print(f"  Launching llama-server for alias '{alias}'...")
        process = start_llamacpp_server(alias, log_path)
        try:
            wait_for_llamacpp_server(args.llamacpp_host, args.llamacpp_ready_timeout, process, log_path)
            for warmup_idx in range(args.warmup):
                run_benchmark_llamacpp_server(
                    host=args.llamacpp_host, prompt=unique_prompt(args.prompt, f"warmup-{warmup_idx}"),
                    max_new_tokens=min(16, args.max_new_tokens), do_sample=False,
                    temperature=0.7, top_p=0.9,
                )
            runs = [
                run_benchmark_llamacpp_server(
                    host=args.llamacpp_host, prompt=unique_prompt(args.prompt, f"run-{run_idx}"),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=0.7, top_p=0.9,
                )
                for run_idx in range(args.runs)
            ]
        finally:
            stop_llamacpp_server(process)

    avg_prefill_tps = sum(r["prefill_tps"] for r in runs) / len(runs)
    avg_decode_tps = sum(r["decode_tps"] for r in runs) / len(runs)
    return {"avg_prefill_tps": avg_prefill_tps, "avg_decode_tps": avg_decode_tps}


def plot_results(dataframe: pd.DataFrame, out_png: str) -> None:
    models = sorted(dataframe["model"].unique())
    backends = sorted(dataframe["backend"].unique())
    width = 0.8 / max(len(backends), 1)

    figure, (ax_prefill, ax_decode) = plt.subplots(1, 2, figsize=(max(10, len(models) * 3), 5))
    for ax, metric, title in [
        (ax_prefill, "avg_prefill_tps", "Prefill tokens/s"),
        (ax_decode, "avg_decode_tps", "Decode tokens/s"),
    ]:
        for offset_index, backend in enumerate(backends):
            subset = dataframe[dataframe["backend"] == backend].set_index("model").reindex(models)
            positions = [i - 0.4 + width / 2 + offset_index * width for i in range(len(models))]
            bars = ax.bar(positions, subset[metric].fillna(0), width=width, label=backend)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("Tokens per second")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    figure.suptitle("Model Speed by Backend", fontsize=13)
    figure.tight_layout()

    parent = os.path.dirname(out_png)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.savefig(out_png, dpi=180)
    print(f"Saved plot to: {out_png}")


def main() -> None:
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        args.prompt = handle.read()
    print(f"Loaded prompt from {args.prompt_file} ({len(args.prompt)} chars)")

    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for model in args.models:
        for backend in args.backends:
            print(f"=== {model} | {backend} ===")
            try:
                summary = run_backend_for_model(backend=backend, model=model, args=args)
            except Exception as exc:
                print(f"  FAIL: {model} | {backend}: {exc}")
                continue
            if summary is None:
                continue
            rows.append({
                "timestamp_utc": timestamp,
                "backend": backend,
                "model": model,
                "avg_prefill_tps": round(summary["avg_prefill_tps"], 2),
                "avg_decode_tps": round(summary["avg_decode_tps"], 2),
                "runs": args.runs,
                "max_new_tokens": args.max_new_tokens,
            })
            print(f"  avg prefill: {summary['avg_prefill_tps']:.2f} tok/s | avg decode: {summary['avg_decode_tps']:.2f} tok/s")

    dataframe = pd.DataFrame(rows)
    parent = os.path.dirname(args.out_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    dataframe.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to: {args.out_csv}")

    plot_results(dataframe, args.out_png)


if __name__ == "__main__":
    main()
