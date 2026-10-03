#!/usr/bin/env python3
"""Measure aggregate throughput and per-request latency under concurrent load.

benchmark_llm_speed.py / sweep_models.py only ever have one request in
flight. That's exactly the regime vLLM's continuous batching + PagedAttention
are *not* built to win -- it should pull ahead here, where many requests are
in flight at once: aggregate tokens/s should keep climbing with concurrency,
while a backend that serializes requests one at a time flatlines near its
single-stream decode speed (and per-request latency balloons as requests
queue behind each other).

llama.cpp's server defaults to a single parallel slot (see LLAMACPP_PARALLEL
in launch_local_llm.sh) -- bump it to give llama.cpp a fair comparison
instead of testing it in a 1-slot config it was never tuned for.
"""
import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_llm_speed import (
    run_benchmark_llamacpp_server,
    run_benchmark_ollama,
    run_benchmark_vllm_server,
)
from sweep_models import (
    LLAMACPP_ALIASES,
    VLLM_ALIASES,
    start_local_server,
    stop_local_server,
    unique_prompt,
    wait_for_local_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark aggregate throughput/latency under concurrent load for llama.cpp, vLLM, and Ollama."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Ollama-style tag, e.g. qwen2.5:1.5b-instruct-q8_0 (looked up in sweep_models.LLAMACPP_ALIASES/VLLM_ALIASES).",
    )
    parser.add_argument("--backends", nargs="+", choices=["ollama", "llamacpp", "vllm"], default=["llamacpp", "vllm"])
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=0, help="0 = 2x concurrency.")
    parser.add_argument("--prompt-file", type=str, default="prompts/prompt_2048.txt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--llamacpp-host", type=str, default="http://localhost:8010")
    parser.add_argument("--llamacpp-ready-timeout", type=int, default=900, help="Seconds to wait for llama-server to come up.")
    parser.add_argument("--vllm-host", type=str, default="http://localhost:8020")
    parser.add_argument(
        "--vllm-ready-timeout", type=int, default=1200,
        help="Seconds to wait for vLLM to come up (first run also builds the local GGUF-plugin image).",
    )
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--out-csv", type=str, default="logs/concurrency_sweep.csv")
    parser.add_argument("--out-png", type=str, default="logs/concurrency_sweep.png")
    return parser.parse_args()


def fire_request(backend: str, host: str, model_name: str, prompt: str, max_new_tokens: int, nonce: str) -> dict:
    start = time.perf_counter()
    if backend == "llamacpp":
        metrics = run_benchmark_llamacpp_server(
            host=host, prompt=unique_prompt(prompt, nonce), max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.7, top_p=0.9,
        )
    elif backend == "vllm":
        metrics = run_benchmark_vllm_server(
            host=host, model_name=model_name, prompt=unique_prompt(prompt, nonce), max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.7, top_p=0.9,
        )
    else:
        metrics = run_benchmark_ollama(
            host=host, model_name=model_name, prompt=unique_prompt(prompt, nonce), max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.7, top_p=0.9, device="cuda",
        )
    metrics["request_latency_s"] = time.perf_counter() - start
    return metrics


def run_concurrency_level(
    backend: str, host: str, model_name: str, prompt: str, max_new_tokens: int, concurrency: int, num_requests: int,
) -> dict:
    wall_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(fire_request, backend, host, model_name, prompt, max_new_tokens, f"c{concurrency}-r{i}")
            for i in range(num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = time.perf_counter() - wall_start

    total_generated_tokens = sum(r["generated_tokens"] for r in results)
    latencies = sorted(r["request_latency_s"] for r in results)
    p95_idx = min(len(latencies) - 1, int(len(latencies) * 0.95))

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time_s": round(wall_time_s, 3),
        "aggregate_decode_tps": round(total_generated_tokens / wall_time_s, 2) if wall_time_s > 0 else 0.0,
        "avg_request_latency_s": round(statistics.mean(latencies), 3),
        "p50_latency_s": round(latencies[len(latencies) // 2], 3),
        "p95_latency_s": round(latencies[p95_idx], 3),
        "avg_per_request_decode_tps": round(statistics.mean(r["decode_tps"] for r in results), 2),
    }


def run_backend(backend: str, model: str, args: argparse.Namespace) -> list[dict]:
    if backend == "ollama":
        host, model_name = args.ollama_host, model
    else:
        aliases = LLAMACPP_ALIASES if backend == "llamacpp" else VLLM_ALIASES
        alias = aliases.get(model)
        if alias is None:
            print(f"  Skipping {backend} backend for '{model}': no local GGUF/alias configured.")
            return []
        host = args.llamacpp_host if backend == "llamacpp" else args.vllm_host
        model_name = alias

    process = None
    if backend in {"llamacpp", "vllm"}:
        log_path = f"logs/concurrency_{backend}_{model_name}.log"
        os.makedirs("logs", exist_ok=True)
        print(f"  Launching {backend} server for alias '{model_name}'...")
        process = start_local_server(model_name, log_path, backend_env="vllm" if backend == "vllm" else None)
        ready_path = "/v1/models" if backend == "vllm" else "/health"
        timeout = args.vllm_ready_timeout if backend == "vllm" else args.llamacpp_ready_timeout
        wait_for_local_server(host, ready_path, timeout, process, log_path, f"{backend} server")

    rows = []
    try:
        # Warm up once at concurrency=1 so weights/CUDA graphs are resident
        # before the first measured level.
        fire_request(backend, host, model_name, args.prompt, min(16, args.max_new_tokens), "warmup")

        for concurrency in args.concurrency_levels:
            num_requests = args.requests_per_level or concurrency * 2
            print(f"  {backend} | concurrency={concurrency} | {num_requests} requests...")
            level_result = run_concurrency_level(
                backend, host, model_name, args.prompt, args.max_new_tokens, concurrency, num_requests,
            )
            level_result["backend"] = backend
            level_result["model"] = model
            rows.append(level_result)
            print(
                f"    aggregate: {level_result['aggregate_decode_tps']:.2f} tok/s | "
                f"avg latency: {level_result['avg_request_latency_s']:.2f}s | "
                f"p95 latency: {level_result['p95_latency_s']:.2f}s"
            )
    finally:
        if process is not None:
            stop_local_server(process)

    return rows


def plot_results(dataframe: pd.DataFrame, out_png: str) -> None:
    backends = sorted(dataframe["backend"].unique())
    figure, (ax_tps, ax_latency) = plt.subplots(1, 2, figsize=(12, 5))

    for backend in backends:
        subset = dataframe[dataframe["backend"] == backend].sort_values("concurrency")
        ax_tps.plot(subset["concurrency"], subset["aggregate_decode_tps"], marker="o", label=backend)
        ax_latency.plot(subset["concurrency"], subset["p95_latency_s"], marker="o", label=backend)

    ax_tps.set_title("Aggregate decode throughput vs. concurrency")
    ax_tps.set_xlabel("Concurrent requests")
    ax_tps.set_ylabel("Tokens/s (summed across in-flight requests)")
    ax_tps.legend()
    ax_tps.grid(alpha=0.25)

    ax_latency.set_title("p95 request latency vs. concurrency")
    ax_latency.set_xlabel("Concurrent requests")
    ax_latency.set_ylabel("Seconds")
    ax_latency.legend()
    ax_latency.grid(alpha=0.25)

    figure.suptitle("Concurrent-load Benchmark", fontsize=13)
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

    all_rows = []
    for backend in args.backends:
        print(f"=== {args.model} | {backend} ===")
        try:
            all_rows.extend(run_backend(backend, args.model, args))
        except Exception as exc:
            print(f"  FAIL: {backend}: {exc}")

    dataframe = pd.DataFrame(all_rows)
    parent = os.path.dirname(args.out_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    dataframe.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to: {args.out_csv}")

    if not dataframe.empty:
        plot_results(dataframe, args.out_png)


if __name__ == "__main__":
    main()
