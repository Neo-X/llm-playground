#!/usr/bin/env python3
"""Benchmark how well a server handles a "team of agents" hitting it at
once -- the actual motivation for adding the vLLM backend (see README.md
section 7): llama-server serves requests with much less request-level
concurrency than vLLM's continuous batching + PagedAttention.

Fires a batch of distinct small prompts (standing in for N agents each
working on a different task) at an OpenAI-compatible /v1/chat/completions
endpoint two ways:

  1. sequential -- one request at a time (the worst case: no concurrency).
  2. concurrent -- all requests in flight at once via a thread pool.

...and reports per-request latency plus aggregate tokens/sec for each, so
you can see the speedup concurrency gets you (or doesn't, for a backend
that just queues requests one at a time).

Usage:
    # vLLM (see: BACKEND=vllm ./launch_local_llm.sh qwen2.5-3b)
    uv run python bench_agent_concurrency.py --host http://localhost:8020 --model qwen2.5-3b

    # llama.cpp, for a side-by-side comparison
    uv run python bench_agent_concurrency.py --host http://localhost:8010 --model qwen2.5-3b

    # More simulated agents, all in flight together
    uv run python bench_agent_concurrency.py --num-requests 16 --concurrency 16
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Distinct small tasks standing in for a team of agents each working on
# something different -- deliberately varied so this isn't just N copies of
# the same prompt (which would let prefix caching flatter the result).
AGENT_TASKS = [
    "Write a Python function that checks if a string is a palindrome, ignoring case and spaces.",
    "Explain the difference between TCP and UDP in 3 bullet points.",
    "Write a haiku about debugging code at 2am.",
    "What is the time complexity of binary search, and why?",
    "Convert this pseudocode to Python: for each item in a list, if item > 10, add it to a result list.",
    "List 3 common causes of memory leaks in long-running server processes.",
    "Write a regex that matches a valid IPv4 address.",
    "Summarize what a load balancer does in 2 sentences.",
    "Write a SQL query to find the second-highest salary in an employees table.",
    "Explain what a race condition is, with a one-sentence example.",
    "Write a bash one-liner to find the 5 largest files under the current directory.",
    "What's the difference between a process and a thread?",
]


def build_prompts(num_requests: int) -> list[str]:
    """Cycle through AGENT_TASKS to reach num_requests, tagging repeats with
    an agent index so duplicate-task runs still get distinguishable prompts
    (avoids identical-prefix requests all hitting the same cache entry)."""
    prompts = []
    for i in range(num_requests):
        task = AGENT_TASKS[i % len(AGENT_TASKS)]
        if i >= len(AGENT_TASKS):
            task = f"[agent-{i}] {task}"
        prompts.append(task)
    return prompts


def send_request(host: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    url = f"{host.rstrip('/')}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"ok": False, "error": f"HTTP {exc.code}: {error_body or exc.reason}", "latency_s": time.perf_counter() - start}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Failed to reach {url}: {exc.reason}. Is the server running?", "latency_s": time.perf_counter() - start}

    latency_s = time.perf_counter() - start
    usage = body.get("usage", {})
    text = ""
    choices = body.get("choices") or []
    if choices:
        text = (choices[0].get("message") or {}).get("content", "")

    return {
        "ok": True,
        "latency_s": latency_s,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "text": text,
    }


def run_batch(host: str, model: str, prompts: list[str], max_tokens: int, timeout: int, concurrency: int) -> tuple[list[dict], float]:
    """Runs all prompts with up to `concurrency` in flight at once (pass
    concurrency=1 for a strictly sequential run) and returns
    (per-request results, total wall-clock time for the whole batch)."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: send_request(host, model, p, max_tokens, timeout), prompts))
    total_wall_s = time.perf_counter() - start
    return results, total_wall_s


def summarize(label: str, results: list[dict], total_wall_s: float) -> dict:
    ok_results = [r for r in results if r["ok"]]
    failed = len(results) - len(ok_results)

    print(f"\n=== {label} ===")
    for i, r in enumerate(results):
        if r["ok"]:
            print(f"  [{i}] {r['latency_s']:.2f}s | {r['completion_tokens']} tok generated")
        else:
            print(f"  [{i}] FAILED after {r['latency_s']:.2f}s: {r['error']}")

    total_completion_tokens = sum(r["completion_tokens"] for r in ok_results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok_results)
    avg_latency_s = sum(r["latency_s"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    aggregate_tps = total_completion_tokens / total_wall_s if total_wall_s > 0 else 0.0

    print(f"  -- {len(ok_results)}/{len(results)} succeeded ({failed} failed)")
    print(f"  -- total wall time: {total_wall_s:.2f}s | avg per-request latency: {avg_latency_s:.2f}s")
    print(f"  -- aggregate throughput: {aggregate_tps:.1f} completion tok/s across all requests")

    return {
        "label": label,
        "num_requests": len(results),
        "num_failed": failed,
        "total_wall_s": round(total_wall_s, 3),
        "avg_latency_s": round(avg_latency_s, 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tps": round(aggregate_tps, 2),
    }


def append_csv(file_path: str, row: dict) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://localhost:8020", help="OpenAI-compatible server URL (vLLM default: 8020; llama.cpp docker: 8010).")
    parser.add_argument("--model", default="qwen2.5-3b", help="Model/alias as served by the endpoint.")
    parser.add_argument("--num-requests", type=int, default=8, help="Number of simulated agent tasks to send.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max requests in flight at once for the concurrent run.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    parser.add_argument("--skip-sequential", action="store_true", help="Only run the concurrent batch (sequential gets slow as --num-requests grows).")
    parser.add_argument("--csv", default="logs/agent_concurrency.csv", help="Append summary rows here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = build_prompts(args.num_requests)

    print(f"Target: {args.host} | model: {args.model} | {args.num_requests} simulated agent tasks | concurrency: {args.concurrency}")

    timestamp = datetime.now(timezone.utc).isoformat()
    summaries = []

    if not args.skip_sequential:
        results, total_wall_s = run_batch(args.host, args.model, prompts, args.max_tokens, args.timeout, concurrency=1)
        summaries.append(summarize("sequential (1 at a time)", results, total_wall_s))

    results, total_wall_s = run_batch(args.host, args.model, prompts, args.max_tokens, args.timeout, concurrency=args.concurrency)
    summaries.append(summarize(f"concurrent ({args.concurrency} in flight)", results, total_wall_s))

    if len(summaries) == 2:
        speedup = summaries[0]["total_wall_s"] / summaries[1]["total_wall_s"] if summaries[1]["total_wall_s"] > 0 else 0.0
        print(f"\n=== speedup ===\nconcurrent was {speedup:.2f}x faster than sequential for this batch")

    for summary in summaries:
        row = {"timestamp_utc": timestamp, "host": args.host, "model": args.model, **summary}
        append_csv(args.csv, row)
    print(f"\nSummary rows appended to: {args.csv}")


if __name__ == "__main__":
    main()
