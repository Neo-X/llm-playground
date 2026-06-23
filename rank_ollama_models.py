#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

from benchmark_llm_speed import ollama_check_gpu_layers, ollama_pull_model, ollama_unload_model, run_benchmark_ollama


def run_ollama_benchmark_with_helpful_errors(
    *,
    host: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str,
    runs_command_hint: str,
) -> dict:
    try:
        return run_benchmark_ollama(
            host=host,
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message and "model" in message:
            raise RuntimeError(
                f"Ollama model '{model_name}' is not available locally. "
                f"Run one of these first:\n"
                f"  ollama pull {model_name}\n"
                f"  {runs_command_hint}"
            ) from exc
        raise


def format_table(rows: list[dict]) -> str:
    headers = [
        "Rank",
        "Model",
        "Avg Prefill tok/s",
        "Avg Decode tok/s",
        "Avg Prompt toks",
        "Avg Gen toks",
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["rank"]),
                row["model"],
                f"{row['avg_prefill_tps']:.2f}",
                f"{row['avg_decode_tps']:.2f}",
                f"{row['avg_prompt_tokens']:.1f}",
                f"{row['avg_generated_tokens']:.1f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_line(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    header_line = render_line(headers)
    separator = "-+-".join("-" * width for width in widths)
    data_lines = [render_line(row) for row in table_rows]
    return "\n".join([header_line, separator, *data_lines])


def append_csv(file_path: str, rows: list[dict]) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model",
                "avg_prefill_tps",
                "avg_decode_tps",
                "avg_prompt_tokens",
                "avg_generated_tokens",
                "runs",
                "prompt",
                "max_new_tokens",
                "device",
                "timestamp_utc",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and rank multiple Ollama models by prompt/decode token speed."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:1b"],
        help="Two or more Ollama model tags to evaluate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in 5 short bullet points how transformer inference latency is measured.",
        help="Prompt to benchmark.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "amd", "cpu"],
        help="Device preference for Ollama model execution. Use 'amd' for ROCm-enabled AMD GPUs.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-pull", action="store_true")
    parser.add_argument("--rank-by", choices=["decode", "prefill"], default="decode")
    parser.add_argument(
        "--out-csv",
        "--csv",
        dest="out_csv",
        type=str,
        default="logs/ollama_model_rankings.csv",
        help="Output CSV path for ranked model results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    command_hint = (
        "python rank_ollama_models.py "
        f"--models {' '.join(args.models)} --runs {args.runs} --warmup {args.warmup} "
        f"--max-new-tokens {args.max_new_tokens} --rank-by {args.rank_by} "
        f"--device {args.device} --ollama-pull"
    )

    if len(args.models) < 2:
        raise ValueError("Please provide at least 2 models using --models.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Please provide distinct model names with --models.")

    model_summaries = []
    ollama_device = "cuda" if args.device == "amd" else args.device

    for model_name in args.models:
        print(f"\n=== Model: {model_name} ===")
        print(f"Device preference: {args.device}")
        if args.ollama_pull:
            ollama_pull_model(args.ollama_host, model_name)

        effective_device = ollama_device
        for warmup_idx in range(args.warmup):
            try:
                _ = run_ollama_benchmark_with_helpful_errors(
                    host=args.ollama_host,
                    model_name=model_name,
                    prompt=args.prompt,
                    max_new_tokens=min(16, args.max_new_tokens),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=effective_device,
                    runs_command_hint=command_hint,
                )
            except RuntimeError as exc:
                if effective_device == "cuda" and "out of memory" in str(exc).lower():
                    print(
                        f"Warning: '{model_name}' does not fully fit in VRAM. "
                        "Retrying with Ollama auto-fit (partial GPU offload)."
                    )
                    effective_device = "auto"
                    _ = run_ollama_benchmark_with_helpful_errors(
                        host=args.ollama_host,
                        model_name=model_name,
                        prompt=args.prompt,
                        max_new_tokens=min(16, args.max_new_tokens),
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=effective_device,
                        runs_command_hint=command_hint,
                    )
                else:
                    raise
            if warmup_idx == 0 and args.device in {"cuda", "amd"}:
                ollama_check_gpu_layers(args.ollama_host, model_name)

        run_metrics = []
        for run_idx in range(1, args.runs + 1):
            metrics = run_ollama_benchmark_with_helpful_errors(
                host=args.ollama_host,
                model_name=model_name,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=effective_device,
                runs_command_hint=command_hint,
            )
            run_metrics.append(metrics)
            print(
                f"Run {run_idx}: prefill={metrics['prefill_tps']:.2f} tok/s, "
                f"decode={metrics['decode_tps']:.2f} tok/s"
            )

        avg_prefill_tps = sum(item["prefill_tps"] for item in run_metrics) / len(run_metrics)
        avg_decode_tps = sum(item["decode_tps"] for item in run_metrics) / len(run_metrics)
        avg_prompt_tokens = sum(item["prompt_tokens"] for item in run_metrics) / len(run_metrics)
        avg_generated_tokens = sum(item["generated_tokens"] for item in run_metrics) / len(run_metrics)

        model_summaries.append(
            {
                "model": model_name,
                "avg_prefill_tps": avg_prefill_tps,
                "avg_decode_tps": avg_decode_tps,
                "avg_prompt_tokens": avg_prompt_tokens,
                "avg_generated_tokens": avg_generated_tokens,
            }
        )

        ollama_unload_model(args.ollama_host, model_name)

    sort_key = "avg_decode_tps" if args.rank_by == "decode" else "avg_prefill_tps"
    ranked = sorted(model_summaries, key=lambda item: item[sort_key], reverse=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["runs"] = args.runs
        row["prompt"] = args.prompt
        row["max_new_tokens"] = args.max_new_tokens
        row["device"] = args.device
        row["timestamp_utc"] = timestamp

    print("\nRanked Results")
    print(format_table(ranked))

    append_csv(args.out_csv, ranked)
    print(f"\nSaved rankings to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
