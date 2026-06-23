#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_supports_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def resolve_transformers_device(device_arg: str) -> tuple[torch.device, str]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            backend_label = "rocm" if torch_supports_rocm() else "cuda"
            return torch.device("cuda"), backend_label
        return torch.device("cpu"), "cpu"

    if device_arg in {"cuda", "amd"}:
        if not torch.cuda.is_available():
            requested = "AMD GPU (ROCm)" if device_arg == "amd" else "CUDA"
            raise RuntimeError(f"{requested} requested but no compatible GPU runtime is available.")
        if device_arg == "amd" and not torch_supports_rocm():
            raise RuntimeError("AMD GPU requested but this PyTorch build does not include ROCm support.")
        backend_label = "rocm" if torch_supports_rocm() else "cuda"
        return torch.device("cuda"), backend_label

    return torch.device("cpu"), "cpu"


def resolve_ollama_device(device_arg: str) -> str:
    if device_arg == "amd":
        return "cuda"
    return device_arg


def format_device_label(device: torch.device, accelerator_label: str) -> str:
    if device.type != "cuda":
        return str(device)
    return accelerator_label


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def reset_output_file(file_path: str) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8"):
        pass


def append_csv(file_path: str, row: dict) -> None:
    ensure_parent_dir(file_path)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(file_path: str, row: dict) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ollama_post(host: str, path: str, payload: dict, timeout: int = 600) -> dict:
    url = f"{host.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        raise RuntimeError(
            f"Ollama request failed ({exc.code}) at {url}. Response: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to reach Ollama at {url}. Is the Ollama server running?"
        ) from exc


def ollama_pull_model(host: str, model: str) -> None:
    print(f"Pulling Ollama model: {model}")
    _ = ollama_post(host, "/api/pull", {"name": model, "stream": False}, timeout=3600)


def ollama_unload_model(host: str, model: str, timeout: int = 10) -> None:
    """Evict a model from Ollama's memory (VRAM + RAM) and wait until it's gone."""
    print(f"Unloading Ollama model: {model}")
    try:
        ollama_post(host, "/api/generate", {"model": model, "keep_alive": 0})
    except RuntimeError as exc:
        print(f"Warning: could not unload '{model}': {exc}")
        return

    # Poll /api/ps until the model no longer appears as loaded
    url = f"{host.rstrip('/')}/api/ps"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            model_base = model.split(":")[0]
            still_loaded = any(
                e.get("name", "").startswith(model_base)
                for e in data.get("models", [])
            )
            if not still_loaded:
                return
        except Exception:
            pass
        time.sleep(0.2)
    print(f"Warning: '{model}' may still be in memory after {timeout}s")


def ollama_check_gpu_layers(host: str, model_name: str) -> None:
    """Query /api/ps and warn if the model has no GPU layers loaded."""
    url = f"{host.rstrip('/')}/api/ps"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return  # non-fatal: can't reach /api/ps, skip check

    for entry in data.get("models", []):
        if entry.get("name", "").startswith(model_name.split(":")[0]):
            gpu_layers = entry.get("size_vram", 0)
            total_size = entry.get("size", 1)
            if gpu_layers == 0:
                print(
                    f"WARNING: '{model_name}' appears to be running entirely on CPU "
                    f"(0 bytes in VRAM). Pass --device cuda and ensure Ollama has GPU access."
                )
            else:
                pct = 100 * gpu_layers / total_size if total_size else 0
                print(f"GPU check: {gpu_layers / (1024**3):.2f} GiB in VRAM ({pct:.0f}% of model)")


def run_benchmark_ollama(
    host: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str = "auto",
) -> dict:
    options = {
        "num_predict": max_new_tokens,
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p,
    }

    if device == "cpu":
        options["num_gpu"] = 0
    elif device in {"cuda", "amd"}:
        options["num_gpu"] = 999

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    response = ollama_post(host, "/api/generate", payload)

    prompt_tokens = int(response.get("prompt_eval_count", 0))
    generated_tokens = int(response.get("eval_count", 0))
    prefill_time_s = float(response.get("prompt_eval_duration", 0)) / 1e9
    decode_time_s = float(response.get("eval_duration", 0)) / 1e9

    prefill_tps = (prompt_tokens / prefill_time_s) if prefill_time_s > 0 else 0.0
    decode_tps = (generated_tokens / decode_time_s) if decode_time_s > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "generated_text": response.get("response", ""),
    }


def run_benchmark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        sync_if_cuda(device)
        prefill_start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        sync_if_cuda(device)
        prefill_time_s = time.perf_counter() - prefill_start

        logits = outputs.logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        past_key_values = outputs.past_key_values

        # Move the single unavoidable CPU sync outside the timed decode section.
        first_token_id = int(next_token.item())

        generate_kwargs: dict = dict(
            input_ids=next_token,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=do_sample,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        sync_if_cuda(device)
        decode_start = time.perf_counter()
        output_ids = model.generate(**generate_kwargs)
        sync_if_cuda(device)
        decode_time_s = time.perf_counter() - decode_start

    # output_ids shape: [1, 1 + new_tokens]; index 0 is next_token (already timed in prefill).
    all_generated = [first_token_id] + output_ids[0, 1:].tolist()
    generated_tokens = len(all_generated)

    prefill_tps = (prompt_tokens / prefill_time_s) if prefill_time_s > 0 else 0.0
    decode_tps = (generated_tokens / decode_time_s) if decode_time_s > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "generated_text": tokenizer.decode(all_generated, skip_special_tokens=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prompt (prefill) and generation (decode) token throughput for a local LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model ID for selected backend (HF model for transformers, model tag for ollama).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "ollama"],
        help="Inference backend to benchmark.",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--ollama-pull",
        action="store_true",
        help="Pull model from Ollama registry before benchmarking.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how a GPU processes transformer attention in 5 short bullet points.",
        help="Prompt to benchmark.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "amd", "cpu"],
        help=(
            "Execution target. Use 'cuda' for NVIDIA, 'amd' for ROCm-enabled AMD GPUs, "
            "or 'auto' to pick the first available accelerator."
        ),
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model in 4-bit quantization via bitsandbytes (required for 7B on <=6 GB VRAM).",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help=(
            "Enable Flash Attention 2 for the transformers backend (requires flash-attn). "
            "For the ollama backend, restart the Ollama server with OLLAMA_FLASH_ATTENTION=1 instead."
        ),
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--csv", type=str, default="logs/benchmark_metrics.csv")
    parser.add_argument("--jsonl", type=str, default="logs/benchmark_metrics.jsonl")
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Clear output files before writing new benchmark rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reset_output:
        reset_output_file(args.csv)
        reset_output_file(args.jsonl)

    model = None
    tokenizer = None
    device = None
    accelerator_label = "cpu"
    ollama_device = resolve_ollama_device(args.device)
    if args.backend == "transformers":
        device, accelerator_label = resolve_transformers_device(args.device)

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        use_4bit = args.load_in_4bit and device.type == "cuda"

        print(f"Backend: transformers | Loading model: {args.model}")
        print(
            f"Device: {format_device_label(device, accelerator_label)} | "
            f"dtype: {'int4 (bitsandbytes)' if use_4bit else dtype}"
        )

        attn_impl = None
        if args.flash_attention and device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print("Flash Attention 2: enabled")
            except ImportError:
                print("Warning: --flash-attention requested but flash-attn is not installed. Falling back to default attention.")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        fa_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                **fa_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **fa_kwargs,
            )
            model.to(device)
        model.eval()

        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            runtime_name = "ROCm" if accelerator_label == "rocm" else "CUDA"
            print(
                f"GPU ({runtime_name}): {props.name} | "
                f"VRAM: {props.total_memory / (1024**3):.2f} GiB"
            )
    else:
        print(f"Backend: ollama | Model: {args.model} | Host: {args.ollama_host}")
        if args.device == "amd":
            print("Device: amd | Ollama will use GPU offload if its ROCm build is installed.")
        if args.flash_attention:
            print("Note: Flash Attention for Ollama is server-side. Start Ollama with OLLAMA_FLASH_ATTENTION=1 to enable it.")
        if args.ollama_pull:
            ollama_pull_model(args.ollama_host, args.model)

    print(f"Running {args.warmup} warmup run(s)...")
    for _ in range(args.warmup):
        if args.backend == "transformers":
            _ = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=min(16, args.max_new_tokens),
                device=device,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            _ = run_benchmark_ollama(
                host=args.ollama_host,
                model_name=args.model,
                prompt=args.prompt,
                max_new_tokens=min(16, args.max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=ollama_device,
            )

    print(f"Running {args.runs} measured run(s)...")
    all_rows = []

    for run_idx in range(1, args.runs + 1):
        if args.backend == "transformers":
            metrics = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            device_label = format_device_label(device, accelerator_label)
        else:
            metrics = run_benchmark_ollama(
                host=args.ollama_host,
                model_name=args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=ollama_device,
            )
            if args.device == "amd":
                device_label = "ollama-amd"
            elif ollama_device == "cuda":
                device_label = "ollama-gpu"
            else:
                device_label = f"ollama-{ollama_device}"

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run": run_idx,
            "backend": args.backend,
            "model": args.model,
            "device": device_label,
            "prompt_chars": len(args.prompt),
            "prompt_tokens": metrics["prompt_tokens"],
            "generated_tokens": metrics["generated_tokens"],
            "prefill_time_s": round(metrics["prefill_time_s"], 6),
            "decode_time_s": round(metrics["decode_time_s"], 6),
            "prefill_tps": round(metrics["prefill_tps"], 2),
            "decode_tps": round(metrics["decode_tps"], 2),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        append_csv(args.csv, row)
        append_jsonl(args.jsonl, row)
        all_rows.append(row)

        print(
            f"Run {run_idx}: prompt={row['prompt_tokens']} tok in {row['prefill_time_s']}s "
            f"({row['prefill_tps']} tok/s), gen={row['generated_tokens']} tok in "
            f"{row['decode_time_s']}s ({row['decode_tps']} tok/s)"
        )

    avg_prefill = sum(r["prefill_tps"] for r in all_rows) / len(all_rows)
    avg_decode = sum(r["decode_tps"] for r in all_rows) / len(all_rows)

    print("\nAverages:")
    print(f"  Prompt/prefill speed: {avg_prefill:.2f} tok/s")
    print(f"  Generation/decode speed: {avg_decode:.2f} tok/s")
    print(f"\nMetrics saved to: {args.csv} and {args.jsonl}")


if __name__ == "__main__":
    main()
