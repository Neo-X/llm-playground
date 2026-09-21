#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Regression test for sweep_models.py's vLLM backend, using qwen2.5:1.5b
(small/fast, already confirmed working -- see README.md section 7, "Serving
with vLLM").

This complements tests/update_and_test_vllm_image.py, which checks that the
vLLM *server* (BACKEND=vllm in launch_local_llm.sh) comes up and serves real
completions. This test instead checks the *benchmarking* tooling on top of
it: that `sweep_models.py --backends vllm` launches the server, drives it via
benchmark_llm_speed.run_benchmark_vllm_server, tears the server down again,
and reports non-zero prefill/decode throughput in its output CSV.

Usage:
    uv run tests/test_benchmark_vllm.py
"""

import csv
import os
import subprocess
import sys
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_TAG = "qwen2.5:1.5b"
MODEL_FILE = f"{REPO_DIR}/models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf"
# Generous: covers a from-scratch vllm-openai-gguf image build (first run
# only) plus model load; typically well under a minute once cached.
SWEEP_TIMEOUT_S = 1200


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_DIR, check=False, capture_output=True, text=True, **kwargs)


def cleanup_server() -> None:
    run(["docker", "rm", "-f", "vllm-server", "llama-vulkan-server", "llama-cuda-server"])


def main() -> int:
    if not os.path.exists(MODEL_FILE):
        print(f"SKIP: model file not found: {MODEL_FILE}")
        return 0

    cleanup_server()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_csv = os.path.join(tmp_dir, "sweep.csv")
        out_png = os.path.join(tmp_dir, "sweep.png")

        print(f"Running: sweep_models.py --models {MODEL_TAG} --backends vllm ...")
        try:
            result = subprocess.run(
                [
                    "uv", "run", "python", "sweep_models.py",
                    "--models", MODEL_TAG,
                    "--backends", "vllm",
                    "--max-new-tokens", "32",
                    "--runs", "1",
                    "--warmup", "1",
                    "--vllm-ready-timeout", str(SWEEP_TIMEOUT_S),
                    "--out-csv", out_csv,
                    "--out-png", out_png,
                ],
                cwd=REPO_DIR,
                check=False,
                capture_output=True,
                text=True,
                timeout=SWEEP_TIMEOUT_S + 120,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"FAIL: sweep_models.py timed out: {exc}")
            return 1
        finally:
            cleanup_server()

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"FAIL: sweep_models.py exited with code {result.returncode}")
            return 1

        if not os.path.exists(out_csv):
            print("FAIL: sweep_models.py did not write the output CSV")
            return 1

        with open(out_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        vllm_rows = [r for r in rows if r.get("backend") == "vllm" and r.get("model") == MODEL_TAG]
        if not vllm_rows:
            print(f"FAIL: no vllm row for '{MODEL_TAG}' in output CSV. Rows: {rows}")
            return 1

        row = vllm_rows[0]
        prefill_tps = float(row["avg_prefill_tps"])
        decode_tps = float(row["avg_decode_tps"])

        if prefill_tps <= 0 or decode_tps <= 0:
            print(f"FAIL: non-positive throughput reported: prefill={prefill_tps} decode={decode_tps}")
            return 1

        print(f"PASS: {MODEL_TAG} via vllm -- prefill {prefill_tps:.1f} tok/s, decode {decode_tps:.1f} tok/s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
