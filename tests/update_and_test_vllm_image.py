#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Regression test for the `BACKEND=vllm` path in launch_local_llm.sh (see
README.md section 7, "Serving with vLLM").

This is the vLLM counterpart to update_and_test_llama_image.py: it launches
launch_local_llm.sh with BACKEND=vllm (which builds the local
vllm-openai-gguf:local image on first run -- vLLM's official image doesn't
ship GGUF support), waits for the OpenAI-compatible server to come up on
port 8020, sends it a direct /v1/completions request, and then runs the
same opencode-based README-summarization regression check the llama.cpp
tests use, against the "vllm" provider (see opencode.json).

qwen2.5-3b and qwen3.8-27b are exercised here. qwen3.6-35b-a3b/qwen3.6-27b
use the same qwen3_5 GGUF architecture as qwen3.8-27b (see the plugin-pin
note in launch_local_llm.sh) so they're expected to work too but aren't
included, to keep this test's runtime down. llama3.2-3b needs a gated
HF_TOKEN, so it isn't included by default either.

Usage:
    uv run tests/update_and_test_vllm_image.py
    uv run tests/update_and_test_vllm_image.py --skip-opencode   # health + completion only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import requests

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = f"{REPO_DIR}/models"
LAUNCH_SCRIPT = os.path.join(REPO_DIR, "launch_local_llm.sh")
CONTAINER_NAME = "vllm-server"
OPENCODE_PROVIDER = "vllm"
PORT = 8020
# Generous: first run also builds the vllm-openai-gguf:local image (pulls
# the ~8GB base image, installs vllm-gguf-plugin) before the model loads.
STARTUP_TIMEOUT_S = 1200
OPENCODE_TIMEOUT_S = 600
LOG_PATH = "/tmp/vllm-image-test.log"

MODELS = [
    {
        "name": "qwen2.5-3b",
        "path": f"{MODELS_DIR}/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf",
        "alias": "qwen2.5-3b",
    },
    {
        "name": "qwen3.8-27b",
        "path": f"{MODELS_DIR}/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "alias": "qwen3.8-27b",
    },
]

README_REGRESSION_KEYWORDS = ["benchmark", "prompt", "token", "generation"]
# qwen2.5-3b is much smaller than the models the llama.cpp regression tests
# use, and paraphrases more instead of reusing these exact words -- 1 hit is
# still enough to confirm it actually read and summarized the file.
README_REGRESSION_MIN_HITS = 1


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kwargs)


def remove_test_container() -> None:
    run(["docker", "rm", "-f", CONTAINER_NAME])


def start_server(model_name: str) -> subprocess.Popen:
    remove_test_container()
    env = os.environ.copy()
    env["BACKEND"] = "vllm"
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    return subprocess.Popen(
        ["bash", LAUNCH_SCRIPT, model_name],
        cwd=REPO_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def stop_server(process: subprocess.Popen) -> None:
    remove_test_container()
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_ready(process: subprocess.Popen, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://localhost:{PORT}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def check_completion(alias: str) -> str | None:
    """Direct /v1/completions sanity check -- confirms the server is
    actually generating text, independent of opencode's own plumbing."""
    payload = {
        "model": alias,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        resp = requests.post(f"http://localhost:{PORT}/v1/completions", json=payload, timeout=60)
    except requests.RequestException as e:
        print(f"FAIL: completion request failed: {e}")
        return None
    if resp.status_code != 200:
        print(f"FAIL: completion request returned {resp.status_code}: {resp.text[:500]}")
        return None
    choices = resp.json().get("choices") or []
    if not choices or not choices[0].get("text", "").strip():
        print(f"FAIL: completion response had no text: {resp.text[:500]}")
        return None
    return choices[0]["text"]


def opencode_binary() -> str:
    return shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")


def passes_regression_check(summary: str, keywords: list[str], min_hits: int) -> bool:
    lower = summary.lower()
    hits = sum(1 for keyword in keywords if keyword in lower)
    return hits >= min_hits


def summarize_readme(alias: str) -> str | None:
    cmd = [
        opencode_binary(), "run",
        "--model", f"{OPENCODE_PROVIDER}/{alias}",
        "--dir", REPO_DIR,
        "-f", "README.md",
        "--auto",
        "Summarize this file in 3-4 sentences.",
    ]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=OPENCODE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def test_model(model: dict, skip_opencode: bool) -> bool:
    print(f"\n=== {model['name']} (BACKEND=vllm) ===")
    if not os.path.exists(model["path"]):
        print(f"SKIP: model file not found: {model['path']}")
        return False

    print(f"launching launch_local_llm.sh {model['name']} with BACKEND=vllm...")
    process = start_server(model["name"])
    try:
        print(f"waiting up to {STARTUP_TIMEOUT_S}s for vLLM to build/load (first run also builds the image)...")
        if not wait_for_ready(process, STARTUP_TIMEOUT_S):
            print("FAIL: server did not become ready in time")
            print(run(["tail", "-n", "60", LOG_PATH]).stdout)
            return False

        print("server ready, sending a direct /v1/completions request...")
        completion = check_completion(model["alias"])
        if completion is None:
            return False
        print(f"PASS: completion text: {completion!r}")

        if skip_opencode:
            return True

        print("asking opencode to summarize README.md via the 'vllm' provider...")
        summary = summarize_readme(model["alias"])
        if summary is None:
            print("FAIL: opencode did not return a summary")
            return False

        if not passes_regression_check(summary, README_REGRESSION_KEYWORDS, README_REGRESSION_MIN_HITS):
            print(f"FAIL: summary didn't mention enough expected keywords {README_REGRESSION_KEYWORDS}:\n{summary}")
            return False

        print(f"PASS: opencode summary:\n{summary}")
        return True
    finally:
        stop_server(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-opencode", action="store_true",
        help="only run the health + direct /v1/completions check, skip the opencode regression check",
    )
    args = parser.parse_args()

    results = {}
    for model in MODELS:
        results[model["name"]] = test_model(model, args.skip_opencode)

    print("\n=== summary ===")
    all_passed = True
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        all_passed = all_passed and passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
