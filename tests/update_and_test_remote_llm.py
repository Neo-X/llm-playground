#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Text + vision regression test for the *remote* llama.cpp server (onyx),
reached the same way connect-remote-llm.sh reaches it.

This is the remote counterpart to update_and_test_llama_image.py, which
tests a Docker container started on this machine. Here there is no local
container: instead this script opens the same SSH tunnel
connect-remote-llm.sh uses (ssh config alias "<REMOTE_HOST>-llamacpp",
LocalForward 8001 -> remote:8000), launches launch_local_llm.sh on the
remote host non-interactively (no long-lived interactive shell, unlike
connect-remote-llm.sh itself), waits for the tunneled port to come up, and
then runs the same opencode-based checks against the "llama-cpp-onyx"
provider (see ~/.config/opencode/opencode.json).

Usage:
    uv run tests/update_and_test_remote_llm.py
    uv run tests/update_and_test_remote_llm.py --skip-launch   # server already running remotely
    uv run tests/update_and_test_remote_llm.py --keep-server   # leave remote server running after test
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

import requests

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(REPO_DIR, ".env")


def load_env() -> dict:
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


_ENV = load_env()
REMOTE_HOST = _ENV.get("REMOTE_HOST", "onyx")
KERB_PRINCIPAL = _ENV.get("KERB_PRINCIPAL", "")
# connect-remote-llm.sh assumes the repo lives at the same absolute path on
# both machines (it interpolates its own local SCRIPT_DIR into the remote
# ssh command), so we do the same here.
REMOTE_REPO_DIR = REPO_DIR

LLAMACPP_TUNNEL_HOST = f"{REMOTE_HOST}-llamacpp"  # ssh config alias: LocalForward 8001 -> localhost:8000
OPENCODE_PROVIDER = "llama-cpp-onyx"
PORT = 8001
STARTUP_TIMEOUT_S = 600
OPENCODE_TIMEOUT_S = 600
REMOTE_LOG = "/tmp/llama-server-remote-test.log"

# Only models actually wired up under the "llama-cpp-onyx" provider in
# opencode.json can be exercised end-to-end here.
MODELS = [
    {
        "name": "qwen3.6-35b-a3b",
        "quant": "",
        "alias": "qwen3.6-35b-a3b",
        "vision": True,
    },
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kwargs)


def kill_stale_tunnel() -> None:
    run(["pkill", "-f", "ssh.*L 8001"])
    run(["pkill", "-f", "ssh.*8001:localhost"])
    time.sleep(1)


def open_tunnel() -> None:
    kill_stale_tunnel()
    if KERB_PRINCIPAL:
        run(["kinit", "-r", "28d", KERB_PRINCIPAL])
    result = run([
        "ssh", "-f", "-N",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
        "-o", "ExitOnForwardFailure=yes",
        LLAMACPP_TUNNEL_HOST,
    ])
    if result.returncode != 0:
        raise RuntimeError(f"failed to open ssh tunnel {LLAMACPP_TUNNEL_HOST}: {result.stderr.strip()}")


def close_tunnel() -> None:
    kill_stale_tunnel()


def stop_remote_server() -> None:
    run(["ssh", REMOTE_HOST, "docker rm -f llama-vulkan-server llama-cuda-server"], timeout=30)


def start_remote_server(model: dict) -> None:
    stop_remote_server()
    remote_cmd = (
        f"cd {shlex.quote(REMOTE_REPO_DIR)} && "
        f"nohup bash launch_local_llm.sh {shlex.quote(model['name'])} {shlex.quote(model['quant'])} "
        f"> {shlex.quote(REMOTE_LOG)} 2>&1 < /dev/null & disown"
    )
    result = run(["ssh", REMOTE_HOST, remote_cmd], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"failed to launch remote server: {result.stderr.strip()}")


def wait_for_health(timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"http://localhost:{PORT}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


README_REGRESSION_KEYWORDS = ["benchmark", "prompt", "token", "generation"]
README_REGRESSION_MIN_HITS = 2

# Same ground-truth fixture as update_and_test_llama_image.py.
VISION_TEST_IMAGE = os.path.join(REPO_DIR, "tests", "vlaps.png")
VISION_EXPECTED_TEXT = (
    "Pickup the black bowl on the stove and place it on the plate. "
    "root simulated task success VLA Policy VLAPS Action Chunk Sampling Selection world model"
)
VISION_REGRESSION_MIN_RATIO = 0.30


def opencode_binary() -> str:
    return shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")


def _significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2}


def passes_regression_check(summary: str, keywords: list[str], min_hits: int) -> bool:
    lower = summary.lower()
    hits = sum(1 for keyword in keywords if keyword in lower)
    return hits >= min_hits


def vision_overlap_ratio(transcription: str) -> float:
    expected = _significant_words(VISION_EXPECTED_TEXT)
    found = _significant_words(transcription)
    return len(expected & found) / len(expected)


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


def transcribe_vision_image(alias: str) -> str | None:
    cmd = [
        opencode_binary(), "run",
        "--model", f"{OPENCODE_PROVIDER}/{alias}",
        "--dir", REPO_DIR,
        "What text appears in this image? Transcribe it exactly.",
        "-f", VISION_TEST_IMAGE,
        "--auto",
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


def test_model(model: dict, skip_launch: bool) -> bool:
    print(f"\n=== {model['name']} (remote: {REMOTE_HOST}) ===")
    if not skip_launch:
        print("launching remote llama-server via launch_local_llm.sh...")
        try:
            start_remote_server(model)
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return False

    print(f"waiting up to {STARTUP_TIMEOUT_S}s for remote model to load...")
    if not wait_for_health(STARTUP_TIMEOUT_S):
        print("FAIL: remote server did not become healthy in time")
        print(run(["ssh", REMOTE_HOST, f"tail -n 40 {shlex.quote(REMOTE_LOG)}"]).stdout)
        return False

    print("health OK, asking opencode to summarize README.md...")
    summary = summarize_readme(model["alias"])
    if summary is None:
        print("FAIL: opencode did not return a summary")
        return False

    if not passes_regression_check(summary, README_REGRESSION_KEYWORDS, README_REGRESSION_MIN_HITS):
        print(f"FAIL: summary didn't mention enough expected keywords {README_REGRESSION_KEYWORDS}:\n{summary}")
        return False

    print(f"PASS: opencode summary:\n{summary}")

    if model.get("vision"):
        print("asking opencode to transcribe the vision regression image (tests/vlaps.png)...")
        transcription = transcribe_vision_image(model["alias"])
        if transcription is None:
            print("FAIL: opencode did not return an image transcription")
            return False

        ratio = vision_overlap_ratio(transcription)
        if ratio < VISION_REGRESSION_MIN_RATIO:
            print(f"FAIL: only {ratio:.0%} word overlap with expected text (need >= {VISION_REGRESSION_MIN_RATIO:.0%}):\n{transcription}")
            return False

        print(f"PASS: opencode transcription ({ratio:.0%} word overlap):\n{transcription}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-launch", action="store_true", help="assume the remote server is already running")
    parser.add_argument("--keep-server", action="store_true", help="don't stop the remote server after testing")
    args = parser.parse_args()

    print(f"opening ssh tunnel {LLAMACPP_TUNNEL_HOST} (localhost:{PORT} -> remote:8000)...")
    try:
        open_tunnel()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1

    results = {}
    try:
        for model in MODELS:
            results[model["name"]] = test_model(model, args.skip_launch)
    finally:
        if not args.keep_server:
            stop_remote_server()
        close_tunnel()

    print("\n=== summary ===")
    all_passed = True
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        all_passed = all_passed and passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
