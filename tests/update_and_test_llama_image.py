#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Standard process for updating the llama.cpp Vulkan Docker image and
verifying it still serves the models used day-to-day.

Vulkan is used (rather than CUDA/ROCm images) so the exact same image works
unmodified on both the AMD and NVIDIA machines in rotation.

Usage:
    uv run update_and_test_llama_image.py                  # pull + test all models
    uv run update_and_test_llama_image.py --skip-pull       # test the image already on disk
    uv run update_and_test_llama_image.py --keep-good-only  # pull, test, and only replace
                                                              # the "known-good" tag on success
"""

import argparse
import grp
import os
import re
import shutil
import subprocess
import sys
import time

import requests

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = f"{REPO_DIR}/models"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-vulkan"
KNOWN_GOOD_TAG = "ghcr.io/ggml-org/llama.cpp:server-vulkan-known-good"
CONTAINER_NAME = "llama-vulkan-image-test"
OPENCODE_PROVIDER = "llama-cpp-test"
PORT = 8079
STARTUP_TIMEOUT_S = 600
OPENCODE_TIMEOUT_S = 180

MODELS = [
    {
        "name": "qwen3.6-35b-a3b",
        "path": f"{MODELS_DIR}/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "mmproj": f"{MODELS_DIR}/qwen3.6-35b-a3b/mmproj-F16.gguf",
        "alias": "qwen3.6-35b-a3b",
    },
    {
        "name": "qwen3.6-27b",
        "path": f"{MODELS_DIR}/qwen3.6-27b/Qwen3.6-27B-Q8_0.gguf",
        "mmproj": f"{MODELS_DIR}/qwen3.6-27b/mmproj-F16.gguf",
        "alias": "qwen3.6-27b",
    },
    {
        "name": "deepseek-v4-flash-q8",
        "path": f"{MODELS_DIR}/DeepSeek-V4-Flash-Q8/Q8_0/DeepSeek-V4-Flash-Q8_0-00001-of-00007.gguf",
        "mmproj": None,
        "alias": "deepseek-v4-flash-q8",
    },
]


def gpu_docker_flags() -> list[str]:
    flags = ["--device", "/dev/dri"]
    for group_name in ("render", "video"):
        try:
            flags += ["--group-add", str(grp.getgrnam(group_name).gr_gid)]
        except KeyError:
            pass
    if shutil.which("nvidia-smi"):
        flags += ["--gpus", "all"]
    return flags


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kwargs)


def remove_test_container() -> None:
    run(["docker", "rm", "-f", CONTAINER_NAME])


def start_container(image: str, model: dict) -> None:
    remove_test_container()
    cmd = [
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        *gpu_docker_flags(),
        "-v", f"{MODELS_DIR}:{MODELS_DIR}",
        "-p", f"{PORT}:8000",
        image,
        "-m", model["path"],
        "--alias", model["alias"],
    ]
    if model.get("mmproj"):
        cmd += ["--mmproj", model["mmproj"], "--image-min-tokens", "1024"]
    cmd += [
        "-ngl", "999", "--no-mmap", "--ctx-size", "32768", "-np", "1",
        "--host", "0.0.0.0", "--port", "8000", "--jinja",
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr.strip()}")


def wait_for_health(timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        still_running = run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
        if still_running.stdout.strip() != "true":
            return False
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

# Ground-truth text for the vision regression fixture (tests/vlaps.png). A
# transcription is considered a pass once it recovers at least 30% of these
# words, since exact OCR-style transcription varies across quants/models.
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


def test_model(image: str, model: dict) -> bool:
    print(f"\n=== {model['name']} ===")
    print(f"starting container with {model['path']}...")
    try:
        start_container(image, model)
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return False

    print(f"waiting up to {STARTUP_TIMEOUT_S}s for model to load...")
    if not wait_for_health(STARTUP_TIMEOUT_S):
        print("FAIL: server did not become healthy in time")
        print(run(["docker", "logs", "--tail", "40", CONTAINER_NAME]).stdout)
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

    if model.get("mmproj"):
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
    parser.add_argument("--image", default=IMAGE, help="image tag to pull and test")
    parser.add_argument("--skip-pull", action="store_true", help="test the locally cached image as-is")
    args = parser.parse_args()
    image = args.image

    if not args.skip_pull:
        print(f"pulling {image}...")
        result = run(["docker", "pull", image])
        if result.returncode != 0:
            print(f"FAIL: docker pull failed: {result.stderr.strip()}")
            return 1

    results = {}
    try:
        for model in MODELS:
            results[model["name"]] = test_model(image, model)
    finally:
        remove_test_container()

    print("\n=== summary ===")
    all_passed = True
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\nall models OK, tagging {image} as {KNOWN_GOOD_TAG}")
        run(["docker", "tag", image, KNOWN_GOOD_TAG])
    else:
        print(f"\nnot promoting to known-good tag; {KNOWN_GOOD_TAG} unchanged")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
