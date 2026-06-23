# LLMs Performance Analysis

This note outlines performance analysis for the best open weight LLM that can be used on a laptop with 128GB of memory. Either the AMD AI Max 395 or a MacBook Pro. The goal is to get the best local Claude Code performance.

---

## Current Laptop: ASUS ProArt PX13 (HN7306)

| Spec | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M |
| RAM | 32 GB |
| GPU | RTX 4050 Laptop GPU, 6 GB VRAM |
| GPU Memory Bandwidth | ~192 GB/s (VRAM only) |
| System RAM Bandwidth | ~68 GB/s (LPDDR5X) |

### What You Can Run Now

| Model Size | Fits Where | Speed | Quality |
|---|---|---|---|
| 7B (Q4_K_M, ~4 GB) | GPU VRAM (6 GB) | ~35-50 t/s | Decent for simple tasks |
| 13B (Q4_K_M, ~8 GB) | Partially GPU + RAM split | ~10-20 t/s | Degraded (CPU bottleneck) |
| 32B (Q4_K_M, ~18 GB) | RAM only (CPU inference) | ~3-6 t/s | OK quality, painful speed |
| 70B (Q4_K_M, ~40 GB) | **Does NOT fit** — 40 GB > 32 GB RAM | — | — |

**Hard limits of your current machine:**
- 6 GB VRAM means only 7B models run fully on GPU (fast). Larger models spill to RAM/CPU and become slow.
- 32 GB RAM means 70B+ models are completely out of reach at any quantization.
- Best realistic coding model right now: **Qwen2.5-Coder-7B at Q8_0** on GPU (~4-5 GB VRAM, ~40-50 t/s).
- Running 32B models is possible but at ~4 t/s — barely usable for agentic coding workflows.

---

## Hardware Comparison

| Spec | Apple M4 Max (128GB) | AMD Ryzen AI Max+ 395 (128GB) | NVIDIA RTX Pro 6000 (96GB VRAM) |
|---|---|---|---|
| Memory Bandwidth | ~546 GB/s | ~215 GB/s (real-world) | ~1,792 GB/s (VRAM only) |
| Total Memory | 128 GB unified | 128 GB unified | 96 GB GDDR7 VRAM + host RAM |
| GPU | 40-core GPU (Metal) | Radeon 8060S, 40 CUs (RDNA 3.5) | Blackwell, 24,064 CUDA cores |
| NPU | 38 TOPS | 50 TOPS (XDNA 2) | — |
| Power draw (AI load) | ~48W | ~95-130W | ~600W TDP |
| Form factor | Laptop / integrated | Laptop / integrated | Desktop workstation PCIe |

**Key insight:** LLM inference is memory-bandwidth-bound. The RTX Pro 6000's 1,792 GB/s VRAM bandwidth is 3.3× the M4 Max and nearly 8× the AMD laptop — translating directly to higher tokens/sec for any model that fits in VRAM. With 96 GB, it holds all mainstream models (up to ~80B at Q8_0, or ~190B at Q4_K_M) fully in fast VRAM with no CPU offloading required.

---

## Best Models for 128GB Systems

### Tier 1: Top Picks for Coding (early 2026)

**Qwen3-Coder-Next 80B MoE** — Best overall for agentic coding
- SWE-Bench: >70% (rank #1, beats Claude Opus 4)
- MoE architecture: only ~3B active params/token → fast despite 80B total
- Fits at Q4_K_M (~46GB) or Q8_0 (~85GB) in 128GB
- Built specifically for agentic coding workflows

**Qwen2.5-Coder-32B-Instruct** — Best practical workhorse
- HumanEval: ~72%, LiveCodeBench: 70.7, SWE-bench: competitive with GPT-4o
- Fits at any quantization; ~30-45 t/s on M4 Max with MLX
- Best choice for code generation and review

**DeepSeek-R1-Distill-Llama-70B** — Best for reasoning/debugging
- LiveCodeBench coding score: 57.5 (highest of all R1 distill variants)
- ~40GB at Q4_K_M; ~70GB at Q8_0 — plenty of headroom in 128GB
- Best for complex algorithmic problems and architecture reasoning

**Qwen3.5-35B-A3B MoE** — Best for code chat
- 35B total, ~3B active per token — very fast
- 262K context window, multimodal support
- Great for PR reviews, explaining code, design discussions

### Tier 2: Notable Alternatives

| Model | Notes |
|---|---|
| Llama 3.3 70B | Strong general model, solid at code chat, ~40GB at Q4 |
| Qwen2.5-72B-Instruct | Good general/code hybrid; fits at Q4 (~45GB) |
| Qwen2.5-Coder 7B/14B | Fast autocomplete; excellent for resource-constrained slots |
| DeepSeek V3/V3.1 | 671B MoE — too large for 128GB at useful quants |

---

## Performance Benchmarks

### Apple M4 Max 128GB

| Model | Quantization | Runtime | Tokens/sec |
|---|---|---|---|
| Qwen3-Coder-Next 80B MoE | Q4_K_M | MLX | ~60 t/s |
| Qwen3-Coder-Next 80B MoE | Q4_K_M | llama.cpp | ~24 t/s |
| Qwen2.5-Coder-32B | Q4_K_M | MLX | ~30-45 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q4_K_M | llama.cpp | ~20-25 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q8_0 | llama.cpp | ~15-18 t/s |
| Qwen2.5-Coder-14B | Q8_0 | MLX | ~55-65 t/s |
| Llama 3.1 8B | Q4_K_M | MLX | ~96-100 t/s |

Time to first token: 1-3s for 32B models, 3-6s for 70B models.

### AMD Ryzen AI Max+ 395 128GB

| Model | Quantization | Runtime | Tokens/sec |
|---|---|---|---|
| Qwen3-Coder-Next 80B MoE | Q4_K_M | llama.cpp | ~15-20 t/s |
| Qwen2.5-Coder-32B | Q4_K_M | llama.cpp | ~12-18 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q4_K_M | llama.cpp + Flash Attn | ~5-7 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q8_0 | LM Studio | ~3-5 t/s |
| Qwen2.5-Coder-14B | Q8_0 | llama.cpp | ~25-35 t/s |
| 7-8B models | Q4_K_M | llama.cpp | ~45-60 t/s |

**Critical AMD note:** Always enable `--flash-attn` in llama.cpp/LM Studio for 70B models. Without it, performance drops to ~3 t/s.

#### Measured Benchmarks — Qwen3.5 Series (ROCm 7.2, llama.cpp, HIP backend)

Source: [Reddit r/LocalLLaMA — Ryzen AI Max+ 395 128GB benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rpw17y/ryzen_ai_max_395_128gb_qwen_35_35b122b_benchmarks/)
System: Fedora, kernel 6.18.13, ROCm 7.2, llama.cpp commit d417bc43

| Model | Quant | Model Size | Prompt (pp512 t/s) | Gen (tg128 t/s) |
|---|---|---|---|---|
| Qwen3.5-0.8B | Q4_K_XL | 0.52 GB | 5,968 | 176 |
| Qwen3.5-0.8B | Q8_K_XL | 1.09 GB | 5,845 | 106 |
| Qwen3.5-0.8B | BF16 | 1.40 GB | 5,537 | 87 |
| Qwen3.5-4B | Q4_K_XL | 2.70 GB | 1,408 | 45 |
| Qwen3.5-9B | Q4_K_XL | 5.55 GB | 918 | 29 |
| Qwen3.5-27B | Q4_K_XL | 16.40 GB | 264 | 10 |
| Qwen3.5-35B A3B (MoE) | Q4_K_XL | 20.70 GB | 887 | **40** |
| Qwen3.5-122B A10B (MoE) | Q4_K_XL | 63.65 GB | 268 | **21** |
| GLM-4.7-Flash (30B) | Q4_K_XL | 16.31 GB | 917 | 46 |
| GPT-OSS-120B | Q8_K_XL | 60.03 GB | 499 | 42 |

**Key takeaway:** MoE models punch far above their weight on AMD. The 35B MoE hits 40 t/s — same as a small 4B dense model — because only 3B params are active per token. The 122B MoE at 21 t/s is competitive with the M4 Max for the same model. Dense 27B at only 10 t/s confirms AMD struggles with dense large models due to lower memory bandwidth.

#### Measured Benchmarks — Large Models (Vulkan + FlashAttention backend)

Vulkan + FA outperforms HIP for prompt processing on this hardware:

| Model | Quant | Size | Prompt (pp512 t/s) | Gen (tg128 t/s) |
|---|---|---|---|---|
| Qwen3-30B-A3B (MoE) | UD-Q4_K_XL | 16.5 GB | 70.03 | **75.32** |
| Llama 4 Scout (109B MoE) | — | 57.93 GB | 102.61 | 20.23 |

**Qwen3-30B-A3B at 75 t/s** is the standout: a 30B MoE model with only 3B active params, running faster than most 7B dense models on this hardware.

#### Backend Comparison (Llama-2-7B Q4_0, pp512/tg128)

| Backend | Prompt (t/s) | Gen (t/s) |
|---|---|---|
| CPU only | 295 | 29 |
| HIP (ROCm) | 349 | 49 |
| Vulkan + FlashAttention | **884** | **53** |
| HIP + WMMA + FA | 344 | 51 |

**Use Vulkan + FlashAttention** for prompt-heavy workloads. HIP + WMMA + FA is comparable for generation. Vulkan FA is the best all-round choice on this hardware.

### NVIDIA RTX Pro 6000 Blackwell (96GB VRAM)

Runtime: llama.cpp (CUDA), vLLM, or TensorRT-LLM. With 96 GB GDDR7 and 1,792 GB/s bandwidth, virtually all mainstream open-weight models run entirely in VRAM at full speed. Benchmarks below are measured single-user generation speeds.

| Model | Quantization | VRAM fit? | Runtime | Tokens/sec |
|---|---|---|---|---|
| Qwen3-Coder-Next 80B MoE | Q4_K_M | ✅ ~46 GB — ~50 GB free for KV cache | llama.cpp CUDA | ~130 t/s |
| Qwen2.5-Coder-32B | Q4_K_M | ✅ ~18 GB | llama.cpp CUDA | ~60-70 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q4_K_M | ✅ ~40 GB — plenty of KV headroom | llama.cpp CUDA | ~25-35 t/s |
| DeepSeek-R1-Distill-Llama-70B | Q8_0 | ✅ ~70 GB — fits with ~26 GB to spare | llama.cpp CUDA | ~15-20 t/s |
| Qwen2.5-Coder-14B | Q8_0 | ✅ ~14 GB | llama.cpp CUDA | ~82 t/s |
| Llama 3.1 8B | Q4_K_M | ✅ ~4.5 GB | llama.cpp CUDA | ~244 t/s |

**VRAM headroom with 96 GB:**
- ✅ All models up to ~80B at Q8_0 (~80 GB) fit fully — no CPU offloading at any mainstream model size
- ✅ 80B MoE Q4_K_M (~46 GB) leaves ~50 GB for KV cache → can sustain very long contexts (100K+ tokens)
- ✅ 125B MoE Q4_K_M (~62 GB) fits with ~34 GB to spare
- ⚠️ 80B dense Q8_0 (~85 GB) is tight; use Q6_K (~66 GB) for headroom
- ❌ Models above ~190B at Q4 or ~120B at Q8 exceed 96 GB

**Recommended runtime for RTX Pro 6000:**
- **llama.cpp (CUDA build)** — best for interactive single-user use; easy model switching
- **vLLM** — best for multi-user or batch throughput; significantly higher concurrency throughput
- **TensorRT-LLM** — maximum single-stream performance; requires model compilation per version
- **Ollama** — simplest setup; CUDA auto-detected; compatible with Claude Code via Anthropic API mode

---

## Quantization Guide

| Format | Bits | 70B size | Quality loss | Recommendation |
|---|---|---|---|---|
| Q8_0 | 8-bit | ~70 GB | ~0% | Best quality; use when headroom allows |
| Q6_K | 6-bit | ~54 GB | ~0.1% | Near-lossless, excellent tradeoff |
| Q5_K_M | 5-bit | ~48 GB | ~0.2% | Best for reasoning/debugging tasks |
| Q4_K_M | 4-bit | ~40 GB | ~0.5% | **Best practical default** |
| Q3_K_M | 3-bit | ~31 GB | ~1.5% | Acceptable only for simple tasks |
| Q2_K | 2-bit | ~23 GB | >5% | Avoid for code generation |

With 128GB, you can run 70B at Q8_0 (~70GB) and still have 50+GB free. Use the headroom — run Q8_0 or Q5_K_M rather than Q4.

---

## Recommendations by Task

| Task | Model | Quant | Why |
|---|---|---|---|
| Agentic coding (Claude Code-style) | Qwen3-Coder-Next 80B MoE | Q4_K_M | #1 SWE-bench, MoE = fast |
| Code generation / single file | Qwen2.5-Coder-32B | Q8_0 | Best proven code gen, fast at 32B |
| Debugging / algorithm reasoning | DeepSeek-R1-Distill-Llama-70B | Q5_K_M | Best chain-of-thought for code |
| Inline autocomplete (FIM) | Qwen2.5-Coder-14B | Q8_0 | Fast, state-of-art FIM |
| Code chat / architecture review | Qwen3.5-35B-A3B MoE | Q4_K_M | Long context, fast due to MoE |

---

## Runtimes & Tools

### Apple Silicon

| Tool | Notes |
|---|---|
| **MLX** | Top recommendation; 2-3x faster than llama.cpp on Apple Silicon. `pip install mlx-lm`. Models at `mlx-community/` on HuggingFace. |
| **LM Studio** | Best GUI; supports MLX backend natively. Easy OpenAI-compatible API server. |
| **Ollama** | Best for dev toolchain integration. Supports Anthropic Messages API (Jan 2026) — Claude Code can connect directly. Metal GPU built in. |
| **llama.cpp** | Maximum control; use for benchmarking or custom KV cache tuning. |

### AMD Ryzen AI Max+ 395

| Tool | Notes |
|---|---|
| **LM Studio + llama.cpp** | Best option; ROCm support matured in 2025. Always enable Flash Attention. |
| **Ollama** | Simpler setup, ROCm built in, slightly less tunable. |
| **llama.cpp (ROCm build)** | Best raw performance; may need `HSA_OVERRIDE_GFX_VERSION=11.0.0`. |
| MLX | Apple Silicon only — not available on AMD. |

### NVIDIA RTX Pro 6000 Blackwell

| Tool | Notes |
|---|---|
| **llama.cpp (CUDA build)** | Best for interactive single-user inference. Straightforward setup, wide model support. |
| **vLLM** | Best for high-throughput / multi-user serving. Significantly better concurrency than llama.cpp. |
| **TensorRT-LLM** | Highest single-stream peak with Blackwell 5th-gen Tensor Core optimizations; requires model compilation. |
| **Ollama** | Simplest setup; CUDA auto-detected; supports Claude Code via Anthropic Messages API mode. |
| **LM Studio** | GUI option; CUDA backend, easy OpenAI-compatible API server. |

### IDE Integration

| Tool | Best for | Backend |
|---|---|---|
| **Continue.dev** | VS Code / JetBrains inline chat + autocomplete | Ollama, LM Studio |
| **Aider** | Terminal agentic editing, git-aware | Any OpenAI-compatible API |
| **Claude Code** (local) | Agentic multi-file editing | Ollama (Anthropic API mode) |
| **Cline** | VS Code agentic agent | Any OpenAI-compatible API |

---

## Current vs New: What You Gain

### Model Access

| Model | Current PX13 (32GB) | New 128GB Machine |
|---|---|---|
| Qwen2.5-Coder 7B Q8_0 | ✅ GPU, ~45 t/s | ✅ faster |
| Qwen2.5-Coder 14B Q4 | ⚠️ RAM-only, ~8 t/s | ✅ ~55 t/s |
| Qwen2.5-Coder 32B Q4 | ⚠️ RAM-only, ~4 t/s | ✅ ~35-45 t/s |
| DeepSeek-R1-Distill 70B Q4 | ❌ doesn't fit | ✅ ~18-25 t/s (M4 Max) |
| Qwen3-Coder-Next 80B MoE Q4 | ❌ doesn't fit | ✅ ~60 t/s (M4 Max MLX) |
| Any model at Q8_0 for quality | ❌ only 7B fits | ✅ up to 70B at Q8_0 |

### Speed Comparison (Qwen2.5-Coder 7B — same model, best case on current)

| Machine | Tokens/sec | Notes |
|---|---|---|
| Current PX13 (RTX 4050, 6GB) | ~45 t/s | GPU inference, 7B only |
| AMD AI Max+ 395 (128GB) | ~60 t/s | Any model up to 80B |
| M4 Max (128GB) | ~95-100 t/s | MLX backend |
| M5 Max (128GB) | ~107-115 t/s | MLX backend |
| RTX Pro 6000 Blackwell (96GB VRAM) | ~244 t/s | llama.cpp CUDA; fully in VRAM |

**RTX Pro 6000 vs 128GB laptops:** The Pro 6000 wins at every model size due to its 1,792 GB/s VRAM bandwidth (3.3× M4 Max). Unlike the laptop platforms, it also runs 70B Q8_0 and 80B MoE fully in VRAM with no offloading. The trade-off is 600W power draw and a desktop workstation requirement.

### Practical Impact for Claude Code-style Workflows

| Capability | Current PX13 | New 128GB Machine |
|---|---|---|
| Inline autocomplete | ✅ Good (7B fast) | ✅ Better (14B fast) |
| Single-function generation | ⚠️ OK (7B quality limits) | ✅ Excellent (32B+) |
| Multi-file agentic editing | ❌ Too slow at 32B (~4 t/s) | ✅ Practical (60+ t/s) |
| Code reasoning / debugging | ❌ 70B unreachable | ✅ DeepSeek-R1 70B |
| Long context (100K+ tokens) | ❌ 7B context limits | ✅ Qwen 262K context |

**Bottom line:** Your current machine is limited to 7B models for any real-time use, and 32B models at speeds too slow (~4 t/s) for agentic workflows. A 128GB machine unlocks 70-80B models at 18-60 t/s — the difference between a slow 7B assistant and a state-of-the-art coding agent that matches or beats GPT-4o.

---

## Recommended Stack

```
Model:   Qwen3-Coder-Next 80B MoE @ Q4_K_M  (agentic)
         + Qwen2.5-Coder-14B @ Q8_0          (inline autocomplete)
Runtime: MLX (Apple) or llama.cpp/ROCm with --flash-attn (AMD)
Server:  Ollama or LM Studio (OpenAI API mode)
IDE:     Aider or Cline (agentic) + Continue.dev (completion)
```

For Claude Code specifically: Ollama's Anthropic Messages API support (Jan 2026) allows Claude Code to use any local Ollama model as its backend.

---

## Models from Artificial Analysis Leaderboard (open-source)

Source: https://artificialanalysis.ai/models/open-source — Intelligence Index v4.0 (10 evals: reasoning, knowledge, math, coding). ~247 models total.

### Which Models Can Run Locally on Each Machine?

Key sizing rule at Q4_K_M: ~0.5 bytes/param → 128GB fits up to ~250B total params; 32GB fits up to ~55B total params; 6GB VRAM fits up to ~12B.

| Model                             | Total Params | Active Params | Q4 Size | Current PX13 (32GB)  | MacBook Pro 128GB     | AMD AI 395 128GB      | Intelligence Index |
| --------------------------------- | ------------ | ------------- | ------- | -------------------- | --------------------- | --------------------- | ------------------ |
| Qwen3.5 4B                        | 4.7B         | 4.7B          | ~2.4 GB | ✅ GPU (fast)         | ✅ ~100+ t/s (MLX)    | ✅ ~60-80 t/s          | 27                 |
| Qwen3.5 9B                        | 9.7B         | 9.7B          | ~4.9 GB | ✅ GPU (fast)         | ✅ ~80-90 t/s (MLX)   | ✅ ~50-65 t/s          | 32                 |
| Apriel-v1.6-15B                   | 15B          | 15B           | ~7.5 GB | ⚠️ CPU-only (~6 t/s) | ✅ ~60-70 t/s (MLX)   | ✅ ~35-45 t/s          | 28                 |
| Qwen3.5 27B                       | 27.8B        | 27.8B         | ~14 GB  | ⚠️ CPU-only (~3 t/s) | ✅ ~40-50 t/s (MLX)   | ✅ ~18-25 t/s          | 42                 |
| Qwen3.5 35B A3B (MoE)             | 36B          | 3B            | ~18 GB  | ⚠️ CPU-only (~3 t/s) | ✅ ~150+ t/s (MLX)    | ✅ ~50-60 t/s          | 37                 |
| Qwen3-30B-A3B (MoE)               | 30B          | 3B            | ~16 GB  | ⚠️ CPU-only (~3 t/s) | ✅ ~120+ t/s (MLX)    | ✅ **75 t/s** (Vulkan) | ~35 (est.)         |
| Qwen3 Coder Next 80B A3B (MoE)    | 80B          | 3B            | ~40 GB  | ❌ too large          | ✅ ~60 t/s (MLX)      | ✅ ~15-20 t/s          | 28                 |
| Qwen3 Next 80B A3B                | 80B          | 3B            | ~40 GB  | ❌ too large          | ✅ ~60 t/s (MLX)      | ✅ ~15-20 t/s          | 27                 |
| NVIDIA Nemotron 3 Super 120B A12B | 120B         | 12.7B         | ~60 GB  | ❌ too large          | ✅ ~20-30 t/s (MLX)   | ✅ ~8-12 t/s           | 36                 |
| Qwen3.5 122B A10B (MoE)           | 125B         | 10B           | ~62 GB  | ❌ too large          | ✅ ~40-50 t/s (MLX)   | ✅ ~15-20 t/s          | 42                 |
| MiniMax-M2.5 230B (MoE)           | 230B         | 10B           | ~115 GB | ❌ too large          | ⚠️ barely fits Q4    | ⚠️ barely fits Q4     | 42                 |
| Qwen3 235B A22B (MoE)             | 235B         | 22B           | ~117 GB | ❌ too large          | ⚠️ barely fits Q4    | ⚠️ barely fits Q4     | 30                 |
| MiMo-V2-Flash 309B (MoE)          | 309B         | 15B           | ~155 GB | ❌ too large          | ❌ too large           | ❌ too large            | 41                 |
| GLM-4.7 357B (MoE)                | 357B         | 32B           | ~178 GB | ❌ too large          | ❌ too large           | ❌ too large            | 42                 |
| DeepSeek V3.2 685B (MoE)          | 685B         | 37B           | ~342 GB | ❌ too large          | ❌ too large           | ❌ too large            | 42                 |
| Kimi K2.5 1T (MoE)                | 1000B        | 32B           | ~500 GB | ❌ too large          | ❌ too large           | ❌ too large            | 47                 |

### Standout Models for 128GB Local Use

**Qwen3.5 122B A10B** (Intelligence Index: 42 — same score as DeepSeek V3.2)
- MoE with only 10B active params → fast despite 125B total
- ~62GB at Q4_K_M, leaves plenty of headroom
- 262K context window
- Likely ~40-50 t/s on M4 Max via MLX — strong intelligence-to-speed tradeoff

**Qwen3.5 35B A3B** (Intelligence Index: 37)
- Only 3B active params = extremely fast (~150+ t/s on good hardware)
- ~18GB at Q4 — leaves room for large context
- Best option if you want fast responses without sacrificing much quality
- Currently NOT runnable at useful speed on your PX13 (CPU-only ~3 t/s)

**NVIDIA Nemotron 3 Super 120B A12B** (Intelligence Index: 36)
- Unique: 1 million token context window
- ~60GB at Q4_K_M, 12.7B active params
- Good for very long codebase analysis sessions

**MiniMax-M2.5 230B** (Intelligence Index: 42)
- Borderline fit: ~115GB at Q4_K_M (cuts it close on 128GB)
- Matches DeepSeek V3.2 quality at a fraction of the size
- Would need to use 3-bit quantization to run comfortably

### What Stays Out of Reach Even at 128GB

The top-ranked models (GLM-5, Kimi K2.5, DeepSeek V3.2, GLM-4.7) all have 350B-1T total parameters. At Q4_K_M they require 175-500GB — beyond any single 128GB machine. These are cloud-only models regardless of your hardware upgrade.

The practical ceiling for local 128GB inference is **~235B total params at Q4** or **~125B at Q8**.

---

## Session Notes — 2026-04-16 (AMD AI Max 395 Setup)

### What Was Done

Set up a full local LLM inference stack on the AMD Ryzen AI Max+ 395 (Strix Halo) machine using distrobox containers to isolate from the stable Ubuntu 24.04 host.

**Infrastructure:**
- Created two distrobox containers from [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes): `llama-vulkan-radv` and `llama-vulkan-amdvlk`
- Each container ships a pre-compiled `llama-server` + `llama-bench` with the respective Vulkan driver
- Benchmarking via `llama-cpp-bencher.py` from [lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing/tree/main/llm-bench) — wraps `llama-bench` and sweeps pp/tg token counts
- GRUB params (`iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856`) were already correctly configured
- `tuned accelerator-performance` profile **not yet applied** — expected to add 5-8% more t/s

**Models downloaded:**
- `Qwen3-Coder-30B-A3B-Instruct BF16` (57GB, 2-part GGUF) — already present
- `Qwen3-30B-A3B-UD-Q4_K_XL` (18GB) — downloaded today; this is the target model

### Benchmark Results (2026-04-16/17, plugged in)

| Model | Backend | pp512 (t/s) | tg128 (t/s) | Notes |
|---|---|---|---|---|
| Qwen3-30B-A3B UD-Q4_K_XL | **RADV Vulkan** | **596** | **70.19** | Best result; no FA |
| Qwen3-30B-A3B UD-Q4_K_XL | RADV Vulkan + FA | 609 | 70.22 | FA no difference for tg |
| Qwen3-30B-A3B UD-Q4_K_XL | AMDVLK Vulkan | 470 | 66.5 | On power, no FA |
| Qwen3-30B-A3B UD-Q4_K_XL | AMDVLK Vulkan + FA | 450 | 66.3 | FA makes no difference here |
| Qwen3-30B-A3B UD-Q4_K_XL | AMDVLK Vulkan | 280 | 39.0 | On battery (power-throttled) |
| Qwen3-Coder-30B-A3B BF16 | AMDVLK Vulkan | 67 | 3.4 | Near theoretical limit for 57GB model |
| Qwen3-Coder-30B-A3B BF16 | RADV Vulkan | — | 6.7 | RADV slower for BF16; no native bf16 |
| Qwen3-30B-A3B UD-Q4_K_XL | ROCm 7.2.1 | — | — | Segfault on model load (gfx1151/gfx1100 mismatch) |

### Key Findings

**RADV beats AMDVLK for Q4 quantized MoE:** RADV gave 70.19 tok/s vs AMDVLK's 66.5 tok/s on Q4_K_XL — ~6% faster. Use the `llama-vulkan-radv` container for this model. (RADV is only slower for BF16, which has no native bf16 path in RADV.)

**Power throttling matters:** On battery, tg128 dropped from 66.5 → 39 tok/s. Always plug in for inference benchmarks.

**BF16 is memory-bandwidth-limited:** The 57GB BF16 model maxes out at ~3.4 tok/s generation — near the theoretical limit of 215 GB/s ÷ 57 GB ≈ 3.77 tok/s. Quantization is not just compression; it's what enables MoE speed by reducing active-weight reads per token.

**Gap to 75 tok/s target:** Currently at 70.19 tok/s with RADV. Remaining ~7% likely recoverable with:
1. `sudo tuned-adm profile accelerator-performance` (5-8% boost, not yet applied)
2. Confirming `-b 256` batching is active for MoE (bencher's `--moe` flag should enable this)

**ROCm status:** The `llama-rocm-7.2.1` container's llama-bench segfaults when loading the Q4_K_XL model. ROCm identifies the GPU as `gfx1100` instead of `gfx1151`, suggesting a driver/arch mismatch. Vulkan (RADV) is the reliable path.

### How to Re-run Benchmark

```bash
MODELS=/home/gberseth/playground/llama.cpp/models
BENCH=/home/gberseth/playground/llm-playground

distrobox enter llama-vulkan-radv -- bash -c "cd $BENCH && uv run python llama-cpp-bencher.py --moe --skip-gpu-mon --rerun --builds llama.cpp-vulkan-radv --build-root /home/gberseth/playground -m $MODELS/qwen3-30B-A3B/Qwen3-30B-A3B-UD-Q4_K_XL.gguf -p 512 -n 128"
```

---

## Summary: M4 Max vs AMD AI Max 395 vs RTX Pro 6000

**Winner overall: RTX Pro 6000 Blackwell.** At 1,792 GB/s VRAM bandwidth it beats every laptop platform at every model size — and with 96 GB VRAM, 70B Q8_0 and 80B MoE Q4_K_M run fully in VRAM with no CPU offloading. The cost is 600W TDP and a desktop workstation setup (street price ~$8-11K).

**Best portable option: M4 Max.** 2.5× the bandwidth of the AMD laptop and 128 GB unified memory keeps large models fully in fast memory. Significantly more power-efficient than the Pro 6000 (~48W vs 600W). Best choice if mobility or power matter.

**AMD AI Max 395** is weaker for dense models but surprises with MoE: measured 40 t/s on Qwen3.5-35B MoE and 75 t/s on Qwen3-30B-A3B (Vulkan backend). MoE models are the right choice here — dense 27B+ is painfully slow at ~10 t/s due to lower memory bandwidth. Use Vulkan + FlashAttention, not HIP.

¹ MoE speeds depend on the model having low active param count (~3B). Dense 80B would be much slower.

| Machine | Best for | 70B Dense Q4 | 32B Dense Q4 | 80B MoE Q4 | 70B Dense Q8 |
|---|---|---|---|---|---|
| RTX Pro 6000 96GB | All models, maximum speed | ✅ ~25-35 t/s | ✅ ~60-70 t/s | ✅ ~130 t/s | ✅ ~15-20 t/s |
| M4 Max 128GB | Large models, portability | ✅ ~20-25 t/s | ✅ ~30-45 t/s | ✅ ~60 t/s | ✅ ~15-18 t/s |
| AMD AI Max 395 128GB | Budget portable 128GB | ✅ ~5-10 t/s | ✅ ~12-18 t/s | ✅ ~40-75 t/s (MoE!¹) | ✅ ~3-5 t/s |
