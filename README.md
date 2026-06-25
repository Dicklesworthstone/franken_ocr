# franken_ocr

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![toolchain: nightly](https://img.shields.io/badge/toolchain-nightly-purple.svg)](./rust-toolchain.toml)
[![unsafe: forbidden*](https://img.shields.io/badge/unsafe-forbidden*-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![model: Baidu Unlimited-OCR](https://img.shields.io/badge/model-Baidu_Unlimited--OCR-teal.svg)](https://huggingface.co/baidu/Unlimited-OCR)
[![status: pre-Phase-0](https://img.shields.io/badge/status-pre--Phase--0%20scaffold-lightgrey.svg)](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md)

</div>

**A pure-Rust, memory-safe, CPU-only OCR engine that runs exactly one model, Baidu Unlimited-OCR, with no general ML framework, no Python, no FFI, and no GPU. The bet: a single fixed model with compile-time-known shapes, quantized to a custom int4/int8 form and driven by model-specific tiled GEMM kernels, can close the CPU gap to ONNX/MLAS while keeping generated-token KV bounded during long document parses.**

> **Status: pre-Phase-0 kickoff scaffold. This does not run yet.** No inference path is implemented. What exists today is the master plan, the repository skeleton, the agent conventions, and the support files in this directory. Every number and behavior below is a *target*, not a measured result. The full roadmap, kernel strategy, and verification methodology live in [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md); contributor and agent conventions live in [`AGENTS.md`](./AGENTS.md).

---

## TL;DR

**The problem.** Baidu Unlimited-OCR is a strong document-parsing model (markdown, tables, LaTeX, reading order, dozens of pages in one pass), but the official stack is Python plus CUDA. Most machines that need OCR (laptops, CI runners, agent hosts, edge boxes) have no usable GPU, and a Python plus CUDA dependency is heavy to ship and awkward to embed.

**The solution.** `franken_ocr` is a library plus a single-binary CLI (`focr`) that runs this one model on CPU, fast, with nothing but a Rust binary. It transforms the bf16 checkpoint into a custom quantized format and runs it through kernels written for this model's exact shapes.

**Why `focr`:**

| | `franken_ocr` (planned) |
|---|---|
| Runtime | One static Rust binary. No Python, no CUDA, no FFI at inference time. |
| Hardware | CPU only, tuned for Apple Silicon (NEON/SDOT/i8mm-SMMLA) and Intel/AMD x86 (AVX2/AVX-VNNI/AVX-512-VNNI/AMX). |
| Quantization | Mixed int4/int8, custom on-disk format, vision tower kept high precision. |
| Memory | R-SWA keeps generated-token KV bounded; the reference block still grows with page count and is capped by the 32K context. |
| Embedding | Synchronous, blocking library API. The async runtime is an owned internal detail. |
| Agents | Versioned NDJSON robot mode, stable exit codes, deterministic output under fixed sampling. |
| Safety | `#![forbid(unsafe_code)]` everywhere except audited SIMD islands, each with a bit-identical scalar fallback. |

---

## The wedge

A general ML framework pays a generality tax on every operation: dynamic dtype dispatch, arbitrary shapes, autograd bookkeeping, broadcast machinery, a device abstraction, and a scheduler that knows nothing about your specific graph. `franken_ocr` runs exactly one model whose every dimension is known at compile time (hidden 1280, 10 heads, head_dim 128, 64 experts, top-6 routing, MoE intermediate 896, R-SWA window 128, vocab 129280). That buys several things a general runtime cannot:

- **Shape-specialized kernels.** `const`-generic tile sizes, no remainder handling for dims that tile cleanly, no runtime shape branching in the hot loop.
- **Offline weight transformation.** The bf16 checkpoint becomes a custom `.focrq` artifact: int8 first, int4 later, vision/projector/embeddings/router/norms held at high precision, arch-specific pre-packing so kernels load contiguous register tiles with zero runtime shuffle.
- **Mixed int4/int8 pushed hard.** Decode is memory-bandwidth-bound, so int4 group-quantized expert weights (the bulk of the parameters) roughly halve the bytes streamed per token. There is no CPU int4 matmul instruction, so int4 unpacks to int8 in-register and feeds the same SMMLA/VNNI/SDOT engine; the win is bandwidth and footprint. Accuracy-sensitive tensors stay int8, and the per-tensor split is chosen by a rate-distortion allocator rather than picked by hand.
- **Pre-allocation.** R-SWA bounds generated-token KV, so each layer gets a fixed ring buffer and a pre-sized reference block. Allocation can be stable, but the live reference length still grows with page count.

**Honest scope.** This is not a leaderboard play. Unlimited-OCR sits around 93.9% on OmniDocBench v1.6, behind PaddleOCR-VL and MinerU. The pitch is fidelity to this model, bounded generated-token KV for long-document parsing on CPU, and speed on commodity hardware, not topping a benchmark.

## How it works

Unlimited-OCR is a ~3B-parameter Mixture-of-Experts vision-language model and a DeepSeek-OCR derivative. The forward path:

```
image ──► DeepEncoder ──► linear projector ──► DeepSeek-V2 MoE decoder ──► text
          (vision tower)   (2048 → 1280)        (12 layers, R-SWA attention)
```

- **DeepEncoder (vision tower).** SAM-ViT-B (width 768, 12 layers, global attention at blocks [2,5,8,11]), then 16x conv token compression, then a CLIP-L/14 cascade (24 layers, width 1024, SDPA with `quick_gelu` FFN). SAM and CLIP features concatenate to 2048 dims. Kept at high precision; quantizing the vision tower hurts OCR, a result both community quants confirm.
- **Linear projector.** A single 2048 to 1280 map bridges the vision tower to the decoder hidden size.
- **DeepSeek-V2 MoE decoder.** 12 layers, hidden 1280. Layer 0 is a dense MLP; layers 1 to 11 are MoE: a router (1280 to 64) selects the top 6 of 64 experts (each a 1280 to 896 SiLU-gated MLP) plus 2 always-on shared experts. Final RMSNorm and `lm_head` (1280 to 129280) feed autoregressive sampling.
- **R-SWA (Reference Sliding Window Attention).** The one architectural novelty. Every decoder attention layer is replaced with R-SWA: each generated token attends to all reference tokens (visual plus prompt prefix, kept as a frozen, never-evicted global KV) plus only the previous 128 generated tokens via a ring-buffer KV cache. The generated-token KV memory stays constant instead of growing with output length; the reference block still grows with page/input length. That, not arbitrary input resolution, is what "Unlimited" means.

The checkpoint is a single 6.67 GB bf16 safetensors shard under the MIT license. The plan carries the verified config census and the exact tensor-name map.

## How it compares

Honest framing against the alternatives. `franken_ocr` is the only one of these built for one model on CPU.

| | `franken_ocr` (planned) | Official Unlimited-OCR | llama.cpp GGUF build | ONNX Runtime |
|---|---|---|---|---|
| Language / runtime | Pure Rust, one binary | Python + HF transformers | C++ | C++ |
| Primary target | CPU | CUDA GPU | CPU/GPU | CPU/GPU |
| Scope | This one model | This model | Many models | Many models |
| int8/int4 kernels | Model-specific tiled SMMLA/VNNI | n/a | Generic K-quant | MLAS |
| Vision encoder | First-class, kept high precision | First-class | Kept F16 (mmproj) | Depends on export |
| Ships as | Single static binary, no FFI | Python env + CUDA | Binary + model | Library + model |
| Constant-memory long docs | Yes (R-SWA preserved) | Yes | Depends on PR support | Depends |

## The `focr` CLI

> These commands describe the intended surface. The diagnostic commands work today; `ocr`, `convert`, and `doctor` return a clear "not yet implemented" pointing at the plan phase that lands them.

```bash
# Parse a document image into Markdown (human default) or structured JSON
focr ocr scan.png
focr ocr scan.png --json

# Stream NDJSON pipeline events for agents (run_start / stage / page / run_complete / run_error)
focr ocr scan.png --robot

# Offline weight transformation: bf16 safetensors into a custom quantized .focrq
focr convert model-00001-of-000001.safetensors -o unlimited-ocr.focrq \
  --quant int8 --arch aarch64-smmla

# Self-describing, versioned event/contract schema (machine-readable)
focr robot schema

# Diagnostics: model present? arch features detected? thread budget?
focr robot health
focr robot backends

# Idempotent self-check / repair (model resolution, format version, permissions)
focr doctor
```

The robot NDJSON stream carries one JSON object per line, each tagged with a `schema_version`; `focr robot schema` describes every event type so an agent can validate the stream against a frozen contract. Input is image-only in v1; PDFs are rasterized out of band (see the plan, section 7.7).

## Build

`franken_ocr` requires the nightly Rust toolchain (pinned in [`rust-toolchain.toml`](./rust-toolchain.toml); it auto-selects on first build). The current scaffold builds the `focr` CLI stub, whose diagnostic subcommands work; inference lands in later phases.

```bash
cargo build --release
```

Model weights are not bundled and are never downloaded at inference time. Fetch them out of band with [`scripts/fetch_model.sh`](./scripts/fetch_model.sh), which documents how to obtain the Unlimited-OCR safetensors shard, `tokenizer.json`, and `config.json` into `$FOCR_MODEL_DIR`, then run `focr convert` to produce the quantized `.focrq` artifact the engine loads.

## Roadmap

The plan is staged so correctness always precedes speed. Each phase has hard exit gates.

| Phase | Goal |
|---|---|
| -1 | Source/Oracle Truth Pack: pin the exact model commits, hash the sources, answer every open question from pinned source, decide the oracle strategy. |
| 0 | Scaffold: the crate skeleton, runtime, robot mode, persistence, CI matrix. |
| 1 | fp32 reference parity: a framework-free forward that matches the bf16 reference, end to end. Correctness before speed. |
| 2 | int8: staged weight-only quantization (experts first, then attention and `lm_head` behind kill switches), each its own parity gate. |
| 3 | SIMD kernels: the per-arch tiled int8/int4 GEMM (SMMLA/AMX prefill, SDOT/VNNI decode), MoE token-grouping, online-softmax R-SWA, NUMA-aware pool sizing. |
| 4 | int4: group-quantized expert weights at a Q4_K_M-class footprint, inside a measured accuracy budget. |
| 5 | CLI hardening, the three-pillar gauntlet certification, and a cross-platform release. |
| 6 | CUDA stretch (deferred; CPU stays the product). |

## Limitations

Being clear about what this is and is not:

- **It does not run yet.** This is a kickoff scaffold. The inference engine, the kernels, and the converter are unimplemented.
- **One model only.** This is a deliberate non-goal to be general. `franken_ocr` will never be a model zoo or a generic inference runtime.
- **Not benchmark SOTA.** Unlimited-OCR is strong but not the OmniDocBench leader; `franken_ocr` aims for fidelity to it, not for beating it.
- **CPU only in v1.** No GPU. CUDA is a deferred stretch goal, and CPU remains the product.
- **Image input in v1.** PDFs are rasterized out of band until a rasterization-parity path is scoped.
- **Quantization has an accuracy cost to measure.** Dense numeric content, tables, and code are exact-token-sensitive; the int4 target stays above the sub-4-bit cliff and every quant choice is gated on a measured character-error-rate budget.

## FAQ

**Is this affiliated with Baidu?** No. It is an independent pure-Rust reimplementation that runs Baidu's openly-licensed (MIT) model weights. The weights and any quantized derivative this project distributes carry Baidu's MIT notice.

**Why not just use llama.cpp or ONNX?** Both are excellent general runtimes. `franken_ocr` is a focused experiment: a single fixed model lets the kernels specialize to its exact shapes and skip the generality tax, and the whole thing ships as one Rust binary with no FFI. The community llama.cpp GGUF path for this model also currently depends on an unmerged PR.

**Why Rust, and why forbid `unsafe`?** Memory safety for a multi-gigabyte weight loader and a tight decode loop, with `unsafe` confined to small, audited SIMD modules that each carry a bit-identical scalar fallback.

**Will quantization wreck accuracy on tables and small numbers?** That is the central risk, and the plan treats it as such: parity gates per quant stage, a tail-risk (worst-case) character-error-rate bound rather than a mean, and the vision tower kept at full precision.

**Can I embed it in my Rust program?** That is a primary goal. The library API is synchronous and blocking, and the engine owns its runtime internally, so there is no async plumbing to thread through your code.

**When will it work?** No timeline is promised. Follow the phase gates in the plan.

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

The `franken_ocr` source code is licensed under the [MIT License](./LICENSE), Copyright (c) 2026 The franken_ocr authors.

The model weights are a separate matter. The Baidu Unlimited-OCR weights, and any quantized derivative this project distributes, are under the MIT License, Copyright (c) 2026 Baidu, reproduced in full in [`LICENSE`](./LICENSE) under "THIRD-PARTY MODEL WEIGHTS, NOTICE". That notice travels with any distributed weight artifact.

## See also

- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md), the master plan: architecture census, quantization format, the full kernel-optimization catalog, the alien-artifact math families, the three-pillar verification gauntlet, and the phased roadmap.
- [`AGENTS.md`](./AGENTS.md), conventions for human and agent contributors, including the engineering doctrine.
- [`CHANGELOG.md`](./CHANGELOG.md), the project history.
- [`docs/PERF_LEDGER.md`](./docs/PERF_LEDGER.md), the honest measured perf-ratio log (seeded empty).
- [`docs/NEGATIVE_EVIDENCE.md`](./docs/NEGATIVE_EVIDENCE.md), what did not work, including results inherited from sibling projects.
- [`docs/DISCREPANCIES.md`](./docs/DISCREPANCIES.md), known measured divergences from the reference model (seeded empty).
