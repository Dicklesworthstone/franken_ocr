# franken_ocr

<div align="center">
  <img src="franken_ocr_illustration.webp" alt="franken_ocr - Pure-Rust CPU-only OCR for Baidu Unlimited-OCR">
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![source tag: v0.7.2](https://img.shields.io/badge/source_tag-v0.7.2-blue.svg)](https://github.com/Dicklesworthstone/franken_ocr/tree/v0.7.2)
[![binary release: v0.7.2](https://img.shields.io/badge/binary_release-v0.7.2-blue.svg)](https://github.com/Dicklesworthstone/franken_ocr/releases/tag/v0.7.2)
[![status: working](https://img.shields.io/badge/status-working-success.svg)](#quick-example)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![toolchain: nightly](https://img.shields.io/badge/toolchain-nightly-purple.svg)](./rust-toolchain.toml)
[![unsafe: forbidden*](https://img.shields.io/badge/unsafe-forbidden*-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![default model: Baidu Unlimited-OCR](https://img.shields.io/badge/default_model-Baidu_Unlimited--OCR-teal.svg)](https://huggingface.co/baidu/Unlimited-OCR)
[![v0.7.0: release payload verified](https://img.shields.io/badge/v0.7.0-release_payload_verified-success.svg)](#conformance-and-release-evidence)

</div>

**A pure-Rust, memory-safe, CPU-only OCR engine for a small family of hand-ported vision-language models.** Baidu Unlimited-OCR is the fast default for document OCR, GOT-OCR2 handles specialized structured formats, SmolVLM2 handles image description and VQA, OneChart extracts chart data, and Polyphonic-TrOMR turns full scanned sheet-music pages or staff crops into MusicXML through `--task music`. All five runtime models are available through `focr pull`; TrOMR publishes both the 61 MB int8 default artifact and an 86 MB f32 reference artifact. The v0.7.0 Unlimited-OCR artifact uses the conservative exact recipe and passes the hard-page termination and complete 20-page corpus budget. The models run through model-specific Rust kernels and need no general ML framework, Python, CUDA, FFI at inference, or GPU.

<div align="center">
<h3>Quick Install</h3>

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
```

</div>

The installer detects your platform, resolves the latest published GitHub binary release (currently `v0.7.2`), verifies the downloaded asset by SHA256, and puts `focr` on your PATH. Install compatible model weights separately; after they are present, inference is offline.

### Current Release

`v0.7.2` ships raw executables for macOS Apple Silicon, macOS Intel, Linux x86-64, Linux ARM64, Windows x86-64, and Windows ARM64, each with a SHA256 sidecar. Its embedded schema-v2 manifest continues to pin the `v0.7.0` 4,157,448,783-byte Unlimited-OCR artifact using recipe `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`; `focr pull` verifies all three part hashes and the reassembled file before installation.

The exact-recipe artifact emits EOS on `page0590` and passes the complete 20-page corpus budget at aggregate normalized CER 0.19307925. The release does not claim the strict three-party OpenPGP certificate: the committed fail-closed finalizer requires three independently controlled registry-pinned signers and production audit receipts, which are not available in this release process. The local corpus, model-census, installer, and binary checks are evidence, not a substitute for that governance claim.

---

## TL;DR

**The problem.** Baidu Unlimited-OCR is a strong document-parsing model: Markdown, tables, LaTeX, reading order, many pages in one pass. The official stack is Python plus CUDA. Most machines that need OCR (laptops, CI runners, agent hosts, edge boxes) have no usable GPU, and a Python plus CUDA dependency is heavy to ship and awkward to embed.

**The solution.** `franken_ocr` is a library plus a single-binary CLI (`focr`) that runs the ready model set on CPU with nothing but a Rust binary. It transforms bf16 checkpoints into custom `.focrq` int8 artifacts and runs them through kernels written for each model's exact shapes. A historical real-page Unlimited-OCR run measured end-to-end character-error-rate **0.0094** and matched the reference decode to within a single token. The v0.7.0 corpus receipt records 20/20 pages within budget and a still-high `page0590` tail; the release publishes that measured artifact without claiming the unavailable three-party signing certificate.

### Why `focr`?

| Feature | What it does |
|---|---|
| **One portable binary** | No Python, no CUDA, no FFI at inference, no GPU. Current release assets are about 13 to 17 MB; portable to hosts where `ort`/CUDA cannot build. |
| **Works offline** | Once compatible weights are installed, inference never touches the network. Zoo pulls install under `~/.cache/franken_ocr/models`; the v0.7.0 default is the versioned conservative Unlimited-OCR artifact. Weight bytes are owned by default; trusted immutable deployments can opt into mmap with `FOCR_MMAP=1`. |
| **Embeddable Rust API** | `OcrEngine` exposes synchronous, blocking calls for Markdown, structured layout, figure extraction, in-memory images, and load-once batches. |
| **Native PDFs and figures** | Scanned PDFs are rasterized in process with pure Rust; page `/Rotate` and image-placement rotations are honored, `--pages` selects exact PDF pages, `--split-spreads` can split two-page book scans, and `--extract-figures` saves chart/photo regions beside the Markdown or JSON output. |
| **Cross-page parsing** | `--multi-page` runs the Unlimited-OCR `infer_multi` contract over selected PDF pages or an image list, producing one document with `<PAGE>` boundaries instead of unrelated page parses. |
| **Model zoo with compatibility status** | Ready engines: Unlimited-OCR, GOT-OCR2, SmolVLM2, OneChart, and Polyphonic-TrOMR, including full-page sheet-music OMR. `focr models` reports compatible and blocked pull entries with recipe metadata. All five runtime models are pullable in v0.7.0. Planned descriptors: TrOCR and pix2tex. |
| **Measured int8 decode path** | Historical rows show the custom int8 kernels can accelerate decoder work. The v0.7.0 exact-recipe artifact passes the hard-page termination and corpus-CER budget. Raw BF16 is only an all-high-precision native diagnostic when paired with `FOCR_DECODE_STATELESS=1`; the default cached decode applies the conservative int8 FFN/expert cache regardless of source storage. |
| **Runtime ISA dispatch** | One binary per architecture detects every available int8 ISA tier and reports the effective ordinary dense-GEMM route separately. Apple Silicon currently defaults to measured-faster LLVM autovec while retaining forced SDOT/SMMLA proof paths; x86 selects AVX-512-VNNI, AVX-VNNI, AVX2, or scalar. |
| **Measured zoo gauntlet** | `docs/PERF_LEDGER.md` records paired HF CPU reference rows. In the historical forced-Apple-SDOT rows at thread parity, decode-per-token ratios are 3.37x for GOT-OCR2, 2.58x for OneChart, and 1.67x for SmolVLM2; end-to-end rows are also kept, including slower cases. |
| **Batch throughput path** | `focr ocr-batch` loads weights once; the optional continuous-batch spine can prefill/decode multiple pages together while preserving per-page bytes. The dense zoo path covers GOT-OCR2, SmolVLM2, and OneChart with per-page token budgets. SAM, CLIP, and SigLIP towers plus their connector/embed tables are hydrated once, then reused across pages and batches. |
| **Stage timing instrumentation** | `FOCR_TIMING=1` reports nested forward timings, including SAM hydrate/forward/block/attention/MLP splits, so perf work can separate artifact load, vision attention, decode, and output costs. |
| **Durable run history** | `focr ocr` records best-effort local telemetry in fsqlite; `focr runs` and `focr sync` expose the run log as plain text, JSON, NDJSON, or locked JSONL. |
| **Bounded long-doc memory** | R-SWA keeps generated-token KV constant (window 128) while the reference block is held as a frozen, never-evicted global KV. |
| **Agent-first** | Versioned NDJSON robot mode, a self-describing `robot schema`, one-shot `robot triage`, TrOMR `staff` and `music_warning` events for music runs, stable documented exit codes, deterministic output under fixed sampling. |
| **Provable kernels** | `focr robot selftest` re-runs the dispatched int8 GEMM against a bit-identical scalar oracle on your CPU, including per-model verdicts for Unlimited-OCR, GOT-OCR2, SmolVLM2, and OneChart. |
| **Real-scan music gate** | The TrOMR lane has public-domain Spohr page/staff fixtures, human-verifiable attributes, a frozen MusicXML anchor, and a model-gated NDJSON runner so real engraved scans are measured separately from synthetic examples. |
| **Release evidence** | `scripts/ladder_scorecard.sh` folds the L0-L5 parity ladder, `docs/FEATURE_PARITY.md` accounts the surface area, `tests/property_suite.rs` exercises generator-driven invariants, the committed fuzz corpus seeds four libFuzzer targets, `scripts/bench_guardrail.py` compares stage timings against frozen baselines, and `scripts/gauntlet_cert.py` computes the three-pillar scorecard, invariant monitors, and 13/13 release-readiness gate. |
| **Memory-safe** | `#![forbid(unsafe_code)]` everywhere except small audited islands: SIMD kernels with bit-identical scalar fallbacks, plus the explicitly opted-in read-only mmap loader. |

---

## System Map

| Layer | What is live today |
|---|---|
| **Inputs** | PNG/JPG-style images, scanned PDFs, selected PDF page ranges, split book spreads, image batches, full sheet-music pages, and single staff crops. |
| **Model routing** | Unlimited-OCR for general documents, GOT-OCR2 for structured OCR modes, SmolVLM2 for describe/VQA, OneChart for chart data, and TrOMR for MusicXML. |
| **Runtime core** | `OcrEngine` owns one asupersync runtime, caches one model per process, exposes blocking Rust calls, and records best-effort run telemetry through fsqlite. |
| **CPU execution** | Fixed-shape Rust forwards call scalar-or-proven SIMD kernels. Apple Silicon advertises SDOT/SMMLA hardware but uses measured-faster LLVM autovec for ordinary dense int8 by default; Intel/AMD selects AVX-512-VNNI, AVX-VNNI, AVX2, or scalar. |
| **Outputs** | Markdown, structured JSON with layout boxes, cropped figure assets, MusicXML, versioned robot NDJSON, run-history JSON/NDJSON/JSONL, and release evidence artifacts. |

---

## Quick Example

```bash
# 1. Install the versioned conservative Unlimited-OCR artifact.
focr pull

# 2. OCR a page into Markdown (the human default).
focr ocr page.png

# 3. Same page as structured JSON (markdown + every span's bounding boxes).
focr ocr page.png --json

# 4. Write the result to a file; format follows the extension (.md or .json).
focr ocr page.png -o page.md
focr ocr page.png -o page.json            # structured JSON with bounding boxes

# 5. Also save figures the model can't transcribe (charts/photos) next to the .md.
focr ocr page.png -o page.md --extract-figures   # -> page.md + page_figures/

# 6. Use a specialized model when the task is not plain document OCR.
focr pull got-ocr2
focr ocr --model got-ocr2.int8.focrq --task tables table.png

focr pull smolvlm2
focr ocr --model smolvlm2.int8.focrq --task describe photo.jpg
focr ocr --model smolvlm2.int8.focrq --task describe --question "How many people are visible?" photo.jpg

focr pull onechart
focr ocr --model onechart.int8.focrq --task chart-data chart.png

focr pull tromr
focr ocr --model tromr.int8.focrq --task music score-page-or-staff.png -o score.musicxml
focr pull tromr --quant f32                 # bit-exact reference artifact when needed
focr ocr --model tromr.focrq --task music score-page-or-staff.png -o score.musicxml

# 7. OCR only the pages you need from a scanned PDF; optionally split two-page spreads.
focr ocr book.pdf --pages 3,5-9 --split-spreads -o excerpt.md

# 8. Stream NDJSON pipeline events for an agent (music runs also emit `staff` events).
focr ocr page.png --robot
focr robot triage

# 9. Prove the int8 kernels on THIS CPU are bit-identical to the scalar oracle.
focr robot selftest

# 10. Generate the parity-ladder and release-readiness scorecards used by release gates.
scripts/ladder_scorecard.sh --self-test
scripts/ladder_scorecard.sh --out scorecard.json
python3 scripts/gauntlet_cert.py --self-test
python3 scripts/bench_guardrail.py --self-test
python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md \
  --scorecard-out /tmp/focr-release-scorecard.json
# From a clean final source/evidence commit, produce the red provisional bundle.
# It exits 1 until fresh CI artifacts and three trusted signers finalize it.
python3 scripts/gauntlet_cert.py --bundle .gauntlet-output/bundle
# After downloading the exact CI, dist, Model Parity, and Performance Gauntlet
# artifacts for this HEAD, run --finalize-bundle as documented in
# docs/gauntlet/RELEASE_CERTIFICATION_TEMPLATE.md.
python3 scripts/gauntlet_cert.py --release-readiness \
  --readiness-out .gauntlet-output/release_readiness.json

# 11. (optional) Convert the exact pinned Unlimited-OCR bf16 shard.
# Same-shaped or reserialized inputs fail before an unloadable artifact is written.
focr convert model.safetensors -o unlimited-ocr.focrq --quant int8

# TrOMR conversion is available for local sheet-music artifacts.
focr convert /path/to/tromr/model.safetensors -o tromr.int8.focrq --quant int8 --model-id tromr
```

With `FOCR_MODEL_PATH` set, default-model inference runs fully offline. Compatible specialized pulls live under `~/.cache/franken_ocr/models` and are also offline after installation.

---

## Design Philosophy

**A few models, every dimension fixed.** A general ML framework pays a generality tax on every operation: dynamic dtype dispatch, arbitrary shapes, autograd bookkeeping, broadcast machinery, and a device abstraction. `franken_ocr` runs a small set of hand-ported models whose important dimensions are known up front. For the default Unlimited-OCR path that means hidden 1280, 10 heads, head_dim 128, 64 experts, top-6 routing, MoE intermediate 896, R-SWA window 128, and vocab 129280. That buys shape-specialized kernels with no runtime shape branching in the hot loop. The scope is a few hand-tuned, certified models, not any model; there is no generic runtime underneath.

**Model status is explicit.** `focr models` is the source of truth for the runtime zoo. Ready models have a runtime forward arm and can be used with `focr ocr`; planned models are visible so agents and humans can see the roadmap without accidentally running the wrong architecture. The table distinguishes runtime readiness from distribution compatibility. Unlimited-OCR, GOT-OCR2, SmolVLM2, OneChart, and TrOMR are pullable in v0.7.0. Non-primary models install into per-model cache subdirectories with their tokenizer sidecars beside the `.focrq` artifact. TrOMR publishes both int8 and f32 artifacts: the default pull is the smaller int8 artifact, while `focr pull tromr --quant f32` restores the bit-exact reference path for audits.

**Scanned books are addressable.** PDF input no longer means "the whole document or nothing." `--pages 3,5-9` selects source pages with 1-based ranges, reports out-of-range requests against the document page count, and keeps source page numbers in JSON/robot output. The native renderer applies both PDF page `/Rotate` metadata and the rotation implied by the image placement matrix, which fixes scanned-book files that store portrait page images but display them landscape through a rotated `cm` transform. `--split-spreads` is opt-in for common two-page book scans: a wide rasterized page with a near-blank center gutter becomes left and right logical pages, while pages without a qualifying gutter pass through unsplit.

**TrOMR page OMR is resilient.** TrOMR accepts either a staff crop or a full printed/scanned page. The page path runs pure-Rust staff detection with global deskew, trims horizontal page margins to the detected ink extent, grows wide staff bands toward neighbor midlines so they fit the model's 1280-column position budget where possible, orders crops top-to-bottom, recognizes each staff sequentially through the certified ResNetV2 plus ViT encoder and four-head decoder, merges the semantic streams, and emits partwise MusicXML. If a band is still too wide, setting `FOCR_TROMR_SPLIT=1` arms an experimental rescue that detects thin full-span barlines, splits the band into in-budget segments, and splices the streams (off by default: isolated segments measurably lose absolute pitch registration, so this trades content fidelity for recognition coverage; the skip with a named reason is the honest default). If one detected staff fails, the page still succeeds with the staves that recognized and logs the skipped staff's bbox and reason; if every staff fails, the error names each staff reason. JSON output includes the ordered `staves` array for music runs, and robot mode emits one `staff` event for each recognized or skipped staff before `run_complete`. The MusicXML emitter also adds annotate-only sanity comments for overfull bars, underfull middle bars, impossible durations, and cross-staff key mismatches; JSON and robot mode expose the same observations as `warnings` and `music_warning` events. Remaining TrOMR work is narrower: optional `**kern` export, camera-photo dewarp, and broader corpus-quality metrics.

**Performance claims are ledgered.** The A11 zoo gauntlet records native runs beside pinned Hugging Face CPU references for GOT-OCR2, SmolVLM2, and OneChart. Decode-per-token speedups are documented where the native path is ahead, and full end-to-end rows stay in the ledger even when artifact loading or preprocessing makes the total slower. The CPU backend also records losing rows, such as the f32 TrOMR baseline, instead of turning them into marketing claims. The README summarizes measured rows; `docs/PERF_LEDGER.md` is the audit trail.

**Offline at inference.** Network access is limited to explicit compatible pull commands; inference itself never downloads. Raw BF16 or locally converted Unlimited weights can be supplied through `FOCR_MODEL_PATH`. There is no Python, no CUDA, no FFI, and no GPU in the inference path. The async runtime that orchestrates I/O and cancellation is an owned internal detail; the library API is synchronous and blocking, so there is no async plumbing to thread through your code.

**Correctness before speed (always).** A parity gate comes first and a faster kernel that drifts the OCR output is reverted, no source landed, and recorded in the negative-evidence ledger. Speed is shipped on top of parity, never instead of it. The conservative recipe keeps the vision tower, projector, embeddings, attention, `lm_head`, MoE router, and all norms high precision. The v0.7.0 artifact passes `page0590` termination and the 20-page corpus budget, with the hard-page CER tail documented. Those measurements do not imply the separate three-party signing certificate.

**Conformance has receipts.** The L0-L5 parity ladder emits structured NDJSON, and `scripts/ladder_scorecard.sh` folds that stream into one `focr-ladder-scorecard/v1` artifact. An unarmed no-weights run is recorded as `skipped_no_model:true`, never as green. `docs/FEATURE_PARITY.md` splits the release surface into the numbered FeatureUniverse and the CLI/robot SurfaceMatrix; `tests/surface_matrix.rs` fails if a live subcommand, robot event, exit code, or rollup cell is missing from that ledger. The three-pillar gauntlet in `docs/gauntlet/METHODOLOGY.md` and `scripts/gauntlet_cert.py` turns those rows into a `franken_ocr.gauntlet.scorecard.v1` artifact, while the release ratchet compares lower bounds rather than point estimates. A change that improves the aggregate while lowering one category is blocked.

**Runtime ISA dispatch, one binary per arch.** There is no per-CPU-feature variant to choose. One `x86_64` binary covers AVX2 / AVX-VNNI / AVX-512-VNNI; one `aarch64` binary covers NEON / SDOT / SMMLA. Hardware capability and the effective ordinary dense-int8 route are distinct: Apple Silicon exposes SDOT/SMMLA, but measured decode results select the LLVM-autovectorized scalar loop by default. Forced SDOT and SMMLA remain available for parity and benchmark sweeps. Non-Apple ARM64 can select SMMLA when it is actually faster. On a Threadripper 5995WX, a Zen 3 part whose ceiling is AVX2, dispatch correctly selects AVX2.

**Bounded generated-token KV.** Every decoder attention layer is R-SWA (Reference Sliding Window Attention). Each generated token attends to all reference tokens (visual plus prompt prefix, kept as a frozen, never-evicted global KV) plus only the previous 128 generated tokens through a ring-buffer KV cache. Generated-token KV memory stays constant instead of growing with output length. That, not arbitrary input resolution, is what "Unlimited" means.

---

## How `franken_ocr` Compares

`franken_ocr` is the only one of these built for a fixed, hand-tuned set of models on CPU.

| | `franken_ocr` v0.7.2 | Official Unlimited-OCR | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| Language / runtime | Pure Rust, one binary | Python + HF transformers | C++ | C++ |
| Primary target | CPU | CUDA GPU | CPU/GPU | CPU/GPU |
| Scope | A few hand-tuned models | This model | Many models | Many models |
| int8 kernels | Model-specific tiled SDOT/SMMLA/VNNI | n/a | Generic K-quant | MLAS |
| Vision encoder | First-class, kept high precision | First-class | Kept F16 (mmproj) | Depends on export |
| Network at inference | None | None | None | None |
| Ships as | Single static binary, no FFI | Python env + CUDA | Binary + model | Library + model |
| Constant generated-token KV | Yes (R-SWA preserved) | Yes | Depends on PR support | Depends |
| Runs with no GPU | Yes | No (needs CUDA) | Yes | Yes |

**When to use `franken_ocr`:**
- You need this model's output and your host has no usable GPU.
- You want to embed OCR in a Rust program with no Python or FFI.
- You want one binary you can drop on a CI runner, an agent host, or an edge box.

**When `franken_ocr` might not be ideal:**
- You need a generic inference runtime that loads arbitrary models. `franken_ocr` runs a few hand-ported, certified models by design.
- You need the OmniDocBench leader; Unlimited-OCR is strong but not the top of the board.

---

## Installation

### Quick install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
```

The script detects your OS and CPU architecture, downloads the matching asset from the latest published binary release, verifies the SHA256 sidecar, and installs `focr`. At the time this source tag was cut, the latest published binary release was `v0.7.2`. Under WSL it proceeds as Linux. Under native Git-Bash, MSYS, or Cygwin it points you at the PowerShell installer below and exits cleanly.

On native Windows, install from PowerShell:

```powershell
irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
```

This selects `focr-x86_64-pc-windows-msvc.exe` on AMD64 or `focr-aarch64-pc-windows-msvc.exe` on ARM64, verifies it by SHA256, and puts `focr` on your PATH. Both installers stage and execute the verified binary before an atomic replacement while holding a destination-scoped process lock, so a failed or concurrent update does not clobber a working install. Both Windows architectures are part of the v0.7.2 release matrix.

### Manual binary download

Release binaries are raw executables, not tar.gz archives. Each one is a single portable file that dispatches the ISA tier at runtime, so there is one binary per architecture (no per-CPU-feature variant). Linux assets are GNU binaries linked for **glibc 2.17 or newer**; the dist producer independently replays `readelf` against the exact staged bytes and refuses a higher required symbol version.

| Platform | Asset |
|---|---|
| macOS Apple Silicon | `focr-aarch64-apple-darwin-neon-sdot-i8mm` |
| macOS Intel | `focr-x86_64-apple-darwin` |
| Linux x86-64 (glibc 2.17+) | `focr-x86_64-unknown-linux-gnu` |
| Linux ARM64 (glibc 2.17+) | `focr-aarch64-unknown-linux-gnu` |
| Windows x86-64 | `focr-x86_64-pc-windows-msvc.exe` |
| Windows ARM64 | `focr-aarch64-pc-windows-msvc.exe` |

Each asset has a `<asset>.sha256` sidecar in the standard `"<hex>  <asset>"` format. Download the binary and its sidecar from the release base URL, verify, then install. Example for Apple Silicon:

```bash
base=https://github.com/Dicklesworthstone/franken_ocr/releases/download/v0.7.2
asset=focr-aarch64-apple-darwin-neon-sdot-i8mm

curl -fsSLO "$base/$asset"
curl -fsSLO "$base/$asset.sha256"

# macOS: shasum -a 256 -c   |   Linux: sha256sum -c
shasum -a 256 -c "$asset.sha256"

chmod +x "$asset"
mv "$asset" /usr/local/bin/focr
```

On Linux, swap the asset name and use `sha256sum -c "$asset.sha256"`.

### From source (advanced; not a one-liner)

`franken_ocr` requires the nightly Rust toolchain pinned in [`rust-toolchain.toml`](./rust-toolchain.toml). `cargo build --locked --release` builds both the `focr` and `franken_ocr` binaries from one shared entrypoint.

The catch: `franken_ocr` path-depends on sibling repositories that are not published on crates.io (`../asupersync`, `../frankentorch`, and `../frankensqlite`). A fresh-clone `cargo build --locked` or a `cargo install --git` will fail to resolve those dependencies. There is no working `cargo install` from crates.io. Prebuilt binaries are the supported path; build from source only if you have those sibling repositories laid out as the workspace expects.

```bash
cargo build --locked --release
# produces target/release/focr and target/release/franken_ocr (identical behavior)
```

---

## Quick Start

1. **Install** the binary with the curl one-liner, or download and verify a release asset by hand.
2. **Install compatible weights:** run `focr pull` for the versioned conservative Unlimited-OCR artifact. Specialized models use `focr pull got-ocr2`, `focr pull smolvlm2`, `focr pull onechart`, or `focr pull tromr`. Raw BF16 and locally converted exact-recipe artifacts remain available through `FOCR_MODEL_PATH`.
3. **Verify your CPU kernels** (optional but reassuring): `focr robot selftest`. Exit 0 means the dispatched int8 GEMM cases, including the model-specific rows for Unlimited-OCR, GOT-OCR2, SmolVLM2, and OneChart, are bit-identical to the scalar oracle on this host.
4. **OCR a page:** `focr ocr page.png` for Markdown, add `--json` for structured output (markdown + bounding boxes), `-o out.md` / `-o out.json` to write a file, or `--robot` for an NDJSON event stream.
5. **Batch many pages** in one process (model loaded once): `focr ocr-batch page1.png page2.png page3.png --json`. Add `--multi-page` to parse them as ONE cross-referencing document instead of independent pages.

---

## Library API

`franken_ocr` is a library as well as a CLI. The public API is synchronous and
blocking: `OcrEngine` owns the internal asupersync runtime, loads the model on
first use, caches that model behind an `Arc`, and returns ordinary
`FocrResult<T>` values. Callers do not need to build or pass an async runtime.

```rust
use std::path::Path;

use franken_ocr::{OcrEngine, FocrResult};

fn main() -> FocrResult<()> {
    let engine = OcrEngine::new()?;

    let markdown = engine.recognize(Path::new("page.png"))?;
    println!("{markdown}");

    let document = engine.recognize_with_layout(Path::new("page.png"))?;
    println!("{} layout spans", document.layout.len());

    Ok(())
}
```

The same engine exposes the surfaces the CLI uses:

| Method | Use case |
|---|---|
| `recognize(path)` | Return Markdown for one image or document page. |
| `recognize_with_model(model, path)` | Pin a specific `.focrq` artifact without changing environment variables. |
| `recognize_with_layout(path)` | Return Markdown plus grounded bounding boxes, matching `focr ocr --json`. |
| `recognize_with_figures(path)` | Return Markdown/layout plus cropped figure regions, matching `--extract-figures`. |
| `recognize_dynamic(image)` | Run an already-decoded `image::DynamicImage`, the same in-memory path used by PDF rasterization. |
| `recognize_batch(&[paths])` | Load weights once and return one result per input path, in order. |
| `request_shutdown()` / `reset_shutdown()` | Drive cooperative cancellation for long-running embedders. |

Model resolution follows the CLI rules: `FOCR_MODEL_PATH` wins, then the default
cache location is searched. For path-explicit calls, pass the model artifact
directly. The engine keeps the one-live-forward discipline from the CLI, so a
single process can reuse weights without fanning out concurrent model forwards.

---

## Command Reference

Both `focr ocr` and `focr robot run` accept the same image plus inference-tuning flags. The default crop mode is `base` (the certified single 1024-pixel global view); `--crop-mode gundam` selects the reference dynamic-resolution tiling, whose end-to-end certification is still pending.

### `focr ocr <image>`

Parse a document image into Markdown (default), JSON, or an NDJSON stream.

```bash
focr ocr page.png                          # Markdown to stdout
focr ocr page.png --json                   # structured JSON to stdout
focr ocr page.png -o page.md               # write Markdown to a file
focr ocr page.png -o page.json             # write structured JSON (markdown + boxes) to a file
focr ocr page.png -o page.md --extract-figures   # also save figures, referenced from the .md
focr ocr page.png --robot                  # NDJSON pipeline events
focr ocr page.png --crop-mode gundam       # reference dynamic-resolution tiling (uncertified)
focr ocr page.png --max-length 4096 --temperature 0.0
focr ocr page.png --model /path/to/unlimited-ocr.focrq
focr ocr eq.png --task formula --model got-ocr2.int8.focrq   # specialized task routing (see below)
focr ocr chart.png --task chart-data --model onechart.int8.focrq
focr ocr book.pdf --pages 3,5-9 --split-spreads -o excerpt.md
```

**Output (`-o`/`--output FILE`).** Writes the result to a file instead of stdout; the
format follows the extension: `.json` emits structured JSON, any other extension
(e.g. `.md`) emits Markdown. `--json` forces JSON regardless of extension. The
structured JSON carries the rendered `markdown` plus a `layout` array, one
`{label, boxes}` entry per grounded span, where each box is `[x1, y1, x2, y2]` in
source-image pixels. A PDF nests these under a per-page `pages` array; split
spreads become separate logical page entries with the same source `page` number
and a `"half": "left"` or `"right"` marker. This is the same shape `--json`
prints to stdout. TrOMR music runs also include a `staves` array in source order;
each entry carries the 1-based staff number, page-space bbox, `recognized` or
`skipped` status, and an optional skip reason.

**Figures (`--extract-figures`).** The model sees figures/photos/diagrams it does not
transcribe to text. With `--extract-figures`, those regions are cropped out of the
source image and saved into a subfolder (default `<output-stem>_figures/`, or set
`--figures-dir DIR`), then referenced from the output: the Markdown gets a real
`![figure N](report_figures/page1_figure_1.jpg)` in place of each figure, and the
JSON gains a `figures` array of `{label, page, bbox, path}`. Each figure's format is
chosen by content: JPG (quality 85) for photographic regions, lossless PNG for
line-art / charts / screenshots. Requires `-o` (or `--figures-dir` for a stdout run).
PDFs name figures per page (`page{N}_figure_{M}`).

**PDF controls.** `--pages SPEC` is for PDFs only. `SPEC` is a comma-separated
list of 1-based pages and inclusive ranges, such as `3`, `3-7`, or
`1,5-9,218`; selected pages run in source order with duplicates removed.
Out-of-range pages are usage errors that name the document page count. Before
OCR, the pure-Rust PDF path applies both the page `/Rotate` entry and any
axis-aligned rotation from the content stream's image placement matrix, so scans
stored sideways but displayed upright are normalized before OCR or spread
splitting. `--split-spreads` is also PDF-only and off by default. It looks for
wide raster pages with a near-blank vertical gutter near the center, then OCRs
the left and right halves as separate logical pages; no qualifying gutter means
the page remains unsplit.

**Multi-page cross-page parsing (`--multi-page`).** By default every page is
parsed independently. `--multi-page` instead runs ONE pass over the whole
document using the Unlimited-OCR `infer_multi` contract, so page N can reference
content on pages 1..N-1, and the output is a single Markdown document with
`<PAGE>` separators. On `focr ocr` it is PDF-only and composes with `--pages`;
for an image list use `focr ocr-batch page1.png page2.png --multi-page`. The
whole document must fit the 32K context (roughly 290 pages); over-budget
requests fail with an actionable error instead of truncating. It does not
compose with `--split-spreads` or `--extract-figures` (per-page semantics).

**Tasks (`--task`).** Convenience routing over the model zoo (`focr models`). `--task ocr`
(the default) is plain document OCR, unchanged. `--task formula`, `tables`, `chart`,
`molecular`, and `geometry` are served by GOT-OCR2's `OCR with format:` mode, so each
implies `--format` (an explicit `--format` composes idempotently) and needs the
got-ocr2 model: `focr pull got-ocr2`, then `--model got-ocr2.int8.focrq`. `--task music`
has two valid lanes: TrOMR is the native OMR path and returns MusicXML, while GOT-OCR2
can still run its sheet-music format mode. Use `focr pull tromr`, then `--model tromr.int8.focrq --task music`
for a full printed/scanned page or a staff crop. Use `focr pull tromr --quant f32`
and `--model tromr.focrq` when you need the bit-exact reference artifact. You can also use the GOT lane
(`--model got-ocr2.int8.focrq --task music`). `--task describe` (photo description / VQA) is served by the smolvlm2
model: `focr pull smolvlm2`, then `--model smolvlm2.int8.focrq --task describe`, optionally with `--question "What
color is the car?"`. SmolVLM2 has no instruction modes; the task is the question
(default: the model-card caption prompt). `--task chart-data` is the OneChart path: it
runs chart-to-dict extraction with the number-head reliability check and needs a
onechart artifact, for example `focr pull onechart`, then `--model onechart.int8.focrq`. Pointing a
specialized task at the wrong model family fails with usage guidance before any weights
load.

Tuning flags: `--base-size` and `--image-size` (preprocessing resolution), `--crop-mode` (`gundam` or `base`), `--max-length` (decode token cap), `--temperature` (sampling), `--no-repeat-ngram` and `--ngram-window` (the sliding no-repeat n-gram decode guard).

### `focr ocr-batch <images...>`

OCR many images in one process, loading the model once and reusing it across all pages.

```bash
focr ocr-batch a.png b.png c.png --json     # one load, many pages
FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
  focr ocr-batch *.png --experimental-full-int8  # explicitly arm the gated cache
FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 FOCR_BATCH_SPINE=1 \
  focr ocr-batch *.png --experimental-full-int8 --json
FOCR_BATCH_SPINE=0 focr ocr-batch *.png --json   # force the sequential oracle path
```

`ocr-batch` loads the model once either way. Unlimited-OCR defaults to the
conservative recipe path. Its continuous-batch R-SWA spine currently belongs to
the explicitly gated all-int8 experiment, so `FOCR_BATCH_SPINE=1` uses that
spine only with `--experimental-full-int8` and both accuracy switches shown
above. The dense zoo path batches GOT-OCR2, SmolVLM2, and OneChart decode streams. GOT-OCR2 and OneChart
reuse hydrated SAM weights, projectors, and widened embedding tables; SmolVLM2
does the same for its SigLIP tower, modality projection, and text embeddings.
The same statics are cached on the loaded model for ordinary sequential pages
too, so a multi-page PDF or repeated in-process calls no longer re-widen those
large tensors for each page.
`FOCR_BATCH_SIZE` defaults to 128 active streams and clamps at 256;
`FOCR_BATCH_PACK=1` groups similar prefill lengths before restoring output
order. Without the experimental flag, attention and `lm_head` stay high
precision by construction.

### `focr pull`

Download a manifest-compatible model and tokenizer into the cache, verifying
every byte. The embedded v0.7.0 Unlimited-OCR entry declares the
runtime-compatible conservative recipe
`unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`; the historical schema-v1
full-int8 entry remains incompatible and is rejected.

```bash
focr pull                                   # versioned conservative Unlimited-OCR artifact
focr pull got-ocr2                          # structured OCR model
focr pull smolvlm2                          # image description / VQA model
focr pull onechart                          # chart-to-data model
focr pull tromr                             # sheet-music OMR model (int8 default)
focr pull tromr --quant f32                 # bit-exact reference artifact
focr pull got-ocr2 --quant int8 --json      # explicit quant, machine-readable
focr pull --manifest ./manifest.json        # compatible custom default manifest
```

Downloads run over asupersync's native HTTP stack (rustls + webpki-roots), reassemble GitHub split parts, verify against a committed SHA256 manifest, and cache to `~/.cache/franken_ocr/models`. Manifest schema v2 requires an exact recipe for every quant; size, paths, counts, declared sizes, HTTPS URLs, and hashes are validated before installation. The release-bound schema-v2 manifest is embedded in each binary; a local or HTTPS override is explicit.
The v0.7.0 primary Unlimited-OCR artifact uses the conservative exact recipe and installs directly under the cache root as `unlimited-ocr.v0.7.0.int8.focrq`. The embedded manifest pins its three release parts, byte count, and final SHA256; the historical full-int8 artifact remains only in the schema-v1 manifest used by already-released v0.6 binaries.
Other models install under `~/.cache/franken_ocr/models/<model-id>/` so their
tokenizer sidecars cannot collide. The resolver searches those subdirectories,
so `--model onechart.int8.focrq`, `--model tromr.int8.focrq`, and
`--model tromr.focrq` work after their pulls. TrOMR publishes `int8` and `f32`;
the default `focr pull tromr` fetches the int8 artifact, while
`focr pull tromr --quant f32` fetches the bit-exact f32 reference artifact.

### `focr models`

List the model zoo with each id, runtime status, task set, model families, default artifact name, and compatible or blocked pull recipes.

```bash
focr models                                 # human table
focr models --json                          # machine-readable list
```

**Which model do I use?**

| Model | Status | Use it for | Notes |
|---|---:|---|---|
| **`unlimited-ocr`** *(default)* | runtime ready, pull ready | Plain-text document OCR for general documents and PDFs. This is what `focr ocr` runs by default. | `focr pull` installs the v0.7.0 conservative exact-recipe artifact; raw BF16 and locally converted compatible artifacts are also supported. |
| **`got-ocr2`** | ready | Specialized structured output the default cannot produce: formulas, tables, charts, molecular diagrams, geometry, and sheet music. | Heavier per page; `--task formula|tables|chart|molecular|geometry|music` implies `--format`. |
| **`smolvlm2`** | ready | Photo description and VQA through `--task describe [--question "..."]`. | `focr pull smolvlm2` installs `smolvlm2.int8.focrq` plus `tokenizer.json` in the model subdirectory. |
| **`onechart`** | ready | Chart-to-data extraction through `--task chart-data`. | `focr pull onechart` installs `onechart.int8.focrq`, `vocab.json`, `merges.txt`, and `added_tokens.json`. The native path runs the fixed chart prompt, SAM/projector splice, OPT KV-cache decode, `num_decoder`, JSON repair, and reliability distance check. |
| **`tromr`** | ready | Polyphonic sheet-music OMR through `--task music`. | `focr pull tromr` installs `tromr.int8.focrq` plus the rhythm, pitch, lift, and note tokenizer tables; `focr pull tromr --quant f32` installs the bit-exact `tromr.focrq` reference artifact. Staff crops and full pages both run; bad staves are skipped with reasons when at least one staff recognizes, and over-budget systems can split at detected barlines. |
| **`trocr`**, **`pix2tex`** | planned | Handwriting and LaTeX OCR lanes. | Registered descriptors only until their forward paths ship. |

`unlimited-ocr` is the fast default for ordinary text. Reach for `got-ocr2` only when you need format extraction (formulas, tables, charts, etc.); it is a much larger decode and is not meant to replace the default for plain text. Install and run it with:

```bash
focr pull got-ocr2                                    # download the weights + tokenizer
focr ocr --model got-ocr2.int8.focrq page.png         # plain-text OCR
focr ocr --model got-ocr2.int8.focrq --format eq.png  # structured .mmd: LaTeX / tables / charts / music
focr ocr --model got-ocr2.int8.focrq --task tables page.png  # task shorthand (implies --format)
```

Run SmolVLM2 (photo description / VQA):

```bash
focr pull smolvlm2
focr ocr --model smolvlm2.int8.focrq --task describe photo.jpg
focr ocr --model smolvlm2.int8.focrq --task describe --question "How many dogs are there?" photo.jpg
```

Run OneChart (chart-to-data extraction):

```bash
focr pull onechart
focr ocr --model onechart.int8.focrq --task chart-data chart.png
```

Run TrOMR sheet-music OMR:

```bash
focr pull tromr
focr ocr --model tromr.int8.focrq --task music score-page.png -o score.musicxml
focr pull tromr --quant f32
focr ocr --model tromr.focrq --task music score-page.png -o score.musicxml
focr convert /path/to/tromr/model.safetensors -o tromr.int8.focrq --quant int8 --model-id tromr
scripts/tromr_convert_e2e.sh
scripts/tromr_music_e2e.sh
scripts/realscan_music_gate.sh
```

The TrOMR runtime accepts a single staff crop or a full printed/scanned page. If the detector finds multiple staves, it deskews the page globally, crops staves top-to-bottom, recognizes each staff through the same certified single-staff path, and emits one MusicXML part per staff; if it finds fewer than two staves, it treats the whole image as the staff input. For genuinely over-wide systems, the page path uses the five detected staff-line anchors to find thin barline columns, splits at the farthest usable barline within the positional budget, recognizes each segment sequentially, and concatenates the semantic stream while suppressing continuation clef/key/time artifacts. A multi-staff page succeeds when at least one staff recognizes, logging skipped staves with bbox and reason; an all-fail page errors with every staff reason named. The published int8 artifact is 61 MB and quantizes the 40 decoder-GEMM candidate tensors; the 86 MB f32 artifact stays published for bit-exact reference work. The int8 default is accepted with a ledgered discrepancy on one degraded Spohr page where both int8 and f32 garble differently while the corpus gate verdict stays unchanged. Local conversion is still available when you have your own TrOMR checkpoint.

The real-scan music corpus lives in `tests/fixtures/realscan_music/`. It starts with public-domain Spohr page and staff fixtures, tier-1 attributes that can be verified by eye, and one frozen MusicXML regression anchor. `scripts/realscan_music_gate.sh` is model-gated and exits successfully when local TrOMR weights are absent; when weights are present it checks attributes, frozen anchors, and page-level `staff` event floors. This is the real-scan regression base for the TrOMR lane, not a claim that broad note-level SER coverage is complete.

### `focr convert <input>`

Offline weight transformation: a bf16 safetensors checkpoint into a custom `.focrq` artifact. Decoder tensors are int8 where that policy is validated for the model; tiny models such as TrOMR can stay f32 by default while still using the same one-binary runtime.

```bash
focr convert model.safetensors -o unlimited-ocr.focrq --quant int8
focr convert got.safetensors -o got-ocr2.int8.focrq --quant int8 --model-id got-ocr2
focr convert smolvlm2.safetensors -o smolvlm2.int8.focrq --quant int8 --model-id smolvlm2
focr convert onechart.safetensors -o onechart.int8.focrq --quant int8 --model-id onechart
focr convert tromr/model.safetensors -o tromr.int8.focrq --quant int8 --model-id tromr
focr convert model.safetensors -o out.focrq --quant int8 --json
```

For Unlimited-OCR, conversion stamps recipe `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`, quantizes exactly 2,148 FFN/expert matrices, and copies all 48 attention projections plus `lm_head` at source precision. The source SHA256 is stamped into the header, and the loader rejects any default-model `.focrq` whose tensor dtypes violate that recipe. `--model-id` selects the architecture descriptor, tied-head policy, tensor-name prefixes, tokenizer expectation, and license notice for each zoo model. `--arch <target>` records architecture-specific prepacking intent (default: `generic`). Int4 remains gated and unvalidated.

### `focr robot <subcommand>`

Agent-facing surface: versioned NDJSON, self-describing, line-oriented, easy to pipe.

```bash
focr robot run page.png                      # stream OCR events as NDJSON (same flags as ocr)
focr robot schema                            # self-describing, versioned event/contract schema
focr robot health                            # model present? arch features? thread budget?
focr robot backends                          # detected SIMD tiers + core count
focr robot selftest                          # int8 kernel vs scalar oracle on this host
focr robot triage                            # quick_ref + health + recommendations + commands
```

`focr robot selftest` runs the selected dense-int8 implementation against the bit-identical scalar oracle across a shape battery, including the worst-case `K=6848` i32-accumulation overflow case, and emits one JSON verdict; it exits 1 on either numeric divergence or a route-trace mismatch. The payload separates `hardware_selected` from the effective `selected` route and lists the branch-derived `executed_routes`. `oracle_independent=false` is explicit for scalar/autovec because those paths invoke the oracle implementation itself. Set `FOCR_FORCE_ARCH` to verify each available intrinsic tier independently.

`focr robot triage` is the first command an agent should run in an unfamiliar
checkout or host. It returns one JSON object with a compact command reference,
the live health payload, state-aware next commands, command templates, and the
exit-code dictionary.

For TrOMR music runs, robot mode adds a `staff` event before `run_complete` for
each detected staff. The event carries the 1-based staff index, total detected
staff count, page-space bbox, `recognized` or `skipped` status, and optional
reason. Music runs can also emit `music_warning` events for annotate-only
musical sanity observations: overfull bars, underfull non-final bars,
impossible note durations, and cross-staff key mismatches. Existing consumers
can ignore these additive events; strict consumers should refresh their schema
with `focr robot schema`.

### `focr runs` and `focr sync`

`focr ocr` records each run in a local fsqlite store on a best-effort basis. A
store failure prints a note to stderr and never fails the OCR run itself. The
default store is `~/.cache/franken_ocr/runs.db`; set `FOCR_RUN_STORE` to use a
different path.

```bash
focr runs --limit 10                         # recent runs, plain table
focr runs --limit 10 --format json           # one JSON object with a runs array
focr runs --format ndjson                    # one run record per line
focr runs --id <run-id> --json               # inspect one run

focr sync export-jsonl --json                # export the canonical audit log
focr sync export-jsonl --file runs.jsonl     # choose the export path
focr sync import-jsonl --file runs.jsonl --json
```

Exports are canonical and atomic: the writer takes an exclusive lock, writes a
temporary file beside the target, fsyncs it, then renames it into place. Imports
are idempotent because records replace by `run_id`.

### `focr doctor`

`doctor` is the local self-check and reversible repair surface. Detect-only mode
is read-only and exits 0 when healthy or 1 when findings are present. `--fix`
applies only safe repairs through one mutation chokepoint: every touched file is
backed up first under `.doctor/runs/<run-id>/backups/`, hashes and modes are
logged to `actions.jsonl`, and `doctor undo <run-id>` restores from that log.
Irreversible work, such as regenerating model artifacts, is refused with the
exact command to run manually.

```bash
focr doctor                                 # human-readable detect-only report
focr doctor --json                          # one JSON object with typed findings
focr doctor --dry-run                       # planned blast radius, no mutation
focr doctor --fix                           # safe reversible repairs only
focr doctor undo <run-id>                   # restore a prior --fix run
focr doctor capabilities                    # detector/fixer/exit-code contract
focr doctor robot-docs                      # paste-ready agent handbook
```

---

## CPU Backend and Optimizations

The hot path is not a generic tensor interpreter. Each model is converted into a `.focrq` artifact, then executed through fixed-shape Rust code that calls model-specific kernels and keeps allocations out of the decode loop.

The backend keeps handwritten SIMD narrow. Ordinary Rust loops stay in place
where LLVM autovectorizes well; audited SIMD is reserved for int8 matmul kernels
whose scalar oracle is bit-identical and whose speedup is measured. That is why
`robot backends` reports both hardware ISA selection and the effective ordinary
dense-int8 route. It does not overclaim the separately dispatched packed-int4 or
offline-packed SMMLA paths; AMX remains absent until a real kernel lands.

Model artifacts load through a byte-range directory over one backing blob. The
safe default is an owned `Vec<u8>`, so a concurrent writer cannot truncate a
mapping and fault the process. Deployments that keep artifacts on immutable
inodes, such as read-only image layers, can set `FOCR_MMAP=1` to opt into lazy
page-in and OS page-cache reuse. The parser and tensor accessors are identical.

The current optimization stack is deliberately specific:

| Surface | Optimization |
|---|---|
| **Decoder GEMMs** | Offline `.focrq` conversion packs recipe-selected decoder weights for int8 matmul while keeping accuracy-sensitive tensors high precision. Artifact-level accuracy remains a release gate. |
| **Weight loading** | Owned bytes are the safe default for `.focrq` and safetensors artifacts. `FOCR_MMAP=1` explicitly opts trusted immutable artifact inodes into lazy page-in and OS page-cache reuse. |
| **Per-token decode** | R-SWA generated-token ring KV and a cached decoder forward. The fused QKV int8 projection remains an explicitly gated experiment until its independent accuracy gates pass. |
| **Vision work** | SAM, CLIP, and SigLIP linears are pre-transposed at hydration, then cached on the model. GOT-OCR2, SmolVLM2, and OneChart cache their vision towers, projectors, and widened embed tables once per loaded model. SAM attention windows run independently across Rayon; relative-position work is hoisted and walked in cache-friendly rows. |
| **Batch throughput** | `FOCR_BATCH_SPINE=1` advances active streams through shared decode steps while preserving each stream's KV cache, position, output cap, and byte identity. |
| **TrOMR** | Staff detection, deskew, ink-extent trim, neighbor-bounded band growth, barline segmentation for over-budget systems, ResNetV2/ViT glue, and MusicXML emission are native Rust. |

**Apple Silicon / ARM64.** The aarch64 backend detects NEON dot-product (`SDOT`) and matrix-multiply int8 (`SMMLA` / i8mm) at runtime. Ordinary dense decoder linears currently use the LLVM-autovectorized scalar loop by default on macOS: interleaved real-decode measurements beat forced SDOT, while forced SMMLA is slower still. `FOCR_INT8_AUTOVEC=0` restores the natural SDOT choice for a bounded comparison, and `FOCR_FORCE_ARCH=sdot|smmla|scalar focr robot selftest` forces and verifies each available branch. Packed int4 and offline-packed SMMLA have separate dispatch and are not labeled by the ordinary-dense route. On non-Apple ARM64, the effective route can favor SMMLA. Norms, softmax, preprocessing, and TrOMR's f32 glue remain simple loops for LLVM. The vision path reduces repeat work by hydrating SAM, CLIP, and SigLIP weights once, storing GEMM-ready pre-transposed linears, exposing independent SAM attention windows to Rayon, and walking relative-position rows in cache-friendly order.

**Intel / AMD x86-64.** The x86 backend detects AVX-512-VNNI, AVX-VNNI, and AVX2 in that order, then falls back to scalar. AVX-VNNI and AVX-512-VNNI take the native dot-product path for int8 decode; AVX2 uses an exact non-saturating implementation rather than a shortcut that could corrupt accumulation. AMX is not advertised until there is a real AMX backend. Forced-tier selection reaches the named implementation, and `robot selftest` records that executed route rather than trusting capability metadata. The same binary therefore runs correctly on older AVX2-only Zen/Intel hosts and selects wider VNNI kernels on newer CPUs. The model-level SAM/CLIP/SigLIP hydration cache and the pre-transposed linear layout also help x86 hosts by removing repeated full-weight transposes before the BLAS-like matmul calls. The non-kernel glue is kept in plain, predictable loops so LLVM can vectorize it for the host CPU instead of forcing a brittle hand-written SIMD path.

**Quantization policy.** The conservative candidate set is exact: the dense layer-0 MLP and the routed/shared MoE expert FFN matrices. Attention q/k/v/o and `lm_head` are not part of that set; they stay high precision unless both independent experimental gates are explicitly armed. The vision tower, projector, embeddings, MoE router, and all norms also stay high precision. This storage policy does not itself certify an artifact; the v0.7.0 artifact separately passes its hard-page and corpus gates.

**Experimental fused QKV decode.** The all-int8 decoder stacks the q/k/v projection
weights into one `[3*qkv_dim, hidden]` panel and runs one quantize-plus-GEMV pass
per token instead of three separate projection calls. It is reachable only when
`FOCR_DECODE_INT8=1`, `FOCR_INT8_ATTN=1`, and `FOCR_INT8_LMHEAD=1` are all truthy.
Set `FOCR_QKV_FUSED=0` to restore the older three-call path for comparison.

**TrOMR-specific CPU primitives.** The sheet-music lane has native staff preprocessing, global deskew and staff grouping, ink-extent crop trimming, neighbor-bounded band growth for wide systems, barline-column detection for over-budget bands, TF-SAME padding arithmetic, TF-SAME max-pool support, a Torch-parity GroupNorm kernel with optional fused ReLU, and a hybrid ResNetV2 plus ViT encoder checked at cosine 1.0 against oracle seams. Weight-standardized convolutions are folded during checkpoint export so the runtime can keep the conv path simple. The decoder includes self-attention, cross-attention, GEGLU, stream embeddings, and four output heads, then merges the rhythm/pitch/lift streams into semantic music tokens and partwise MusicXML. Opaque-alpha RGBA staff images take the RGB luma path rather than upstream's blanket inverted-alpha path, because the literal upstream rule blanks fully opaque demo staves; `docs/DISCREPANCIES.md` records the measured SER impact. The TrOMR closeout pins mean SER at 0.211 on committed single-staff examples and shows detection-lossless full-page reads at 0.125 / 0.040 SER for stacked staves. These pieces are scalar Rust loops designed for LLVM autovectorization across Apple Silicon and Intel/AMD CPUs; hand-written wide SIMD stays limited to int8 matmul kernels where the proof and measurement support it.

**Zoo performance evidence.** The latest zoo gauntlet keeps paired reference rows for GOT-OCR2, SmolVLM2, and OneChart. On an aarch64 host with NEON dotprod at eight threads, the native decode-per-token path measured 3.37x over Hugging Face CPU for GOT-OCR2, 2.58x for OneChart, and 1.67x for SmolVLM2. The ledger also records the full end-to-end rows, including the current artifact-load tax, so the project can improve throughput without hiding unfavorable totals.

**Instrumentation and batch-spine bring-up.** `FOCR_TIMING=1` prints nested timing rows for the native forward, including SAM hydrate, SAM forward, per-block attention, and per-block MLP stages. That makes the current bottleneck visible: large pages can spend their wall time in vision attention rather than model artifact loading. The vision hydration path now builds SAM, CLIP, and SigLIP weight bundles once per loaded model, including GEMM-ready pre-transposed linears, so later pages skip repeated transpose work. GOT-OCR2, SmolVLM2, and OneChart apply the same model-cache pattern to their vision towers, projectors, and widened embed tables. The SAM attention pass keeps the math bit-identical while reducing overhead: independent windows run across Rayon, relative-position tables are hoisted once per block, Q/K/V head splits copy contiguous slices, and the bias add walks `(ky, kx)` rows directly. The dense decoder batch spine also has a byte-identity-gated helper for Qwen/Llama and OPT-family decode steps. Prefill stays per stream; active non-EOS streams then advance through one batch step while each stream keeps its own KV cache, absolute position, and per-stream emission cap. The public `ocr-batch` path keeps `FOCR_BATCH_SPINE` as an opt-in switch while new batch plumbing earns correctness and perf evidence.

**Correctness gates.** Every accelerated int8 GEMM has a bit-identical scalar fallback. `focr robot selftest` includes the doctrine worst case, `K=6848`, plus model-specific overflow rows for the ready int8 decoder families. It verifies parity and that every call executed the selected route; forced subprocess coverage sweeps every host-available ISA tier. The batch scheduler and decode cache are guarded by byte-identity tests against the proven sequential path.

---

## Conformance and Release Evidence

The conformance harness is built to leave reviewable artifacts rather than only
a green test line.

| Artifact | Command or location | What it proves |
|---|---|---|
| **Parity scorecard** | `scripts/ladder_scorecard.sh --out scorecard.json` | Runs L0-L5 in order, folds each rung's NDJSON into `focr-ladder-scorecard/v1`, and writes the raw stream beside the summary. |
| **Skip-honest mode** | `scripts/ladder_scorecard.sh --out scorecard.json` without weights | Produces `all_green:false` and `skipped_no_model:true`, so a missing-model run cannot masquerade as a certified release. |
| **FeatureUniverse / SurfaceMatrix** | `docs/FEATURE_PARITY.md`, `tests/surface_matrix.rs` | Accounts modeling features, ops, CLI surfaces, robot events, parity gates, and alien-artifact families as `present`, `partial`, `missing`, `n/a`, or `excluded`; partial never rounds up. |
| **Three-pillar gauntlet cert** | `python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md --scorecard-out /tmp/focr-release-scorecard.json` | Scores the surface pillar from the live parity ledger and emits `franken_ocr.gauntlet.scorecard.v1`; performance and conformance gates stay separate, so one green pillar cannot hide another regression. |
| **Release certification** | `--bundle`, then `--finalize-bundle` with downloaded workflow manifests and three trusted signer inputs, then `--release-readiness` | The provisional bundle is always uncertified. The production finalizer live-replays one exact-HEAD run of CI, dist, Model Parity, and Performance Gauntlet, derives every strict field, verifies three registry-pinned OpenPGP signatures, proves the candidate in memory, and only then persists `certified:true`. |
| **Benchmark guardrail** | `python3 scripts/bench_guardrail.py --stages artifacts/.../focr_stages.json --baseline benches/.bench-history/baseline.json --parity-receipt tests/fixtures/ladder_scorecard/scorecard_armed.json` | Compares fresh stage timings against committed per-regime baselines, refuses performance reporting without an all-green parity receipt, marks noisy or posture-mismatched rows ineligible, and exits successfully when model fixtures are absent. |
| **Real-scan music corpus** | `scripts/realscan_music_gate.sh`, `tests/fixtures/realscan_music/` | Runs TrOMR over public-domain Spohr staff/page fixtures, checks human-verifiable attributes and frozen MusicXML anchors, and uses robot `staff` events for page-level floors. |
| **TrOMR page resilience** | `cargo test --locked --lib native_engine::tromr::tests::tromr_page` | Checks stacked-staff order, measured single-staff SER floors, one fittable-plus-one-unfittable staff page, and all-fail pages whose error names every skipped staff. |
| **Property invariants** | `cargo test --locked --test property_suite` | Runs bounded proptest cases for SIMD-vs-scalar int8 GEMM identity, K=6848 i32 accumulation, preprocess geometry, `.focrq` parser totality under byte mutation, and tokenizer round-trip when a tokenizer artifact is present. |
| **Fuzz seed corpus** | `fuzz/corpus/`, `cargo fuzz run focrq_parse\|safetensors_parse\|image_decode\|pretok_split` | Seeds four untrusted-input fuzz targets covering `.focrq` parsing, safetensors parsing, image ingest, and pretokenizer splitting. The latest post-head-to-head sweep ran 3.65M cases with zero crashes, and the repo keeps the resulting corpus growth under `fuzz/corpus/`. |
| **Conformal ratchet** | `docs/conformance/RATCHET.md`, `src/conformance.rs` | Computes per-category lower bounds from Jeffreys posterior and Hoeffding instruments, then rejects any category that drops below its committed floor. |
| **Ville e-process monitors** | `python3 scripts/gauntlet_cert.py --eprocess-fold test-log.ndjson --eprocess-state /tmp/focr-eprocess-state.json` | Folds live invariant observations for KV capacity, `K=6848` i32 no-overflow, same-input determinism, and SIMD-vs-scalar bit identity into persistent anytime-valid monitors. |
| **Convergence gate** | `python3 scripts/gauntlet_cert.py --convergence docs/gauntlet/ROUNDS.jsonl` | Requires at least 10 gauntlet rounds and a clean tail before declaring the investigation converged. |
| **Capacity certificate** | `cargo test --locked --test many_pages_without_deadlock capacity_certificate_bounded_stream_soak -- --nocapture` | Exercises the bounded `stream_pages` channel, records queueing percentiles, proves the channel bound, and checks that the kernel pool width stays stable. |
| **Fixture provenance** | `tests/fixtures/MANIFEST.toml`, `tests/fixtures/PROVENANCE.md` | Records how committed fixtures were generated, including armed and unarmed scorecard examples. |

For an armed release receipt, provide the real model and fixture roots:

```bash
FOCR_FIXTURES_DIR=/path/to/fixtures/native_f32 \
FOCR_MODEL_PATH=/path/to/model-00001-of-000001.safetensors \
scripts/ladder_scorecard.sh --out scorecard.json
```

The scorecard is the compact receipt. The adjacent `scorecard.raw.ndjson` file is the audit trail.

The broader gauntlet has its own self-test and scorecard path:

```bash
python3 scripts/gauntlet_cert.py --self-test
python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md \
  --scorecard-out /tmp/focr-release-scorecard.json
python3 scripts/gauntlet_cert.py --bundle .gauntlet-output/bundle
# Finalize only with the workflow manifests and signer identities described in
# docs/gauntlet/RELEASE_CERTIFICATION_TEMPLATE.md.
python3 scripts/gauntlet_cert.py --release-readiness \
  --readiness-out .gauntlet-output/release_readiness.json
```

The committed gauntlet bundle stays conservative: its surface lower bound is not inflated beyond the evidence in `docs/FEATURE_PARITY.md`, and its release certificate remains `certified:false` unless the strict finalizer verifies fresh workflow evidence, production audit receipts, and three registry-pinned OpenPGP signers. v0.7.0 publishes locally and remotely hash-verified release bytes without claiming that unavailable governance certificate.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `FOCR_MODEL_PATH` | Override the model artifact path (a `.focrq` blob or a safetensors directory). For the current default, point this at raw BF16 weights or an artifact using exact recipe `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`; the historical schema-v1 full-int8 pull is rejected. |
| `FOCR_MODEL_DIR` | Add one or more model search roots before the default cache. A bare `focr ocr` can resolve pulled artifacts from this directory without `--model`. |
| `FOCR_QUANT` | Pick the quant-suffixed artifact name during cache resolution when multiple variants are present. |
| `FOCR_MANIFEST_URL` | Override the release-bound embedded manifest with a local path or an HTTPS URL. |
| `FOCR_RUN_STORE` | Override the local run-history database path. Defaults to `~/.cache/franken_ocr/runs.db`. |
| `FOCR_MMAP` | Opt into read-only mmap for trusted artifact inodes that are guaranteed immutable for the process lifetime. Unset is the safe owned-buffer default. |
| `FOCR_NO_REPEAT_NGRAM` | Override the sliding no-repeat n-gram size for decode (default 35). |
| `FOCR_GOT_NO_REPEAT_NGRAM` | Override the GOT-OCR2 global no-repeat n-gram size (default 20, matching the upstream model; `0` disables the repetition guard). |
| `FOCR_GOT_FORMAT` | Force GOT-OCR2's `OCR with format:` structured-output mode, the env analog of `--format` and the format-implying `--task` values. |
| `FOCR_SMOLVLM2_QUESTION` | The smolvlm2 describe/VQA question (the env analog of `--question`; the CLI flag outranks it; default: the model-card caption prompt). |
| `FOCR_TROMR_SAMPLE` | Enable TrOMR's upstream top-k/T=0.2 sampling arithmetic from a deterministic PCG32 seed. Unset uses the default per-head argmax path. |
| `FOCR_TROMR_SEED` | Seed for `FOCR_TROMR_SAMPLE`; defaults to `0`. Same seed means the same TrOMR decode stream on every supported CPU. |
| `FOCR_TROMR_SPLIT` | `1` arms the experimental barline-split rescue for staff bands too wide for the 1280-column position budget. Off by default: split segments lose absolute pitch registration (measured), so the default is an honest per-staff skip. |
| `FOCR_MAX_NEW_TOKENS` | Cap the number of generated tokens (the engine's `max_length`; default 32768). An explicit `--max-length` flag outranks it. Capping never changes the per-step math, so a capped run's tokens are a true prefix of the full run's. |
| `FOCR_DECODE_INT8` | Request the experimental all-int8 Unlimited-OCR decoder. Truthy values require both `FOCR_INT8_ATTN=1` and `FOCR_INT8_LMHEAD=1`; otherwise the command fails with a usage error. The default is the conservative recipe path. |
| `FOCR_INT8_ATTN` | Independent accuracy gate for int8 attention q/k/v/o weights. Default off. It does not select the all-int8 decoder by itself. |
| `FOCR_INT8_LMHEAD` | Independent accuracy gate for int8 `lm_head`. Default off. It does not select the all-int8 decoder by itself. |
| `FOCR_DECODE_STATELESS` | Force the stateless re-prefill decoder, kept as a parity oracle for the cached decode path. |
| `FOCR_QKV_FUSED` | Controls fused q/k/v projection inside the experimental all-int8 decoder. Default is on within that experiment; set `0`, `off`, `false`, or `no` for its older three-call comparison path. |
| `FOCR_BATCH_SPINE` | Arm continuous batching. For Unlimited-OCR, the current R-SWA spine requires the explicitly gated all-int8 decoder; without it, pages use the conservative sequential path. The dense zoo spine covers GOT-OCR2, SmolVLM2, and OneChart. Set `1`, `on`, or `true` to arm it. |
| `FOCR_BATCH_SIZE` | Maximum in-flight stream count for the continuous-batch spine. Defaults to 128 when the spine is armed; blank, invalid, or `0` also use that default, and larger values are capped at 256. |
| `FOCR_BATCH_PACK` | When present, admit pending streams by similar prefill length inside the batch scheduler. Output order is restored before return, and each stream's tokens stay independent; unset preserves submission-order admission. |
| `FOCR_BATCH_VISION` | Inside the batch spine, run the vision tower batched across pages (the default). `0`/`off`/`false`/`no` reverts to the per-page vision loop. Read only when the spine is armed. |
| `FOCR_UNLIMITED_VISION_CACHE` | Cache Unlimited-OCR's immutable SAM weights and pretransposed bridge projector on the loaded model (default). `0`/`off`/`false`/`no` restores per-page hydration for diagnostics and memory comparisons. |
| `FOCR_TIMING` | Emit nested native-forward timing rows for performance work, including SAM hydrate/forward/block/attention/MLP splits and decode/output stages. Interactive progress is disabled while this line-oriented trace is active. |
| `FOCR_NO_PROGRESS` | Disable the interactive stderr progress bar shown during multi-page PDF and sequential `ocr-batch` runs (any value other than empty or `0`). The bar already auto-disables when stderr is not a TTY and in `--robot`/`--json` modes; this is the explicit opt-out for TTY sessions. |
| `FOCR_FORCE_ARCH` | Force an available SIMD tier (`sdot`/`smmla`/`scalar`/`avx2`/`avxvnni`/`avx512vnni`) for CPU dispatch; used by `robot selftest`, `robot backends`, and pinned perf runs. |
| `FOCR_INT8_AUTOVEC` | Apple-only ordinary dense-int8 route switch. Default uses measured-faster LLVM autovec; `0`/`off`/`false`/`no` restores SDOT when no valid `FOCR_FORCE_ARCH` override is active. It does not affect packed-int4 or offline-packed SMMLA paths. |
| `FOCR_RESAMPLE` | Preprocess resampling kernel. Unset (default): the `image` crate's CatmullRom. `pil-bicubic`: a Pillow-bit-exact fixed-point BICUBIC at every resize site, for reference-exact comparison against the PIL/torch oracle (DISC-001 in `docs/DISCREPANCIES.md`). |
| `FOCR_STAGE_BUDGET_FORWARD_MS` | Override the forward stage budget in milliseconds (default 600000, i.e. 10 minutes). |
| `HOME` | Required for cache resolution; the model cache installs to `~/.cache/franken_ocr/models`. |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Successful completion |
| 1 | Generic error or a not-yet-implemented surface |
| 2 | Usage or CLI argument error |
| 3 | Model artifact not found or could not be resolved |
| 4 | Input image or page could not be decoded |
| 5 | Budget or timeout exceeded |
| 6 | Operation cancelled cooperatively |
| 7 | Format or version mismatch |

---

## Architecture

Unlimited-OCR is a roughly 3B-parameter Mixture-of-Experts vision-language model and a DeepSeek-OCR derivative. The forward path:

```
  page.png
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DeepEncoder (vision tower, kept high precision)                       │
│    SAM-ViT-B  ──►  16x conv token compressor  ──►  CLIP-L/14           │
│    (SAM + CLIP features concatenate to 2048 dims)                      │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Linear projector   2048 ──► 1280                                      │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DeepSeek-V2 MoE decoder   12 layers, hidden 1280, 10 heads, hd 128    │
│    layer 0 : dense MLP                                                 │
│    layers 1..11 : MoE  ── router 1280 ──► 64, top-6 of 64 experts      │
│                        (each expert 1280 ──► 896, SiLU-gated)          │
│                        + 2 always-on shared experts                    │
│    attention replaced by R-SWA (window 128)                           │
│    final RMSNorm  ──►  lm_head  1280 ──► 129280                        │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
  autoregressive text  ──►  Markdown / JSON / NDJSON

  ── ISA dispatch (chosen at load, one binary per arch) ──
     aarch64 : NEON / SDOT / SMMLA (i8mm)
     x86_64  : AVX2 / AVX-VNNI / AVX-512-VNNI
```

- **DeepEncoder (vision tower).** SAM-ViT-B, then 16x conv token compression, then a CLIP-L/14 cascade. SAM and CLIP features concatenate to 2048 dims. Kept high precision; quantizing the vision tower hurts OCR, a result both community quants confirm.
- **Linear projector.** A single 2048 to 1280 map bridges the vision tower to the decoder hidden size.
- **DeepSeek-V2 MoE decoder.** 12 layers, hidden 1280, 10 heads, head_dim 128. Layer 0 is a dense MLP; layers 1 to 11 are MoE, with a router (1280 to 64) selecting the top 6 of 64 experts (each a 1280 to 896 SiLU-gated MLP) plus 2 always-on shared experts. Final RMSNorm and `lm_head` (1280 to 129280) feed autoregressive sampling.
- **R-SWA.** The decoder's attention novelty; generated-token KV is bounded to a 128-token window while the reference block stays as a frozen, never-evicted global KV.

The upstream checkpoint is a single 6.67 GB bf16 safetensors shard under the MIT license. `focr convert` turns the decoder FFN/expert GEMMs into int8 inside the `.focrq` format and leaves the vision tower, projector, embeddings, router, and norms high precision.

---

## Troubleshooting

### "model artifact was not found" (exit 3)

For the default Unlimited-OCR runtime, point to the raw BF16 checkpoint or a locally converted artifact with the exact conservative recipe:

```bash
export FOCR_MODEL_PATH=/path/to/Unlimited-OCR
focr ocr page.png
```

Run `focr pull` to install the v0.7.0 conservative exact-recipe artifact. If a download is interrupted, rerun the command: the pull lock is process-scoped, orphan staging files are diagnosed by `focr doctor`, and no artifact becomes visible until every part and the final reassembled hash verify. Use `focr models` to inspect the complete model zoo.

### Wrong or missing SIMD tier

Check what the binary detected on this host, then confirm the int8 kernel is correct:

```bash
focr robot backends     # effective dense route + hardware tier/capabilities
focr robot health       # model present, arch features, thread budget
focr robot selftest     # int8 GEMM bit-identical to scalar oracle (exit 1 on divergence)
```

To force a specific tier for verification, set `FOCR_FORCE_ARCH` (for example `FOCR_FORCE_ARCH=scalar focr robot selftest`).

### Checksum mismatch on a manual download

Compatible `focr pull <model>` commands verify every part and the reassembled file automatically. If a manual download fails `shasum -a 256 -c` (or `sha256sum -c`), it is corrupt or truncated. A manifest, recipe, format, or version mismatch surfaces as exit code 7; do not bypass it by renaming an artifact.

### Running fully offline

Once compatible weights are installed, inference touches no network. To pin raw or locally converted Unlimited-OCR weights explicitly:

```bash
export FOCR_MODEL_PATH=/path/to/unlimited-ocr.focrq
focr ocr page.png
```

`FOCR_MODEL_PATH` accepts a `.focrq` blob or a safetensors directory.

### Experimental all-int8 repetition on a dense table

The legacy all-int8 experiment can repeat on a few hard, dense tables. The v0.7.0 conservative exact-recipe artifact with exact Torch routing terminates on `page0590` and clears the corpus budget without the optional runaway controller, though the hard-page CER tail remains high. When reproducing the rejected legacy experiment, one mitigation is a tighter n-gram guard; it is not a release certification:

```bash
# Reproduce the gated all-int8 path with a tighter guard.
FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
  FOCR_NO_REPEAT_NGRAM=20 focr ocr hard_table.png
```

Drop `FOCR_DECODE_INT8` to return to the conservative default. The native
all-high-precision diagnostic is explicitly stateless and quadratic:

```bash
FOCR_DECODE_STATELESS=1 \
  FOCR_MODEL_PATH=/path/to/unlimited-ocr-bf16 \
  focr ocr page.png
```

---

## Limitations

What this is and is not:

- **Unlimited-OCR has a documented hard-page tail.** The v0.7.0 exact-recipe artifact keeps attention and `lm_head` high precision, emits EOS on `page0590`, and passes the complete 20-page CER budget, but `page0590` remains high at normalized CER 0.6164886. The release is hash-verified and does not claim the unavailable three-party OpenPGP certificate. For the slow native all-high-precision diagnostic, combine raw BF16 with `FOCR_DECODE_STATELESS=1`; raw storage alone still takes the conservative cached int8 FFN/expert route.
- **TrOMR int8 is default, with a documented fallback.** `focr pull tromr` fetches the smaller `tromr.int8.focrq` artifact. The corpus gate stays green, but the discrepancy ledger records one degraded public-domain Spohr page where int8 and f32 diverge into different bad token streams. Use `focr pull tromr --quant f32` and `--model tromr.focrq` when you need bit-exact reference behavior.
- **Image and PDF input.** PNG, JPG, and similar, plus native PDF: `focr ocr file.pdf` rasterizes pages in process (pure Rust, no FFI, no out-of-band `pdftoppm`) and OCRs the document. The renderer applies page-level and image-placement rotations before OCR, so common scanned-book PDFs that display a rotated image through the content stream are not fed sideways to the model. `--pages` lets you run only the source pages you need, and `--split-spreads` can split common two-page book scans when the gutter is clear enough. The fast path covers common scanned-PDF codecs: JPEG (`DCTDecode`), CCITT Group 4 fax, and `FlateDecode`/LZW raw rasters. Two image codecs with no production-quality pure-Rust decoder, `JPXDecode` (JPEG 2000) and `JBIG2Decode`, plus born-digital vector/text pages, are reported with a precise error naming what was unsupported. Rasterize that PDF out of band and retry.
- **Native Windows x86_64 is proven end-to-end; ARM64 has a narrower proof.** The `x86_64-pc-windows-msvc` binary runs full OCR on real Windows 10 with the same model, vision tower, and DeepSeek-V2 decoder as macOS and Linux. The ARM64 MSVC binary builds and passes the no-weights CLI/robot smoke suite on a native hosted Windows 11 ARM runner (dist run `29048523156`), and the PowerShell installer selects its stable asset name. Full model OCR and installer e2e on an ARM64 Windows machine remain release-verification obligations. `focr.exe robot selftest` covers the int8 GEMM scalar-oracle contract, including K=6848. `focr pull` performs native multi-part download, reassembly, and SHA-256 verification. The model cache resolves to `%LOCALAPPDATA%\franken_ocr\models`, falling back to `%USERPROFILE%\.cache\franken_ocr\models`; on macOS and Linux it stays at `~/.cache/franken_ocr/models`.
- **A few models, not any model.** Generality is a deliberate non-goal. `franken_ocr` runs a small family of hand-ported models: Unlimited-OCR by default, GOT-OCR2 for structured formats, SmolVLM2 for image description/VQA, OneChart for chart-to-data extraction, and TrOMR for sheet-music OMR on full printed/scanned pages or staff crops. TrOCR and pix2tex remain descriptor-only roadmap items. Each ready model is transformed offline, distributed through the manifest with required sidecars, and certified against its reference before it ships. It will not become a generic inference runtime that loads arbitrary checkpoints.
- **Not benchmark SOTA.** Unlimited-OCR is strong but not the OmniDocBench leader. The aim is fidelity to this model, bounded generated-token KV for long-document parsing on CPU, and speed on commodity hardware, not topping a benchmark.
- **CPU only.** No GPU. CUDA is a deferred stretch goal; CPU stays the product.

---

## FAQ

**Is this affiliated with Baidu?** No. It is an independent pure-Rust reimplementation that runs Baidu's openly-licensed (MIT) model weights. The weights and any quantized derivative this project distributes carry the model notice surfaced by the binary and `.focrq` metadata: `Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License`.

**Why use this instead of llama.cpp or ONNX?** Both are excellent general runtimes. `franken_ocr` is a focused build: a small fixed set of hand-ported models lets the kernels specialize to each model's exact shapes and skip the generality tax. The whole thing ships as one Rust binary with no FFI and is portable to targets where `ort` or CUDA cannot build.

**Why Rust, and why forbid `unsafe`?** Memory safety for a multi-gigabyte weight loader and a tight decode loop, with `unsafe` confined to small audited islands: SIMD modules with bit-identical scalar fallbacks and the optional read-only mmap call. Owned weight bytes are the default.

**Does int8 hurt accuracy?** A historical real-page run measured CER 0.0094 and matched the Baidu reference decode to within one token, but that single row is not a release-wide certificate. The legacy full-int8 artifact repeats on hard dense tables. The v0.7.0 conservative exact-recipe artifact instead emits EOS on `page0590` at 4,033 generated tokens and passes 20/20 pages at aggregate normalized CER 0.19307925 within the 0.25 budget; `page0590` itself remains a high-tail case at CER 0.6164886. Native all-high-precision comparison requires raw BF16 plus `FOCR_DECODE_STATELESS=1`; the default cached route remains mixed precision.

**Can I embed it in my Rust program?** Yes. The library API is synchronous and blocking, and the engine owns its runtime internally, so there is no async plumbing to thread through your code.

**Where do the weights live, and do they ever download at inference?** Compatible zoo pulls cache under `~/.cache/franken_ocr/models`; raw or locally converted Unlimited weights can live anywhere and be selected with `FOCR_MODEL_PATH`. Weights are never bundled or downloaded during `focr ocr`; inference is offline.

**Which binary do I download?** One per architecture. The `x86_64` binary covers AVX2/AVX-VNNI/AVX-512-VNNI; the `aarch64` binary covers NEON/SDOT/SMMLA. The right tier is chosen at load. Or just use the curl installer and let it pick.

---

## About Contributions

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

The `franken_ocr` source code is licensed under the [MIT License with an OpenAI/Anthropic Rider](./LICENSE), Copyright (c) 2026 Jeffrey Emanuel.

The model weights are a separate matter. The Baidu Unlimited-OCR weights, and any quantized derivative this project distributes, are under the MIT License, Copyright (c) 2026 Baidu, reproduced in full in [`LICENSE`](./LICENSE) under "THIRD-PARTY MODEL WEIGHTS, NOTICE". That notice travels with any distributed weight artifact.

## See also

- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md), the master plan: architecture census, quantization format, the kernel-optimization catalog, the alien-artifact math families, the verification gauntlet, and the phased roadmap.
- [`AGENTS.md`](./AGENTS.md), conventions for human and agent contributors, including the engineering doctrine.
- [`CHANGELOG.md`](./CHANGELOG.md), the project history.
- [`docs/PERF_LEDGER.md`](./docs/PERF_LEDGER.md), the honest measured perf-ratio log.
- [`docs/FEATURE_PARITY.md`](./docs/FEATURE_PARITY.md), the FeatureUniverse and SurfaceMatrix scoreboard read by the release gauntlet.
- [`docs/gauntlet/METHODOLOGY.md`](./docs/gauntlet/METHODOLOGY.md), the three-pillar release certification design.
- [`docs/gauntlet/RELEASE_SCORECARD.json`](./docs/gauntlet/RELEASE_SCORECARD.json), the current surface-pillar gauntlet scorecard artifact.
- [`docs/gauntlet/EPROCESS_STATE.json`](./docs/gauntlet/EPROCESS_STATE.json), the persisted invariant-monitor state.
- [`docs/NEGATIVE_EVIDENCE.md`](./docs/NEGATIVE_EVIDENCE.md), what did not work, including results inherited from sibling projects.
- [`docs/DISCREPANCIES.md`](./docs/DISCREPANCIES.md), known measured divergences from the reference model.
- [`docs/conformance/LADDER_HARNESS.md`](./docs/conformance/LADDER_HARNESS.md), the L0-L5 scorecard runner and parity-receipt contract.
- [`docs/conformance/RATCHET.md`](./docs/conformance/RATCHET.md), the conformal lower-bound release ratchet.
