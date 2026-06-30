# franken_ocr

<div align="center">
  <img src="franken_ocr_illustration.webp" alt="franken_ocr - Pure-Rust CPU-only OCR for Baidu Unlimited-OCR">
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![version: v0.2.0](https://img.shields.io/badge/version-v0.2.0-blue.svg)](https://github.com/Dicklesworthstone/franken_ocr/releases/tag/v0.2.0)
[![status: working](https://img.shields.io/badge/status-working-success.svg)](#quick-example)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![toolchain: nightly](https://img.shields.io/badge/toolchain-nightly-purple.svg)](./rust-toolchain.toml)
[![unsafe: forbidden*](https://img.shields.io/badge/unsafe-forbidden*-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![model: Baidu Unlimited-OCR](https://img.shields.io/badge/model-Baidu_Unlimited--OCR-teal.svg)](https://huggingface.co/baidu/Unlimited-OCR)

</div>

**A pure-Rust, memory-safe, CPU-only OCR engine that runs exactly one model, Baidu Unlimited-OCR, with no general ML framework, no Python, no CUDA, no FFI at inference, and no GPU.** It parses document images into Markdown, JSON, or a versioned NDJSON event stream, on a single static binary that fits in about 5 MB.

<div align="center">
<h3>Quick Install</h3>

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
```

</div>

The installer detects your platform, downloads the right prebuilt binary from the `v0.2.0` release, verifies it by SHA256, and puts `focr` on your PATH. Then `focr pull` fetches the weights once and you run offline forever after.

---

## TL;DR

**The problem.** Baidu Unlimited-OCR is a strong document-parsing model: Markdown, tables, LaTeX, reading order, many pages in one pass. The official stack is Python plus CUDA. Most machines that need OCR (laptops, CI runners, agent hosts, edge boxes) have no usable GPU, and a Python plus CUDA dependency is heavy to ship and awkward to embed.

**The solution.** `franken_ocr` is a library plus a single-binary CLI (`focr`) that runs this one model on CPU with nothing but a Rust binary. It transforms the bf16 checkpoint into a custom int8 format and runs it through kernels written for this model's exact shapes. On a real page measured against the Baidu reference, the end-to-end character-error-rate is **0.0094**; the decode matched the reference to within a single token, and on one token it beat the reference. That is a measured result on the 6.67 GB model, not a target.

### Why `focr`?

| Feature | What it does |
|---|---|
| **One static binary** | No Python, no CUDA, no FFI at inference, no GPU. About 5 MB; portable to hosts where `ort`/CUDA cannot build. |
| **Works offline** | `focr pull` fetches and verifies the weights once into `~/.cache/franken_ocr/models`; inference never touches the network. |
| **int8 decode, ~2.5x faster** | Custom `.focrq` int8 expert/FFN weights, byte-identical to the f32 path on typical pages. The vision tower stays high precision, where quantizing it would wreck OCR. |
| **Runtime ISA dispatch** | One binary per architecture selects the best int8 kernel tier at load via CPU feature detection: ARM SDOT / SMMLA (i8mm), x86 AVX2 / AVX-VNNI / AVX-512-VNNI. |
| **Bounded long-doc memory** | R-SWA keeps generated-token KV constant (window 128) while the reference block is held as a frozen, never-evicted global KV. |
| **Agent-first** | Versioned NDJSON robot mode, a self-describing `robot schema`, stable documented exit codes, deterministic output under fixed sampling. |
| **Provable kernels** | `focr robot selftest` re-runs the dispatched int8 GEMM against a bit-identical scalar oracle on your CPU and emits a single JSON verdict. |
| **Memory-safe** | `#![forbid(unsafe_code)]` everywhere except small audited SIMD islands, each with a bit-identical scalar fallback. |

---

## Quick Example

```bash
# 1. Fetch the int8 weights once (~3.9 GB) into the local cache, verified by SHA256.
focr pull

# 2. OCR a page into Markdown (the human default).
focr ocr page.png

# 3. Same page as structured JSON (markdown + every span's bounding boxes).
focr ocr page.png --json

# 4. Write the result to a file — format follows the extension (.md or .json).
focr ocr page.png -o page.md
focr ocr page.png -o page.json            # structured JSON with bounding boxes

# 5. Also save figures the model can't transcribe (charts/photos) next to the .md.
focr ocr page.png -o page.md --extract-figures   # -> page.md + page_figures/

# 6. Stream NDJSON pipeline events for an agent (run_start ... run_complete; full event set via `focr robot schema`).
focr ocr page.png --robot

# 7. Prove the int8 kernel on THIS CPU is bit-identical to the scalar oracle.
focr robot selftest

# 8. (optional) Convert your own bf16 safetensors into the int8 .focrq format.
focr convert model.safetensors -o unlimited-ocr.focrq --quant int8
```

After step 1 the weights live in `~/.cache/franken_ocr/models` and every later command runs fully offline.

---

## Design Philosophy

**One model, every dimension fixed.** A general ML framework pays a generality tax on every operation: dynamic dtype dispatch, arbitrary shapes, autograd bookkeeping, broadcast machinery, a device abstraction. `franken_ocr` runs exactly one model whose every dimension is known at compile time (hidden 1280, 10 heads, head_dim 128, 64 experts, top-6 routing, MoE intermediate 896, R-SWA window 128, vocab 129280). That buys shape-specialized kernels with no runtime shape branching in the hot loop.

**Offline at inference.** The only network step is `focr pull`, which runs ahead of time. There is no Python, no CUDA, no FFI, and no GPU in the inference path. The async runtime that orchestrates I/O and cancellation is an owned internal detail; the library API is synchronous and blocking, so there is no async plumbing to thread through your code.

**Correctness before speed (always).** A parity gate comes first and a faster kernel that drifts the OCR output is reverted, no source landed, and recorded in the negative-evidence ledger. Speed is shipped on top of parity, never instead of it. The int8 expert/FFN quantization is validated against the f32 path; the vision tower, projector, embeddings, MoE router, and all norms stay high precision.

**Runtime ISA dispatch, one binary per arch.** There is no per-CPU-feature variant to choose. One `x86_64` binary covers AVX2 / AVX-VNNI / AVX-512-VNNI; one `aarch64` binary covers NEON / SDOT / SMMLA. At load, CPU feature detection picks the fastest available int8 kernel tier. On a Threadripper 5995WX (a Zen 3 part whose ceiling is AVX2), dispatch correctly selects AVX2.

**Bounded generated-token KV.** Every decoder attention layer is R-SWA (Reference Sliding Window Attention). Each generated token attends to all reference tokens (visual plus prompt prefix, kept as a frozen, never-evicted global KV) plus only the previous 128 generated tokens through a ring-buffer KV cache. Generated-token KV memory stays constant instead of growing with output length. That, not arbitrary input resolution, is what "Unlimited" means.

---

## How `franken_ocr` Compares

`franken_ocr` is the only one of these built for a single model on CPU.

| | `franken_ocr` v0.2.0 | Official Unlimited-OCR | llama.cpp | ONNX Runtime |
|---|---|---|---|---|
| Language / runtime | Pure Rust, one binary | Python + HF transformers | C++ | C++ |
| Primary target | CPU | CUDA GPU | CPU/GPU | CPU/GPU |
| Scope | This one model | This model | Many models | Many models |
| int8 kernels | Model-specific tiled SDOT/SMMLA/VNNI | n/a | Generic K-quant | MLAS |
| Vision encoder | First-class, kept high precision | First-class | Kept F16 (mmproj) | Depends on export |
| Network at inference | None | None | None | None |
| Ships as | Single static binary, no FFI | Python env + CUDA | Binary + model | Library + model |
| Constant generated-token KV | Yes (R-SWA preserved) | Yes | Depends on PR support | Depends |
| Runs with no GPU | Yes | No (needs CUDA) | Yes | Yes |

**When to use `franken_ocr`:**
- You need this model's output and your host has no usable GPU.
- You want to embed OCR in a Rust program with no Python or FFI.
- You want a single binary you can drop on a CI runner, an agent host, or an edge box.

**When `franken_ocr` might not be ideal:**
- You need a model zoo or a generic inference runtime. `franken_ocr` runs exactly one model, by design.
- You need the OmniDocBench leader; Unlimited-OCR is strong but not the top of the board.

---

## Installation

### Quick install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
```

The script detects your OS and CPU architecture, downloads the matching binary from the `v0.2.0` release, verifies the SHA256 sidecar, and installs `focr`. Under WSL it proceeds as Linux. Under native Git-Bash, MSYS, or Cygwin it points you at the PowerShell installer below and exits cleanly.

On native Windows, install from PowerShell:

```powershell
irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
```

This downloads the `focr-x86_64-pc-windows-msvc.exe` binary, verifies it by SHA256, and puts `focr` on your PATH.

### Manual binary download

Release binaries are raw executables, not tar.gz archives. Each one is a single portable file that dispatches the ISA tier at runtime, so there is one binary per architecture (no per-CPU-feature variant). Sizes are roughly 4.7 to 5.9 MB. Linux binaries are gnu (glibc), not musl.

| Platform | Asset |
|---|---|
| macOS Apple Silicon | `focr-aarch64-apple-darwin-neon-sdot-i8mm` |
| macOS Intel | `focr-x86_64-apple-darwin` |
| Linux x86-64 (glibc) | `focr-x86_64-unknown-linux-gnu` |
| Linux ARM64 (glibc) | `focr-aarch64-unknown-linux-gnu` |
| Windows x86-64 | `focr-x86_64-pc-windows-msvc.exe` |

Each asset has a `<asset>.sha256` sidecar in the standard `"<hex>  <asset>"` format. Download the binary and its sidecar from the release base URL, verify, then install. Example for Apple Silicon:

```bash
base=https://github.com/Dicklesworthstone/franken_ocr/releases/download/v0.2.0
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

`franken_ocr` requires the nightly Rust toolchain pinned in [`rust-toolchain.toml`](./rust-toolchain.toml). `cargo build --release` builds both the `focr` and `franken_ocr` binaries from one shared entrypoint.

The catch: `franken_ocr` path-depends on sibling repositories that are not published on crates.io (`asupersync`, patched to `/dp/asupersync`; `../frankentorch`; `../frankensqlite`). A fresh-clone `cargo build` or a `cargo install --git` will fail to resolve those dependencies. There is no working `cargo install` from crates.io. Prebuilt binaries are the supported path; build from source only if you have those sibling repositories laid out as the workspace expects.

```bash
cargo build --release
# produces target/release/focr and target/release/franken_ocr (identical behavior)
```

---

## Quick Start

1. **Install** the binary with the curl one-liner, or download and verify a release asset by hand.
2. **Fetch the weights once:** `focr pull`. This downloads about 3.9 GB of int8 weights plus `tokenizer.json` into `~/.cache/franken_ocr/models`, verifying every byte by SHA256 against a committed manifest.
3. **Verify your CPU kernel** (optional but reassuring): `focr robot selftest`. Exit 0 means the dispatched int8 GEMM is bit-identical to the scalar oracle on this host.
4. **OCR a page:** `focr ocr page.png` for Markdown, add `--json` for structured output (markdown + bounding boxes), `-o out.md` / `-o out.json` to write a file, or `--robot` for an NDJSON event stream.
5. **Batch many pages** in one process (model loaded once): `focr ocr-batch page1.png page2.png page3.png --json`.

---

## Command Reference

Both `focr ocr` and `focr robot run` accept the same image plus inference-tuning flags. The default crop mode is `gundam` (dynamic-resolution tiling); the alternative is `base`.

### `focr ocr <image>`

Parse a document image into Markdown (default), JSON, or an NDJSON stream.

```bash
focr ocr page.png                          # Markdown to stdout
focr ocr page.png --json                   # structured JSON to stdout
focr ocr page.png -o page.md               # write Markdown to a file
focr ocr page.png -o page.json             # write structured JSON (markdown + boxes) to a file
focr ocr page.png -o page.md --extract-figures   # also save figures, referenced from the .md
focr ocr page.png --robot                  # NDJSON pipeline events
focr ocr page.png --crop-mode base         # disable dynamic-resolution tiling
focr ocr page.png --max-length 4096 --temperature 0.0
focr ocr page.png --model /path/to/unlimited-ocr.focrq
```

**Output (`-o`/`--output FILE`).** Writes the result to a file instead of stdout; the
format follows the extension — `.json` emits structured JSON, any other extension
(e.g. `.md`) emits Markdown. `--json` forces JSON regardless of extension. The
structured JSON carries the rendered `markdown` plus a `layout` array — one
`{label, boxes}` entry per grounded span, where each box is `[x1, y1, x2, y2]` in
source-image pixels (a PDF nests these under a per-page `pages` array). This is the
same shape `--json` prints to stdout.

**Figures (`--extract-figures`).** The model sees figures/photos/diagrams it does not
transcribe to text. With `--extract-figures`, those regions are cropped out of the
source image and saved into a subfolder (default `<output-stem>_figures/`, or set
`--figures-dir DIR`), then referenced from the output: the Markdown gets a real
`![figure N](report_figures/page1_figure_1.jpg)` in place of each figure, and the
JSON gains a `figures` array of `{label, page, bbox, path}`. Each figure's format is
chosen by content — JPG (quality 85) for photographic regions, lossless PNG for
line-art / charts / screenshots. Requires `-o` (or `--figures-dir` for a stdout run).
PDFs name figures per page (`page{N}_figure_{M}`).

Tuning flags: `--base-size` and `--image-size` (preprocessing resolution), `--crop-mode` (`gundam` or `base`), `--max-length` (decode token cap), `--temperature` (sampling), `--no-repeat-ngram` and `--ngram-window` (the sliding no-repeat n-gram decode guard).

### `focr ocr-batch <images...>`

OCR many images in one process, loading the model once and reusing it across all pages.

```bash
focr ocr-batch a.png b.png c.png --json     # one load, many pages
focr ocr-batch *.png --f32                  # use the high-precision f32 decode path
```

`--f32` runs the f32 decode path instead of int8 (see [Limitations](#limitations) for when this matters).

### `focr pull`

Download the int8 weights and tokenizer into the cache, verifying every byte.

```bash
focr pull                                   # default int8, built-in manifest
focr pull --quant int8 --json               # explicit quant, machine-readable
focr pull --manifest ./manifest.json        # override the manifest source
```

Downloads run over asupersync's native HTTP stack (rustls + webpki-roots), reassemble GitHub split parts, verify against a committed SHA256 manifest, and cache to `~/.cache/franken_ocr/models`. Verified in production against this repo's `models-v1` GitHub release and a Hugging Face mirror.

### `focr models`

List the models this build can run — the "model zoo" — with their id, the tasks they serve, and status.

```bash
focr models                                 # human table
focr models --json                          # machine-readable list
```

Today this lists the Baidu Unlimited-OCR model (`ready`). The roadmap (epic `bd-3jo6`) adds specialized models — GOT-OCR2 (math/tables/charts/molecular/geometry/music), SmolVLM2 (photo description / VQA), OneChart (chart → data), Polyphonic-TrOMR (sheet music) — each transformed to the same int8 CPU performance bar; they appear here (`planned`, then `ready`) as they land.

### `focr convert <input>`

Offline weight transformation: a bf16 safetensors checkpoint into a custom int8 `.focrq` artifact.

```bash
focr convert model.safetensors -o unlimited-ocr.focrq --quant int8
focr convert model.safetensors -o out.focrq --quant int8 --json
```

The output is proven byte-identical to the load-time int8 path on the real 6.67 GB model (6,672,547,120 bytes of bf16 become 3,914,093,440 bytes of int8, about 3.9 GB), with the source SHA256 stamped into the header. `--arch <target>` pre-packs register tiles for a specific architecture (defaults to `generic`, an architecture-neutral packing; other targets: `aarch64-smmla`, `x86-vnni`, `x86-amx`). Only int8 is validated today (int4 is not yet validated).

### `focr robot <subcommand>`

Agent-facing surface: versioned NDJSON, self-describing, line-oriented, easy to pipe.

```bash
focr robot run page.png                      # stream OCR events as NDJSON (same flags as ocr)
focr robot schema                            # self-describing, versioned event/contract schema
focr robot health                            # model present? arch features? thread budget?
focr robot backends                          # detected SIMD tiers + core count
focr robot selftest                          # int8 kernel vs scalar oracle on this host
```

`focr robot selftest` runs the dispatched int8 GEMM against a bit-identical scalar oracle across a shape battery, including the worst-case `K=6848` i32-accumulation overflow case, and emits a single JSON verdict; it exits 1 if any parity case diverges. It has passed 24/24 on Apple SDOT and on a real x86 AVX2 box. Set `FOCR_FORCE_ARCH` to verify a specific ISA tier.

### Scaffolded subcommands (not yet implemented)

These emit a JSON structure but return a not-yet-implemented status (exit code 1). They are Phase-0/Phase-5 scaffolds.

```bash
focr runs --limit 10 --format json          # query durable run history
focr sync export-jsonl --json               # export run-state audit records
focr sync import-jsonl --json               # import run-state audit records
focr doctor --json                          # idempotent self-check / repair
```

---

## Environment Variables

| Variable | Effect |
|---|---|
| `FOCR_MODEL_PATH` | Override the model artifact path (a `.focrq` blob or a safetensors directory). When unset, the model cache is searched for `unlimited-ocr.focrq` and the quant-suffixed names a `focr pull` installs (`unlimited-ocr.int8.focrq`, `unlimited-ocr.int4.focrq`), so a freshly-pulled model resolves with no `--model` flag. |
| `FOCR_MANIFEST_URL` | Override the manifest source (a local path or an `https` URL). Defaults to the built-in repo manifest. |
| `FOCR_NO_REPEAT_NGRAM` | Override the sliding no-repeat n-gram size for decode (default 35). |
| `FOCR_FORCE_ARCH` | Force the SIMD tier (`sdot`/`smmla`/`scalar`/`avx2`/`vnni`/`amx`) for CPU dispatch; used by `robot selftest` and SIMD detection. |
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

The weights are not in the cache yet. Fetch them once:

```bash
focr pull
```

On an interactive terminal, `focr ocr` with no model present offers to download. In robot mode and any non-TTY context it never auto-downloads; it returns a model-not-found result plus a `focr pull` hint, so automated callers stay predictable.

### Wrong or missing SIMD tier

Check what the binary detected on this host, then confirm the int8 kernel is correct:

```bash
focr robot backends     # detected SIMD tiers (SMMLA/SDOT/VNNI/AMX/scalar) + core count
focr robot health       # model present, arch features, thread budget
focr robot selftest     # int8 GEMM bit-identical to scalar oracle (exit 1 on divergence)
```

To force a specific tier for verification, set `FOCR_FORCE_ARCH` (for example `FOCR_FORCE_ARCH=scalar focr robot selftest`).

### Checksum mismatch on a manual download

`focr pull` verifies every byte automatically, so prefer it. If you downloaded an asset by hand and `shasum -a 256 -c` (or `sha256sum -c`) fails, the download is corrupt or truncated; re-download the binary and its `.sha256` sidecar from the `v0.2.0` release. A format or version mismatch on a model artifact surfaces as exit code 7.

### Running fully offline

After `focr pull`, nothing else touches the network. To pin the artifact explicitly or to use weights you converted yourself:

```bash
export FOCR_MODEL_PATH=/path/to/unlimited-ocr.focrq
focr ocr page.png
```

`FOCR_MODEL_PATH` accepts a `.focrq` blob or a safetensors directory.

### int8 repetition on a dense table

On most pages int8 is byte-identical to f32, but a few hard, dense tables can send int8 decode into a repetition run. Two mitigations:

```bash
# Tighten the no-repeat n-gram guard.
FOCR_NO_REPEAT_NGRAM=20 focr ocr hard_table.png

# Or fall back to the high-precision f32 decode path.
focr ocr-batch hard_table.png --f32
```

For the single-page `ocr` command, point `FOCR_MODEL_PATH` at the bf16 safetensors directory to run f32 end to end.

---

## Limitations

What this is and is not:

- **int8 can repeat on hard tables.** int8 decode is roughly 2.5x faster and byte-identical to f32 on typical pages, but some dense tables (for example `page_0590`) can trigger repetition runs. The no-repeat n-gram guard and the f32 fallback are the documented kill-switches. The vision tower stays high precision because quantizing it breaks OCR.
- **Image and PDF input.** PNG, JPG, and similar, plus native PDF: `focr ocr file.pdf` rasterizes each page (pure-Rust, no FFI, no out-of-band `pdftoppm`) and OCRs the document. The fast path covers the common scanned-PDF codecs — JPEG (`DCTDecode`), CCITT Group 4 fax, and `FlateDecode`/LZW raw rasters. Two image codecs with no production-quality pure-Rust decoder, `JPXDecode` (JPEG 2000) and `JBIG2Decode`, plus born-digital vector/text pages, are reported with a precise error naming what was unsupported (rasterize that PDF out of band and retry).
- **Native Windows (x86_64) is supported and proven end-to-end; ARM64 is not yet.** The `x86_64-pc-windows-msvc` binary runs full OCR on real Windows 10: the same 3.9 GB int8 weights, vision tower, and DeepSeek-V2 decoder produce the same markdown a Mac or Linux host does. `focr.exe robot selftest` passes 24/24 (int8 GEMM bit-identical to the scalar oracle, including the K=6848 overflow case). `focr pull` works on Windows too — the full 3.9 GB multi-part download, reassembly, and SHA-256 verify complete over the native async HTTP/TLS stack (an earlier send-path bug that surfaced as `WSAENOTCONN` / os error 10057, `bd-15ow`, is fixed). The model cache resolves to `%LOCALAPPDATA%\franken_ocr\models`, falling back to `%USERPROFILE%\.cache\franken_ocr\models`; on macOS and Linux it stays at `~/.cache/franken_ocr/models`. The one remaining gap, tracked under epic `bd-3u97`, is that ARM64 Windows is not published.
- **One model only.** This is a deliberate non-goal to be general. `franken_ocr` will not become a model zoo or a generic inference runtime.
- **Not benchmark SOTA.** Unlimited-OCR is strong but not the OmniDocBench leader. The aim is fidelity to this model, bounded generated-token KV for long-document parsing on CPU, and speed on commodity hardware, not topping a benchmark.
- **CPU only.** No GPU. CUDA is a deferred stretch goal; CPU stays the product.

---

## FAQ

**Is this affiliated with Baidu?** No. It is an independent pure-Rust reimplementation that runs Baidu's openly-licensed (MIT) model weights. The weights and any quantized derivative this project distributes carry the model notice surfaced by the binary and `.focrq` metadata: `Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License`.

**Why not just use llama.cpp or ONNX?** Both are excellent general runtimes. `franken_ocr` is a focused build: a single fixed model lets the kernels specialize to its exact shapes and skip the generality tax, and the whole thing ships as one Rust binary with no FFI. It is portable to targets where `ort` or CUDA cannot build.

**Why Rust, and why forbid `unsafe`?** Memory safety for a multi-gigabyte weight loader and a tight decode loop, with `unsafe` confined to small, audited SIMD modules that each carry a bit-identical scalar fallback.

**Does int8 hurt accuracy?** On a real page the end-to-end character-error-rate is 0.0094 versus the Baidu reference, matching the reference decode to within a single token. The known failure mode is repetition on a few dense tables, mitigated by the no-repeat n-gram guard or the f32 fallback. The vision tower is never quantized.

**Can I embed it in my Rust program?** Yes. The library API is synchronous and blocking, and the engine owns its runtime internally, so there is no async plumbing to thread through your code.

**Where do the weights live, and do they ever download at inference?** They cache to `~/.cache/franken_ocr/models` after `focr pull`. Weights are never bundled and never downloaded during `focr ocr`; inference is offline.

**Which binary do I download?** One per architecture. The `x86_64` binary covers AVX2/AVX-VNNI/AVX-512-VNNI; the `aarch64` binary covers NEON/SDOT/SMMLA. The right tier is chosen at load. Or just use the curl installer and let it pick.

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

The `franken_ocr` source code is licensed under the [MIT License with an OpenAI/Anthropic Rider](./LICENSE), Copyright (c) 2026 Jeffrey Emanuel.

The model weights are a separate matter. The Baidu Unlimited-OCR weights, and any quantized derivative this project distributes, are under the MIT License, Copyright (c) 2026 Baidu, reproduced in full in [`LICENSE`](./LICENSE) under "THIRD-PARTY MODEL WEIGHTS, NOTICE". That notice travels with any distributed weight artifact.

## See also

- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md), the master plan: architecture census, quantization format, the kernel-optimization catalog, the alien-artifact math families, the verification gauntlet, and the phased roadmap.
- [`AGENTS.md`](./AGENTS.md), conventions for human and agent contributors, including the engineering doctrine.
- [`CHANGELOG.md`](./CHANGELOG.md), the project history.
- [`docs/PERF_LEDGER.md`](./docs/PERF_LEDGER.md), the honest measured perf-ratio log.
- [`docs/NEGATIVE_EVIDENCE.md`](./docs/NEGATIVE_EVIDENCE.md), what did not work, including results inherited from sibling projects.
- [`docs/DISCREPANCIES.md`](./docs/DISCREPANCIES.md), known measured divergences from the reference model.
