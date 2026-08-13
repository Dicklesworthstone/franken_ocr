---
license: mit
library_name: franken_ocr
pipeline_tag: image-to-text
tags:
  - ocr
  - document-understanding
  - quantized
  - int4
  - rust
  - cpu
  - webassembly
---

# franken_ocr weights

Quantized weight artifacts for [`franken_ocr`](https://github.com/Dicklesworthstone/franken_ocr)
— a pure-Rust, memory-safe, CPU-only OCR engine that runs a small family of
hand-ported vision-language models with no general ML framework: no PyTorch, no
Python, no CUDA, no FFI at inference, no GPU.

These are **runtime artifacts in the project's own `.focrq` container**, not
Hugging Face `transformers` checkpoints. They are consumed by the `focr` CLI,
the browser playground at [franken-ocr.com](https://franken-ocr.com), and the
FrankenOCR iOS app. They will not load in `transformers`.

## Why this mirror exists

GitHub release assets return 503 under load and cap a single asset at 2 GiB,
which is why the largest artifact ships there as byte-split parts. This mirror
serves the same bytes from a CDN, with ranged requests and CORS, so a browser
can fetch a model directly and a phone can resume an interrupted download.

Every file is pinned by SHA-256 in
[`models/manifest-v2.json`](https://github.com/Dicklesworthstone/franken_ocr/blob/main/models/manifest-v2.json)
and verified byte-for-byte before installation. A mismatched or truncated
download can never hydrate.

## Files

| file | bytes | SHA-256 | recipe |
|---|---:|---|---|
| `unlimited-ocr.wasm-int4.focrq` | 3,003,988,117 | `2653831ccd7f481f898f80ae5c95fa1ec7ee2a5a18005d3c927ddf64ed75e187` | `unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1` |
| `tokenizer.json` | 9,979,544 | `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4` | — |

### `unlimited-ocr.wasm-int4.focrq`

Baidu Unlimited-OCR (a DeepSeek-OCR derivative: SAM-ViT-B → 16× conv token
compressor → CLIP-L/14 → linear projector → 12-layer DeepSeek-V2 MoE decoder
with Reference Sliding Window Attention) transformed from the 6.67 GB bf16
checkpoint to 3.00 GB:

- **MoE routed + shared experts → int4**, group size 16 for `gate_proj` and
  `down_proj`, 32 for `up_proj` — the measured sensitivity order.
- **Attention q/k/v/o, `lm_head`, `embed_tokens` → int8** per-channel.
- **Vision tower, projector, MoE router gate, and all norms stay BF16.**
  Quantizing the vision encoder wrecks OCR; every downstream token is
  conditioned on its 256 outputs, so error there is not local.

Quantization is calibration-aware (importance-weighted clip search plus an AWQ
`down_proj` fold over a 13-page activation-statistics run), measured strictly
better than plain round-to-nearest on every corpus page.

**Honest accuracy note.** int4 costs something real on dense material: on an
1886 newsprint page the measured character error rate is 0.156. Calibration
bought a 10.9% relative reduction on the hardest page without regressing any
page — a real improvement, not a fix. The per-page receipts live in
[`docs/DISCREPANCIES.md`](https://github.com/Dicklesworthstone/franken_ocr/blob/main/docs/DISCREPANCIES.md).

This artifact is **not** the native CLI default. `focr pull` installs a separate
conservative 4.16 GB artifact that keeps attention and `lm_head` at high
precision. This one exists for memory-constrained targets: the browser, where
the whole model must fit WebAssembly's 4 GiB linear memory, and phones.

## Usage

```bash
# CLI
focr pull                      # installs the conservative native default
FOCR_MODEL_PATH=/path/to/unlimited-ocr.wasm-int4.focrq focr ocr page.png
focr ocr book.pdf --pages 3,5-9 -o excerpt.md
```

The tokenizer sidecar must sit beside the artifact.

## Licenses

The engine is MIT (with an OpenAI/Anthropic rider), Copyright (c) 2026 Jeffrey
Emanuel.

The weights are a separate matter. This artifact is a quantized derivative of
Baidu's openly-licensed model, and its notice travels with it:

> **Baidu Unlimited-OCR — Copyright (c) 2026 Baidu, MIT License**

The original checkpoint is [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
(source SHA-256 `2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6`).

## Not affiliated with Baidu

This is an independent reimplementation that runs Baidu's openly-licensed
weights. No endorsement is implied.
