# franken_ocr — Performance Ledger

> Head-to-head, **MEASURED** performance log for the `focr` engine. Every row is
> a real wall-clock measurement against a real reference on the same machine.
> No row is added without a number; projections and targets do not go here (they
> live in `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`). Levers that show ~0 gain or a
> regression are reverted and recorded in `docs/NEGATIVE_EVIDENCE.md`, not here.

## Measurement protocol

- **Correctness reference** is the pinned CUDA PyTorch / `transformers`
  Unlimited-OCR model (bf16) from `scripts/gen_reference_fixtures.py`.
- **CPU performance reference** is whichever CPU baseline Phase -1 proves is
  valid and runnable on the same host: CPU-patched HF if it reproduces the CUDA
  oracle's tokens, otherwise llama.cpp GGUF / ONNX Runtime / MLAS labeled as such.
- **`focr`** is measured in the `release-perf` profile (`debug=line-tables-only,
  lto=thin, codegen-units=1`), warm, with a fixed thread budget recorded per row.
- **Precision column** states what is being compared: `focr-int8` (or `-int4`)
  vs `torch-bf16`. A speed ratio is only meaningful alongside the accuracy delta
  for that precision (see `docs/DISCREPANCIES.md`).
- **ratio** = reference_time / focr_time (>1.0 means focr is faster). Stages are
  measured per the pipeline boundary they name.

## Ratio table

| date | claim_id | evidence_id | model_commit | fixture_hash | arch/cpu_features | stage | focr | reference | ratio | precision (focr vs ref) | thread budget | command/env | fallback/kill-switch state | notes |
|------|----------|-------------|--------------|--------------|-------------------|-------|------|-----------|------:|-------------------------|---------------|-------------|----------------------------|-------|
| _—_  | _—_      | _—_         | _—_          | _—_          | _—_               | _—_   | _—_  | _—_       |  _—_  | _—_                     | _—_           | _—_         | _—_                        | _no measurements yet_ |

**Stage vocabulary:** `preprocess` (image decode/resize/normalize) · `vision-encode`
(DeepEncoder + projector, per page) · `prefill` (build reference KV: visual + prompt) ·
`decode-per-token` (R-SWA + MoE, amortized per output token) · `end-to-end`
(`focr ocr`, image in → text out).

---

_No performance numbers recorded yet. The inference path is not implemented, so
there is nothing to measure. This table stays empty until a real head-to-head
ratio exists — no fabricated or projected numbers._
