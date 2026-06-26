# Decoder + end-to-end parity: franken_ocr vs baidu/Unlimited-OCR

Continues [`VISION_PARITY_RESULTS.md`](VISION_PARITY_RESULTS.md). The DeepSeek-V2
MoE decoder + `lm_head` were wired to the real `Weights` accessors and validated
against the pinned baidu reference (truth-pack modeling code, rev `3a7f4dbb`,
bf16) on a real scanned page (`royalnavy02clow.pdf` p.9, 200 DPI). The same
decouple-via-hooked-fixture method as the vision proof is used: franken_ocr runs
on baidu's **exact** intermediate tensors so each stage is isolated.

## Parity ladder — the COMPLETE forward, stage by stage (page 9)

| Stage | Entry point | vs baidu |
|---|---|---|
| Preprocess | `preprocess_image(Base{1024})` | cosine 0.99891 |
| SAM ViT-B | `vision_sam::forward` | cosine 0.99992 |
| CLIP-L/14 | `vision_clip::forward` | cosine 0.99921 |
| Projector (vision tower) | `vision_bridge::forward` | cosine 0.99964 |
| **Vision-token scatter** | `connector::masked_scatter` + `image_newline`/`view_seperator` | **cosine 0.99964** |
| **Decoder hidden** [277,1280] | `decoder::forward` (12L MoE, no-cache prefill) | **cosine 0.99995** |
| **lm_head logits** [129280] | `decoder::lm_head` | **cosine 0.99990** |
| **First-token argmax** | greedy | **baidu 128818 == franken 128818, EXACT** |

The scatter cosine being bit-identical to the bridge cosine (0.99964, max|Δ| 0.119)
proves the `image_newline`/`view_seperator` placement and the masked-position
ordering are exactly right (273 = 16×17 + 1 slots → 273 mask-True positions).
Top-5 logits are an identical set `[128818, 100855, 128819, 122875, 20852]`.
Residuals everywhere are pure bf16(baidu)→f32(franken) numeric drift (the engine
runs fp32 here; quantized kernels are a separate, later comparison).

### Name trap resolved
The DeepSeek-V2 MHA path (`use_mla=false`, `SlidingWindowLlamaAttention`) uses
standard NEOX `rotate_half` RoPE (`_llama_apply_rotary_pos_emb`), **not** the
interleaved `apply_rotary_pos_emb` defined elsewhere in `modeling_deepseekv2.py`.
Using the interleaved variant would silently corrupt attention.

## End-to-end OCR (true pipeline, franken's own preprocess → text)

`examples/e2e_ocr.rs` assembles the standalone pipeline (preprocess → SAM/CLIP/
bridge → `embed_tokens` + `masked_scatter` → `decoder::forward` → `lm_head` →
greedy no-KV-cache decode), bypassing the still-`NotImplemented` `mod.rs` CLI glue.
Output ids are detokenized with the reference tokenizer and scored by
`scripts/baseline/compare_ocr.py` against the baidu oracle.

**Page 9 result: CER 3.36 %** (9 edits / 268 chars). Body text is essentially
perfect — e.g. `WILLIAM CLOVES AND SONS, LIMITED,` exact; franken even reads
`CHARING` (correct) where the oracle has `CHAIRING`. With baidu's exact vision
input, franken reproduces baidu **token-for-token for 73 tokens with identical
bounding boxes**.

## Known residual gaps (both OUTSIDE the proven decoder/scatter math)
1. **Preprocess fidelity** — franken's `preprocess_image` differs slightly from
   baidu's pad/normalize (`sam_in` cosine 0.99891, max|Δ| 0.84), which is what
   raises the true-e2e scatter from 0.99964 (decoupled) to 0.9893 and accounts for
   most of the 3.36 % CER. A preprocessing nuance, not a forward bug.
2. **Decode policy** — the correctness-first greedy loop omits baidu's
   `no_repeat_ngram_size=35` processor, so it can enter a tail repetition loop on
   some pages. A decode-loop policy gap, trivially added; does not affect the
   proven prefill/argmax parity.

## Performance note
The no-KV-cache decoder is O(n²) (full recompute per step): **~5.3 s/tok** fp32 on
M4 — fine for short pages (page 9 ≈ 99 tokens) but the R-SWA bounded KV cache
(window 128, `bd-1gv.17`) is the headline decode-speed lever for long documents,
and int8 quantization of the decoder FFN/expert GEMMs (the validated quant recipe)
is the throughput lever — both are the next phase, on top of this proven-correct
forward.
