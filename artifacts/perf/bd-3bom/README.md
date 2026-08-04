# bd-3bom — GOT-OCR2 decode: lm_head per-token 622 MB re-transpose (95 % of decode)

**Lever:** precompute the tied `lm_head`/`embed_tokens` transpose **once** instead of
re-transposing the full `[vocab=151860, hidden=1024]` matrix (~0.6 GB) on every decode
step. **Verdict: WIN, bit-identical.** 1 lever, 0 revert.

## What was wrong

`focr ocr` with `got-ocr2.int8.focrq` took **487 s** for one page (`page_0107.png`). The
`FOCR_TIMING` stage breakdown (added in this change, permanent, gated) isolated it:

| stage | before | after |
|---|--:|--:|
| vision + splice | 5.74 s | 5.84 s |
| seed prefill | 1.13 s | 1.16 s |
| decoder layers (attn + gemv) | 12.09 s | 12.49 s |
| **lm_head** | **463.74 s** | **16.18 s** |
| decode tok/s (688 tok) | 1.4 | **23.9** |
| **page total** | **487 s** | **41 s** |

The decode itself (layers/attn/gemv) and vision were already fast. `decoder::lm_head_proj`
→ `decoder::linear_no_bias` transposes the entire `[vocab, hidden]` head weight from
scratch **on every call**, and the O(n)-per-token KV-cache decode calls it once per
generated token (688×). ~688 × ~0.9 s strided 622 MB transpose ≈ 590 s.

## The fix

- `decoder::norm_and_lm_head_pretransposed(hidden, norm_w, head_wt, eps)` — `rms_norm`
  then the **same** `nn::matmul` against an already-`[hidden, vocab]` weight.
- `GotDecodeWeights::build` transposes the tied embedding to `[hidden, vocab]` **once**
  (`lm_head_t`), reusing it for every decode step. The one-shot seeding prefill keeps the
  naive path.

Same `rms_norm`, same `nn::matmul`, same values → **bit-identical logits → identical
argmax → identical token stream**. `lm_head` stays **F32** (NOT quantized), so doctrine #2
(int8-lm_head only behind a measured-CER kill-switch) is not touched and the L4 torch-oracle
cert (`kvcache_greedy_matches_oracle_l4`) is unaffected.

## Correctness proof

- `OUTPUT_IDENTICAL=yes` in both `perf_before_page0107.txt` and `perf_after_page0107.txt`:
  `cmp -s` of the decoded text vs the committed baseline (`page_0107.got.txt`).
- Unit/parity gate: `cargo test` green (incl. `decoder_matches_torch_oracle`,
  `kvcache_greedy_matches_oracle_l4`) — the pretransposed path is exercised by the KV-cache
  decode the L4 gate drives.

## Provenance

- **model:** `got-ocr2.int8.focrq` (GOT-OCR2.0, `bd-3jo6` sub-epic).
- **host/arch:** Apple M4, `aarch64+neon+dotprod` (SDOT) dispatch.
- **fixture:** `page_0107.png` (navy-history book page; the same page the B8 CER used).
- **command:** `FOCR_TIMING=1 focr ocr --model got-ocr2.int8.focrq page_0107.png`.

Raw before/after timing lines are in the two `*.txt` files; `SHA256SUMS` anchors them.
