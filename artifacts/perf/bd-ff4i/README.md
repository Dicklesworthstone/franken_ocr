# bd-ff4i evidence — GOT global no-repeat-ngram=20 guard

Fix: wire the upstream HF-builtin **global** `no_repeat_ngram_size=20` (spec §12
OQ-8, hard-coded in GOT's `chat()`) into BOTH `decoder_qwen2::generate_greedy`
and `generate_greedy_kvcache` via the shared `argmax_no_repeat` pick, which
reuses the sampler's existing `window == 0` global n-gram scan
(`masked_sliding_window_logits_if_needed`). Kill-switch:
`FOCR_GOT_NO_REPEAT_NGRAM` (`0` disables; unset = config default 20).

Host: Apple M4, `got-ocr2.int8.focrq` (int8 + top-K-refine lm_head default ON),
release build, 2026-07-01.

## The bug (before)

`page_0019.png` (a "list of plates" index page): unguarded greedy decode locks
into a ` . ` cycle and emits 8,379 bytes / 4,096 tokens (max_new cap) of
repetition — see `page_0019_unguarded_FOCR_GOT_NO_REPEAT_NGRAM_0.txt`,
reproduced with the kill-switch (`FOCR_GOT_NO_REPEAT_NGRAM=0`), byte-count
matching the original bd-2dlz sweep report exactly.

## The fix (after)

- `page_0019_guarded.txt` — default guarded decode: 1,460 bytes of real page
  content, clean EOS stop at 577 tokens in 9.22 s (62.6 tok/s).
- `page_0107_guarded.txt` — the committed-CER clean page: **byte-identical** to
  the pre-guard fixture `tests/fixtures/got/cer/page_0107.got.txt` (the guard
  never fires on a repeat-free stream); CER vs ground truth 0.0247 (unchanged).
- `oracle_certs.log` — all four env-gated torch-oracle certs PASS on the guarded
  decode with the real int8 model: `kvcache_greedy_matches_oracle_l4`,
  `decoder_matches_torch_oracle`, `recognize_reads_the_sample_image_e2e`
  (committed e2e golden), `prompt_id_oracle_cross_check`.
- `sweep_20_pages.txt` — the bead's required re-sweep: all 20 navy pages decode
  rc=0 with NO runaway (heuristic: zlib ratio of the 600-byte tail < 0.08 flags
  a repetition run; every page passes, including the historically loop-prone
  dense-table page_0590).

Gates: `cargo fmt --check`, `clippy --all-targets -D warnings`, `cargo test
--lib` 748 passed / 0 failed (2 new guard unit tests + the census assert),
ubs-critical proxy 0 hits.
