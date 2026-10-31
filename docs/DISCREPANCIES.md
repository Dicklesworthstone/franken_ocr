# franken_ocr — Known Conformance Discrepancies

This document is the honest-divergence ledger: every place where `franken_ocr`'s
output or behavior **intentionally or measurably differs** from the reference
Baidu Unlimited-OCR model (the PyTorch `transformers` oracle pinned by
`scripts/gen_reference_fixtures.py`).

A discrepancy is only recorded once its impact has been **measured** against the
reference. Speculation does not belong here; the cost of a divergence must be a
real number tied to a real test before it is accepted. Every accepted divergence
carries a **kill-switch** (an environment variable that restores reference
behavior) so it can be toggled off for bit-exact comparison.

This is an **artifact-graph ledger** (plan §8.4): every entry carries the same
FrankenSuite provenance fields as `NEGATIVE_EVIDENCE.md` / `PERF_LEDGER.md`, so a
divergence is reproducible and traceable to the exact model version and command
that measured it.

## Canonical provenance source (the truth pack)

Every entry's `claim_id`/`evidence_id` and provenance fields resolve against the
**Phase −1 truth pack**:

- **Model source commit:** Hugging Face `baidu/Unlimited-OCR`
  **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25 — see
  `docs/truth-pack/PINNED_SOURCES.md`.
- **Source / fixture hashes:** SHA-256 of every load-bearing source in
  `docs/truth-pack/SOURCE_HASHES.md`. The `Reference behavior` of every entry
  cites the oracle code by `(file_sha256, line range)` against that table (e.g.
  `modeling_deepseekv2.py` 74e36e6b… for R-SWA semantics), and the measured impact
  cites the **fixture hash** of the parity corpus it ran against.
- **Runtime pin:** the oracle stack is `torch==2.10.0`, `transformers==4.57.1`,
  `Pillow==12.1.1` (`PINNED_SOURCES.md`). A "measured impact" produced against any
  other stack is **not comparable** and may not be recorded as ACCEPTED.

If `SOURCE_HASHES.md` fails to verify, the model moved: STOP, re-pin, and
re-confirm every entry. A DISC entry whose `Reference behavior` cannot be
resolved to a truth-pack source line is **incomplete**.

## Per-entry schema

```
## DISC-NNN: <short title>
- claim_id / evidence_id: <CLAIM-… → artifacts/perf/<bead>/ or artifacts/parity/<bead>/>
- Provenance (model commit + fixture hash): HF 3a7f4db… + <oracle file_sha256:lines from
    SOURCE_HASHES.md> + parity corpus fixture sha256 + <.focrq sha256 for the precision under test>
- CPU feature string: <dispatched SIMD tier the divergence was observed on, e.g.
    aarch64+neon+dotprod or aarch64+neon+i8mm — a divergence can be arch-specific
    (rounding/order)>
- Exact command + env: <gauntlet/parity invocation + FOCR_*/OMP_NUM_THREADS set>
- Reference behavior: <what the torch/transformers oracle does — quote the source line>
- Our impl: <what franken_ocr does, and where (file:fn)>
- Fallback / kill-switch state: <FOCR_* var, default value, and what the ON value restores>
- Measured impact: <real numbers vs reference — CER / token diff / TEDS / ULP / timing,
    plus the AF-2 tail figure (CVaR_0.1 / EVT_p999) for accuracy divergences>
- Resolution: ACCEPTED / INVESTIGATING / REVERT
- Tests affected: <test names / fixture corpus> (XFAIL, never SKIP — §8.6)
- Review date: <YYYY-MM-DD>
```

`Kill switch` is folded into **Fallback / kill-switch state** (the same field the
other two ledgers carry) so the three ledgers share one provenance vocabulary:
the env var name, its default, and exactly what restoring it gives back
(reference-bit-exact behavior).

Quantization-induced divergences (int8, then int4) are the expected source of
most future entries: each will record the per-bit-width measured accuracy delta
against the bf16 reference, the kill switch (e.g. forcing a layer back to higher
precision via `FOCR_INT8_ATTN=0` / `FOCR_INT8_LMHEAD=0` / dropping a tensor one
tier under AF-1), and the corpus slice (dense text / tables / formulas / numbers)
where the impact was measured — with the AF-2 tail bound, not just the mean,
since exact-token OCR fails in the tail.

---

DISC-009 to DISC-021 are the Unlimited-OCR-2 entries. They resolve against the v2 truth
pack, not the v1 pins above: Hugging Face `baidu/Unlimited-OCR-2`
`5632329a32e103d5f1add270681e0f5c1233c6e4` and the v2 reference stack (torch 2.13.0,
transformers 4.52.1), both in `docs/truth-pack/PINNED_SOURCES.md` under "Unlimited-OCR-2
(v2)". Each title opens with the entry's plan name, U2-k for DISC-U2-k
(`docs/planning/UNLIMITED_OCR_2_PLAN.md` §5.11). Source tags follow the plan's §5:
`MU` `modeling_unlimitedocr.py`, `MD` `modeling_deepseekv2.py`, `SB` `_sparse_base_impl.py`,
`DA` `dash_attention.py`.

## DISC-009: U2-1 — router entmax domain: HF's top-M window by default, SGLang's all-candidates domain selectable

- claim_id / evidence_id: CLAIM-uocr2-entmax-domain → `src/native_engine/srswa.rs`
    (`EntmaxDomain`, `entmax15_topm`, `entmax15_all`, `route`) +
    `srswa::tests::all_candidates_equals_top_m_under_its_preconditions` +
    `real_v2_entmax_domains_decode_s1_alike` (`src/native_engine/mod.rs`) +
    the U2-B2 and U2-B6 close comments (bd-uocr2-epic-cgux.11, bd-uocr2-epic-cgux.15).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_deepseekv2.py` sha256
    `24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2` lines 346-362;
    `dash_attention.py` sha256
    `960b460acd816df8e561e73dccd5683acb95445eddc723c4f0d6ed1f462b2f77` lines 762-813;
    SGLang wheel `sglang-0.5.20.dev10+g8d62f5fb9c` sha256
    `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`route_triton.py:129-212`); oracle `tests/fixtures/uocr2_oracle/ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90`; shipping artifact
    `unlimited-ocr-2.v0.1.0.int8.focrq` sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: model-free:
    `cargo test --lib all_candidates_equals_top_m_under_its_preconditions`. Model-gated:
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages
    cargo test --profile release-perf --lib real_v2_entmax_domains_decode_s1_alike -- --nocapture`.
- Reference behavior: HF runs entmax-1.5 over the top-M window of all blocks, M = min(16, nblk),
    and reads the candidates among the first K' = min(12, M) slots (`MD:346-362`). SGLang's Triton
    router runs entmax-1.5 over every candidate block and no other block. It solves τ with a 24-step
    bisection (fp64 sums) and finds the read set, the 12 highest-score candidates, with a 40-step
    count bisection, so a tie at 12th place can read fewer than 12 blocks (`route_triton.py:129-212`).
    The two agree when a query has at most 16 candidates (every own-page query in base mode), up to
    bisection precision and ties. They can differ on fallback queries (BOS, instruction, the early
    image rows of pages 2..N, text before the first `<PAGE>`) and on gundam pages (plan §5.6.6).
- Our impl: both domains (D21). `EntmaxDomain::TopM` is HF's form and the default.
    `EntmaxDomain::AllCandidates` (`entmax15_all`) follows SGLang's rule with an exact sort-based τ in
    place of both bisections, so a tie never shortens its read set. `DecodeParams.entmax_domain`
    carries the choice into the v2 document loop.
- Fallback / kill-switch state: `FOCR_SRSWA_ENTMAX`, read once per process. Unset, empty or `topm`
    selects HF's top-M (the parity default); `all` selects the all-candidates domain. Any other value
    exits 2 at admission.
- Measured impact:
    * S1-page_0009 (one base page): top-M and all-candidates decode the same 50 ids, equal to the
      f32 oracle's (`real_v2_entmax_domains_decode_s1_alike`, U2-B6, 2026-10-30). A base page has at
      most 10 blocks, so this run checks the `all` wiring; it cannot separate the domains.
    * Model-free: on 2,000 random rows that meet the preconditions (at most 16 candidates, no tie at
      12th place, best candidate input ≥ −1e4+2, none ≤ −1e4) the domains read the same blocks, with
      biases within 1e-5. With 20 near-equal candidates both read the same 12 blocks, but the support
      is 20 vs 16 and the biases differ by more than 1e-4.
    * Where they can differ on real prompts: U2-C1 counted fallback queries per layer at 232 on S2,
      109 on S3-p9_p14, 1,037 on S3-p10 and 2,161 on S3-p20. No decode has compared the domains on
      these documents: U2-B6 did not run the S2 comparison, and U2-C5b owns it.
    * Our exact τ against SGLang's bisections: not measured. No SGLang run exists in this repo.
- Resolution: ACCEPTED for parity: top-M is the default and the f32 oracle's domain. The default for
    long documents stays open until U2-C5b scores both domains against ground truth (D21).
- Tests affected: `all_candidates_equals_top_m_under_its_preconditions`,
    `top_m_slots_break_ties_by_lower_block`, `the_entmax_domain_env_names_a_domain_or_is_refused`
    (`srswa.rs`); `real_v2_entmax_domains_decode_s1_alike` (`mod.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

## DISC-010: U2-2 — n-gram ban vs page guard order: a forced token is emitted even when the n-gram bans it

- claim_id / evidence_id: CLAIM-uocr2-forced-token-wins → `src/native_engine/sampler.rs`
    `decode_step_guarded` (`GuardedStep.hf_all_neg_inf`) +
    `sampler::tests::{forced_token_is_authoritative_over_the_ngram_ban,
    guard_and_ngram_replay_the_v2_oracle_counters}` + the `disc_u2_2_steps` field of the
    `real_v2_l4_*` NDJSON lines (U2-C2) + the U2-B6 close comment (bd-uocr2-epic-cgux.15).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 583-623 (guard and
    n-gram) and 1168-1173 (their order); SGLang wheel dev10 sha256
    `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`custom_logit_processor.py:194-262,325-338`); `ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90`; shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4). The rule is integer control flow.
- Exact command + env: `cargo test --lib -- forced_token_is_authoritative_over_the_ngram_ban
    guard_and_ngram_replay_the_v2_oracle_counters`. Model-gated:
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages
    FOCR_E2E_LONG=1 cargo test --profile release-perf --lib -- real_v2_l4
    real_v2_document_limits_stop_where_the_contract_says --nocapture`.
- Reference behavior: HF's `_uocr_generate` applies the n-gram ban first, then the page guard
    (`MU:1168-1173`). If the guard allows a single token and the n-gram has banned it, every logit is
    −inf and argmax emits id 0. SGLang applies the guard first, and a forced token skips the n-gram
    (`custom_logit_processor.py:194-262,325-338`).
- Our impl: SGLang's order. `decode_step_guarded` returns a forced token without consulting the
    n-gram. It then runs the n-gram scan only to set `hf_all_neg_inf` on a step where HF would emit
    id 0, and `DocumentStats.hf_all_neg_inf_steps` counts those steps. EOS is never appended to the
    generated ids, so the n-gram cannot ban it: only a cap-forced `<PAGE>` can reach the case.
- Fallback / kill-switch state: none. No switch restores HF's order, which would emit id 0 on such a
    step.
- Measured impact: never reached on any run.
    * HF oracle: `disc_u2_2_all_neg_inf_steps` is 0 on all nine fixtures (`ORACLE.json`). The
      reference forced no token on any of them (`guard_cap_forced_steps` and
      `guard_eos_only_steps` are 0), so the case cannot arise there.
    * franken_ocr: `hf_all_neg_inf_steps` is 0 on S1, on S3, and on S2 with the page cap lowered to
      50, the one run that forced tokens (2 forced steps; U2-B6). It is 0 on all 28 U2-C2 L4 runs,
      which assert it.
    * The unit test builds the case: a cap-forced `<PAGE>` whose 2-gram is banned is still emitted,
      with `hf_all_neg_inf` set.
- Resolution: ACCEPTED. OQ-V2-7 is RESOLVED: the forced token wins.
- Tests affected: `forced_token_is_authoritative_over_the_ngram_ban`,
    `guard_and_ngram_replay_the_v2_oracle_counters`,
    `guarded_loop_keeps_the_page_contract_under_random_logits` (`sampler.rs`);
    `real_v2_l4_s1_pages_decode_the_oracle_ids`, `real_v2_l4_documents_decode_the_oracle_ids`,
    `real_v2_l4_long_documents_decode_the_oracle_ids`,
    `real_v2_document_limits_stop_where_the_contract_says` (`mod.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

## DISC-011: U2-3 — `<PAGE>` ban on the last page: HF's rule when unwindowed, SGLang's when windowed

- claim_id / evidence_id: CLAIM-uocr2-last-page-marker-ban → `src/native_engine/sampler.rs`
    (`PageGuard::ban_last_page_marker`) + `src/native_engine/window.rs` (`window_guard`) +
    `postprocess::split_pages_by_token` (`extra_markers`) + `tests/fixtures/uocr2_windowed/` +
    the U2-D1a close comment (bd-uocr2-epic-cgux.24).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 583-605; SGLang wheel
    dev10 sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`custom_logit_processor.py:227,256-262`, `uocr_doc_reuse.py:1101-1109`); `ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90`;
    `degenerate-s2-w3-t12.json` sha256
    `af019ec1f81f8c60bbede053066277ceefc134ba183a7351525d016a54d6ce91`;
    `w2-n7-recompute-t12.json` sha256
    `5f9953e6d4e500c59cd9ae7d935fc7800e1253d41c1a468bad55a340172afcaa`; shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: `cargo test --lib -- guard_last_page_ban_masks_the_marker
    only_a_recomputed_last_window_bans_the_marker`. Model-gated:
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages
    cargo test --profile release-perf --lib real_v2_windowed_decode_matches_the_windowed_oracle --
    --nocapture`. The S2 check below:
    `python3 -I -c "import json; a=json.load(open('tests/fixtures/uocr2_windowed/degenerate-s2-w3-t12.json'))['windows'][0]['generated_ids']; b=[int(t) for l in open('tests/fixtures/uocr2_oracle/S2-p9_p14_p19.ids.txt') if not l.startswith('#') for t in l.split()]; print(len(a), len(b), a == b)"`
    prints `2152 2152 True`.
- Reference behavior: HF's guard never bans `<PAGE>` (`MU:583-605`). Once every page has its
    marker, another `<PAGE>` is allowed, and the guard then forces EOS. SGLang bans `<PAGE>` on the
    last page unless `end_on_marker` (`custom_logit_processor.py:256-262`), where
    `end_on_marker = lookahead > 0 or reused` (`:227`; a reused window sets it,
    `uocr_doc_reuse.py:1101-1109`). The ban therefore reaches only an unsplit request and a last
    window that was recomputed.
- Our impl: an unwindowed decode (every document of N ≤ W pages, D13) follows HF: no ban. An extra
    marker is counted in `extra_markers` and the empty trailing body is dropped, so N pages give N
    bodies (`split_pages_by_token`). A windowed decode follows SGLang: `window_guard` adds the ban
    only for a last window that was recomputed.
- Fallback / kill-switch state: none. The rule follows from whether the document is windowed.
- Measured impact: no id changed on any measured run.
    * HF oracle: `guard_eos_only_steps` is 0 on all nine fixtures, so HF never emitted a `<PAGE>`
      past the page count. On the oracle's own logits the ban would have changed no id.
    * S2 (3 pages), direct: the U2-D2a windowed oracle's unsplit run applies SGLang's ban on 1,842
      steps of the last page and generates the same 2,152 ids as the HF oracle (command above,
      run 2026-10-30).
    * franken_ocr: `extra_markers` is 0 on all 28 U2-C2 L4 runs (unwindowed). The recomputed last
      window of `w2-n7-recompute-t12` (ban active on 595 steps) is token-exact against the windowed
      oracle on the f32 cache.
    * The 25-page W=10 run had 1 extra marker, in its reused last window, which neither stack bans.
      It was counted as extra, its body was dropped, and the document kept 25 bodies (U2-D1a).
- Resolution: ACCEPTED.
- Tests affected: `guard_last_page_ban_masks_the_marker_once_every_page_has_one` (`sampler.rs`);
    `only_a_recomputed_last_window_bans_the_marker` (`window.rs`);
    `split_pages_by_token_keeps_n_bodies_and_reports_head_and_extras` (`postprocess.rs`);
    model-gated `real_v2_l4_*`, `real_v2_windowed_decode_matches_the_windowed_oracle`,
    `real_v2_windowed_25_pages_own_every_page_once` (`mod.rs`). No XFAIL.
- Review date: 2026-10-30

## DISC-012: U2-4 — windowed stitching never drops a leading blank body

- claim_id / evidence_id: CLAIM-uocr2-no-leading-blank-drop →
    `postprocess::split_window_pages` + `postprocess::tests::split_window_pages_keeps_own_pages_only`
    + `window::tests::windowed_bodies_match_the_oracle_stitching` + the U2-D1a close comment
    (bd-uocr2-epic-cgux.24).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; SGLang wheel dev10 sha256
    `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` (`doc_split.py:237-257`,
    the drop at `:253-254`; `serving_chat.py:1114-1124`); v2 `tokenizer.json` sha256
    `c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589`; windowed fixtures
    `w2-n7-t12.json` `3bbb36eeb764e474f2c16f1b4c34577668052e0be027d4f5e1b137cae784d79d`,
    `w3-n7-t12.json` `9f748c896dce0e260304fb5e6142f91ffc79fe482f844c7e1a617299b3b9f36d`,
    `w2-n7-recompute-t12.json` `5f9953e6d4e500c59cd9ae7d935fc7800e1253d41c1a468bad55a340172afcaa`,
    `w2-n5-markeronly-t12.json` `152e0626c7c7ffd172643a5f012090ed33b244e4ce2821795bffabc08c73ff27`,
    `degenerate-s2-w3-t12.json` `af019ec1f81f8c60bbede053066277ceefc134ba183a7351525d016a54d6ce91`;
    shipping artifact sha256 `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4). The split is byte-level text handling.
- Exact command + env: `cargo test --lib split_window_pages_keeps_own_pages_only`;
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> cargo test --lib
    windowed_bodies_match_the_oracle_stitching` (it reads only `tokenizer.json` from that dir).
- Reference behavior: HF never splits a document, so it has no counterpart. SGLang's
    `select_batch_output` drops leading blank bodies while a window has more bodies than own pages:
    `while len(bodies) > input_pages and not bodies[0].strip(): bodies = bodies[1:]`
    (`doc_split.py:253-254`). It runs on the non-streaming path and in the streaming handler's
    end-of-window call (`serving_chat.py:1114-1124`); the incremental `stream_owned_text`
    (`:260-275`) has no such drop. When a window's first own page is blank and the window ended on
    the lookahead marker, the drop removes that page's body and the blank page moves to the end of
    the window's bodies.
- Our impl: `split_window_pages` keeps every body in page order and keeps only the own pages (D5). A
    blank page stays a blank body at its own index.
- Fallback / kill-switch state: none. The drop shifts page alignment, so nothing restores it.
- Measured impact: not reached. On all 5 committed U2-D2a runs, our split of the oracle's own
    generated ids stitches to the wheel's document bodies byte for byte
    (`windowed_bodies_match_the_oracle_stitching`). So no window there had a blank first own page,
    and the drop never fired. The unit test pins the rule on synthetic ids: a blank first body is
    kept ("no leading-blank drop"). The 40-60-page S5 document (U2-A6b, U2-D2b) is the next corpus
    that can contain a blank page.
- Resolution: ACCEPTED (D5).
- Tests affected: `split_window_pages_keeps_own_pages_only` (`postprocess.rs`);
    `windowed_bodies_match_the_oracle_stitching` (`window.rs`, tokenizer-gated);
    `real_v2_windowed_decode_matches_the_windowed_oracle` (`mod.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

## DISC-013: U2-5 — numerics: f32 throughout, compared with an f32 oracle; routing tie-flips at near-ties

- claim_id / evidence_id: CLAIM-uocr2-f32-numerics → `tests/parity_ladder.rs` (L1/L2, L3) +
    `real_v2_l4_*` (`src/native_engine/mod.rs`) + `tests/metamorphic.rs` (MR-6) +
    `tests/fixtures/uocr2_oracle/FLOOR.json` + the U2-C1 and U2-C2 close comments
    (bd-uocr2-epic-cgux.17, .18) + the U2-C4 comment of 2026-10-30 04:19 UTC (bd-uocr2-epic-cgux.20).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_deepseekv2.py` sha256
    `24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2` lines 447-450, 530;
    `dash_attention.py` sha256 `960b460acd816df8e561e73dccd5683acb95445eddc723c4f0d6ed1f462b2f77`
    lines 1230-1261; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 1165-1167; `ORACLE.json`
    sha256 `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90`; `FLOOR.json` sha256
    `0b9978731408e7ddc51587feb975b42da9116efa56502def81a75b71d28e93ed` (torch 2.13.0, transformers
    4.52.1, f32 CPU, 12 threads); shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4); the oracle ran on macOS arm64.
- Exact command + env: with `FOCR_UOCR2_FIXTURES_DIR=~/.cache/franken_ocr_local_models/uocr2_oracle/f32/t12
    FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages`:
    `cargo test --profile release-perf --test parity_ladder -- uocr2_l1_l2_prefill_matches_the_f32_oracle
    uocr2_l3_decode_steps_match_the_f32_oracle --nocapture` (about 140 s and 1,020 s);
    `FOCR_E2E_LONG=1 cargo test --profile release-perf --lib real_v2_l4 -- --nocapture`;
    `cargo test --profile release-perf --test metamorphic mr6_uocr2_reprefill_equals_incremental`.
- Reference behavior: every path computes the same math; only rounding differs (plan §5.7). GPU
    prefill (Triton, `MD:447-450`) keeps the bias and online softmax in fp32 but rounds q·scale and P
    to bf16. CPU prefill runs SDPA in the activation dtype (`DA:1230-1261`). Decode does fp32 math over
    activation-dtype q/K/V and casts back before `o_proj` (`MD:530`). Logits come from `lm_head` in
    the activation dtype, then `.float()` (`MU:1165-1167`). SGLang serves bf16 Triton.
- Our impl: activations, router, attention, norms and `lm_head` run in f32 over the bf16 weights.
    The gathered `srswa::attend` equals the dense `attend_dense` bit for bit (`FOCR_SRSWA_DENSE=1`
    selects the dense one). The parity target is HF `infer()` in f32 on CPU (D4). The shipping
    cache's int8 FFN/expert storage is DISC-019 (U2-11), not this entry.
- Fallback / kill-switch state: none. v2 has no bf16 compute path; f32 is its only precision.
- Measured impact:
    * Reference floor (f32, 12 vs 4 threads, nine fixtures): identical ids, logit spread ≤ 4.8e-4,
      layer-input max_rel ≤ 2.3e-4, and 4 rows that change their selected block set. HF bf16 vs f32:
      ids leave within 7-94 steps and 12,169 routing rows differ.
    * Prefill (L1/L2): every stage of every layer at cosine ≥ 0.9999; worst 0.99999992 on S1-S3 and
      0.99999946 on S4. The router run on the oracle's scores has 0 selection mismatches.
    * Prefill routing, chained: 5 single-row tie-flips over 9 fixtures × 12 layers, each swap gap
      within 2× the layer's score noise: S3-p10 L3 row 2245 (gap 1.5e-8) and S3-p20 L3 row 2245
      (gap 0), the block 56/64 swap the oracle itself makes between 12 and 4 threads; S3-p20 L4 row
      5037 (1.9e-7) and L7 row 5041 (1.2e-7); S4-page_0590-gundam L11 row 823 (1.2e-4).
    * The S4 flip follows an MoE expert near-tie: at layer 8, row 823's 6th and 7th experts (53 and
      26) score 0.031048 each, margin 4.5e-7, the smallest of the page's 909 rows by 4×. From layer
      9 on, row 823 holds the largest native-vs-oracle difference. The MoE code is v1's (DISC-006).
    * Decode (L3, stateless f32, S1 ×3, S2, S3-p9_p14, S4 ×2): logit cosine ≥ 0.9999865, no
      raw-argmax mismatch on any dumped step. Decode routing differs only on S4-page_0590-gundam:
      3 tie-flips at layers 5, 7 and 10, steps 11, 11 and 59, swap gaps 1.6e-7, 3.5e-4 and 4.3e-6,
      each within 2× the layer's score noise.
    * Ids (L4): the f32 paths are exact on all 7 base fixtures (50 to 16,880 ids), retirement on and
      off, so the tie-flips change no generated id on S1-S3. L4 does not run S4. MR-6 on the f32
      cache: max relative logit gap 8.95e-7 over 2,152 steps (S2), tokens equal.
    * Not measured: the bf16 ledger tolerance (OQ-V2-3 is PARTIAL) and any run against SGLang.
- Resolution: ACCEPTED. The math is the same, and the tie-flips sit within the measured score noise.
- Tests affected: `uocr2_l1_l2_prefill_matches_the_f32_oracle`,
    `uocr2_l3_decode_steps_match_the_f32_oracle`, `real_v2_l4_s1_pages_decode_the_oracle_ids`,
    `real_v2_l4_documents_decode_the_oracle_ids`, `real_v2_l4_long_documents_decode_the_oracle_ids`,
    `mr6_uocr2_reprefill_equals_incremental`; all model-gated. No XFAIL on the f32 paths.
- Review date: 2026-10-30

## DISC-014: U2-6 — crop mode for multi-image requests: base views, not SGLang's gundam default

- claim_id / evidence_id: CLAIM-uocr2-document-crop-mode → `src/native_engine/mod.rs`
    (`paged_document_view`, `OcrModel::document_page`) +
    `gundam_is_refused_for_a_v2_document_and_warns_on_a_v2_page` + plan D12 and OQ-V2-14.
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 541 and 1117; SGLang
    wheel dev10 sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`multimodal/processors/unlimited_ocr.py:23`); `ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90` (S1-page_0590 ids sha256
    `666cd4633da039c5cf085b0b918c43868e9f779d84cbf54d196518108da6d982`, S4-page_0590-gundam ids
    sha256 `a6b19ee98007eee6f9f740988eb88ebf143a76c7c97b19939e845e0e40d50fc7`); shipping artifact
    sha256 `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: `cargo test --lib gundam_is_refused_for_a_v2_document_and_warns_on_a_v2_page`.
    The token counts below are `fixtures.<name>.token_count` in `ORACLE.json`. The gundam parity rungs
    run as in DISC-013.
- Reference behavior: HF `infer_multi` reads every page of a multi-image request as one base 1024
    view (`MU:1117`). HF `infer` with `crop_mode=None` uses gundam only for a single image
    (`MU:541`). SGLang's processor defaults to gundam for every request, multi-image included
    (`unlimited_ocr.py:23`). Baidu's report does not state the crop mode behind its long-document
    scores, and its latency table uses gundam (plan §2, OQ-V2-14).
- Our impl: a v2 document reads each page as a base 1024 letterbox (`OcrModel::document_page`), as
    HF `infer_multi` does. A v2 `--multi-page` document given `--crop-mode gundam` exits 2
    (`paged_document_view`, U2-E1). A single v2 page also defaults to base (D12); `--crop-mode
    gundam` there runs gundam with an `uncertified_crop_mode` warning.
- Fallback / kill-switch state: none. No environment variable selects the crop mode. A single page can
    still run gundam through `--crop-mode gundam`; there is no gundam document mode.
- Measured impact: not measured for documents. OQ-V2-14 is OPEN; U2-C5b decides it against ground
    truth, or hands its gundam cells to U2-G2.
    * The HF reference itself decodes a page differently by crop mode: page_0590 gives 2,329 ids in
      base (S1) and 2,971 in gundam (S4); page_0009 gives 50 and 78. No ground-truth score ranks the
      two yet.
    * Gundam parity holds where it was measured: S4 preprocessing is bit-exact, S4 prefill stages
      reach cosine ≥ 0.99999946, and S4 decode steps pass L3 (DISC-013).
    * Cost: plan D12 estimates gundam's CPU vision cost at 1.7-6.4× base from §8.1's per-view
      figures. Not measured on v2.
- Resolution: ACCEPTED: base is the default (D12). OQ-V2-14 stays open.
- Tests affected: `gundam_is_refused_for_a_v2_document_and_warns_on_a_v2_page` (`mod.rs`);
    the S4 legs of `uocr2_l0_preprocess_is_bit_exact_against_the_oracle`,
    `uocr2_l1_l2_prefill_matches_the_f32_oracle` and `uocr2_l3_decode_steps_match_the_f32_oracle`
    (`tests/parity_ladder.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

## DISC-015: U2-7 — page2d tok_pos after an extra `<PAGE>`: HF's reset, not SGLang's running count

- claim_id / evidence_id: CLAIM-uocr2-extra-marker-tok → `src/native_engine/srswa.rs`
    (`DecodePageState::feed`) + `srswa::tests::an_extra_generated_marker_clamps_to_the_last_page` +
    `window::tests::windowed_rows_match_the_oracle_dumps` (its `decode_rows` helper) + the
    `disc_u2_7_steps` field of the `real_v2_l4_*` NDJSON lines (U2-C2).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `_sparse_base_impl.py` sha256
    `86da9c5638704bb9338575040568a97b3e10820bbeb426f51c07e866f3423ccd` lines 1484-1489 and
    2555-2561; SGLang wheel dev10 sha256
    `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` (`page2d_rope.py:579-591`);
    `ORACLE.json` sha256 `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90`;
    `w2-n7-t12.json` `3bbb36eeb764e474f2c16f1b4c34577668052e0be027d4f5e1b137cae784d79d`,
    `w3-n7-t12.json` `9f748c896dce0e260304fb5e6142f91ffc79fe482f844c7e1a617299b3b9f36d`,
    `w2-n5-markeronly-t12.json` `152e0626c7c7ffd172643a5f012090ed33b244e4ce2821795bffabc08c73ff27`;
    shipping artifact sha256 `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4). The state machine is integer arithmetic.
- Exact command + env: `cargo test --lib -- an_extra_generated_marker_clamps_to_the_last_page
    windowed_rows_match_the_oracle_dumps`. Model-gated: the L4 command of DISC-013, and
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages
    cargo test --profile release-perf --lib real_v2_windowed_decode_matches_the_windowed_oracle --
    --nocapture`.
- Reference behavior: a fed `<PAGE>` moves the page state to `page = min(page+1, n_pg−1)`, and HF
    then sets `tok = pf_cnt[page]` (`SB:1484-1489`). A marker past the page count therefore clamps
    to the last page and repeats the tok_pos of that page's response `<PAGE>`. SGLang keeps the
    running count, `tok + 1` (`page2d_rope.py:579-591`). HF's prefill builder has no upper clamp
    (`SB:2555,2561`).
- Our impl: HF's reset (`DecodePageState::feed`). `PageCoords::build` has no upper clamp either, so
    "state machine ≡ prefill builder" holds only for sequences with at most N generated markers.
- Fallback / kill-switch state: none.
- Measured impact: no id can change. Under the guard, a marker past the page count forces EOS on the
    next step, and that forced step is the only one that reads the reset coordinate.
    * Unwindowed: the reset never fired. The HF oracle emitted no extra marker on any of the nine
      fixtures (`guard_eos_only_steps` 0), and `disc_u2_7_steps` is 0 on all 28 U2-C2 L4 runs.
    * Windowed: it fires in each reused last window that ends on an extra `<PAGE>`. That is 3 of the
      15 committed U2-D2a windows (the last window of `w2-n7-t12`, `w3-n7-t12` and
      `w2-n5-markeronly-t12`), and the last window of the 25-page W=10 run. All four committed
      windowed runs are token-exact (`real_v2_windowed_decode_matches_the_windowed_oracle`, f32
      cache). `windowed_rows_match_the_oracle_dumps` gives a clamped `<PAGE>` row the running count
      before hashing. With that substitution (and DISC-020's on the marker-only run) every row array
      hashes equal to the oracle's.
- Resolution: ACCEPTED.
- Tests affected: `an_extra_generated_marker_clamps_to_the_last_page` (`srswa.rs`);
    `windowed_rows_match_the_oracle_dumps` (`window.rs`); model-gated `real_v2_l4_*`,
    `real_v2_windowed_decode_matches_the_windowed_oracle`,
    `real_v2_windowed_25_pages_own_every_page_once` (`mod.rs`). No XFAIL.
- Review date: 2026-10-30

## DISC-016: U2-8 — page guard lag: none here; SGLang can lag one step under overlap scheduling

- claim_id / evidence_id: CLAIM-uocr2-guard-lag → `OcrModel::document_loop` (calls
    `sampler::decode_step_guarded` on every step) + `window::cut_at_extra_marker` +
    `window::tests::text_tail_and_extra_marker_cut_follow_the_wheel` + plan §5.10.7.
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 583-605; SGLang wheel
    dev10 sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`uocr_doc_reuse.py:807-820`, `_cut_at_extra_marker`, per
    `docs/truth-pack/oq/uocr2-windowed-sglang.md` §3a); windowed oracle script
    `scripts/gen_reference_fixtures_uocr2.py` sha256
    `5edc7645e72207c6b02724134907dcd721273e22e303a533344d2c1627780e92` (the U2-D2a runs); shipping
    artifact sha256 `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: `cargo test --lib text_tail_and_extra_marker_cut_follow_the_wheel`. No
    command runs live SGLang.
- Reference behavior: HF applies the guard to every step's logits before the argmax, so it cannot
    lag. SGLang's guard can lag one step under overlap scheduling, so one extra token can precede the
    forced EOS (plan §5.10.7). `_cut_at_extra_marker` cuts a window's output at the first marker
    past its own pages (`uocr_doc_reuse.py:807-820`).
- Our impl: no overlap scheduling. `document_loop` applies the guard to every step before it picks
    the token, so a forced token is emitted on the step it is due. `cut_at_extra_marker` mirrors the
    wheel's cut for windows.
- Fallback / kill-switch state: none.
- Measured impact: not measured. No live SGLang run exists in this repo. The U2-D2a windowed oracle
    replaces SGLang's scheduler with one request at a time and no overlap scheduling
    (`WINDOWED_SUBSTITUTIONS`, `scripts/gen_reference_fixtures_uocr2.py:1067-1068`), so it cannot show
    the lag: in every committed window that ended on a marker, its guard forced EOS on the next
    step (`guard_steps.forced_eos_extra_marker` is 1). Our ids equal its ids on all four windowed
    runs.
- Resolution: ACCEPTED by design. A future comparison against live SGLang must tolerate one extra
    token before a forced EOS.
- Tests affected: `text_tail_and_extra_marker_cut_follow_the_wheel` (`window.rs`). No test runs
    live SGLang.
- Review date: 2026-10-30

## DISC-017: U2-9 — text before the first `<PAGE>` joins page 1 with a `leading_text` warning

- claim_id / evidence_id: CLAIM-uocr2-leading-text → `postprocess::split_pages_by_token`,
    `postprocess::split_window_pages` and `join_head` + the `leading_text` run warning
    (`src/native_engine/mod.rs`) + the U2-B6 and U2-D1a close comments (bd-uocr2-epic-cgux.15,
    bd-uocr2-epic-cgux.24).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` line 1210; SGLang wheel dev10
    sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`doc_split.py:251-255`); `ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90` and the nine
    `tests/fixtures/uocr2_oracle/*.ids.txt`; the five `tests/fixtures/uocr2_windowed/*.json` (hashes
    in DISC-012); shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4). The split is byte-level text handling.
- Exact command + env: `cargo test --lib -- split_pages_by_token_keeps_n_bodies_and_reports_head_and_extras
    split_window_pages_keeps_own_pages_only`; `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> cargo test
    --lib windowed_bodies_match_the_oracle_stitching`; model-gated L4 as in DISC-013. The oracle
    check: `python3 -I -c "import glob; print(sorted({[l for l in open(f) if not
    l.startswith('#')][0].split()[0] for f in glob.glob('tests/fixtures/uocr2_oracle/*.ids.txt')}))"`
    prints `['128827']`.
- Reference behavior: HF's `_uocr_save_results` splits the output on `<PAGE>` and keeps `[1:]`, so
    text before the first marker is dropped (`MU:1210`). SGLang keeps a non-blank head of a
    recomputed window as body 0 (`[head, *tail] if leads_with_body or head.strip() else tail`,
    `doc_split.py:251-255`). With front-anchored ownership that shifts every page by one and drops
    the last own page's body.
- Our impl: a non-blank head is prepended to the first own page's body, and the run adds
    `warnings[{kind:"leading_text"}]`. The rule is the same unwindowed (`split_pages_by_token`) and in
    every window (`split_window_pages`), so N pages give N bodies at any W.
- Fallback / kill-switch state: none. Neither reference keeps both the text and the page alignment.
- Measured impact: not reached.
    * HF oracle: all nine outputs open with `<PAGE>` (id 128827 is the first generated id of every
      committed `.ids.txt`, checked 2026-10-30), so none had a head to drop.
    * franken_ocr: `leading_text` is false on all 28 U2-C2 L4 runs (asserted), on S1, S3 and the S2
      CLI run (U2-B6: 0 heads in 3 documents), and on the 25-page W=10 run (U2-D1a).
    * Windowed: on all 5 committed U2-D2a runs our split stitches to the wheel's bodies byte for
      byte, so no recomputed window there had a non-blank head.
- Resolution: ACCEPTED.
- Tests affected: `split_pages_by_token_keeps_n_bodies_and_reports_head_and_extras`,
    `split_window_pages_keeps_own_pages_only` (`postprocess.rs`);
    `document_reports_warn_in_v2_terms_and_only_a_stopped_one_is_partial` (`mod.rs`);
    `windowed_bodies_match_the_oracle_stitching` (`window.rs`, tokenizer-gated); model-gated
    `real_v2_l4_*`, `real_v2_windowed_25_pages_own_every_page_once`. No XFAIL.
- Review date: 2026-10-30

## DISC-018: U2-10 — EXIF orientation is applied; both v2 references ignore it

- claim_id / evidence_id: CLAIM-uocr2-exif-orientation → `src/preprocess/mod.rs` (`decode_path`,
    `decode_bytes`, `decode_reader`) + the U2-B7 close comment (bd-uocr2-epic-cgux.16) + the U2-C4
    comments of 2026-10-30 02:31 and 10:38 UTC (bd-uocr2-epic-cgux.20).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` line 1070; SGLang wheel dev10
    sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`base_processor.py:215`, `utils/common.py:1834`); v1 `modeling_unlimitedocr.py` sha256
    `268bdcbe12cf37bf5a2debb53faf542e56570958a5d9f3314aab3cab2cf6cb48` lines 27-34; `ORACLE.json`
    sha256 `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90` (PNG pages, input
    sha256s inside); shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: `cargo test --lib decode_applies_the_exif_orientation_tag`. L0 rung:
    `FOCR_UOCR2_FIXTURES_DIR=~/.cache/franken_ocr_local_models/uocr2_oracle/f32/t12
    FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages cargo test --profile release-perf --test
    parity_ladder uocr2_l0_preprocess_is_bit_exact_against_the_oracle -- --nocapture`.
    The U2-C4 measurement (2026-10-30):
    * Inputs: corpus page_0009 and page_0019 stored sideways with Pillow 12.3.0,
      `im.convert("RGB").rotate(90, expand=True).save(out, exif=e)` where `e` is an `Image.Exif()`
      with `e[0x0112] = 6`. `ImageOps.exif_transpose` restores the original pixels exactly. The
      tagged files hash to `2f6428fe953c6306ece8d1a2cc569711915cc919578ce9992db121564b86bfb7` and
      `01ee93a55cd4a29cee812e1db0820f5b6d093bde2382f60b8147d81ef215b375`.
    * HF: `python -I scripts/gen_reference_fixtures_uocr2.py --model-dir <v2 snapshot> --corpus
      <tagged dir> --out <dir> --sets S1C --only S1C-page_0009,S1C-page_0019 --dtype f32
      --threads 8`.
    * franken_ocr: `focr ocr --model <dir>/unlimited-ocr-2.v0.1.0.int8.focrq <tagged page> --json`,
      with v2's `tokenizer.json` in `<dir>`.
    * CER: `scripts/corpus_cer.py`'s `normalize` and `levenshtein`, against the upright S1C f32
      reference. HF texts are rendered by deleting `<PAGE>` and every block prefix.
- Reference behavior: HF v2 opens each image with `Image.open(f).convert('RGB')` (`MU:1070`) and
    applies no EXIF orientation. SGLang's `smart_to_rgb` would transpose (`utils/common.py:1834`),
    but only for a processor that sets `smart_rgb_conversion = True`. The base default is False
    (`base_processor.py:215`) and only glm4v sets it, so Unlimited-OCR gets a plain
    `convert("RGB")`. v1's HF reference did transpose (`load_image` → `ImageOps.exif_transpose`,
    v1 `MU:27-34`). Both v2 references drop alpha in `convert('RGB')`.
- Our impl: the shared decoder applies the Orientation tag for every model (`decode_reader`:
    `decoder.orientation()`, then `apply_orientation`). Every image entry reads through
    `decode_path`: single images, `ocr-batch`, directory pages, and `ocr-batch --multi-page` since
    U2-D0. Alpha is dropped by `to_rgb8`, as in HF.
- Fallback / kill-switch state: none. No switch skips orientation. Stripping the Orientation tag from
    the input gives the references' behavior.
- Measured impact: only an image with a non-identity Orientation tag differs. On one, the
    references read the page as stored and franken_ocr reads it upright.
    * Tagged page_0009: HF v2 f32 sees a 1837×1182 page and reads it sideways, 47 ids against the
      upright page's 50, with CER 0.2500 (22/88): "WILLIAM CLOVIS", "SEAMING PRINTED AND CUTLING
      HOUSE". franken_ocr's markdown equals its markdown for the untagged page: CER 0.
    * Tagged page_0019 (prose): HF v2 reads 1,590 ids against the upright page's 802, with CER
      0.8349 (1,917/2,296). franken_ocr again equals its untagged reading: CER 0.
    * Untagged pages, which include all 20 corpus pages and every oracle fixture: no difference. The
      L0 rung is bit-exact on them.
- Resolution: ACCEPTED. The deviation is deliberate: it keeps v1's behavior and reads a rotated
    photo upright.
- Tests affected: `decode_applies_the_exif_orientation_tag` (`src/preprocess/mod.rs`; a sideways PNG
    with an `eXIf` Orientation 6 chunk decodes upright, the same PNG untagged decodes as stored);
    `uocr2_l0_preprocess_is_bit_exact_against_the_oracle` (`tests/parity_ladder.rs`, untagged pages).
    No XFAIL.
- Review date: 2026-10-30

## DISC-019: U2-11 — int8 FFN/expert weights in the shipping v2 decoder, measured at L5

- claim_id / evidence_id: CLAIM-uocr2-int8-ffn-l5 → `tests/parity_ladder.rs`
    (`uocr2_l5_corpus_gate_admits_v2_against_the_v1_baseline`) + the U2-C2 close comment
    (bd-uocr2-epic-cgux.19, first divergences) + the U2-C4 close comment (bd-uocr2-epic-cgux.20,
    the gate run of 2026-10-30).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; shipping artifact
    `unlimited-ocr-2.v0.1.0.int8.focrq` sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398` (the conservative recipe:
    the 2,148 FFN/expert matrices per-channel int8; attention, router, norms and `lm_head` f32);
    v2 tokenizer sha256 `c55c165e…`; the f32 references are U2-A6's S1C/S2/S3 sets, generated
    by `scripts/gen_reference_fixtures_uocr2.py --dtype f32 --threads 12` (torch 2.13.0 /
    transformers 4.52.1), each fixture's sha256 in its gate row; the v1 baseline lane reads
    `unlimited-ocr.int8.focrq` (sha256 `573340…fb592`) against v1's f32 S1C references.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: pre-registered 2026-10-30, run 18:56–20:23 EDT (5,179 s):
    `FOCR_UOCR2_CORPUS_GATE=1 FOCR_MODEL_PATH=<v1 int8 .focrq>
    FOCR_UOCR2_MODEL_PATH=<dir holding the v2 .focrq and v2's tokenizer.json>/unlimited-ocr-2.v0.1.0.int8.focrq
    FOCR_UOCR2_V1_REFS_DIR=<v1 S1C f32 refs> FOCR_UOCR2_FIXTURES_DIR=<v2 f32 refs, t12>
    FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages FOCR_STAGE_BUDGET_FORWARD_MS=0 cargo test
    --profile release-perf --test parity_ladder uocr2_l5_corpus_gate_admits_v2_against_the_v1_baseline
    -- --exact --nocapture --test-threads=1`. Documents read with `PageWindow::Off` (one pass, as
    the references did).
- Reference behavior: the f32 reference runs every weight in f32 (the bf16 checkpoint upcast).
- Our impl: the shipping decoder keeps FFN and expert matrices in per-channel int8 (the fixed
    recipe, AGENTS.md); everything else is f32. The all-f32 cache (`FOCR_DECODE_F32`, otherwise the
    same ops) is the parity path. U2-C2 traced the int8 path's first divergences, each at a near-tie
    where the f32 cache keeps the reference's token and no layer routes differently: S1 page_0014
    step 47 (the reference's token 0.214 below the int8 pick), page_0590 step 32 (0.156), S2 step 44
    (0.156), S3 p9_p14 step 73 (0.065), S3-p10 step 203 (0.017), S3-p20 step 118 (0.056);
    page_0009 is exact. The f32 cache is exact over S3-p10 (6,272 ids) and S3-p20 (16,880 ids).
- Fallback / kill-switch state: `FOCR_DECODE_F32=1` decodes Unlimited-OCR-2 on the all-f32 cache,
    which reproduces the f32 reference on every measured near-tie. It loads the raw checkpoint (it
    refuses a quantized artifact) and decodes slower.
- Measured impact: the gate run, CER through `scripts/corpus_cer.py`'s normalization, per page
    against the f32 reference; pre-registered budgets 0.45 per page and 0.25 aggregate, documents
    exact in page count with `stop` and no extra marker.
    * 20 corpus pages: 19 at CER ≤ 0.032 (10 exactly 0; page_0014 0.0272 at 441 ids against 256,
      page_0358 0.0318); aggregate **0.0685** (v1's lane in the same invocation: 0.1328 against
      its f32 references; its recorded 0.1931 is against bf16). page_0590 (the one dense table page) reads at
      **0.44996**, inside its 0.45 budget by 0.00004, stopping at 1,644 ids against the reference's
      2,329. The v1 baseline lane on the same page, with the same recipe class, is 0.9504 (4,032 ids
      against 2,327).
    * Documents: S2 (3 pages), S3-p9_p14, S3-p10 and S3-p20 each give exactly one body per page,
      `stop`, no extra marker (the gate). Their CER, recorded and not gated: S2 **0.5272** (1,232
      ids against 2,152), S3-p9_p14 0.0916 (314 against 538), S3-p10 0.0013 (6,277 against 6,272),
      S3-p20 0.0199 (17,165 against 16,880). S2 and S3-p9_p14 leave the reference at a near-tie
      (steps 44 and 73). S2's 0.5272 is mostly dot leaders. On its third page, page_0019 (a
      list of illustrations), the reference spells each leader as a long `. . . .` run (1,841 ids
      there, against 802 when it reads the page alone), and int8 writes `.....` (694 ids).
      Re-measured on the captured v2 golden (`tests/fixtures/uocr2_v2_goldens/s2.md`; 0.5355
      as-is), collapsing every leader to one mark on both sides gives 0.1025: page 2 is exact and
      page 3 is at 0.1383. Most of page 3's remainder is the reference's own gaps, which int8
      reads: five missing page numbers, stray `$` signs, and a run of 17 empty table cells after
      MINORCA. The rest is the superscript int8 renders as `$ ^{1} $` where the reference prints
      `1`, one joined line, and one digit (HAGBOAT 538 against 533).
    * Generated ids per page over the 55 measured pages: int8 p50 735, p90 852, p99 2,892; f32 p50
      740, p90 982, p99 2,892. No page reached 16,000. The one-pass page limit this supports is 102
      (`64 + 274·N + (N − 1)·p90 + p99 ≤ 131,072` on the stricter f32 distribution; 114 on int8).
    * Release decision (conformal ratchet, α = 0.05): v2 passes 20/20 pages, lower bound 0.726333,
      against v1's floor 0.676333 (19/20). The decision does not hinge on page_0590: at 19/20 v2's
      bound would be 0.676333, which still holds the floor.
- Resolution: ACCEPTED. The int8 FFN/expert recipe is the shipping trade (memory and speed) and
    the near-ties it flips are measured; `FOCR_DECODE_F32` is the exact path.
- Tests affected: `uocr2_l5_corpus_gate_admits_v2_against_the_v1_baseline` (`tests/parity_ladder.rs`,
    model-gated); the U2-C2 L4 rungs (`real_v2_l4_*`, `src/native_engine/mod.rs`); the v2 token
    golden lane (`tests/uocr2_goldens.rs`). No XFAIL.
- Review date: when the recipe or the decoder changes; page_0590's margin makes it the first page
    to recheck.

## DISC-020: U2-12 — reused window without a text splice: the last prompt row keeps its page

- claim_id / evidence_id: CLAIM-uocr2-markeronly-seed → `srswa::DecodePageState::reused_window` +
    `window::WindowedOptions::marker_only_splice` (test hook) +
    `window::tests::windowed_rows_match_the_oracle_dumps` +
    `real_v2_windowed_decode_matches_the_windowed_oracle` on `w2-n5-markeronly-t12` + the U2-D1a
    close comment (bd-uocr2-epic-cgux.24).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; SGLang wheel dev10 sha256
    `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`
    (`session/uocr_doc_reuse.py:490-497`, `_seed_continuation`); `w2-n5-markeronly-t12.json` sha256
    `152e0626c7c7ffd172643a5f012090ed33b244e4ce2821795bffabc08c73ff27`; shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4).
- Exact command + env: `cargo test --lib windowed_rows_match_the_oracle_dumps`;
    `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir> FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages
    cargo test --profile release-perf --lib real_v2_windowed_decode_matches_the_windowed_oracle --
    --nocapture`. The oracle run: `gen_reference_fixtures_uocr2.py --windowed 2 --doc S3-p10:5
    --threads 12 --force-fallback 1:splice_alloc_failed --tag w2-n5-markeronly-t12`
    (`tests/fixtures/PROVENANCE.md`).
- Reference behavior: HF has no windows. In SGLang, when a reused window's text KV allocation
    fails, only the forced `<PAGE>` is spliced. The first decode step's carry is then written into
    the slot of the prompt's last row (`_seed_continuation`, `uocr_doc_reuse.py:490-497`), so that
    image row's page reads −1.
- Our impl: the reused-window seed (`DecodePageState::reused_window`) carries the step, and the row
    keeps its page. Both stacks read a row's page only through its 32-row block's (min, max), which
    skips −1. They differ only when that row opens its block, that is when the rewritten prompt
    length is ≡ 1 (mod 32).
- Fallback / kill-switch state: none. franken_ocr's KV is never pool-allocated, so a text splice
    cannot fail in production; only the `marker_only_splice` test hook reaches this path.
- Measured impact: no difference reached. In `w2-n5-markeronly-t12`, window 1 is forced to a
    marker-only splice with a 956-row prompt. The run is token-exact against the windowed oracle
    (windows of 311, 1,506 and 770 generated ids). 956 ≡ 28 (mod 32), so the row does not open its
    block, and the case that differs was not reached. `windowed_rows_match_the_oracle_dumps` sets
    that slot's page to −1 before hashing. With that change (and DISC-015's for a clamped `<PAGE>`)
    all ten per-row arrays hash equal to the oracle's.
- Resolution: ACCEPTED.
- Tests affected: `windowed_rows_match_the_oracle_dumps`,
    `every_fallback_reason_is_detected_where_the_wheel_checks_it` (`window.rs`);
    `real_v2_windowed_decode_matches_the_windowed_oracle` (`mod.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

## DISC-021: U2-13 — v2 block grammar rendered under v1's span rule

- claim_id / evidence_id: CLAIM-uocr2-block-grammar → `src/native_engine/postprocess.rs`
    (`scan_block_spans`, `block_prefix`) +
    `postprocess::tests::every_oracle_v2_block_parses_and_none_reaches_the_markdown` +
    `v2_cli_decodes_a_page_and_a_document_through_every_entry_point_when_armed`
    (`tests/e2e_recognize.rs`) + the U2-E3b close comment (bd-uocr2-epic-cgux.53).
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR-2
    `5632329a32e103d5f1add270681e0f5c1233c6e4`; `modeling_unlimitedocr.py` sha256
    `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` lines 44-59; v2
    `tokenizer.json` sha256 `c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589`
    (`<BLOCK>` 128828, `<FORK>` 128829); `ORACLE.json` sha256
    `41236faf439dce1abcc2f7795f3efbbeccb21314d66d8649464912b6a9154f90` and the nine committed
    `tests/fixtures/uocr2_oracle/*.md`; shipping artifact sha256
    `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- CPU feature string: aarch64+neon+dotprod (Apple M4). The scanner is byte-level text handling.
- Exact command + env: `cargo test --lib -- every_oracle_v2_block_parses_and_none_reaches_the_markdown
    v2_blocks_render_in_v1_shape_with_layout_and_figures
    v2_block_prefixes_off_the_grammar_stay_as_emitted`; `cargo test --test property_suite
    every_v2_image_block_is_referenced_once`; model-gated: `FOCR_UOCR2_SOURCE_DIR=<v2 checkpoint dir>
    FOCR_CORPUS_DIR=~/.cache/franken_ocr/corpus/pages cargo test --profile release-perf --test
    e2e_recognize v2_cli_decodes_a_page_and_a_document_through_every_entry_point_when_armed`.
- Reference behavior: v2 opens every block with `<BLOCK>label[<LEVELn>][x1, y1, x2, y2]<FORK>`. HF's
    `_uocr_save_results` runs `re_match` (`MU:44-59`), which knows only v1's `<|ref|>`/`<|det|>`
    spans, so `result.md` keeps every block prefix raw. SGLang returns the generated text with no
    postprocess.
- Our impl: each prefix is a span under v1's rule (`scan_block_spans`). An `image` block becomes
    `![](images/…)` and a figure; every other prefix is deleted. Titles stay unmarked, one newline
    separates blocks and tables stay HTML, so the text has v1's shape. `--json` `layout` keeps each
    block's label, its `level` (titles only) and its box, rescaled from the 0-999 grid as v1's boxes
    are. A prefix off the grammar, such as one cut by the page token cap, stays in the text.
    `cross_text`/`append_text` continuations stay on their own pages.
- Fallback / kill-switch state: none. No switch emits the raw grammar; bd-v73v (GH #11) tracks a raw
    output surface.
- Measured impact:
    * All 431 blocks (5 of them images) in the nine oracle outputs parse, and none of the grammar
      reaches the markdown. The 12 labels are pinned, `level` appears only on `title`, and boxes are
      ordered on the 999 grid.
    * page_0009 through `focr ocr`: the markdown is 88 characters; with the prefixes left in it was
      125 (U2-E3b).
    * page_0590 through `focr ocr --model <v2> page_0590.png --json` renders
      `( 535 )\nAPPENDIX TO CHAPTER XXIV.\n…`. Its layout is header, title level 1, title level 2,
      text and table, in pixels. v1's `x/999·W` crops land on the header and both titles, in base
      and in gundam.
    * Property tests, 2,000 cases each: every postprocess entry point is total on mixes of both
      grammars, and on block pages each image reference appears exactly once.
- Resolution: ACCEPTED.
- Tests affected: `every_oracle_v2_block_parses_and_none_reaches_the_markdown`,
    `v2_blocks_render_in_v1_shape_with_layout_and_figures`,
    `v2_block_prefixes_off_the_grammar_stay_as_emitted`,
    `document_pages_number_figures_and_map_boxes_per_page` (`postprocess.rs`);
    `a_layout_span_round_trips_its_json` (`src/cli.rs`); `postprocess_is_total_on_grounding_fragments`,
    `every_v2_image_block_is_referenced_once` (`tests/property_suite.rs`);
    `v2_cli_decodes_a_page_and_a_document_through_every_entry_point_when_armed`
    (`tests/e2e_recognize.rs`, model-gated). No XFAIL.
- Review date: 2026-10-30

---

## DISC-006: MoE routing reproduces pinned torch-2.10 CPU `topk(sorted=False)` and production-width reduction

- claim_id / evidence_id: CLAIM-bd-2mo30-moe-route-order ->
    `tests/fixtures/moe_torch_2_10_cpu.json` +
    `moe::tests::{torch_2_10_topk_fixture_matches_exact_slot_permutation,
    torch_2_10_topk_matches_2048_case_oracle_corpus,
    production_six_term_combine_matches_256_case_torch_oracle,
    production_combine_fixture_pins_scalar_tree_difference_and_rollback,
    moe_policy_env_subprocess_matrix}` +
    `decoder::tests::moe_policy_subprocess_probe_int8` +
    `artifacts/perf/bd-2mo.30/profile-recipe-5733407/pass15-exact-torch-corpus-20260710T182636Z/`.
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR
    `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`; pinned
    `modeling_deepseekv2.py` sha256
    `74e36e6bd0ba7bc565ef76464a99baa8e6bccb710ae9c1007b54ac30b855fa4c`
    lines 449-453 and 631-703; oracle fixture sha256
    `84fa230a314ee2722a256b07504f9b319c470e246cb50729d1f49db78a3d01a2`;
    fixture generator `scripts/gen_moe_torch_2_10_cpu_fixture.py` sha256
    `580c3751753b813cbba99076858e0f5f978c68292a316eaaf3f49113292fde2e`;
    `torch==2.10.0` commit
    `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`; PyTorch
    `TopKImpl.h` sha256 `dddf5fc982e7d25f9ba38fe1ebd6645fb8851485f25fd5e975822ae90828635a`;
    LLVM `llvmorg-15.0.7` libc++ `nth_element.h` sha256
    `e0152c1647c275c112fea5fb477b6859a2b2f213d905d28183159731d77ffdd9`
    and `sort.h` sha256
    `33f739d139c79d5467aa69083e696b157c3d73e72711ff7552e6cd66e87d3f36`.
- CPU feature string: Apple M4, arm64+neon+dotprod+i8mm; torch reports its
    wheel was compiled as C++17 with clang 15.0.0 and `NDEBUG`; oracle and Rust
    fixture executed on macOS 26.5 arm64.
- Exact command + env:
    `uv run --python 3.12 --with torch==2.10.0 python scripts/gen_moe_torch_2_10_cpu_fixture.py --check tests/fixtures/moe_torch_2_10_cpu.json`;
    Rust: `rch exec -- cargo test --lib moe_policy -- --nocapture`. The Python
    checker is model-free, uses the fixture's byte-exact SplitMix64 corpus
    specifications, calls `torch.topk(scores, 6, sorted=False)` and contiguous
    `x.view(cases,6,1280).sum(dim=1)`, checks thread counts 1/2/4/8, and hashes
    the resulting bytes. `--check` hashes the fixture before and after and never
    writes it. The Rust tests independently regenerate the same inputs and bind
    the frozen receipts to public f32/batched and private int8 prefill/decode
    paths at hidden width 1280.
- Reference behavior: at `n=64,k=6`, PyTorch 2.10 CPU takes the
    `k*64 > n` branch and calls libc++
    `nth_element(begin, begin + k - 1, end)` with a NaN-first, score-descending
    comparator, then returns queue slots `0..6` without sorting. Exact ties,
    signed zero, and multiple NaNs are comparator-equivalent, so their selected
    ids and permutation expose the libc++ partition algorithm. The model then
    restores those six slots and runs `sum(dim=1)` at hidden width 1280.
- Our impl: safe Rust transcribes the pinned libc++ median-of-three, guard,
    partition, and small-range selection-sort algorithm; it uses no `unsafe` and
    never delegates to the host standard library. f32 and int8 grouped prefill
    retain expert GEMM grouping but restore every result to its original route
    slot. Prefill, batched execution, and single-token decode all call the same
    `moe::combine_routed_rows` primitive. At the production
    `[tokens,6,1280]` geometry torch's reduction is bit-identical to the
    slot-ordered f32 left fold used by that primitive. Runtime routing rejects a
    non-finite softmax row instead of propagating NaNs; the lower-level oracle
    corpus still pins PyTorch's exact NaN ordering semantics.
- Fallback / kill-switch state: default reproduces the pinned torch CPU slot
    permutation and production-width reduction. `FOCR_MOE_SCORE_ORDER=1`
    process-wide restores the immediately preceding deterministic policy:
    descending score with lower expert id for ties, followed by ascending-expert
    reduction. The rollback applies uniformly to f32/int8 prefill, batched, and
    decode paths; it is no longer a decode-only fork. Only trimmed,
    case-insensitive `1`/`true`/`on`/`yes` enable it; unset, empty, falsy,
    unknown, and non-Unicode values fail closed to the pinned default. Fresh
    subprocess tests cover unset, `0`, and `1` so the process-wide cache cannot
    hide an environment parsing regression.
- Measured impact: the safe Rust partition matches all 2,048 torch oracle cases
    exactly (unique values, tie cardinalities 1/2/3/7, monotone permutations,
    signed zero, infinities, and NaNs), producing 12,288 index bytes with SHA-256
    `f39d18c1ef6fad5971b01268acd959bb8b559e2f68b0004386daf759a32d8898`.
    The 256-case x 1,280-channel reduction corpus matches torch bit-for-bit over
    1,310,720 result bytes (SHA-256
    `2a0c796a7940c865bfdf9a8d50830467de87e2818697293ac7120c9422a44e2d`)
    and was invariant at torch thread counts 1/2/4/8. The cancellation probe
    `[2^24,1,-2^24,1,1,1]` is deliberately shape-sensitive: scalar
    `[1,6,1]` torch reduction returns `5.0`, production-width `[1,6,1280]`
    returns the slot left-fold result `3.0`, and rollback ascending-expert order
    returns `4.0`. The port pins the model geometry, not the unrelated scalar
    reduction kernel. Restoring six route rows costs `tokens * 6 * 1280 * 4`
    scratch bytes (about 8 MiB for the 273-token base prefix); profile this cost
    after the correctness A/B rather than weakening route-order parity. The
    current-tree conservative-model receipt then emitted EOS on `page_0590` at
    4,033 generated tokens with normalized CER `0.6164885983`, and the complete
    20-page batch succeeded 20/20 with aggregate normalized CER
    `0.1930792459 <= 0.25`. This accepts the routing correction and clears the
    local termination/corpus budget; it does not describe the hard-page tail as
    exact or release-certified.
- End-to-end receipt identity: binary sha256
    `6a401be3bfd6bfb0a9766e2ff333489a5e02b2a5f0641376b6ab5fc6885e6953`;
    model sha256
    `573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592`;
    `page_0590.png` sha256
    `6d71d9c94f2370f51824fb91e3291ce4c64052979adc8f3b14dfe618683512d3`;
    BF16 page reference sha256
    `6542b1d31b64103e9a56104738bf9038487877e8408b8ac34d8d10d1f5d2c8cd`;
    page output sha256
    `57d9b5a6686e4c8f877b42390c74b85f77ef789036351cfbd0f373194d10997a`;
    page comparison sha256
    `49d0b7e9074348f6e38e44d843c7f3f2e7091de643f0ba331dd874ae61f930bb`;
    20-page batch sha256
    `7d6e1f5d7823c4a297b5fcc07f7ade44a0ce36f0cd89a14f4d6a424345db572c`;
    corpus comparison sha256
    `38680e64677ab30f4842aa53fa98eb9c1ef0caefca3e293163f1e34b501d9f43`;
    summary sha256
    `1b43e4d0bc35054e1b0068a65bcb9a7724ed548e3afb70c0b42da56f263e9bcc`.
- Resolution: ACCEPTED - the pinned CPU primitive is exact across the local
    f32/int8 execution paths and the hash-bound conservative-model page/corpus
    gate passes. The accepted result remains a local candidate receipt: the
    high `page_0590` tail is documented, and distribution, regenerated gauntlet
    bundle, signing, and release publication remain pending.
- Tests affected: the four primitive fixture tests, the fresh-process policy
    matrix and its f32/int8 path probes, plus `tests/batched_moe_parity.rs`; no
    XFAIL or SKIP.
- Review date: 2026-07-10; next review is due when the exact-recipe candidate is
    published and the final release bundle is regenerated.

---

## DISC-005: TrOMR int8 weight storage diverges the token stream on ONE tier-2 degraded page (spohr_p100)

- claim_id / evidence_id: CLAIM-tromr-int8-p100 → bd-av64.12 (measurement log in
    the bead close note; comparison tree preserved in the session scratchpad)
- Provenance (model commit + fixture hash): TrOMR export `model.safetensors` sha256 `41c88802…`
    (`TROMR_EXPORT_MANIFEST.json`); f32 artifact `tromr.focrq` sha256
    `a9d41485…` (models-tromr-v1); int8 artifact `tromr.int8.focrq` sha256
    `cced11c0…` (models-tromr-v1, 40/260 tensors QInt8PerChan = exactly the
    decoder-GEMM candidate set, 61 107 485 bytes vs 86 168 002 f32, −29 %);
    fixture `tests/fixtures/realscan_music/pages/spohr_p100.png` sha256
    `b3004420f8ca5de6…` (tier-2: goldens-labeled-NOT-truth, 1843 real scan).
- CPU feature string: aarch64+neon+i8mm (Apple M4).
- Exact command + env: `focr ocr --model <artifact> --task music
    tests/fixtures/realscan_music/pages/spohr_p100.png` per artifact, byte-diff
    of the two MusicXML outputs; `scripts/realscan_music_gate.sh` with
    `FOCR_TROMR_DIR=<int8 dir>`.
- Reference behavior: the published f32 artifact's forward (itself unreliable
    on this page: repetition-runs — E4 quarters — and an `overfull_bar 448/256`
    sanity annotation).
- Our impl: int8 per-output-channel weight STORAGE with dequant-on-access in
    `Weights::mat()/vec()` (`src/native_engine/weights.rs`) — compute stays f32;
    the divergence is pure weight-rounding (`round(w/scale)*scale`), not an
    int8 kernel.
- Fallback / kill-switch state: artifact choice, not an env var — the f32
    artifact stays published; `focr pull tromr --quant f32` restores the
    reference bytes exactly (the round-trip test proves HP tensors byte-exact).
- Measured impact: 5/6 corpus fixtures byte-identical MusicXML end-to-end,
    including the committed golden `spohr_no17_top` (also byte-identical when
    run from a clean-cache PULLED int8 artifact). Corpus gate delta 0: same
    pass/xfail verdicts, same staff counts (p055 5/5, p100 3 recognized vs
    floor 1). p100 only: token stream forks (int8 garbles with F4
    repetition-runs where f32 garbles with E4 runs; both annotated overfull);
    the fixture has no ground truth, so neither side is "correct" — the page's
    truth floor (≥1 staff recognized) holds identically.
- Resolution: ACCEPTED — int8 published as the default pull quant (matches the
    zoo convention; 25 MB smaller download), f32 retained for bit-exact
    reference work.
- Tests affected: `quant::convert::tests::tromr_real_artifact_roundtrips_byte_exact`
    (accepts the 0-int8 f32 artifact or the 40-int8 artifact, HP remainder
    byte-exact); `weights::tests::qint8_records_dequant_on_access_via_mat_and_vec`;
    `scripts/realscan_music_gate.sh` (green on both artifacts).
- Review date: 2026-07-07.

---

## DISC-007: TrOMR upstream `readimg` blanks fully-opaque RGBA inputs; opaque-alpha images take the RGB path here

- claim_id / evidence_id: CLAIM-e8-alpha-ink → the armed E8/E9 cert logs
    (`tromr::tests::{tromr_preprocess_envelope_and_output_gate,tromr_ser_vs_committed_ground_truth}`,
    src/native_engine/tromr.rs) + `scripts/gen_reference_fixtures_tromr.py`
- Identity note: this entry was originally assigned the already-used
    `DISC-004` identifier. Live source and documentation use `DISC-007`; frozen
    receipts may retain the historical ambiguous label.
- Provenance (model commit + fixture hash): NetEase/Polyphonic-TrOMR
    `img2score_epoch47.pth` sha256 02925259ef… (census pin, tromr-spec §Sources);
    `examples/{1..4}.png` + committed `.txt` ground truths (upstream clone)
- CPU feature string: aarch64+neon+dotprod (Apple M4)
- Exact command + env: `FOCR_TROMR_DIR=<zoo>/tromr cargo test --release --lib --
    --nocapture tromr::tests`
- Reference behavior: upstream `staff2score.py::readimg` applies `img = 255 −
    alpha` to EVERY 4-channel input (the rendered-PNG ink convention). The
    repo's own `examples/*.png` are fully-opaque RGBA (alpha ≡ 255, measured
    2026-07-06), so upstream's literal code feeds the model an ALL-ZERO image
    for its own demo staves; the model then hallucinates a stereotyped
    ~42-token reading that is IDENTICAL across different staves (verified on
    the oracle itself: examples 1 and 2 produce the same argmax stream on
    blank input). Argmax SER vs the committed ground truths on blank input:
    ~1.55 (garbage).
- Our impl: the inverted-alpha ink path fires ONLY when the alpha channel
    actually varies (`min(alpha) < 255`); fully-opaque RGBA takes the
    BGRA→RGB → cv2 fixed-point luma path (`preprocess::tromr_staff_tensor` +
    the fixture script's `readimg`, both sides identical).
- Measured impact: with real (non-blank) input the model reads staves
    correctly — L5 SER vs the four committed ground truths: 0.125 / 0.040 /
    0.375 / 0.304 (mean 0.211, argmax decode), and the recognized opening
    (`clef-F4+keySignature-CM`) matches the ground truth's own opening.
    Decode-mode attribution: per-head ARGMAX and upstream top-k/T=0.2
    sampling produce IDENTICAL streams on real inputs (SER equal to 3 d.p.)
    — the earlier apparent "argmax collapse" was entirely the blank-input
    artifact. L0b preprocess envelope vs the cv2 reference on the luma path:
    maxabs exactly 1 u8 LSB, 0/102400 pixels past 1.5 LSB, and the
    output-level gate (our preprocess → certified encoder+decoder) stays
    TOKEN-EXACT.
- Fallback / kill-switch state: `FOCR_TROMR_SAMPLE=1` enables the upstream
    sampling arithmetic from a pinned PCG32 seed (`FOCR_TROMR_SEED`,
    default 0) — deterministic per seed; the default is per-head argmax.
- Resolution: ACCEPTED as a deliberate, justified divergence from upstream's
    literal code — their convention self-evidently blanks their own demo
    inputs; ours preserves the ink convention exactly where alpha carries ink
    and is measured-superior everywhere else (SER 1.55 → 0.211 on the
    committed examples).
- Tests affected: `tromr::tests::tromr_ser_vs_committed_ground_truth` (pinned
    gates mean ≤ 0.25, per-example ≤ 0.45),
    `tromr::tests::tromr_preprocess_envelope_and_output_gate` (envelope
    reported every run + token-exact output gate); the frozen pre-fix stream
    literals live on in `merge_semantic_matches_upstream_golden` as a
    self-consistent synthetic merge case.
- Review date: when E5 (staff-detection front end) lands — re-measure the
    envelope + SER over its crop corpus, and extend the alpha-variance rule
    if real rendered-PNG (transparent-background) staves appear there.

## DISC-008: the wasm-only int4 browser artifact — quantization tail on degraded newsprint + platform f32 decode drift
- claim_id / evidence_id: CLAIM-WASM-INT4-V2 → scratchpad corpus receipts
    (`corpus/MANIFEST.json`, `cer_v1_vs_v2.tsv`, `corpus/browser/REPORT*.json`,
    session 2026-08-10..11); to be re-homed under `artifacts/parity/bd-b8ta/` at
    the next evidence sweep.
- Provenance (model commit + fixture hash): HF 3a7f4db… source shard sha256
    2bc48a7a… → `unlimited-ocr.wasm-int4.focrq` v2 sha256 2653831c… (recipe
    `unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1`, calibration-aware:
    importance-weighted clip search + AWQ down_proj fold over a 13-page
    activation-stats run, `scratchpad/calib/expert_stats.json`). Corpus: 3 repo
    fixtures + 4 public-domain archive.org scan pages (1873/1886), provenance in
    `corpus/provenance.md`.
- CPU feature string: native side aarch64+neon+dotprod (autovec default); browser
    side wasm32+simd128, serial AND threaded (8 rayon workers) — threaded output
    byte-identical to serial wasm.
- Exact command + env: native `focr ocr --model <artifact> <page>` (defaults;
    `FOCR_STAGE_BUDGET_FORWARD_MS=7200000` for the hard pages); browser via
    `site/harness/differential.mjs` (real Chromium, real `_headers`, persistent
    profile).
- Reference behavior: the certified native bf16-source decode (conservative
    cached int8 route) on the same pages. NOTE: this is a PROXY reference — the
    torch oracle was not re-run on these corpus pages (the pinned oracle stack is
    not installed on this host), and the bf16 decode is itself imperfect OCR. No
    ground truth exists for the archive scans.
- Our impl: (a) offline int4/int8 quantization per the wasm recipe
    (`src/quant/convert.rs::wasm_int4_to_focrq`); (b) browser decode over the
    identical artifact through `focr-wasm` (scalar int8 tier — bit-identical to
    every accelerated tier by the selftest contract; the drift source is
    autovectorized f32 glue codegen, proven by `FOCR_FORCE_ARCH=scalar` native
    runs reproducing the native answer exactly).
- Fallback / kill-switch state: the artifact is wasm-lane-only — the native
    resolver refuses it as a default and `focr pull` never fetches it; native
    users keep the certified conservative artifact. In-browser, the labeled
    repetition-guard `no_repeat_ngram=20` (the README-documented
    `FOCR_NO_REPEAT_NGRAM` mitigation) is applied via the model manifest's
    `decodeGuard`; the native CLI can reproduce it with the same knob.
- Measured impact: (v2 artifact, true Levenshtein)
    * Quantization fidelity, native int4 vs native bf16 decode: CER 0 on 3/7
      pages, ≤0.0006 on 2, 0.0155 on one fixture, **0.1562 on the degraded 1886
      newsprint page** (v1 RTN was 0.1753; calibration closed 10.9% relative;
      corpus micro-CER 0.0962 → 0.0856, nothing regressed on any page).
    * Platform drift, browser vs native on the SAME artifact (v1 measurement):
      byte-identical modulo the CLI trailing newline on 4/7 pages, ≤3 chars on
      2, and on the hardest page the un-guarded wasm decode fell into a
      degenerate repetition loop (CER 0.3489 vs native int4) — the motivation
      for the labeled `decodeGuard`.
    * v2 + `decodeGuard=20` in-browser differential (real Chromium, threaded
      lane, parts-streamed 2.8 GB, `corpus/browser/REPORT-v2.json`): the
      repetition loop is GONE — 5/7 pages at edit distance 1 (the CLI trailing
      newline), 1 at distance 3, and the hard newsprint page lands at
      CER 0.1494 vs native-v2 (distance 2012/13.5 KB, the platform f32-drift
      band, no degenerate loop). Browser wall time 80–406 s/page (8 threads);
      total 1325 s browser vs 1611 s native for the 7 pages on a contended
      host.
- Resolution: ACCEPTED for the wasm-only browser lane (never a native default),
    with per-expert GPTQ error compensation filed as the next quality lever
    (bd-50wo remainder). INVESTIGATING remains open only for the f32-glue drift
    class (bd-ajna).
- Tests affected: `site/harness/differential.mjs` (browser-vs-native
    differential), `site/harness/persistence.mjs`; native corpus receipts in the
    scratchpad paths above.
- Review date: 2026-09-15, or immediately upon a GPTQ-compensated artifact.

## DISC-004: multi-page (640-squash) f32 subject forks from the bf16 oracle after the deterministic plate

- claim_id / evidence_id: CLAIM-l5multi-fork → `tests/parity_ladder.rs`
    `l5_multi_page_matches_infer_multi_oracle` and
    `l5_multi_page_10p_long_horizon` (the armed parity lines) +
    `tests/fixtures/multi_page/{p9_p14,p10}_{raw.md,meta.json}`
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR
    `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` (shard sha256 2bc48a7a…); oracle
    raw sha256 `6fad1a5e0dbb22257f95c805c8ee0e053f9bc8a014737d10bcec8615e72ee54d`
    (2 pages) and `66bd3447c49c904f6fd3bfaed304a929bfc07dc9f1766065d8583c5a07de99b8`
    (10 pages), generated by `scripts/baseline/run_baidu_reference_multi.py`
    (torch 2.10.0 / transformers 4.57.1 bf16-CPU, 640, greedy,
    no_repeat_ngram 35 / window 1024)
- CPU feature string: aarch64+neon+dotprod (Apple M4)
- Exact command + env: `FOCR_MODEL_PATH=<model dir> FOCR_CORPUS_DIR=<pages dir>
    cargo test --release --test parity_ladder l5_multi -- --nocapture`. The
    near-tie probe below: `run_baidu_reference_multi.py --model <HF dir>
    --pages-dir <pages> --out <dir> --dtype f32 --max-new-tokens 60 --margins
    <the ten pages>` (and `--dtype bf16`), torch 2.13.0 / transformers 4.52.1.
- Reference behavior: `infer_multi` at 640 emits the page-1 plate + two
    `<|det|>` footer spans (257 chars, 103 tokens), then EOS. Over ten pages it
    decodes 7117 tokens with 9 `<PAGE>` markers.
- Our impl: the SAME squash-640 pipeline (PIL-bicubic hard-wired at the
    multi-page squash site — the CatmullRom default measurably garbled glyphs
    at this 2.9× downscale and was fixed during this rung's bring-up, as was
    an aspect-preserving pad that should have been the reference's SQUASH).
    Until 2026-10-30 both vision towers also fitted their position tables to
    the 640 grid with plain bicubic, where the reference uses torch's
    antialiased bicubic (bd-p8hu). Part of the fork this entry first recorded
    was that defect, not precision: the 2-page subject then rambled a short
    "page N" run to 266 tokens, and the 10-page subject ran to the 32768
    position cap. With the antialiased resize the subject reproduces the
    plate, reads the footer region and stops on its own: the v1 int8 token
    goldens (`tests/fixtures/uocr2_v1_goldens/`) stop after 90 tokens on 2
    pages (the oracle: 103) and 6435 on 10 (the oracle: 7117). The "f32"
    subject's 10-page text is byte-identical to that golden: the cached
    decode runs the conservative recipe (f32 attention and `lm_head` over
    FFN/expert matrices quantized to int8 at load, even from the bf16 shard;
    `OcrModel::generate_cached`), the same recipe the int8 artifact stores.
- Measured impact: (2026-10-30, armed, f32 subject; CER over the
    oracle-length prefix, both sides through the same `finalize_multi`)
    * 2 pages: **0.0522** (was 0.1791 before bd-p8hu); the three-line plate
      is exact.
    * 10 pages: **0.3186** (was 0.4045); markers 8-vs-9. The 10-page plate's
      third line ("PRINTED BY … LIMITED,") is a near-tie in the REFERENCE
      MODEL itself. At the word after "SONS," torch's bf16 logits tie
      exactly (`' L'` vs `' LIMITED'`, gap 0.0000), and the next token holds
      OND / UTH / US within 0.25; in f32 the gaps are 0.4958, then 0.0132 and
      0.0804. torch 2.13 bf16 reads "LONDON,", torch f32 "LUTHER,", our
      subject "LUSITER,", and the frozen torch 2.10 bf16 oracle "LIMITED,".
      Our subject agrees with torch f32 on the 0.50-gap step and differs only
      inside the three-way tie (0.08 spread). So the 10-page leg anchors on the first two
      plate lines; the 2-page leg keeps all three.
    * AT 20 PAGES THE REFERENCE ITSELF COLLAPSES (frozen:
      tests/fixtures/multi_page/p20_raw.md, sha 0c148a9e25e8…): 10469 tokens
      but only 7 `<PAGE>` markers for 20 pages, ending in a repetition run
      ("The CAME was a strong enough…") — long-horizon degradation past ~10
      pages at 640 is a property of the REFERENCE MODEL, not of this port; no
      CER gate is meaningful against a degenerate oracle, so the committed
      rungs stop at the 10-page leg and the 20-page fixture stands as the
      upper-bound evidence.
    * THE SHIPPING INT8 FFN PATH LOST A PAGE ON SPARSE CJK PAGES (bd-1jx0,
      2026-10-30). The first `tests/fixtures/cjk_stream` layout (two 1024²
      pages of four lines, 56 px glyphs on a 120 px pitch; sha256
      `5ba337cd…`, `ef01486a…`) reads as the same two pages in the f32
      reference, in the stateless decode (over the bf16 checkpoint, and over
      the int8 artifact's dequantized weights), and in the cached decode on
      the all-f32 cache (`FOCR_DECODE_F32`). The shipping cache, int8 weights
      times activations quantized to int8 per row, reads the title, then line
      4, then lines 2–4, and stops after one page (`v1_page_merge`). A second
      text in the same layout fails the same way. The two cached decodes
      differ only in FFN/expert storage and GEMM, so the flip is that path's;
      that the stateless decode over the same int8 weight values reads
      correctly points at the activation quantization. The same text at 40 px
      on a 64 px pitch (the committed fixture since bd-1jx0) reads as the
      reference's two pages, character for character, on every path.
    The remaining fork is the same precision-trajectory class as DISC-003
    (greedy path divergence at a near-tie under a different
    summation/precision), amplified by the lossy 640 squash making footer
    glyphs genuinely ambiguous.
- Fallback / kill-switch state: `FOCR_MAX_NEW_TOKENS` bounds the subject tail
    for tighter comparisons; the multi-page squash kernel itself has NO env
    (PIL-bicubic is parity-mandatory at this site — the shipped-default
    CatmullRom of DISC-001 applies only to the single-image/base sites).
    `FOCR_DECODE_F32=1` (v1 since bd-1jx0) decodes on the all-f32 cache from
    the raw checkpoint: the exact path for the CJK case above.
- Resolution: ACCEPTED as precision-trajectory divergence (reordered/precision
    math, not wrong math — the plate anchor, the probe's measured near-tie and
    the DISC-003 attribution precedent bound the class). Per this entry's
    review clause the budgets were tightened to the new measurements
    (2 pages 0.25 → 0.10; 10 pages 0.50 → 0.40).
- Tests affected: `l5_multi_page_matches_infer_multi_oracle` (3-line plate +
    CER ≤ 0.10); `l5_multi_page_10p_long_horizon` (2-line plate + CER ≤
    0.40); `multi_page_streaming_matches_terminal_assembly_when_armed`;
    `recognize_document_real_model_when_present_else_skip_with_success`
    (`recognize_multi_page_…` until U2-E2);
    `multi_page_streaming_survives_split_cjk_characters_when_armed` (its
    fixture re-rendered for bd-1jx0, the two-marker precondition kept).
- Review date: when an int8 or bf16-matched decode variant lands for the
    multi-page path — re-measure the fork point and the near-tie margins.

## DISC-003: SmolVLM2 f32 describe e2e flips near-tied tokens via summation-order drift

- claim_id / evidence_id: CLAIM-c8-neartie → the armed C8 cert logs
    (`smolvlm2::tests::describe_e2e_matches_oracle_l4`, src/native_engine/smolvlm2.rs)
- Provenance (model commit + fixture hash): HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    model.safetensors (zoo/smolvlm2/SHA256SUMS); vision oracle fixtures
    `tests/fixtures/smolvlm2/vision_oracle_fixtures.json` +
    `sample_photo.png` sha256 c69c42d3… (gen_reference_fixtures_smolvlm2_vision.py,
    transformers in-tree ≥4.50, ALL floors = 0.00e+00)
- CPU feature string: aarch64+neon+dotprod (Apple M4)
- Exact command + env: `FOCR_SMOLVLM2_DIR=<zoo> cargo test --lib -- --nocapture
    describe_e2e_matches_oracle_l4`
- Reference behavior: f32 greedy describe of the committed sample photo — 64 ids,
    "…buildings are primarily rectangular and have multiple windows, suggesting…"
- Our impl: the FULL native pipeline (L0b preprocess maxabs 0.0, SigLIP cert cos
    1.00000000 / maxabs 4.4e-4, connector maxabs 2.6e-4, prompt id-EXACT 876/876)
    matches the oracle for a 22-token exact prefix, then flips one near-tied token
    ("multiple windows…" → "a uniform color scheme…" — both coherent, faithful
    captions of the fixture image) and re-converges structurally.
- Measured impact: (fully attributed, 2026-07-03) divergence is
    decode-trajectory-only and belongs to the **KV-cache fast path**, not the
    decoder math. Three probes isolated it, all with the ORACLE's own
    `connector_out.bin` vision rows:
    * **prefill logits are essentially exact** — at the ledger's step-0 anchors
      our drift is < 5e-5 (our top-2 gap 0.5699 vs oracle 0.5700), argmax exact;
    * **the O(n²) `generate_greedy` path (same sdpa rounding as prefill) is
      64/64 id-EXACT** vs the oracle — prompt + splice + decoder math certified;
    * **the O(n) `generate_greedy_kvcache` path** (bespoke token-major
      decode-attention, a different f32 summation order) first flips at step 20
      on the oracle's rank-2 token at a top-2 gap of 0.353 — the per-step
      rounding difference compounds along the autoregressive chain. The full
      native pipeline (our vision) flips similarly at step 22. On the int8
      artifact the C5 cert's kvcache==greedy BIT-identity still holds
      (activation quantization absorbs the drift), which is why B9 never saw
      this: it is f32-only, long-decode-only behavior.
    ATTRIBUTION GATE: the oracle fixture carries a per-step top-2 logit ledger
    (`l4_describe_greedy.step_top2`); the cert asserts every first divergence
    lands on the oracle's rank-2 token at a gap ≤ 0.5 (median ledger gap ~1.0),
    plus an opt-in `FOCR_SMOLVLM2_CERT_FULL=1` leg re-proving the greedy path
    id-exact — a real defect (wrong math, not reordered math) fails both.
- Fallback / kill-switch state: none needed (f32-vs-f32 numerics, not a quant
    tier); for a trajectory bit-faithful to the sdpa math, `generate_greedy` is
    the O(n²) reference path.
- Resolution: ACCEPTED as reordered-math (not wrong-math), attribution-gated —
    every first divergence must land on the oracle's rank-2 token within the
    ledgered top-2 gap; the greedy path stays id-exact under
    `FOCR_SMOLVLM2_CERT_FULL=1`.
- Tests affected: `smolvlm2::tests::describe_e2e_matches_oracle_l4` +
    `decoder_qwen2::tests::smolvlm2_kvcache_greedy_matches_oracle_l4`
    (ledger-gated acceptance, NOT XFAIL — the attribution gate IS the assert);
    `onechart::tests::opt_kvcache_matches_greedy_and_oracle` (prefix ≥ 12 gate).
- ALSO OBSERVED at OPT geometry (OneChart D4, 2026-07-05): on the SAME int8
    artifact, the kvcache and re-prefill greedy paths agree for a measured
    13-step prefix at ~320 positions, then flip a whitespace/quote-class JSON
    near-tie — the same reduction-order compounding; the D4 cert gates
    prefix ≥ 12 plus the `<Number>`-first and dict-open structural anchors
    (`onechart.rs::opt_kvcache_matches_greedy_and_oracle`).
- Review date: when C8's L5 caption/VQA quality budget lands — score both captions
    under the keyword-containment metric and confirm the flips move no metric.

## DISC-002: SmolVLM2 int8 decode flips a near-tied greedy token vs the f32 oracle

- claim_id / evidence_id: CLAIM-c5-int8-neartie → the armed C5 cert logs
    (`smolvlm2_decoder_matches_torch_oracle` / `smolvlm2_kvcache_greedy_matches_oracle_l4`,
    src/native_engine/decoder_qwen2.rs)
- Provenance (model commit + fixture hash): HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    model.safetensors sha256 b9bfd456… (zoo/smolvlm2/SHA256SUMS);
    smolvlm2.int8.focrq from `focr convert --model-id smolvlm2` (C2, census-verified);
    oracle fixtures `tests/fixtures/smolvlm2/oracle_fixtures.json`
    (gen_reference_fixtures_smolvlm2.py, transformers in-tree ≥4.50, floor = 0.00e+00)
- CPU feature string: aarch64+neon+dotprod (Apple M4)
- Exact command + env: `FOCR_SMOLVLM2_MODEL=<zoo>/smolvlm2.int8.focrq
    FOCR_SMOLVLM2_ORACLE_HIDDEN0=<zoo>/smolvlm2_decoder_input.bin cargo test --release
    smolvlm2_ -- --nocapture`
- Reference behavior: f32 greedy decode of the text-only C5 seam prompt — 24 ids,
    "…France is Paris. It is a city located in the northern part of the country…"
- Our impl: the int8 decode (both `generate_greedy` on the `.focrq` and the kvcache
    path — MUTUALLY bit-identical, the B9 contract) flips the near-tied token at
    step 7 ("It" → "Paris", both coherent continuations) and re-converges structurally.
- Measured impact: int8 last-pos logit cosine 0.998301 vs oracle, argmax EXACT at the
    seam; the f32 path (`model.safetensors`) is token-exact for all 24 ids (cos
    1.000000). Divergence is decode-trajectory-only, first flip at generated index 7.
- Fallback / kill-switch state: run the f32 reference weights
    (`FOCR_MODEL_PATH=<dir with safetensors>`) — the f32 decode is oracle-exact;
    the int8 artifact is the speed path.
- Resolution: ACCEPTED — decode-trajectory-only near-tie flip with argmax exact
    at the seam and cosine 0.998301; the C10 VQA gate later measured int8 == f32
    (7/7 BOTH precisions), so the flips move no quality metric.
- Tests affected: `decoder_qwen2::tests::smolvlm2_decoder_matches_torch_oracle` +
    `decoder_qwen2::tests::smolvlm2_kvcache_greedy_matches_oracle_l4`
    (ledger-gated near-tie acceptance, NOT XFAIL).
- Review date: when C8 (SmolVLM2 e2e quality gate) lands a caption/VQA quality budget —
    re-measure whether near-tie flips move any quality metric.

## DISC-001: L0 resampling kernel — `image` crate CatmullRom in place of PIL BICUBIC

- claim_id / evidence_id: CLAIM-l0-resample-catmullrom → artifacts/parity/bd-30me/
    (directory to be populated when the armed L0 EXACT gate runs; today the evidence is
    the in-tree Pillow-12.1.1 goldens + the differential log below)
- Provenance (model commit + fixture hash): HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    + `modeling_unlimitedocr.py` sha256 268bdcbe…: `ImageOps.pad(image, (base_size,
    base_size), …)` at :872-873 and `dynamic_preprocess`'s `resized_img =
    image.resize((target_width, target_height))` at :197 (BICUBIC is `Image.resize`'s
    default) — SOURCE_HASHES.md; GOT-OCR2's `GOTImageEvalProcessor` squash resize is the
    same kernel (spec §13b). Oracle resampler pin: **Pillow 12.1.1** (PINNED_SOURCES.md
    runtime pin). **No parity-corpus fixture hash yet** — the oracle preprocessed-tensor
    fixture (bd-1gv.3.1) is still blocked, which is exactly why the impact below is TBD.
- CPU feature string: n/a — preprocess is scalar integer/f64 image code with no SIMD
    dispatch; the divergence is kernel-semantics, not arch-rounding (same output on
    aarch64 and x86_64)
- Exact command + env: `cargo test -p franken_ocr preprocess::` (Pillow-12.1.1 goldens +
    default-path regression, no env); reference path armed via `FOCR_RESAMPLE=pil-bicubic`
    (e.g. `FOCR_RESAMPLE=pil-bicubic focr ocr <img>`); golden generator + 370-case
    differential: `scripts/gen_pil_bicubic_goldens.py` (header has the uv-venv recipe)
- Reference behavior: PIL `Image.resize(…, Resampling.BICUBIC)` / `ImageOps.pad(…,
    method=BICUBIC)` — Pillow's `src/libImaging/Resample.c` 8-bit path: two passes
    (horizontal then vertical) with u8 clip between them; per-window f64 coefficients with
    the sample window clamped to the image **before** weighting and renormalized by their
    own sum; coefficients fixed-pointed at 2^22 (`PRECISION_BITS = 32-8-2`) rounded half
    away from zero; accumulation from `1 << 21` with `clip8` (>>22, saturate 0..255).
- Our impl: default `image::imageops::FilterType::CatmullRom` at the single resize funnel
    `src/preprocess/mod.rs::resample_exact_with` (feeding `pad_to_square`,
    `build_gundam_tiles`, `got_view_tensor`). Same `a = -0.5` continuous cubic, but
    clamp-at-edge sampling + float accumulation ⇒ NOT bit-identical to PIL. The reference
    path `src/preprocess/pil_resample.rs::resize_bicubic` is a step-for-step `Resample.c`
    port proven bit-exact against Pillow 12.1.1: 370/370 randomized differential cases
    (sources 1×1..640×480 → targets 1×1..1024×1024, random + solid-extreme pixels,
    2026-07-01, `scripts/gen_pil_bicubic_goldens.py`) + 6 Pillow-generated goldens
    embedded in its unit tests.
- Fallback / kill-switch state: `FOCR_RESAMPLE` (default **unset** ⇒ CatmullRom,
    byte-identical to the pre-DISC-001 pipeline — doctrine #2); `=pil-bicubic` restores
    reference-bit-exact PIL BICUBIC at ALL preprocess resize sites for L0 EXACT
    comparison. Note the polarity: unlike most entries, here the DEFAULT is the divergence
    and the switch arms the reference.
- Measured impact: **TBD — honestly not yet measured.** The CER/token-diff cost of
    CatmullRom-vs-BICUBIC needs (a) the armed L0 EXACT gate over an oracle
    preprocessed-tensor fixture (bd-1gv.3.1, blocked) and (b) an e2e A/B
    (`FOCR_RESAMPLE` unset vs `=pil-bicubic`) over the parity corpus, on the pinned
    oracle stack. Known today: aggregate tensor stats match the torch oracle to
    ~5e-3 (`preprocess::tests::got_preprocess_matches_oracle_l0b` tolerances exist
    BECAUSE of this divergence) — that bounds it as small but does NOT license calling
    it zero, and per §above this entry may not be promoted to ACCEPTED until the real
    numbers land here.
- Resolution: INVESTIGATING (divergence ledgered + bit-exact reference path and
    kill-switch shipped; measurement pending the armed L0 gate)
- Tests affected: `preprocess::pil_resample::tests::*` (Pillow-12.1.1 goldens, copy /
    solid / 1×1 / zero-dim semantics), `preprocess::tests::
    resample_kind_default_and_kill_switch_parse`, `preprocess::tests::
    default_resample_is_catmullrom_byte_identical` (doctrine-#2 byte-identity regression),
    `preprocess::tests::pil_kill_switch_dispatch_routes_to_pil_resampler`,
    `preprocess::tests::got_preprocess_matches_oracle_l0b` (stats-tolerance test; goes
    EXACT under the kill-switch once the L0 fixture exists)
- Review date: 2026-07-01

---

Entry shape reference (a **template**, not a measurement):

```
## DISC-001: <e.g. int8 attention q/k/v/o drifts a sub-script token on dense formulas>
- claim_id / evidence_id: CLAIM-int8-attn-qkvo → artifacts/parity/<bead>/
- Provenance (model commit + fixture hash): HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    + modeling_deepseekv2.py sha256 74e36e6b…: <attn lines>  (SOURCE_HASHES.md)
    + parity corpus fixture sha256 <…>  +  <model>.focrq sha256 <int8-attn build>
- CPU feature string: aarch64+neon+i8mm   (and re-checked on x86_64+avx512vnni)
- Exact command + env: cargo test -p focr --test parity -- disc_int8_attn  /
    OMP_NUM_THREADS=8  (reference torch set_num_threads(8), §9.3)
- Reference behavior: f32 Q·Kᵀ / scores·V bmm (modeling_deepseekv2.py:<lines>)
- Our impl: int8 SMMLA attention in src/decode/attention.rs::<fn>
- Fallback / kill-switch state: FOCR_INT8_ATTN (default 0 = reference f32 attention);
    =1 enables the int8 path under test
- Measured impact: CER Δ <x.xx>%, token diff <n> on the dense-formula slice;
    CVaR_0.1 <x.xx>%, EVT_p999 <x.xx>% (AF-2)
- Resolution: <ACCEPTED|INVESTIGATING|REVERT>
- Tests affected: parity::disc_int8_attn (XFAIL while kill-switch ON)
- Review date: 2026-MM-DD
```
