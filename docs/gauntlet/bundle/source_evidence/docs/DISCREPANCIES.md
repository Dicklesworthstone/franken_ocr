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

## DISC-004: TrOMR upstream `readimg` blanks fully-opaque RGBA inputs; opaque-alpha images take the RGB path here

- claim_id / evidence_id: CLAIM-e8-alpha-ink → the armed E8/E9 cert logs
    (`tromr::tests::{tromr_preprocess_envelope_and_output_gate,tromr_ser_vs_committed_ground_truth}`,
    src/native_engine/tromr.rs) + `scripts/gen_reference_fixtures_tromr.py`
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

## DISC-004: multi-page (640-squash) f32 subject forks from the bf16 oracle after the deterministic plate

- claim_id / evidence_id: CLAIM-l5multi-fork → `tests/parity_ladder.rs`
    `l5_multi_page_matches_infer_multi_oracle` (the armed parity line) +
    `tests/fixtures/multi_page/p9_p14_{raw.md,meta.json}`
- Provenance (model commit + fixture hash): baidu/Unlimited-OCR
    `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` (shard sha256 2bc48a7a…); oracle
    raw sha256 `6fad1a5e0dbb22257f95c805c8ee0e053f9bc8a014737d10bcec8615e72ee54d`
    (generated by `scripts/baseline/run_baidu_reference_multi.py`, torch 2.10.0 /
    transformers 4.57.1 bf16-CPU, page_0009+page_0014 @ 640, greedy,
    no_repeat_ngram 35 / window 1024)
- CPU feature string: aarch64+neon+dotprod (Apple M4)
- Exact command + env: `FOCR_MODEL_PATH=<model dir> FOCR_CORPUS_DIR=<pages dir>
    cargo test --release --test parity_ladder l5_multi -- --nocapture`
- Reference behavior: `infer_multi` at 640 emits the page-1 plate + two
    `<|det|>` footer spans (257 chars, 103 tokens), then EOS.
- Our impl: the SAME squash-640 pipeline (PIL-bicubic hard-wired at the
    multi-page squash site — the CatmullRom default measurably garbled glyphs
    at this 2.9× downscale and was fixed during this rung's bring-up, as was
    an aspect-preserving pad that should have been the reference's SQUASH)
    reproduces the plate block BYTE-EXACTLY, then forks on the fuzzy footer
    region: the bf16 oracle reads footers + stops; the f32 subject rambles a
    short "page N" run until the 35-gram window ban terminates it (266 tokens).
- Measured impact: CER over the oracle-length prefix (both sides through the
    same `finalize_multi`) = **0.1791** (2026-07-07, armed); the plate region
    is exact. Budget pinned at 0.25 in the rung. LONG-HORIZON (bd-1465): the
    10-page leg measures **0.4045** (subject capped at 7600 tokens — a true
    prefix; markers 8-vs-9; plate still exact) — the fork compounds across
    pages exactly as §2.5's graceful-degradation curve expects; the UNCAPPED
    10-page subject runs to the 32768 position cap and terminates cleanly
    (31653 + 1115 prefill = 32768) where the bf16 oracle EOSes at 7117.
    AT 20 PAGES THE REFERENCE ITSELF COLLAPSES (frozen:
    tests/fixtures/multi_page/p20_raw.md, sha 0c148a9e25e8…): 10469 tokens
    but only 7 `<PAGE>` markers for 20 pages, ending in a repetition run
    ("The CAME was a strong enough…") — long-horizon degradation past ~10
    pages at 640 is a property of the REFERENCE MODEL, not of this port; no
    CER gate is meaningful against a degenerate oracle, so the committed
    rungs stop at the 10-page leg and the 20-page fixture stands as the
    upper-bound evidence. The fork is the same
    precision-trajectory class as DISC-003 (greedy path divergence at a
    near-tie under a different summation/precision), amplified here by the
    lossy 640 squash making footer glyphs genuinely ambiguous.
- Fallback / kill-switch state: `FOCR_MAX_NEW_TOKENS` bounds the subject tail
    for tighter comparisons; the multi-page squash kernel itself has NO env
    (PIL-bicubic is parity-mandatory at this site — the shipped-default
    CatmullRom of DISC-001 applies only to the single-image/base sites).
- Resolution: ACCEPTED as precision-trajectory divergence (reordered/precision
    math, not wrong math — the plate-exact anchor + the DISC-003 attribution
    precedent bound the class); the CER budget in the rung is the measured cost.
- Tests affected: `l5_multi_page_matches_infer_multi_oracle` (plate containment
    + CER ≤ 0.25); `multi_page_streaming_matches_terminal_assembly_when_armed`;
    `recognize_multi_page_real_model_when_present_else_skip_with_success`.
- Review date: when an int8 or bf16-matched decode variant lands for the
    multi-page path — re-measure the fork point; if the subject then matches
    the oracle's EOS behavior, tighten the budget toward the plate-exact bound.

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
