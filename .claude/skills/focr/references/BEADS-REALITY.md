# Beads and BV Reality for franken_ocr

## Table of Contents

- [Purpose](#purpose)
- [Refresh Commands](#refresh-commands)
- [Current Signals to Recheck](#current-signals-to-recheck)
- [Evidence Pattern](#evidence-pattern)
- [Caution Patterns](#caution-patterns)
- [How to Use This Evidence](#how-to-use-this-evidence)
- [When to Update This File](#when-to-update-this-file)

## Purpose

This file captures the kind of project-state evidence the skill should look for.
It is not a substitute for live `br` and `bv` output. The franken_ocr tracker
moves quickly, so re-run commands before making current claims.

## Refresh Commands

```bash
cd ~/projects/franken_ocr
br list --json | jq '{total: length, by_status: group_by(.status)|map({status:.[0].status,count:length})}'
bv --robot-triage
br show <id> --json
```

Use only JSON or robot-safe output. Never run bare `bv`.

## Current Signals to Recheck

This file intentionally does not freeze live tracker state. Before a capability
claim, re-run the tracker and inspect current source/tests. Treat any old Beads
summary as history, not proof.

High-risk claims that require fresh proof:

- `convert` producing a usable `.focrq`
- L5 end-to-end OCR parity against the pinned Baidu reference
- any lossy quantization default beyond the documented recipe
- batch-spine byte identity on the target model/corpus
- PDF input regressions or unsupported-codec gaps
- `-o/--output` file contract for markdown and JSON-with-boxes
- `--extract-figures` file contract, JSON `figures`, and no-stray-files failure behavior
- fresh-pull resolver behavior for `unlimited-ocr.int8.focrq`
- model-zoo manifest pullability (`bd-av64.7` / `ece14f9`): current source
  embeds all five ready models, `ModelEntry.sidecars`, `pull.in_manifest`, and
  `pull.quants`; older binaries can still carry a stale embedded manifest
- public zoo release and clean-cache verification (`bd-av64.8` / `bd-av64.9`):
  GitHub releases `models-smolvlm2-v1`, `models-onechart-v1`, and
  `models-tromr-v1` were published and verified by clean-cache `focr pull`
  runs with exact hashes, sidecars, cache subdirectories, idempotent repull,
  and real inference for each model. HF mirror spot-checks returned 401 and
  remain resilience/auth follow-up; describe mirroring as unverified unless
  fresh evidence supersedes this note.
- Model Zoo registry/discovery (`bd-3jo6`, `focr models`) versus shipped
  artifacts and sidecar installation
- GOT-OCR2/SmolVLM2/OneChart/TrOMR ready and pullable support versus quality,
  perf, and task-subcommand claims
- GOT-OCR2 `--format` / `--task` plumbing versus specialized-output accuracy
  budgets (`bd-3kix` phase 1 has real-model smokes; phase 2 metrics still need
  review)
- GOT-OCR2 no-repeat-ngram/repetition guard behavior (`bd-ff4i` is closed; the
  default guard is 20 with `FOCR_GOT_NO_REPEAT_NGRAM` as override)
- decode tuning flags (`bd-3j3p` closed: `--max-length` and
  `FOCR_MAX_NEW_TOKENS` now reach the engine)
- f32 oracle / L0-L5 ladder claims (`bd-31vc` closed after f32 oracle regen)
- batch-spine/vision claims (`bd-1azu.10` and `bd-1azu.14` closed with
  byte-identity/watchdog evidence; `bd-1azu.14` has a fresh control correction
  proving `FOCR_BATCH_SPINE=0` truly disables for the default spine)
- fused q/k/v decode projection (`bd-241s` closed after `98cc790` /
  `5474ae0`): current source makes `FOCR_QKV_FUSED` default-on. Treat
  `FOCR_QKV_FUSED=0` / `off` / `false` / `no` as the old-path kill switch, not
  as the way to enable the optimization. The source anchors are
  `qkv_fused_enabled`, `fuse_qkv`, `CachedLayerI8.qkv`, and
  `fused_qkv_gemv_is_byte_identical_to_three_calls`. Close evidence cites the
  `bd-1waa` ledger-kept win, `bd-3pg7` ledger resolution, page_0590 SHA/CER
  identity, 20-page CER equality, and page_0009 best-of-3 decode speed
  `0.072 -> 0.052 s/tok`. Do not generalize it to prefill qkv fusion.
- dense batched decode spine (`bd-3jo6.1.7.5` closed/current): source after
  `8497080` has `gemm_i8_bias_prequant_batched`, `BatchedQwen2KvCache`,
  `qwen2_batched_decode_step`, `DenseDecoderBatchStep`, and
  `generate_greedy_batched`, with bit-identical per-stream proof for
  Qwen/Llama and OPT families. Source after `cf0b037` routes GOT-OCR2
  `ocr-batch` through `recognize_batch_dense_got` / `got::recognize_batch`
  under `FOCR_BATCH_SPINE`. Source after `4ca1577` broadens the dense route to
  `OcrModel::recognize_batch_dense` for `got-ocr2`, `smolvlm2`, and `onechart`,
  including `smolvlm2::recognize_batch`, `onechart::recognize_batch`,
  `generate_greedy_batched(..., caps: &[usize], ...)`,
  `PageStream::with_max_emit`, `FOCR_BATCH_PACK`, and the 128/256 batch-width
  default/cap in `batch_scheduler.rs`. `fdd1d64` closes the bead and records
  four-level lossless proof: primitive gates, scheduler mixed-cap/EOS identity,
  real binary byte-identical markdown on all three dense zoo lanes, and the GOT
  model-gated e2e gate. Throughput evidence is scoped: SmolVLM2 1.32x,
  OneChart 1.27x, GOT roughly +3% to +16%; broad batched `lm_head`,
  fairness-controlled A11/PERF_LEDGER, and decode-heavy B>=8 rows remain
  follow-ups.
- preprocess reference resampler (`bd-30me` closed: `FOCR_RESAMPLE=pil-bicubic`
  is bit-exact to Pillow 12.1.1 goldens; default stays CatmullRom under
  DISC-001)
- preprocess flag wiring (`bd-1e9n` closed: `--base-size`, `--image-size`, and
  `--crop-mode` now reach the engine; first Gundam e2e exists but is not a full
  corpus parity cert)
- speculative decode (`bd-1azu.36` closed for the LINEAR gate: 20-page
  ON==OFF byte identity in f32 and int8; per-run ON engagement telemetry is
  still an explicit follow-up, and tree/EAGLE follow-ups remain separate)
- runtime/cancellation/thread-budget lane (`bd-223.2` closed at `59d376b`):
  verify `request_shutdown`, `shutdown_requested`, `reset_shutdown`,
  `cancel_checkpoint`, checkpoints in decoder loops, `FOCR_THREADS` /
  `thread_budget()`, robot `threads`, and bounded `stream_pages()`. If any of
  those are missing, suspect a pre-`bd-223.2` binary or stale checkout before
  calling the feature absent.
- run-state / sync lane (`bd-223.4` closed at `03eadd2`, parent Phase 0 closed
  at `d52d344`): current source implements `src/storage.rs`, `RunStore`,
  `_meta`, `SCHEMA_VERSION`, `FOCR_RUN_STORE`,
  `focr runs --format plain|json|ndjson`, and
  `focr sync export-jsonl|import-jsonl`. Older binaries can still be truthful
  `NotImplemented` scaffolds, so classify the exact binary/source boundary
  before changing docs.
- run/sync frozen-contract lane (`bd-wp8.11` closed at `aab7829`): current
  source/tests freeze `tests/fixtures/runs_schema.json`, populated-store and
  empty-history behavior, stdout purity, locked atomic sync, and one-way JSONL
  sync semantics.
- doctor lane (`bd-wp8.4` / `25eadc5` and `bd-wp8.4.1` closed): current source
  implements detect-only, `--fix`, `--dry-run`, `undo`, `capabilities --json`,
  `robot-docs`, `--robot-triage`, pure detectors, a single mutation chokepoint,
  backups/actions ledger, doctor-local exit codes, and an 8-fixture real-binary
  suite. A scaffolded doctor is stale-binary/source evidence.
- agent ergonomics / robot triage (`bd-wp8.7` / `055e513` closed): current
  source exposes `focr robot triage` with `quick_ref`, `health`,
  `recommendations`, `commands`, `exit_codes`, stdout purity, actionable
  errors, did-you-mean, and empty-history success behavior.
- release-readiness and certification bundle (`bd-wp8.8`, `bd-wp8.9`, and
  `bd-wp8.10` closed): `2bdccc5` adds `--release-readiness`, `8cacf52` adds
  `--bundle`, and `9bc715e` makes `certification_bundle` a live
  `release_certificate.json` cell. `c29a78b` is the certifying source commit:
  readiness reports `ship:true`, `green:13`, `red:0`, the bundle certificate is
  `certified:true`, convergence is `rounds=11/10, tail_clean=True`, and
  `7c7bd00` closes the three Beads. `29516b9` publishes that certified state as
  `v0.6.0`, `beaed7c` records the CI/dist supplement, `db02421` refreshes
  README evidence, `5df6395` commits the post-certification fuzz corpus, and
  `592426c` refreshes public README release identity/asset-size/backend prose.
  Treat the bundle files as the release evidence package for the Phase-5 ship
  gate, not as proof of unrelated open epics or a user's installed binary.
- PDF page selection / split-spreads / CTM rotation (`bd-av64.11` closed after
  `11f60ea` / `9546571` / `5679268` / `b3f74b6`): current source exposes
  `--pages` and `--split-spreads`, keeps source page and half metadata, logs
  split decisions under `FOCR_TIMING`, and applies page `/Rotate` plus
  axis-aligned content-stream image-placement rotation through
  `content_rotation` before OCR. `--split-spreads` still refuses to compose
  with `--extract-figures`; treat that as a known clean usage boundary, not a
  regression.
- TrOMR page resilience plus staff observability (`bd-av64.2` closed):
  `3da9dac` landed the runtime core and `8af3887` landed the observability
  half; `4e881d7` closes the bead. Current source has
  `PageRecognition { staves, skips }`, `StaffSkip { index, bbox, reason }`,
  `recognize_page` skip-on-bad-staff behavior, all-fail error aggregation,
  human stderr skip notes, and `FOCR_TIMING` per-staff dims/outcome. It also has
  the machine-readable staff surface: `MusicPageMeta`,
  `OcrModel::take_music_meta`, `OcrEngine::take_music_page_meta()`,
  `robot::staff_event`, event kind `staff`, and `music_meta_to_json`. Robot
  `staff` events are additive schema-v1 events shaped as
  `{staff, total, bbox, status, reason?}`; `--json` / `-o .json` music runs add
  a detection-ordered `staves` array and non-music JSON shapes remain unchanged.
  Do not use the older `staff_detection` / `staff_result` acceptance wording as
  current event names, and do not demand a schema bump. Known caveat:
  PDF+music multi-page runs expose only the last page's staves through the
  side channel; single-image music is the documented flow.
- TrOMR crop geometry now has committed source landings at `eb0c70e` and
  `40ee875`, and `bd-av64.14` is closed for the fit-first geometry lane plus
  Cadwallader p169 5/5 recognized-staff acceptance. The source-current
  mechanics are fit-first: bands that already fit the
  1280-column position budget keep the historic full-width geometry exactly;
  only over-budget bands get horizontal ink-extent trim with line-spacing pad,
  neighbor-bounded vertical extend-to-fit, non-overlapping staff bands, and
  explicit skip/error handling for bands that still cannot fit. Evidence anchors
  are
  `src/preprocess/staff_detect.rs`, `trim_cuts_page_margins_but_keeps_ink`,
  `fitting_bands_keep_the_classic_full_width_geometry`,
  `wide_staff_with_room_fits_the_positional_budget`,
  `packed_staves_stop_at_the_midline`,
  `unpressured_band_keeps_the_generous_margins`,
  `tromr_page_skips_overwide_staff_and_keeps_the_rest`, and
  `tromr_page_all_staves_failing_is_a_named_error`. Do not infer that corpus
  SER is complete, or that camera dewarp, default/lossless barline quality,
  TrOMR int8, or perf evidence exists unless fresh Beads and gate evidence say
  so. Closed `bd-av64.4` is a separate experimental `FOCR_TROMR_SPLIT=1`
  recognition-count rescue, not a broad quality closure.
- TrOMR real-scan corpus (`bd-av64.6` closed for corpus-v1 measurement at
  `af13d3e`/`3ae26b1`) now exists as committed Spohr fixtures under
  `tests/fixtures/realscan_music/` and the model-gated
  `scripts/realscan_music_gate.sh`. Treat `truth/attributes.json` as the
  tier-1 truth, `goldens/*.musicxml` as frozen model-output regression anchors
  not truth, and page-level floors as robustness evidence through robot `staff`
  events. After `40ee875`, Spohr p055 and p100 are promoted floor checks
  (`min_recognized` 5 and 1) rather than page-level XFAILs; after `91d552f`,
  the shell gate reads those floors from `truth/attributes.json` so future
  floor ratchets have one source of truth. The gate emits `realscan_music/v1`
  NDJSON and fails loud on XPASS for any remaining XFAILs such as the
  double-dotted `bd-av64.13` class. `bd-av64.5` is closed for annotate-only
  musical-sanity telemetry (`focr-sanity` XML comments, robot `music_warning`
  events, JSON `warnings`, stderr/`FOCR_TIMING` detail), but it is not
  auto-correction. `bd-av64.13` is closed after `69039c3`: lever 1
  residual-skew refinement landed at `39651e6` and stayed corpus-safe, but it
  did not flip the double-dotted XFAIL; `FOCR_TROMR_TTA=3` micro-rotation
  voting regressed no17_sys at about 2.8x and did not flip the target; and
  single-staff refined-crop routing broke the committed no17_top golden. The
  negative levers were reverted, and they should not be retried without the
  held-out `.15` corpus and a presence-first scorer. Do not inflate the scoped closure:
  remaining work includes 10-20 item expansion, GOT cross-reference outputs,
  ladder-scorecard row wiring, aggregate SER, and model-quality work.
  `bd-av64.12` is closed after `efccce9` as TrOMR quantized storage, not int8
  compute or a performance win. The published default artifact is
  `tromr.int8.focrq`; `focr pull tromr --quant f32` selects the bit-exact
  `tromr.focrq` reference. Source quantizes exactly 40 Seq2SeqDense decoder GEMM suffixes and
  keeps encoder/embeddings/norms/heads high precision; `Weights::mat()` /
  `Weights::vec()` dequant `QInt8PerChan` through `dequant_qint8()`. Close
  evidence includes byte-identical committed golden output, 5/6 truth-tier
  real-scan MusicXML matches, zero same-verdict/count real-scan delta,
  `DISC-005` for the sole no-truth p100 fork, clean-cache pull byte-exactness,
  pulled-artifact inference matching the committed golden, and a 954-test
  library pass. TrOMR stays absent from int8-kernel selftest until a separate
  compute path exists.
  GOT cross-reference runs should use full systems because auto-format can read
  narrow staff strips as SMILES/molecules. The documented music `semantic`
  JSON key is a doc/code gap until current source proves otherwise.
- Benchmark guardrail (`bd-1a6h` closed at `60d8af4`) ships
  `scripts/bench_guardrail.py` and `benches/.bench-history/baseline.json`.
  It compares `gauntlet_focr.sh` stage records against frozen per-regime
  baselines, fails on >10% regression, treats `cv_pct > 5` or posture mismatch
  as ineligible, refuses perf reporting unless the L0-L5 parity receipt is
  all-green, skips-green on absent fixtures/baselines/receipts, and moves the
  baseline only by explicit reviewed `--ratchet`. Do not confuse this with
  roofline floor accounting; `gauntlet_row.py` remains that ledger path.
- Ngram-lmhead fusion (`bd-2mo.24` closed at `a0ad299`) is ledgered as
  correct-but-does-not-pay. `FOCR_FUSE_NGRAM_LMHEAD` remains opt-in and
  unit-gated by `fused_ngram_lmhead_is_byte_identical_to_separate_mask`, but
  the page_0023 A/B was 16.43s -> 16.40s inside noise with byte-identical
  output. Do not treat it as a default or retry target unless the workload has
  multi-image `ngram_window=1024` ban sets or decode has become about 10x
  faster; rerun that A/B first.
- Phase-2/P3 status after the July 7 sweeps: `bd-1es` is closed for the
  validated weight-only int8 decoder recipe, `bd-2mo.1` is closed for runtime
  ISA dispatch/bit-identical scalar-oracle proof, and `bd-2mo.3` /
  `bd-2mo.3.1` are closed for offline SMMLA prepacking. `focr convert --arch
  aarch64-smmla` emits real `[2x8]` panels, `src/simd/pack.rs` is the single
  packing source of truth, and the loader preserves panels on SMMLA dispatch or
  un-permutes with a warning otherwise. VNNI/AMX are tag-only, row-major remains
  the AVX2 zero-shuffle layout, and this is not a speed claim. The `fcb8289`
  sweep closed 13 kernel/dispatch/proof beads, but the parent `bd-2mo` remains
  open: memory/mmap/allocator reuse (`bd-2mo.22`), NUMA/USL pool sizing
  (`bd-2mo.21`), int8 attention, vectorized exp, and multiple fusions remain
  separate open levers.
- GOT/SAM timing triage (`bd-av64.10`, closed by `efd83e8` as measured final
  state) says artifact
  loading is not the e2e tax: `sam.hydrate` is negligible and `Weights::load`
  is a single `fs::read`; the measured gap is in SAM vision, especially
  attention (`sam.block attn(GLOBAL)`, `sam.block attn(win)`) and MLP timing.
  `01f07fe` landed bit-identical SAM attention speedups and `f3d3215` added
  head-parallel global attention without changing output (`attn(win)` 1.88s to
  0.72s; `attn(GLOBAL)` 2.10s to 1.66s; `sam.forward` 5.55s to 4.24s to
  3.4-3.6s; GOT forward 6.7s to 5.7s to 4.6-4.8s; unlimited-OCR real page
  19.3s to 13.5s). Treat this as committed self-relative stage evidence; the
  formal closeout rows are the later `efd83e8` evidence and new
  revision/host/corpus claims still need fresh matched rows.
  `0298651` adds the committed CLIP-tower pass: `LinearParams` stores
  pre-transposed GEMM-ready weights, `OcrModel` caches hydrated `ClipWeights`,
  96 per-forward transposes are gone, and `vision.clip` is reported as
  2.49s -> 0.77s steady-state with byte-identical proof. If prose or old plans
  still blame artifact hydration, treat that as a superseded hypothesis.
  `f65fded` then lands pass 4 as current committed source: `vision_sam::Linear`
  hydrates through `Linear::from_row_major`, stores a GEMM-ready cached
  `[in,out]` matrix, and removes repeated transposes across SAM, GOT,
  OneChart, SmolVLM2, SigLIP, and TrOMR linear/projector consumers. `3c1b1ea`
  records the pass-3/pass-4 Beads evidence; `efd83e8` is the later formal
  closeout.
  `ab6e083` records the SIMD/polynomial-exp softmax lever as negative and
  reverted; `50d5dad` adds the required `artifacts/perf/bd-av64.10-simd-exp/`
  pointer so the release ledger is green again. Do not relabel that numerics
  substitution as a current speed path.
  `f1ac972` lands pass 5 for SmolVLM2 SigLIP: `smolvlm2::vision_rows` uses
  `vision_siglip::forward_frames_batched` by default, `FOCR_SIGLIP_SEQ=1` is the
  sequential kill switch, and `batched_frames_match_sequential_byte_for_byte`
  proves the batched tower matches the sequential loop. The reported wins are
  loaded-host self-relative pairs; the formal G2 state is the later `efd83e8`
  row bundle.
  `c5e535a` briefly makes row-tiled SAM global-attention score buffers public on
  `origin/main`, `8bd4037` restores the untiled baseline, and `b757bc0` records
  `artifacts/perf/bd-av64.10-rowtile/` plus the negative-evidence row after
  same-regime measurement showed the tiled path was byte-identical but slower.
  Do not treat row tiling as a remaining easy win, a Beads closeout, or a formal
  A11/PERF_LEDGER row.
- GOT-OCR2 batch hydration hoist (`d25dbd7`) is committed source evidence, not
  just live-WIP: `got::recognize_batch` hydrates `SamWeights`,
  `mm_projector_vary`, and `model.embed_tokens.weight` once per batch and logs
  `got.hydrate(batch)` plus `got.vision+splice(batch of N)`. The commit's
  honest attribution is same-binary 3-page sequential 14.47s vs batch 13.53s
  (~6.5%) with `recognize_batch_matches_sequential_e2e` byte identity. Keep it
  separate from the larger `01f07fe`/`f3d3215` SAM-attention campaign and from
  any final A11/PERF_LEDGER matched-reference claim.
- GOT model-level statics cache (`3f2878d`, pass 6) is now Beads/comment-backed
  committed source evidence: `GotStatics` caches the SAM tower, projector, and
  widened embed table on `OcrModel` for both sequential and batch GOT paths, with
  `got.hydrate(cached)` timing. Comment 88 reports about 0.8s/page saved in the
  cited 2-page loop, GOT sample byte identity, full lib 959, fmt/clippy/ubs
  clean, and the armed batch==sequential e2e contract. Keep it scoped; it does
  not replace the later `efd83e8` quiet-host G2/PERF_LEDGER rows.
- `8de3674` is a committed source/package version bump to `v0.5.1` and adds
  `memmap2` to Cargo. `4cedacd` then tags `v0.5.2`, and the `v0.5.2` GitHub
  release is real historical release-asset evidence with platform assets, but
  no longer latest after public `v0.6.0`. Do not infer that a user's installed
  binary has those bits from the tag, Cargo version, or README prose; verify
  `focr --version`, help, and the installed asset.
- `507cebe` ships the `bd-2mo.22` mmap half: default read-only mmap loading,
  `Backing::{Owned,Mapped}`, `FOCR_NO_MMAP=1` owned-buffer fallback,
  mmap-failure fallback, the documented `mmap_island`, and
  `mmap_load_is_byte_identical_to_owned_read` with an is-mapped probe. `0401df2`
  records the Beads note that the full bead stays open for 64B scratch
  alignment, decode-loop buffer reuse, and mimalloc opt-in measurement.
- `38ab806` / `a9a406e` add OneChart model-level statics caching:
  `OcrModel::onechart_statics`, `onechart::OnechartStatics`,
  `onechart::hydrate_statics`, cached SAM/projector/embed, and
  `onechart.hydrate(cached)`. Beads comment 90 records the pass-7 proof:
  byte-identical chart-data output, full lib 960, fmt/clippy clean, and about
  0.10s one-time hydrate. Keep it scoped; it does not create standalone formal
  G2/PERF_LEDGER evidence.
- `9b2a03b` adds SmolVLM2 model-level statics caching:
  `OcrModel::smol_statics`, `smolvlm2::SmolStatics`,
  `smolvlm2::hydrate_statics`, cached SigLIP/projector/embed, and
  `smolvlm2.hydrate(cached)`. Beads comment 91 records the pass-8 proof:
  byte-identical describe output, lib green, about 0.14s one-time hydrate, and
  all four zoo lanes hydrating model-constant tensors exactly once per process.
  Keep it scoped; it does not create public VQA quality or standalone formal
  G2/PERF_LEDGER evidence.
- `efd83e8` closes `bd-av64.10` with nine 2026-07-08 PERF_LEDGER rows under
  `artifacts/perf/bd-av64.10-g2r/`: GOT e2e `0.624 -> 0.885`, OneChart
  `0.546 -> 0.755`, SmolVLM2 `0.878 -> 0.890`, and decode-per-token
  `3.046x / 2.249x / 1.499x`. The close reason is exhausted receipts, not an
  e2e `>=1.0x` win.
- `3f2f97e` is committed `bd-2mo.26` gauntlet harness hardening: AppleDouble
  `._*.rs` junk no longer crashes `check_ledgers.py`, `gauntlet_timing.py`
  parses current SAM/CLIP drill-down timing labels, and `gauntlet_runbook.sh`
  can put rerun evidence under `OUT_DIR`.
- `4291807` closes `bd-127v`: SmolVLM2 untied `lm_head` int8+refine is
  certified and default-on, with `FOCR_GOT_INT8_LMHEAD=0` as the f32-head
  kill switch.
- `ae7b8f2` / `c248e6d` / `c4c1684` close `bd-2mo.26`: page_0009 e2e `3.41x`
  and page_0014 e2e `2.81x` vs pinned HF bf16, with CER `0.00943` /
  `0.03529`, cv% under 5, and evidence under `artifacts/perf/bd-2mo.26/` plus
  `artifacts/perf/bd-re8.17/G2-*-20260708/`.
- `c29a78b` / `7c7bd00` / `29516b9` close and publish the
  release-certification trio: `bd-wp8.8`, `bd-wp8.9`, and `bd-wp8.10` are
  closed, readiness is `ship:true` / `green:13` / `red:0`, convergence is
  `rounds=11/10`, and the public release is `v0.6.0`.
- Dirty-worktree feature-looking changes need quarantine before they become
  truth. `0401df2`, `507cebe`, `8de3674`, `3f2878d`, `d25dbd7`, `eb0c70e`,
  `f65fded`, `e1332a7`, `38ab806`, `a9a406e`, `9b2a03b`, `8d6601d`,
  `4cedacd`, `0924479`, `efd83e8`, `3f2f97e`, `4291807`, `ae7b8f2`,
  `c248e6d`, `c4c1684`, `c29a78b`, `7c7bd00`, `29516b9`, `beaed7c`,
  `db02421`, `5df6395`, and `592426c` are committed facts; anything beyond
  those commits in a dirty checkout is not. Run
  `git diff --stat`, then classify uncommitted README/source/test changes as
  `live-WIP` until the relevant tests, Beads, and committed source agree. This
  matters especially for any
  extra TrOMR test rewrites in `src/native_engine/tromr.rs`, because
  `bd-av64.14` is only a scoped fit-first geometry/p169 closure, not a broad
  real-scan SER/camera-dewarp/default-barline-quality/int8/perf closure.
  The same quarantine applies to new dirty TrOMR storage/quality/perf edits
  beyond closed `bd-av64.12`, and to any new perf rewrites after committed
  `f65fded`. The broad `vision_sam::Linear::from_row_major` spread across GOT,
  OneChart, SmolVLM2, TrOMR, SAM, SigLIP, and related connector checks is no
  longer dirty-only; it is committed current source. New uncommitted follow-ons
  remain useful evidence for what an agent should inspect next, but not
  user-facing current capability until committed source, tests, and Beads agree.
- verification infrastructure closures: `bd-zc1o` closes the robot NDJSON schema
  contract around `tests/fixtures/robot_schema_v1.json`; `bd-n68o` closes
  structured test logging with `docs/TEST_LOGGING.md` and
  `tests/fixtures/test_log_schema.json`; `bd-29wv` closes the model-gated e2e
  runner and skip-with-SUCCESS / `native_path_ran` discipline; `bd-re8.7`
  closes the L5 OCR parity fixture gate. These are strong scoped gates, not
  blanket claims that every model/corpus/perf row is finished. Source at/after
  `adb4ee6` refreshes additive `staff` in `tests/fixtures/robot_schema_v1.json`
  and the hard-coded advertised-events assertion in
  `tests/cli_robot_golden.rs`; `0b74af0` does the same for additive
  `music_warning`. If `bd-wp8.2.2` still appears open, treat that as a possible
  stale tracker mismatch and verify the focused schema tests before making a
  closure claim. `staff` and `music_warning` events still do not require schema
  v2.
  `bd-10sb.1` is closed-current after `f9f4c49` / `2dda846` as property/fuzz
  verification plumbing: `tests/property_suite.rs`,
  `tests/support/proptest_support.rs`, `PROPTEST_CASES`, shrink seeds,
  `fuzz/` cargo-fuzz targets (`focrq_parse`, `safetensors_parse`,
  `image_decode`, `pretok_split`), public `tokenizer::pretok`, and the
  `decode_reader` decompression-bomb guard. Treat it as parser/kernel/preprocess
  hardening, not a new `focr` command, runtime behavior, exhaustive fuzz
  coverage, or release-wide CI proof.
  `bd-4yks` is closed for CI/gate hardening after `e80360b`, `cc79d70`,
  `c960b77`, `29aa40a`, `7777e34`, `2e5801b`, `18712cc`, and round-8 follow-on
  `3f3d9d0`: macOS+Ubuntu run
  the full `scripts/check.sh` gate, gate-log artifacts upload, dist is green,
  `bench-guardrail` is advisory, and the advisory matrix was noted 6/6 after
  the aarch64 SMMLA layout-aware comparison fix. `e80360b` / `cc79d70` are in
  the `v0.5.0` tag ancestry; the remaining closure commits plus the round-8
  sweep are post-`v0.5.0` current-source evidence in the July 8
  `592426c` public-origin probe. `3f3d9d0` records 4.7M fuzz runs,
  zero crashes, `PROPTEST_CASES=2048` 8/8 green, and a 6/6 advisory matrix;
  `ab6fa6c` commits the earlier grown fuzz corpus, `5df6395` commits the
  post-certification 3,271-seed fuzz corpus after a 3.65M-run zero-crash sweep,
  and `592426c` refreshes README release identity/asset-size/backend prose for
  `v0.6.0`.
  Keep the caveats with the closure:
  this is bounded
  deep verification rather than exhaustive fuzz proof, no in-repo `TEST_LOG_DIR`
  capture layer, no scheduled full-model self-hosted runner with 6.67GB weights,
  ARM64 Windows support remains `bd-3u97`, and CI/fuzz closure is not release
  approval by itself. Native Windows x86_64 is supported/proven separately.
  Current release approval is OP-SG evidence from `c29a78b` / `7c7bd00` / `29516b9` (`v0.6.0`).
- backend/SIMD claims: `robot backends` is capability reflection and
  `FOCR_FORCE_ARCH` can force `scalar|sdot|smmla|avx2|avxvnni|avx512vnni` where
  supported; `robot selftest` is selected-kernel parity, not performance
  evidence. Current `ad3ad20`/`adb4ee6` selftest output includes a `models`
  rollup for `unlimited-ocr`, `got-ocr2`, `smolvlm2`, and `onechart`, each
  derived from its case rows and model-specific overflow cases. Closed
  `bd-3jo6.1.12` reports 44/44 cases green on scalar, sdot, and smmla. TrOMR is
  intentionally absent because its current published int8 artifact is
  storage-only and runtime dequants through f32 accessors. Use
  `docs/PERF_LEDGER.md` or a fresh gauntlet for timing.
- determinism and fixture-policy closures: `bd-3kge` closes the shared
  `assert_deterministic` / `assert_outputs_deterministic` gate in
  `tests/support/parity_harness.rs`, including injected nondeterminism failure
  and e2e `recognize()`-twice adoption; `bd-2pgf` closes
  `tests/fixtures/PROVENANCE.md`, `tests/fixtures/MANIFEST.toml`, and
  `scripts/check_fixture_manifest.py` wired into `scripts/check.sh`. Treat this
  as infrastructure proof, not an OCR-quality or perf result.
- conformance-accounting closure: `bd-re8.12` / `fb52843` closes the
  `ConformanceTest` trait and coverage matrix. Current source has
  `RequirementLevel`, `ConformanceCategory`, `conformance_registry()`, and
  `tests/conformance_matrix.rs`; the matrix parses `[SPEC-NNN]` clauses from
  `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`, gates MUST coverage >=
  0.95, emits per-clause NDJSON, rejects bare XFAIL without `DISC-NNN` or an
  explicit phase gap, and runs registry entries green in-process. Treat this as
  spec/accounting infrastructure; differential (`bd-re8.9`), metamorphic
  (`bd-re8.10`), golden-artifact (`bd-re8.11`), three-pillar release
  certification (`bd-re8.13`), conformal ratchet (`bd-re8.14`), and e-process
  invariants (`bd-re8.15`) remain separate proof surfaces even when some are
  closed. The conformal ratchet, Ville e-process monitors, and
  `capacity_certificate_bounded_stream_soak` are release-evidence instruments,
  not standalone ship approval; fold them through `OP-SG` and the capstone
  scorecard, especially when `FOCR_FIXTURES_DIR` is missing or unarmed.
- differential-suite closure: `bd-re8.9` / `390d05c` closes the oracle
  differential comparator. Current source has
  `tests/parity_ladder.rs::differential_per_op_vs_bf16_oracle`,
  `differential_row`, an `EngineIdentity` guard against oracle-vs-oracle false
  greens, ULP/L3-L5 tolerance comparisons against the primary bf16 oracle, and
  `DISC-NNN` XFAIL handling for intentional divergences. Treat model-gated
  skip-with-SUCCESS as missing-artifact evidence, not as proof that native e2e
  inference ran.
- metamorphic-suite closure: `bd-re8.10` / `5f2d7ce` closes
  `tests/metamorphic.rs`. Current relations cover identity resize, rotation
  bbox mapping, mean-gray padding through `preprocess::PAD_FILL`, repeat and
  `FOCR_THREADS=1` vs `4` determinism, logged white-padding SHOULD behavior,
  and gated cross-page dependence. Do not assert that multi-page output equals
  concatenated or summed single-page output; R-SWA makes that a false relation.
- golden-artifact closure: `bd-re8.11` / `f879211` closes the golden suite in
  `tests/cli_robot_golden.rs`, `tests/fixtures/golden/PROVENANCE.md`, and
  `docs/conformance/GOLDEN.md`. Current coverage spans exact CLI/help/schema
  JSON, fuzzy ULP numeric artifacts, scrubbed robot NDJSON, canonicalized
  cross-platform output, manual `UPDATE_GOLDENS=1` only after human diff
  review, `.actual` / `.snap.new` ignored, and CI forbidden from auto-updating
  goldens.
- ladder-scorecard runner: `1b84428` adds `scripts/ladder_scorecard.sh` for
  `bd-re8.19`; `1112cf8` plus the close evidence make the bead closed-current.
  It runs `parity_ladder` serially, folds `event=parity` rows and
  `event=result` outcomes into `focr-ladder-scorecard/v1`, records per-gate
  outcomes/worst rows, sets `all_green=false` when `skipped_no_model=true`,
  marks downstream gates `not_meaningful` after the first hard failure, and has
  `--self-test` for no-weights fold validation. Armed close evidence reported
  all six gates green; unarmed no-model runs still report
  `skipped_no_model=true` and `all_green=false`. Keep `bd-re8.18`
  capacity-certificate proof separate from this scorecard.
- SmolVLM2 route proof (`bd-3jo6.3.2` closed for the real-weights conversion
  census; `bd-3jo6.3.3` closed for C3 SigLIP seams; `bd-3jo6.3.4` closed for
  C4 connector seams; `bd-3jo6.3.5` closed for C5 f32/int8 decoder seam
  evidence; `bd-3jo6.3.6` closed for C6 SmolLM2 tokenizer conformance;
  `bd-3jo6.3.7` closed for SmolVLM2 prompt/IO and preprocessing; `bd-3jo6.3.9`
  closed for `--task describe` / `--question` routing; A8/A9 shared leaves are
  also closed; `bd-3jo6.3.8` closed C8 with L0b/L0c/L2/L3/L4 proof, DISC-003
  near-tie ledgering, L5 7/7 f32 and 7/7 int8 oracle-answer guard, and
  informational release-int8 M4 timings; `bd-3jo6.3.10` closed C10 with
  per-module tests and `scripts/smolvlm2_describe_e2e.sh`; `bd-3jo6.3` closed
  the sub-epic. Still keep public-benchmark, manifest-packaging, and A11
  fairness/perf boundaries separate.)
- OneChart runtime landing (`bd-3jo6.4.1` closed the D1 spec/census;
  `bd-3jo6.4.2` closed arch-aware conversion for OneChart-shaped weights with
  384 source records -> 383 `.focrq` records, 72 OPT int8 GEMMs, tied-head
  dedup, K=768/K=3072 overflow proof, and manifest upload deferred;
  `bd-3jo6.4.3` closed D3 vision/projector certification with
  `onechart_view_tensor`, `model.vision_tower`, `model.mm_projector`
  `Linear(1024->768,bias)`, `onechart_preproc.bin`,
  `onechart_proj_out.bin`, `proj_out cos 1.00000000`, and maxabs `6.5e-4`;
  `20ac599` landed D4 half 1:
  `DecoderFamily::Opt`, `DecoderConfig::onechart`, learned offset-2 positions,
  LayerNorm-with-bias, ReLU `fc1`/`fc2`, biased OPT linears,
  `build_inputs_embeds`, and `onechart_final_logits.bin` prefill proof with
  argmax 50268, cos `1.00000000`, maxabs `6.1e-5`, prompt length 308;
  `2c77d21` landed D4 cached decode support and `2769d21` closed
  `bd-3jo6.4.4`: `generate_greedy_kvcache`,
  `opt_kvcache_matches_greedy_and_oracle`, OPT `GotDecodeWeights`,
  `MlpW::ReluFc`, `family_norm`, same-int8 `onechart.int8.focrq` preference,
  24-token KV-cache vs re-prefill greedy, measured 13-step exact prefix with
  gate >=12, first id 50268, and dict-open decoded output;
  `bd-3jo6.4.9` closed D9 tokenizer conformance with `PretokScheme::Gpt2`,
  `vocab.json`/`merges.txt`/`added_tokens.json`, pinned image/number tokens, and
  29/29 id-exact fixtures; `0145419` added the recognize pipeline and
  `prefill_final_hidden`, and `2a56c96` closed `bd-3jo6.4.5` for D5 native
  assembly with `ChartResult`, `recognize`, `complete_json_string`,
  `number_head`, `reliable_distance`, `recognize_reads_the_committed_chart`,
  `reliable_check_matches_upstream_goldens`, `number_head_matches_golden`, and
  `chart_prompt_ids_match_oracle_l0c`; `e926c46` closed D6/D7/D8 and sub-epic D
  with `OcrTask::ChartData`, `model_spec_is_knowably_not_onechart`,
  `forward_onechart`, `model_arch implemented=true`, and
  `scripts/onechart_chart_e2e.sh`; `bd-2lje` / `9cb91f9` added the
  in-distribution SCRM-proxy corpus, with head fires 6/6, mean distance about
  0.015 int8 / 0.014 f32, byte-identical f32-vs-int8 decoded text, and valid
  JSON 1/6 in both precisions. `bd-av64.7` / `ece14f9` later adds
  `focr pull onechart`; broad quality and a separate `focr chart` still remain
  separate.)
- TrOMR runtime/distribution landing (`bd-3jo6.5.2` / `c22b047` closed E2 conversion for the
  WS-folded export: `tromr.focrq`, 260 tensors, `0 int8`, byte-exact roundtrip,
  `model_id=tromr`; `bd-3jo6.5.6` / `7464590` closed E6 decode-only
  `MusicTokenizer` / WordLevel conformance over four token tables;
  `bd-3jo6.5.3` / `6403d4c` landed `tf_same_pad`, `max_pool2d`, `group_norm`,
  and `fuse_relu` helpers; `45da3a3` committed the hybrid ResNetV2+ViT encoder
  with `TromrEncoderW` and `tromr_encoder_matches_torch_oracle`; `bd-3jo6.5.4`
  / `3472c1b` closed E4 deterministic decoder conformance; `bd-3jo6.5.7` /
  `79d715c` closed E7 semantic merge and MusicXML; `bd-3jo6.5.9` / `78a2de3`
  closed E9 `--task music` with `forward_tromr`, `model_arch implemented=true`,
  and `tromr_music_e2e/v1`. Current default-pull runtime uses
  `--model tromr.int8.focrq`; use `--model tromr.focrq` for the f32 reference
  path after `focr pull tromr --quant f32`. `bd-3jo6.5.8` /
  `2cbded9` closed the single-staff parity/SER/e2e ladder; `fc9d88a` added the
  E5 v1 `staff_detect` module; `752f3cd` closed E5 with `recognize_page`,
  `staves_to_musicxml`, full-page `forward_tromr`, and detector-lossless
  stacked-page SER 0.125 / 0.040; `9127676` pinned DISC-004 alpha routing; and
  `ab0bae0` closed E10 plus sub-epic `bd-3jo6.5`. `bd-av64.7` / `ece14f9`
  later adds `focr pull tromr`; `efccce9` / closed `bd-av64.12` now makes the
  default pull `tromr.int8.focrq` plus tokenizer sidecars and keeps
  `tromr.focrq` behind `focr pull tromr --quant f32`. `bd-2sez` / `5430e2c` then adds the
  TrOMR f32 PERF_LEDGER baseline
  row: token streams agree exactly, but focr f32 loses to pinned upstream torch
  on the measured staff example. Do not infer a standalone `focr music`
  subcommand, int8 decoder kernels, `**kern` export, camera dewarp,
  default/lossless barline quality, or a perf win. Experimental
  `FOCR_TROMR_SPLIT=1` is separate over-budget-staff rescue.)
- head-to-head gauntlet (`bd-re8.17` closed with first pinned HF bf16
  PERF_LEDGER rows; decode-per-token is the G2-clearing win, but cite
  per-stage rows exactly because `page_0009` preprocess is 0.916)
- deferred fresh-eyes minors (`bd-2dui`): spec-loop engagement telemetry,
  sub-100ms timing-resolution guard, batched-spine roofline view counting,
  production-constant parity anchors, phase-2 format metrics, SmolVLM2 L4
  contract selection, explicit-default `--max-length` observability, and the
  watchdog gauge's scope remain follow-up work.
- `.focrq` `model_id` conversion/load semantics and license validation
- task subcommands (`focr music`, `focr chart`, `focr describe`) versus shipped
  CLI; `focr ocr --task` itself is current source
- packaged release/install support on a specific platform
- Windows/macOS/Linux model acquisition

## Evidence Pattern

A credible closed Bead for this project should include:

- exact command and env,
- git revision or artifact hash,
- model source hash or `.focrq` hash when model behavior is involved,
- target CPU/SIMD tier when performance is involved,
- parity/CER/TEDS/golden result when output quality is involved,
- fallback or kill-switch state,
- clear distinction between source scaffold, implemented code, and proven
  behavior.

Use Beads as a pointer to evidence; do not use a close reason by itself as the
final authority.

## Caution Patterns

Do not round these up:

- a source module exists, but the CLI path still returns `NotImplemented`,
- a perf lever is present but logged in `docs/NEGATIVE_EVIDENCE.md`,
- a test is model-gated and skipped because artifacts are absent,
- a README example describes the intended product, but `src/cli.rs` disagrees,
- a binary exists in `target/`, but source/help/tests have moved on.

## How to Use This Evidence

When answering "can focr do X?":

1. Check source for the command/API.
2. Check tests for exercised behavior.
3. Search Beads for the feature and read close reasons.
4. If CASS has hits, use them for historical context only.
5. State uncertainty explicitly when a capability is scaffolded or phase-gated.

Command snippets:

```bash
br search "ocr-batch" --json
br search "int8 convert" --json
br search "PDF" --json
br search "multi-page" --json
br show bd-1gv.25 --json
br show bd-2z0y --json
br show bd-1gv.26 --json
br show bd-1465 --json
br show bd-av64.4 --json
br search "FOCR_TROMR_SPLIT" --json
br search "barline" --json
br search "preprocess_dynamic_squash" --json
br search "page_decoded_event" --json
br search "extract-figures" --json
br search "Model Zoo" --json
br search "GOT-OCR2" --json
br search "model_id" --json
br search "L5 parity" --json
br search "bd-31vc" --json
br search "FOCR_RESAMPLE" --json
br search "bd-1e9n" --json
br search "bd-1azu.36" --json
br search "bd-3jo6.3.2" --json
br search "bd-3jo6.3.5" --json
br search "bd-3jo6.3.6" --json
br search "bd-3jo6.3.8" --json
br search "vqa_quality_matches_oracle_l5" --json
br search "smolvlm2_describe_e2e" --json
br search "bd-3jo6.4.2" --json
br search "bd-3jo6.4.3" --json
br search "bd-3jo6.4.4" --json
br search "bd-3jo6.4.9" --json
br search "bd-3jo6.4.6" --json
br search "bd-3jo6.4.7" --json
br search "bd-3jo6.4.8" --json
br search "bd-2lje" --json
br search "FOCR_ONECHART_DIR" --json
br search "onechart" --json
br search "bd-3jo6.5.2" --json
br search "bd-3jo6.5.3" --json
br search "bd-3jo6.5.6" --json
br show bd-3jo6.5.7 --json
br show bd-3jo6.5.8 --json
br show bd-3jo6.5.9 --json
br show bd-3jo6.5.10 --json
br show bd-av64.8 --json
br show bd-av64.9 --json
br show bd-av64.10 --json
br show bd-av64.2 --json
br show bd-av64.14 --json
br search "aarch64-smmla" --json
br show bd-2mo.3 --json
br show bd-2mo.3.1 --json
br search "robot selftest" --json
br show bd-3jo6.1.12 --json
br show bd-3jo6.1.7.5 --json
br search "tromr" --json
br search "bd-3jo6.3.3" --json
br search "bd-3jo6.1.8" --json
br search "bd-3jo6.1.9" --json
br search "bd-re8.17" --json
br search "FOCR_BATCH_VISION" --json
br search "Windows ARM64" --json
```

If `br search` output shape differs, adapt with `jq` after inspecting it. Do
not silently drop failed tracker queries.

## When to Update This File

Update this reference when:

- a major phase gate closes,
- int4 or additional lossy paths become validated defaults,
- robot schema version changes,
- artifact distribution changes,
- Windows/macOS/Linux support claims change,
- source API signatures in `src/lib.rs` change.

Every update should cite source or Beads evidence in the commit message or
adjacent research notes.
