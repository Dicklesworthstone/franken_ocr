---
name: focr
description: >-
  Use when working with focr/franken_ocr: OCR CLI, OcrEngine, robot NDJSON,
  .focrq artifacts, Model Zoo (GOT, SmolVLM2, OneChart, TrOMR), kernels, quantization, or ~/projects/franken_ocr.
dependencies:
  - "franken_ocr repo or installed focr binary"
  - "Rust nightly when building from source"
  - "jq for robot/JSON inspection"
  - "Baidu/GOT/SmolVLM2/OneChart/TrOMR source weights or a .focrq model for real inference; TrOMR also needs tokenizer tables beside the artifact"
---

# focr / franken_ocr

## One Rule
Do not invent capability. First classify a surface as current, scaffolded,
planned, or stale, then give the user commands and code that match that class.

## Truth Stack

Use this order when facts conflict:

1. Current committed `franken_ocr` source and tests. A dirty live worktree can
   identify candidate names and code paths, but it is live-WIP until the diff is
   understood and the matching tests/Beads agree.
2. Live help from the exact binary that will run, after confirming it is not
   stale.
3. `br`/`bv --robot-*` issue evidence for current closure/open status.
4. `AGENTS.md`, `README.md`, `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`, and
   `docs/` as narrative context that can lag source and Beads.
5. `cass` session history, as historical context only.

If a command exists in source but not in `focr --help`, treat the binary as
stale and run from source or rebuild before drawing conclusions.

This repository-local skill is the maintained copy for this checkout. If a
global installed `$focr` skill disagrees with `.claude/skills/focr`, use this
project-local skill plus live `~/projects/franken_ocr` source/Beads evidence.

If `~/projects/franken_ocr` is dirty, run `git status --short --branch` and
`git diff --stat` before trusting changed prose or comments. Uncommitted docs
that advertise a feature, uncommitted source that has not passed the relevant
gate, or extreme diffs that look mechanically corrupted are not current
capability. Label them `live-WIP`, then check `br show <id> --json`, tests, and
committed `origin/main` before updating user-facing claims.

## Cold-Start Boundary Card (July 8, 2026)

> **Dated update (2026-08-18).** The card below is preserved verbatim as the
> July 8 observation; these deltas supersede its point-in-time facts:
>
> - **Latest binary release is `v0.7.2`** (published 2026-07-13), not `v0.6.0`.
>   Six release binaries now ship, adding Windows ARM64.
> - **The "Latest" GitHub release slot can be a weights release.** Weights ship
>   as their own non-prerelease releases (`models-*`), so `/releases/latest`
>   returned `models-unlimited-wasm-v1` in August 2026 and broke bare installer
>   runs (GH #12; installers on `main` now filter for semver `v*` tags). Never
>   equate "latest release" with "latest binary release" here.
> - **The default artifact is versioned:** `focr pull` installs
>   `unlimited-ocr.v0.7.0.int8.focrq` (4.16 GB, 3 verified parts) under the
>   cache root, using the conservative recipe
>   `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`. The old
>   `unlimited-ocr.int8.focrq` name and the ~3.9 GB figure are stale.
> - All five runtime models (unlimited-ocr, got-ocr2, smolvlm2, onechart,
>   tromr) are pullable; TrOMR publishes int8 (default) and f32 artifacts.

If you are fresh to `focr`, keep these five ledgers separate before answering:

- **GitHub release:** latest observed GitHub release is `v0.6.0`, a real,
  non-draft, non-prerelease release named "v0.6.0 — certified faster than
  PyTorch, end to end", published 2026-07-08T14:47:48Z with Apple Silicon,
  macOS x86-64, Linux x86-64, Linux ARM64, and Windows x86-64 binaries plus
  `.sha256` assets. The release tag is `29516b9` / `v0.6.0`. This is proof of
  the release-asset boundary, not proof of the installed binary on the user's
  `PATH`, and it does not include source commits after the tag. Historical
  release name to preserve for archaeology: `v0.5.2 — the cold-start release`.
- **Public source head:** `origin/main` is `592426c`, with clean committed
  source describing as `v0.6.0-4-g592426c`; the inspected checkout had no
  tracked source diff and only untracked `.claude/worktrees/`. The post-`v0.5.2`
  chain is now: `0924479` README source/binary clarification, `efd83e8` formal
  `bd-av64.10` G2 re-measurement and closeout, `3f2f97e` `bd-2mo.26` gauntlet
  harness fixes, `4291807` SmolVLM2 untied `lm_head` int8+top-K-refine
  default-on certification, `91c12fd` README selftest evidence refresh,
  `ae7b8f2` bead-scoped perf evidence bundle, `c248e6d` `bd-2mo.26`
  head-to-head PERF_LEDGER rows, `c4c1684` `bd-2mo.26` plus rounds 9-10
  closeout, `c29a78b` release certification achievement, `7c7bd00` Beads
  closeout for `bd-wp8.8`, `bd-wp8.9`, and `bd-wp8.10`, `29516b9` `v0.6.0`
  release version/tag, `beaed7c` CI/dist supplement notes, `db02421` README
  release-readiness evidence refresh, `5df6395` committed post-certification
  fuzz corpus growth, and `592426c` README release-identity refresh for the
  now-unified `v0.6.0` source tag plus binary release. This is current committed
  source/tracker evidence, not proof that a user's installed binary came from
  the same source without an installed-binary check.
- **Committed source features:** `507cebe` mmap-loader half, `3f2878d` GOT
  `GotStatics`, `38ab806`/`a9a406e` OneChart `OnechartStatics`, `9b2a03b`
  SmolVLM2 `SmolStatics`, and `4291807` SmolVLM2 untied-`lm_head`
  int8+top-K-refine default-on certification are current source evidence with
  scoped proofs. The `4291807` boundary proves paired describe output
  byte-identical f32-head vs int8+refine, armed L5 VQA green under the lever,
  full describe/VQA e2e suite pass, `lm_head` about `6.99 -> ~1.2 ms/tok`, and
  decode `40.4 -> 55-57 tok/s`; `FOCR_GOT_INT8_LMHEAD=0` is the f32-head kill
  switch. This is not a public VQA benchmark, broad quality proof, or guarantee
  that the `v0.5.2` release asset includes the post-tag source. Formal
  G2/PERF_LEDGER rows now exist for both `bd-av64.10` and `bd-2mo.26`: the
  `bd-av64.10` final state records GOT e2e `0.624 -> 0.885`, OneChart
  `0.546 -> 0.755`, SmolVLM2 `0.878 -> 0.890`, and decode-per-token
  `3.046x / 2.249x / 1.499x`; the later `bd-2mo.26` head-to-head rows record
  focr-int8 vs pinned HF bf16 page_0009 e2e `3.41x` and page_0014 e2e `2.81x`
  with CER `0.00943` / `0.03529`, best-of-5 warm, N=8 both sides, and cv%
  under the 5% bar.
- **Post-certification fuzz corpus:** `5df6395` commits the post-head-to-head
  corpus growth that used to look like untracked fuzz WIP: 3,271 seed files
  across `focrq_parse` (1,269), `image_decode` (603), `pretok_split` (967), and
  `safetensors_parse` (432), about 721 KB total. The final convergence sweep
  also records four fuzz targets x four minutes, 3.65M total zero-crash runs,
  and `PROPTEST_CASES=2048` 8/8 green in round 11. `592426c` is README-only
  current-source evidence that replaces stale `v0.5.1`/`v0.5.2` public-facing
  release identity, manual-download URL, binary-size, and CPU-backend summary
  prose with `v0.6.0` facts. Treat only new deltas after committed `592426c` as
  live-WIP; do not delete, clean, or fold untracked files into release/source
  claims.
- **Readiness artifact:** `bd-wp8.8`, `bd-wp8.9`, and `bd-wp8.10` are now
  closed in Beads. `c29a78b` regenerates
  `docs/gauntlet/RELEASE_READINESS.json` as
  `franken_ocr.release_readiness.v1` with `ship:true`, `green:13`, `red:0`,
  `certification_bundle` green, and `gauntlet_convergence` green with
  `rounds=11/10, tail_clean=True`;
  `docs/gauntlet/bundle/release_certificate.json` says `certified:true` at
  `v0.5.2-8-gc4c1684`. That apparent version mismatch is expected: the bundle
  certificate was generated at `c4c1684`; `c29a78b` then folded the live
  certificate into the all-green release-readiness scorecard, and `29516b9`
  tagged the certified state as `v0.6.0`. This is the ship-gate receipt, not
  proof of unrelated open epics such as parent `bd-2mo`, `bd-3gaa`, ARM64
  Windows `bd-3u97`, or a user's installed binary.

Mnemonic: release object, source head/features, installed binary,
post-certification corpus/live-WIP boundary, and readiness artifact are five
different truth surfaces.

## First 90 Seconds

Use this checklist before reading deep references:

1. Name the surface: CLI command, Rust library API, model/artifact, verification
   gate, performance claim, or source edit.
2. Classify the evidence boundary: installed binary, current source, committed
   `main`, live worktree, Beads status, or historical docs.
   Dirty-worktree claims need an explicit `live-WIP`/`committed-main`
   distinction before they become advice.
3. For install/release/version questions, keep four facts separate:
   source/Cargo version, pushed tag, GitHub release object/assets, and the
   actual installed binary help/version. Do not infer one from another.
   If `git describe` says `v0.5.0-N-gHASH`, `v0.5.1-N-gHASH`,
   `v0.5.2-N-gHASH`, or `v0.6.0-N-gHASH`, current `main` is ahead of that tagged
   release. If it says `v0.5.2-dirty` or `v0.6.0-dirty`, the tag/release
   boundary can still be real while the checkout contains uncommitted work.
   Treat source tag, release object, release assets, and installed binary as
   four separate facts.
4. Pick one operator/reference from the router below; do not read every file.
5. If the claim is about "done", "green", "conformant", "fast", or "quality",
   name the exact gate and what it does not prove.
6. End with concrete commands or code and an explicit proof gap when a model,
   fixture, binary, or tracker close is missing.

## Default Answer Shape

When answering from this skill, use this order unless the user asked for a
different format:

1. Boundary: "current source", "release binary", "dirty live-WIP", "Beads-open",
   or "historical note".
2. Action: the exact `focr` command, Rust API, file path, or verification gate.
3. Exclusions: the adjacent thing this does *not* prove, for example "not a
   GitHub release asset", "not full `bd-2mo.22` closure", "not SmolVLM2
   statics caching", or "not a formal PERF_LEDGER row".
4. Proof gap: one command or artifact the next agent should check if the user
   needs stronger certainty.

This keeps answers ergonomic: lead with what works, then attach the smallest
necessary caveat. Do not bury the usable command under the entire history.

## Fresh-Agent Reading Plan

Do not read this skill linearly. Use it like an operator console:

1. Read the Cold-Start Boundary Card and First 90 Seconds checklist.
2. Choose one surface from the Fresh-Agent Router: CLI, library, model/artifact,
   verification, performance, release, or troubleshooting.
3. Open exactly one matching reference first. Use `OPERATORS.md` when the
   question is about proof boundaries, state classification, or avoiding
   overclaims; use `CLI.md`, `LIBRARY.md`, or `ARTIFACTS-AND-ENV.md` for
   concrete commands and APIs.
4. Search within the skill for the exact bead, commit, env var, command, model
   id, or operator tag. If a global/installed copy of the focr skill disagrees
   with this project-local skill, use this project-local skill plus live
   `~/projects/franken_ocr` source/Beads as the fresher evidence.
5. Stop once you can state the boundary, action, exclusion, and proof gap. More
   history is not better unless the user asks for archaeology.

## Fresh-Agent Router

- Cold start if you know nothing: read this router, classify the exact surface
  with OP-LC (Live Contract Probe), check the model row in the Model Surface Map,
  then open only the matching reference. Do not read `RESEARCH.md` first; it is
  historical context after current source/help/Beads are settled.
- Asked "can focr do X?" Start with [CLI.md](references/CLI.md), then confirm
  source/help and Beads before answering yes/no.
- Editing or reviewing `franken_ocr` source? Use [DEVELOPMENT.md](references/DEVELOPMENT.md)
  and [VERIFICATION.md](references/VERIFICATION.md); read repo `AGENTS.md` and
  `README.md` first.
- Handling models, artifacts, env vars, or deployment? Use
  [ARTIFACTS-AND-ENV.md](references/ARTIFACTS-AND-ENV.md), then [FOCRQ.md](references/FOCRQ.md).
- SmolVLM2 work splits by evidence layer: OP-SC conversion/decoder census,
  OP-ST tokenizer, OP-SV vision/connector, OP-SP preprocess/prompt/route,
  OP-VQ Rust VQA guard and C10 CLI e2e script. Sub-epic C is now closed, but
  each claim still needs the right evidence layer. Source at/after `9b2a03b`
  has `smolvlm2::SmolStatics`, `OcrModel::smol_statics`, and
  `smolvlm2.hydrate(cached)` as committed pass-8 evidence: SigLIP tower,
  modality projection, and widened untied text embed hydrate once per model and
  are shared by sequential describe/VQA and batch paths. Keep it scoped to
  byte-identical describe source evidence plus lib-green proof. The parent
  `bd-av64.10` now has a formal closeout at `efd83e8`, but pass 8 by itself is
  not release readiness, public VQA quality, or a standalone PERF_LEDGER row.
- Model-zoo distribution work is OP-ZM: `ece14f9` / closed `bd-av64.7`
  publishes the embedded pull manifest for all five ready models, and
  `bd-av64.8` / `bd-av64.9` add the public GitHub release and clean-cache pull
  verification for `models-smolvlm2-v1`, `models-onechart-v1`, and
  `models-tromr-v1`. `focr pull smolvlm2`, `focr pull onechart`, and
  `focr pull tromr` are now current when the running binary embeds or resolves
  that manifest; clean-cache evidence says the GitHub release assets, hashes,
  sidecars, idempotent repull, and real inference paths work. Use
  `focr models --json` `pull.in_manifest` and `pull.quants`,
  `models/manifest.json`, `BUILTIN_MANIFEST_JSON`, `ModelEntry.sidecars`, and
  separate `br show ... --json` checks for `bd-av64.7`, `bd-av64.8`,
  `bd-av64.9`, and `bd-av64.12` before making pullability/quant claims. Keep
  the TrOMR boundary precise after `efccce9`: `models-tromr-v1` now publishes
  both `tromr.int8.focrq` and `tromr.focrq`; the default `focr pull tromr`
  resolves the `int8` storage artifact, while `focr pull tromr --quant f32`
  gets the bit-exact reference artifact. Do not turn that into an int8-compute,
  speed, or quality-improvement claim. Also keep the mirror
  boundary precise: production pulls are GitHub-first and verified; HF mirror
  spot-checks returned 401 and remain a resilience/auth follow-up, and the
  dedicated one-command pull-e2e script is still deferred.
- OneChart work is OP-OC: D1-D9 are now closed and `--task chart-data` is a
  current runtime route. Distribution is also source-current after
  `bd-av64.7`: `focr pull onechart` installs `onechart.int8.focrq`,
  `vocab.json`, `merges.txt`, and `added_tokens.json` into the OneChart model
  cache subdirectory. Source at/after `38ab806` / `a9a406e` has
  `onechart::OnechartStatics`, `OcrModel::onechart_statics`, and
  `onechart.hydrate(cached)` as committed pass-7 evidence: SAM tower,
  pre-transposed `mm_projector`, and widened OPT embed table hydrate once per
  model and are shared by sequential, batch, and number-head-tap paths. Keep it
  scoped to byte-identical chart-data source evidence plus full-lib/fmt/clippy
  proof. The parent `bd-av64.10` now has a formal closeout at `efd83e8`, but
  pass 7 by itself is not release readiness, broad chart quality, or a
  standalone PERF_LEDGER row. Do not invent a separate `focr chart` subcommand.
- TrOMR work is OP-TM: E2 conversion, E6 WordLevel music tokenizer, E3
  hybrid ResNetV2+ViT encoder, E4 four-head AR decoder, E7 semantic merge /
  MusicXML, E9 `focr ocr --task music --model tromr.int8.focrq` after the
  default pull or `--model tromr.focrq` after `focr pull tromr --quant f32`, E8
  single-staff parity/SER, E5 v1 staff detection/full-page runtime, E10
  coverage, and sub-epic E are closed in current `franken_ocr` `main`
  (`ab0bae0`). Distribution is current after `bd-av64.7` and `bd-av64.12`:
  default `focr pull tromr` installs `tromr.int8.focrq` plus
  `tokenizer_rhythm.json`,
  `tokenizer_pitch.json`, `tokenizer_lift.json`, and
  `tokenizer_note.json`; `focr pull tromr --quant f32` installs
  `tromr.focrq`. Keep the
  boundary precise: TrOMR supports single-staff and v1 printed/scanned
  full-page OMR to partwise MusicXML, and has a `bd-2sez` TrOMR f32
  PERF_LEDGER baseline row. That row is an honest losing baseline against
  pinned upstream torch
  (`vision_encode` ratio 0.308, `decode-per-token` 0.428, `end-to-end` 0.424),
  not a speed win. The TrOMR int8 artifact is weight-storage int8 only:
  `Weights::mat()` / `Weights::vec()` transparently dequant `QInt8PerChan`
  records and TrOMR compute remains f32; `qint8()` consumers are unchanged.
  Still claim no standalone `focr music` subcommand, no int8 compute proof, no
  TrOMR perf win or int8 perf row, no camera dewarp, and no `**kern` export.
  Barline splitting now exists only as the experimental `FOCR_TROMR_SPLIT=1` rescue
  path from closed `bd-av64.4`; it is off by default because the
  detection-lossless quality gate failed on isolated segments.
- TrOMR `bd-av64.2` is closed-current for both control-flow resilience and
  staff-level observability: `3da9dac` landed the resilience core, `8af3887`
  landed robot/JSON observability, and `4e881d7` closed the bead. Current source
  has `PageRecognition { staves, skips }`, `StaffSkip { index, bbox, reason }`,
  `recognize_page` skips failed staff crops instead of aborting the whole page
  when at least one staff succeeds, all-fail pages report every staff reason,
  `forward_tromr` prints a human skip note on stderr, and `FOCR_TIMING` can
  print per-staff dims/outcome. The observability surface is now also current:
  `MusicPageMeta`, `OcrModel::take_music_meta`,
  `OcrEngine::take_music_page_meta()`, `robot::staff_event`, robot event kind
  `staff`, and `music_meta_to_json` feed robot `staff` events plus detection-
  ordered `--json` / `-o .json` `staves` arrays for music runs. The emitted
  robot event is `staff`, not the older bead wording `staff_detection` /
  `staff_result`; its payload is `{staff, total, bbox, status, reason?}` with
  1-based staff numbers, `status: "ok"|"skipped"`, and skip reasons only when
  skipped. Schema stays v1 because adding the `staff` event kind is additive;
  do not invent a schema bump. Tests anchoring this are
  `staff_event_shapes_ok_and_skipped`,
  `music_meta_json_interleaves_in_detection_order`,
  `schema_advertises_all_events`, and the `scripts/tromr_music_e2e.sh` robot
  staff-event arm. The known caveat is PDF+music multi-page runs: the
  side-channel exposes only the last page's staves, while single-image music is
  the documented flow. Resilience warnings and staff metadata are not crop
  geometry evidence by themselves.
- TrOMR crop geometry has a committed source landing at `eb0c70e`, and
  `40ee875`/`91d552f`/`3ae26b1` now close `bd-av64.14` for the fit-first
  geometry lane and Cadwallader p169 5/5 recognized-staff acceptance. The
  committed mechanics are in `src/preprocess/staff_detect.rs`: `40ee875` makes
  the logic fit-first, so staff bands that already fit the 1280-column position
  budget keep the historic full-width geometry byte-for-byte. Only over-budget
  bands get horizontal ink-extent trim with line-spacing pad,
  neighbor-bounded vertical extend-to-fit, non-overlapping bands, and explicit
  skip/error handling for bands that still cannot fit. Focused tests are
  `fitting_bands_keep_the_classic_full_width_geometry`,
  `trim_cuts_page_margins_but_keeps_ink`,
  `wide_staff_with_room_fits_the_positional_budget`,
  `packed_staves_stop_at_the_midline`,
  `unpressured_band_keeps_the_generous_margins`,
  `tromr_page_skips_overwide_staff_and_keeps_the_rest`, and
  `tromr_page_all_staves_failing_is_a_named_error`. Keep the closure scoped:
  this is not camera dewarp, TrOMR int8, perf evidence, or broad note-level SER
  completion; those remain separate follow-up lanes despite `bd-av64.12` and
  `bd-av64.13` closing their scoped experiments.
- TrOMR barline splitting is current only as OP-TM experimental/gated evidence:
  `64edce3` / closed `bd-av64.4` adds `staff_detect::barline_columns`,
  `recognize_split`, and `FOCR_TROMR_SPLIT=1` for over-budget staff bands. The
  measured doctrine outcome is not "lossless quality shipped": isolated
  segments are out of distribution, continuation pitch registration can drift,
  rhythm agreement was about `0.2`, and pixel clef-prepend context measured
  worse. Use it as recognition-count rescue, for example p055 5/7 -> 7/7
  recognized when armed, not as a default route, camera dewarp, broad SER,
  int8 decoder-kernel evidence, or a perf win.
- TrOMR musical sanity and residual-skew work are current but scoped. `d51d7d9`
  closes `bd-av64.5`: `tromr::sanity_warnings` adds annotate-only warnings for
  overfull bars, underfull non-final bars, impossible durations, and cross-staff
  key mismatches. The surfaces are XML `<!--focr-sanity: ...-->` comments that
  strip cleanly without changing musical content, robot `music_warning` events,
  music-run JSON `warnings`, stderr count summaries, and `FOCR_TIMING` lines.
  This is telemetry and deterministic fallback, not auto-correction. `39651e6`
  lands `bd-av64.13` lever 1: `refine_band_skew()` sweeps each detected staff
  band over +/-1.5 degrees at 0.1-degree steps, engages only when the best angle
  is at least 0.2 degrees from flat, re-derives line centers, and abandons the
  refinement if the five-line group cannot be re-found. Straight bands stay
  byte-stable and the corpus gate stayed green, but the no21 double-dotted XFAIL
  did not flip. `69039c3` closes `bd-av64.13` by measuring and reverting the
  remaining levers: `FOCR_TROMR_TTA=3` micro-rotation voting regressed
  `no17_sys` at 2.8x cost because degraded scans dropped time signatures and
  inverted the bar-sum scorer, while one-crop routing through the refined band
  broke the committed golden by re-trimming a tight crop. Do not retry either
  without a held-out `bd-av64.15`-style corpus and a presence-first scorer.
- TrOMR int8 is current after `efccce9` / closed `bd-av64.12`, but only as a
  quantized-storage artifact. `src/quant/convert.rs` now marks exactly the 40
  Seq2SeqDense decoder GEMM suffixes (`to_{q,k,v}`, `to_out.0`,
  `net.0.proj`, `net.3`) as int8; encoder/embeddings/norms/heads stay high
  precision. `src/native_engine/weights.rs` lets f32 accessors dequant
  `QInt8PerChan` via `dequant_qint8()` so the existing f32 TrOMR forward can
  run `tromr.int8.focrq`. Evidence: committed golden byte-identical, 5/6
  real-scan corpus fixtures byte-identical MusicXML, corpus gate delta 0, one
  tier-2 no-truth p100 fork ledgered as `DISC-005`, clean-cache pull
  byte-exact, and pulled-artifact inference byte-matching the committed golden.
  Do not call this int8 compute, a speed win, or a broad quality improvement.
  Use `focr pull tromr --quant f32` / an existing `tromr.focrq` for bit-exact
  reference work; current `focr convert` has no f32 quant mode.
- Property/fuzz infrastructure is now closed-current after `f9f4c49` /
  `2dda846` / closed `bd-10sb.1`. It is verification/safety infrastructure, not
  a user-facing CLI feature. The committed surface is `proptest` dev-dependency
  plumbing,
  `tests/property_suite.rs`, `tests/support/proptest_support.rs`, and the
  `fuzz/` cargo-fuzz workspace with targets `focrq_parse`, `safetensors_parse`,
  `image_decode`, and `pretok_split`; `tokenizer::pretok` is public because it
  is an untrusted-Unicode fuzz surface. The property suite covers
  SIMD-vs-scalar bit identity, i32-vs-i64 accumulator agreement up to K=6848,
  preprocess geometry totality, `.focrq` parser totality under byte mutations,
  and model-gated byte-level-BPE round trip. The day-one fuzz finding was real:
  `image_decode` found a tiny PNG declaring huge dimensions and the fix bounds
  decoded images before allocation with the `pdf.rs` 1 Gpx policy, decoder
  limits, `decompression_bomb_png_is_rejected_before_allocation`, and the typed
  `InputDecode` exit-4 contract. CI/gate closure is now current but still
  scoped: `e80360b` unblocks the GitHub Actions provisioning failure caused by
  frankensqlite moving its asupersync dependency off the old `/dp/asupersync`
  path; `cc79d70` registers the frozen `tests/fixtures/multi_page` oracle
  directory plus narrows `check_release_linkage.py` byte scanning to linkage
  tokens such as `libpython`, `libtorch`, `torch_cpu`, `_PyObject`, and
  `Py_Initialize`; `c960b77` keeps golden-loop `.actual` diff aids out of the
  fixture-manifest gate; `29aa40a` adds always-uploaded gate-log artifacts and
  an advisory `bench-guardrail` job; `7777e34` fixes the aarch64 advisory
  SMMLA layout comparison by un-permuting panels before logical equality; and
  `2e5801b` / `18712cc` close `bd-4yks` with macOS+Ubuntu gate jobs green on
  the full `scripts/check.sh` battery, dist green, and the advisory matrix
  noted 6/6 after the SMMLA fix. Do not describe `bd-10sb.1`, `bd-4yks`, or
  those gate repairs as an exhaustive fuzz campaign, a scheduled full-model
  self-hosted run, ARM64 Windows completion, or all-green release-readiness:
  there is still no in-repo `TEST_LOG_DIR` capture layer, the 6.67GB-weight
  full-model scheduled job needs a self-hosted runner, native Windows x86_64 is
  now supported/proven end-to-end, ARM64 Windows remains `bd-3u97`, and CI/fuzz
  evidence still needs separate release-scorecard approval. Current OP-SG
  release approval is the `c29a78b` / `7c7bd00` / `29516b9` `v0.6.0`
  release-readiness evidence, not the CI gate itself.
- TrOMR real-scan quality work is OP-TM plus OP-VG, with a new current
  boundary after `bd-av64.6` / `af13d3e` / `3ae26b1`: corpus v1 is closed for
  the measuring device, with six public-domain 1843 Spohr fixtures under
  `tests/fixtures/realscan_music/` plus `scripts/realscan_music_gate.sh`. The
  gate emits `realscan_music/v1` NDJSON and uses three truth tiers:
  human-verified attributes in `truth/attributes.json`, frozen
  `goldens/*.musicxml` as regression anchors not truth, and robustness floors
  via robot `staff` events. It is model-gated skip-with-SUCCESS when TrOMR
  weights are absent, but XFAILs are never silent skips: an XPASS is a failure
  that must promote the fixture. Do not inflate this scoped closure into broad
  note-level SER completion, a full 10-20 item expansion, GOT cross-reference
  scorecard completion, or aggregate SER. Known current source refinements:
  `40ee875` promotes Spohr p055 and p100 from page-level XFAILs into
  truth-floor checks (`min_recognized` 5 and 1), and `91d552f` makes
  `scripts/realscan_music_gate.sh` read those full-page floors from
  `truth/attributes.json` instead of duplicated shell literals. Closed
  `bd-av64.4` adds experimental split rescue for some over-budget duet bands,
  but it is not broad SER closure or default quality. GOT auto-format
  misclassifies narrow staff strips as SMILES/molecules, so cross-reference full
  systems; documented `--json` music `semantic` output is still a doc/code gap
  unless source has since added it.
  The double-dotted system XFAIL remains after closed `bd-av64.13`; the
  measured voting and one-crop levers were negative/reverted, so future work
  belongs to held-out corpus/model-quality calibration, not to residual-skew
  refinement by default.
- Runtime/cancellation/thread-budget work is OP-OE/OP-RP current as of
  `59d376b` / closed `bd-223.2`: look for `request_shutdown`,
  `shutdown_requested`, `reset_shutdown`, `cancel_checkpoint`,
  `thread_budget`, `stream_pages`, `FOCR_THREADS`, `ctrlc`, robot `threads`,
  and checkpoint calls in decoder loops. If an installed binary lacks those
  surfaces, classify it as stale/pre-`bd-223.2` before changing docs.
- Run-state and sync work is OP-RS: current source after `03eadd2` / closed
  `bd-223.4` contains `src/storage.rs`, `pub mod storage`, `RunStore`,
  `RunRecord`, `FOCR_RUN_STORE`, live `focr runs`, and live
  `focr sync export-jsonl` / `import-jsonl`. Treat it as closed-current when
  source/help/tests agree; quarantine older binaries that still scaffold or omit
  it. `bd-wp8.11` later froze the run/sync one-way JSONL contract with
  `tests/fixtures/runs_schema.json`, populated-store/empty-history cases,
  stdout purity, and locked atomic sync.
- Doctor work is OP-DR: current source after `25eadc5` / closed `bd-wp8.4`
  implements `focr doctor` detect-only, `--fix`, `--dry-run`, `undo`,
  `capabilities --json`, `robot-docs`, and `--robot-triage` through pure
  detectors plus a single mutation chokepoint. `bd-wp8.4.1` pins the fixture
  suite. Do not call doctor scaffolded; do not run `--fix` casually when the
  user asked only for diagnostics.
- Robot triage and agent ergonomics are OP-RT: `055e513` / closed `bd-wp8.7`
  adds the one-round-trip `focr robot triage` JSON mega-command with
  `quick_ref`, `health`, `recommendations`, `commands`, and `exit_codes`.
  Prefer it for automation over several human/help probes.
- Release ship claims are OP-SG: `2bdccc5` adds
  `scripts/gauntlet_cert.py --release-readiness` and
  `docs/gauntlet/RELEASE_READINESS.json`; `8cacf52` adds `--bundle`; `9bc715e`
  makes `certification_bundle` a live `release_certificate.json` cell and keeps
  `--bundle` from certifying itself. The old red state (`ship:false`,
  `green:11`, `red:2`, `rounds=8/10, tail_clean=False`) is now historical.
  `c29a78b` is the ship-gate receipt and `29516b9` tags that certified state as
  the public `v0.6.0` release. Release readiness is `ship:true`, `green:13`,
  `red:0`; the bundle is certified; convergence is
  `rounds=11/10, tail_clean=True`; and `7c7bd00` closes `bd-wp8.8`,
  `bd-wp8.9`, and `bd-wp8.10` in Beads. `beaed7c` adds the CI/dist supplement,
  `db02421` refreshes README evidence, and `5df6395` commits the
  post-certification fuzz corpus. `592426c` refreshes public README release
  identity/asset-size/backend prose for `v0.6.0`. Keep this exact boundary: it
  proves the
  release certification bundle and Phase-5 ship gate, not arbitrary model
  quality, not open Phase-3/int4/ARM64-Windows epics, and not the user's
  installed binary without a binary/version check.
- PDF page selection, spread splitting, and PDF rotation normalization are
  OP-PS: `bd-av64.11` is closed after `11f60ea` / `9546571` / `5679268` /
  `b3f74b6`. Current committed source exposes `--pages` and `--split-spreads`
  for scanned PDFs, and the pure-Rust PDF renderer applies both page `/Rotate`
  and axis-aligned content-stream image-placement rotation via
  `content_rotation` before OCR/spread splitting. Treat page ranges as PDF-only,
  1-based comma/range syntax; split-spread output is heuristic left/right
  logical pages. `--split-spreads` does not compose with `--extract-figures`
  yet; source should return a clean usage error rather than guessing figure
  names across split halves.
- Multi-page cross-page parsing is OP-MP: `4afcaca` / `f115403` / `b9cc16c`
  close `bd-1gv.25` for the `infer_multi` core and
  `focr ocr-batch <images...> --multi-page`; `a2dd1c9` plus `750a69a` add the
  PDF half as `focr ocr doc.pdf --multi-page`. This is one Unlimited-OCR
  cross-page document pass, not a concatenation of independent page parses.
  Source uses reference-faithful 640x640 squash preprocessing
  (`preprocess_dynamic_squash`, PIL-bicubic hard-wired at this site), one
  111-placeholder visual block per page, a single cross-page prompt/decode with
  `ngram_window=1024`, and `<PAGE>` separators in the final markdown. PDF
  `--multi-page` composes with `--pages`, but not `--split-spreads` or
  `--extract-figures`; for image lists use `ocr-batch --multi-page`. `828ea4c`
  closes the rest of `bd-2z0y`: robot-mode PDF multi-page can now emit additive
  `page` events with `status:"decoded"`, `page`, `chars`, and raw `text` as
  `<PAGE>` boundaries are crossed during token streaming. `727701b` closes
  `bd-1gv.26` for the 2-page L5 multi oracle rung. `3201e8c` and `e1332a7`
  then close `bd-1465`: the 10-page rung
  `l5_multi_page_10p_long_horizon` is committed with fixture `p10`, subject cap
  7600 as a true-prefix comparison, plate byte-exactness, markers 8-vs-9, and
  CER 0.4045 <= 0.50; the uncapped subject terminates cleanly at the 32768
  position cap (31653 generated + 1115 prefill) where the bf16 oracle EOSes at
  7117. The 20-page `p20` oracle fixture is also frozen and shows the reference
  itself collapsing (7 `<PAGE>` markers for 20 pages plus repetition tail), so
  treat it as upper-bound/degradation evidence, not as a meaningful 40-page CER
  gate or proof of arbitrary long-document quality.
- Backend, SIMD, and performance claims are OP-BG/OP-GB territory. Use
  `robot backends` for selected/available CPU tier facts, `robot selftest` for
  selected-kernel parity, and `docs/PERF_LEDGER.md` / gauntlet evidence for
  speed. `ad3ad20` and the `adb4ee6` e2e prove `robot selftest.models`, a
  per-model rollup for `unlimited-ocr`, `got-ocr2`, `smolvlm2`, and `onechart`
  whose verdicts must match their underlying case rows. Closed
  `bd-3jo6.1.12`/sub-epic A evidence reports 44/44 cases green on scalar,
  sdot, and smmla, with model-specific overflow rows. TrOMR is intentionally
  absent because its published int8 artifact is storage-only and runtime
  dequants through f32 accessors rather than int8 decoder kernels. Never turn a
  selftest pass into a throughput claim.
- A11 zoo performance claims are OP-GB, not batch-spine or selftest evidence.
  The historical v0.4.0 README summarizes `docs/PERF_LEDGER.md`
  matched-thread Apple SDOT decode-per-token rows as GOT-OCR2 `3.37x`,
  OneChart `2.58x`, and SmolVLM2 `1.67x` over pinned Hugging Face CPU
  references. Those are decode-per-token ratios; full end-to-end rows are
  still kept, including slower totals from artifact load or preprocessing.
  Prefer the later `efd83e8` final rows for `bd-av64.10` closeout claims. Do
  not mix either family with the scoped dense-batch rows (`1.32x` SmolVLM2,
  `1.27x` OneChart, GOT +3-16%) or with the TrOMR f32 loss row.
- Release-evidence instruments are OP-SG/OP-CM/OP-VG: the scorecard comes from
  `scripts/gauntlet_cert.py --release-readiness`, the conformal ratchet from
  `docs/conformance/RATCHET.md`, the anytime-valid invariants from the Ville
  e-process `--eprocess-fold` / `--eprocess-state` path, and the capacity
  certificate from `capacity_certificate_bounded_stream_soak`. These are
  separate proof families. They do not override a red release-readiness cell,
  an open capstone bead, or a missing armed fixture directory such as
  `FOCR_FIXTURES_DIR`.
- Phase-2/P3 source status must not be rounded up: `bd-1es` is closed for the
  validated weight-only int8 decoder recipe, `bd-2mo.1` is closed for runtime
  ISA dispatch through `FOCR_FORCE_ARCH` and the bit-identical scalar oracle,
  and `bd-2mo.3` / `bd-2mo.3.1` are closed for offline arch-specific SMMLA
  prepacking. `focr convert --arch aarch64-smmla` now emits real SMMLA panels
  through `src/simd/pack.rs`; the loader preserves panels on SMMLA dispatch and
  un-permutes with a warning otherwise. This is a layout/correctness and
  zero-runtime-shuffle fact, not an M-series throughput win and not proof that
  packed-consuming x86 kernels exist for VNNI/AMX. The `fcb8289` tracker sweep
  closed
  13 SIMD kernel/dispatch/proof beads, but `bd-2mo` is still open and important
  levers remain open: the non-mmap remainder of memory/allocator reuse
  (`bd-2mo.22`), NUMA/USL pool sizing (`bd-2mo.21`), int8 attention, vectorized
  exp, and several fusions. `8de3674` adds `memmap2` to the committed `v0.5.1`
  Cargo dependency graph; the `v0.5.1` GitHub release now exists with
  platform binaries and SHA256 assets, but release publication is still
  separate from source work after the tag. `507cebe` ships the read-only mmap
  loader half:
  `Weights::load` defaults to `Backing::Mapped`, `FOCR_NO_MMAP=1` forces the
  owned-buffer fallback, mmap failures also fall back to owned bytes, and
  `mmap_load_is_byte_identical_to_owned_read` proves mapped-vs-owned tensor and
  directory identity. `0401df2` adds the Beads note that `bd-2mo.22` is still
  open for 64B alignment, decode-loop buffer reuse, and mimalloc measurement.
  Treat closed kernel parity and mmap-loader correctness as current capability
  evidence, not as proof every perf lever or P3 exit gate is done.
- Default fused-QKV decode is OP-BG/OP-GB territory too: `98cc790` /
  `5474ae0` close `bd-241s` by making the int8 decoder's fused
  q/k/v projection the default path. `FOCR_QKV_FUSED` is no longer an opt-in
  speed lever; it is a kill switch where `FOCR_QKV_FUSED=0` / `off` / `false`
  / `no` restores the older three-call projection path for parity/profiling.
  Source anchors are `qkv_fused_enabled`, `fuse_qkv`, `CachedLayerI8.qkv`, and
  `fused_qkv_gemv_is_byte_identical_to_three_calls`. Evidence anchors are
  closed `bd-1waa`, `bd-3pg7`, `bd-241s`, `docs/NEGATIVE_EVIDENCE.md`, the
  page_0590 SHA/CER checks, the 20-page CER equality, and the controlled
  page_0009 A/B row (`0.072 -> 0.052 s/tok`, best-of-3). Keep the boundary
  precise: this fuses decode q/k/v projection only; prefill still uses
  per-projection paths, and stale adjacent source comments must not override
  `qkv_fused_enabled()` or the close evidence.
- Ngram-lmhead fusion is a measured negative, not a likely next win:
  `bd-2mo.24` / `a0ad299` ledgered `FOCR_FUSE_NGRAM_LMHEAD` as
  correct-but-does-not-pay. The code stays harmless and opt-in behind
  `FOCR_FUSE_NGRAM_LMHEAD` with unit proof
  `fused_ngram_lmhead_is_byte_identical_to_separate_mask`, but the measured
  page_0023 A/B moved decode best only 16.43s -> 16.40s (0.2%, inside noise)
  with byte-identical outputs. Keep it off by default and do not retry unless
  the workload changes the arithmetic: multi-image `ngram_window=1024` ban sets
  or a roughly 10x faster decode step; rerun the same A/B first.
- Dense batched decode work is OP-BS/OP-GB territory: `bd-3jo6.1.7.5` is now
  closed/current after `8497080`, `cf0b037`, `4ca1577`, and `fdd1d64`, and it
  ships in the `v0.4.0` source boundary. Preserve the layering because it is how
  agents avoid false speed claims. `8497080` added the reusable dense pieces:
  `gemm_i8_bias_prequant_batched`, `BatchedQwen2KvCache`,
  `qwen2_batched_decode_step`, `DenseDecoderBatchStep`, and
  `generate_greedy_batched`. `cf0b037` first routed GOT-OCR2 `ocr-batch`
  through `OcrModel::recognize_batch` -> `recognize_batch_dense_got` ->
  `got::recognize_batch` under value-parsed `FOCR_BATCH_SPINE`.
  `4ca1577` broadens the dense zoo route through
  `OcrModel::recognize_batch_dense` for `got-ocr2`, `smolvlm2`, and `onechart`,
  adds per-model `recognize_batch` helpers, and makes
  `generate_greedy_batched(weights, cfg, inputs_embeds, caps: &[usize], eos)`
  preserve each page's generation budget through `PageStream::with_max_emit`.
  Dense zoo vision/splice/prefill stay model-specific per page; the active
  decode streams then share one continuous-batch greedy decode and per-page
  finalization stays the same as sequential recognition. The scheduler default
  from `src/native_engine/batch_scheduler.rs` is `DEFAULT_BATCH_SIZE = 128`,
  capped at `MAX_BATCH_SIZE = 256`; `FOCR_BATCH_PACK` groups similar prefill
  lengths and must restore input order; `FOCR_BATCH_VISION` is default-on inside
  the original batch spine and can be killed with `0`/`off`/`false`/`no`.
  Lossless proof is four-layered: bit-identical per stream f32::to_bits step
  gates for Qwen/Llama and OPT families, scheduler identity including mixed caps
  `[6,3,5]` and real mid-batch EOS retirement, real binary byte-identical
  markdown for GOT/SmolVLM2/OneChart, and the durable model-gated GOT e2e gate
  `recognize_batch_matches_sequential_e2e` with skip-with-SUCCESS NDJSON when
  unarmed. Honest throughput is scoped: SmolVLM2 decode 700 tok is 74.6s vs
  98.3s (1.32x), OneChart 438 tok is 2.39s vs 3.04s (1.27x), and GOT remains
  vision-dominated at roughly +3% to +16% on the cited fixtures. Still do not
  claim broad batched `lm_head`, a fairness-controlled A11/PERF_LEDGER row, or
  decode-heavy B>=8 throughput until follow-up evidence lands.
  Follow-on `d25dbd7` improves the GOT-OCR2 batch path itself:
  `got::recognize_batch` now hydrates `SamWeights`, `mm_projector_vary`, and
  `model.embed_tokens.weight` once per batch, then logs `got.hydrate(batch)`
  and `got.vision+splice(batch of N)`. The commit reports same-binary 3-page
  sequential 14.47s vs batch 13.53s (~6.5%) with
  `recognize_batch_matches_sequential_e2e` byte identity. Attribute that narrow
  hoist honestly; do not roll it into the larger SAM-attention win or into
  final A11 throughput. Current public source at `3f2878d` extends this into
  model-level `GotStatics`: the GOT SAM tower, projector, and widened embed
  table hydrate once on `OcrModel` and both sequential and batch page paths log
  `got.hydrate(cached)`. Treat that as committed pass-6 evidence; the parent
  `bd-av64.10` closeout is now separate formal G2 evidence at `efd83e8`, not
  something pass 6 alone proves.
- GOT e2e performance triage has moved, and `efd83e8` now closes `bd-av64.10`
  with formal G2 re-measurement rows. The earlier `bd-av64.10` measurement
  overturned the older artifact-hydration premise, `01f07fe` landed pass 1 of the
  bit-identical SAM attention optimization work, `f3d3215` landed pass 2 with
  head-parallel global attention across the 12 heads, `0298651` landed the
  committed CLIP-tower pass 3, `f65fded` / `3c1b1ea` make the broader
  shared-`Linear` pre-transpose battery pass-4/current evidence, and `f1ac972`
  lands SmolVLM2 SigLIP frame batching as pass 5. Public `origin/main` also
  contains the row-tile experiment and its reversal: `c5e535a` added
  row-tiled SAM global-attention score buffers, `8bd4037` restored the untiled
  baseline, and `b757bc0` ledgered the measured negative result in
  `docs/NEGATIVE_EVIDENCE.md` plus
  `artifacts/perf/bd-av64.10-rowtile/`. The tiled path was byte-identical on the
  cited real Unlimited-OCR and GOT-OCR2 fixtures, but slower in all four
  interleaved Apple-Silicon pairs because the cache-local score-buffer idea
  multiplied small GEMM dispatches. Treat the SAM vision lever list as
  exhausted unless a fresh target-specific profile proves a different hardware
  regime and the replacement avoids the dispatch explosion.
  With
  `FOCR_TIMING`,
  GOT sample text showed `sam.hydrate` around 0.03s and the forward gap
  dominated by SAM vision, especially `sam.block attn(GLOBAL)`,
  `sam.block attn(win)`, and MLP timing. Pass 1 parallelized the 25 window
  attentions, hoisted window rel-pos tables, removed per-element division from
  bias add, and made the QKV split contiguous-copy based. Pass 2 parallelizes
  global-attention heads across rayon disjoint output spans while preserving
  per-head arithmetic and byte identity. Cumulative evidence reports
  `attn(win)` 1.88s -> 0.72s, `attn(GLOBAL)` 2.10s -> 1.66s, `sam.forward`
  5.55s -> 4.24s -> 3.4-3.6s, GOT forward 6.7s -> 5.7s -> 4.6-4.8s, and
  unlimited-OCR real page 19.3s -> 13.5s (-30%), with byte-identical output,
  armed GOT certs 8/8, `vision_sam` 37/37, and `clippy -D` clean. Pass 3
  pre-transposes `vision_clip::LinearParams` into GEMM layout once
  at hydration, caches `ClipWeights` on `OcrModel`, removes 96 per-forward
  transposes (~1.2 GB data movement), and reports `vision.clip` 2.49s -> 0.77s
  steady-state on page_0009 with byte-identical output, armed L2 certs, 41/41
  `vision_clip`, and 957 full-lib tests. Pass 4 moves
  `vision_sam::Linear` to a validating `Linear::from_row_major` constructor
  that stores a GEMM-ready cached `[in,out]` `Mat` once at hydration and checks
  matrix metadata on `apply`; it removes repeated row-major transpose work
  across SAM block qkv/proj/mlp weights, GOT and OneChart projectors,
  SmolVLM2 modality projection, SigLIP linears, and TrOMR
  encoder/decoder/feed-forward/head linears. Treat that as committed
  self-relative optimization evidence, not final matched-reference
  A11/PERF_LEDGER closure. Pass 5 stacks SmolVLM2 SigLIP frames through
  `vision_siglip::forward_frames_batched`; `FOCR_SIGLIP_SEQ=1` forces the old
  per-frame loop, and `batched_frames_match_sequential_byte_for_byte` is the
  proof anchor. Treat the reported loaded-host SmolVLM2 vision+splice wins as
  pass evidence; the later closeout rows below are the formal G2 state. Do not list SIMD-exp
  softmax as a remaining easy path: `ab6e083` measured it dead and reverted it,
  and `50d5dad` records `artifacts/perf/bd-av64.10-simd-exp/`. Do not list
  row-tiled SAM attention either: `b757bc0` makes the negative/reverted result a
  public current-main fact, not just local checkout lore.
  `d25dbd7` first hoists GOT batch hydration once per batch; count that as the
  predecessor batch-path amortization win, not as closure of `bd-av64.10`'s
  wider e2e lane. `3f2878d` then lands pass 6: `got::GotStatics` hydrates the
  GOT SAM tower, `mm_projector_vary` projector, and widened `embed_tokens` table
  (~1 GB f32) once via an `OcrModel` `OnceLock`, routes both sequential and
  batch GOT page paths through that cache, and logs `got.hydrate(cached)`.
  Honest scope: measured about 0.8s/page saved on the sequential page loop
  (`got.vision+splice` 4.15 -> 3.31s/page on a 2-page batch, one 0.14s hydrate
  total), GOT sample output byte-identical, full lib 959 green, fmt/clippy/ubs
  clean, and the armed batch-vs-sequential e2e gate covers the contract. Treat
  pass 6 as committed source evidence, not as the formal closeout by itself.
  `38ab806` / `a9a406e` then land pass 7:
  `onechart::OnechartStatics` hydrates the OneChart SAM tower, `mm_projector`,
  and widened OPT embed table once on `OcrModel`; `onechart::recognize`,
  `recognize_batch`, `vision_features`, `build_inputs_embeds`, and the
  Number-head finish pass all reuse it. Evidence is chart-data output
  byte-identical to the pre-statics reference, full lib 960 green, fmt/clippy
  clean, and `onechart.hydrate(cached)` about 0.10s once per model. Treat pass 7
  as scoped committed source evidence, not as release readiness, broad chart
  quality, or the formal closeout by itself.
  `9b2a03b` then lands pass 8: `smolvlm2::SmolStatics` hydrates the SigLIP
  tower, `modality_projection`, and widened untied text embed once on
  `OcrModel`; `recognize`, `recognize_batch`, `vision_rows`, and
  `build_inputs_embeds` reuse it. Evidence is describe output byte-identical to
  the pre-statics reference, lib green, and Beads comment 91 reports
  `smolvlm2.hydrate(cached)` about 0.14s once per model. Treat pass 8 as scoped
  committed source evidence, not as release readiness, public VQA quality, or
  the formal closeout by itself. All four zoo lanes now hydrate model-constant
  tensors exactly once per process.
  `4291807` then certifies and flips the SmolVLM2 untied `lm_head`
  int8+top-K-refine path on by default. Evidence: paired describe outputs
  byte-identical f32-head vs int8+refine, armed L5 VQA green under the lever,
  full describe/VQA e2e suite pass, `lm_head` about `6.99 -> ~1.2 ms/tok`, and
  decode `40.4 -> 55-57 tok/s`; `FOCR_GOT_INT8_LMHEAD=0` remains the f32-head
  kill switch. Treat it as a scoped runtime/head certification, not a public VQA
  benchmark or a reason to store the SmolVLM2 `.focrq` `lm_head` as int8.
  `efd83e8` is the formal closeout: nine 2026-07-08 PERF_LEDGER rows under
  `artifacts/perf/bd-av64.10-g2r/`, hash-anchored by `SHA256SUMS`, at matched
  8 threads against the frozen 2026-07-05 references, with cv<=1.7%. Final
  end-to-end ratios improved but stayed below the original `>=1.0x` target:
  GOT `0.624 -> 0.885`, OneChart `0.546 -> 0.755`, and SmolVLM2
  `0.878 -> 0.890`. Decode-per-token remains positive and should be cited as
  the new final rows, not the older v0.4.0 summary: GOT `3.046x`, OneChart
  `2.249x`, SmolVLM2 `1.499x`. The honest close reason is "lever list
  exhausted with receipts": 8 landed bit-identical passes, 3 measured negatives
  (`SIMD exp`, `row tiling`, per-head batching absorbed by head-parallelism),
  and remaining e2e gap attributed to load-inclusion bias plus vision f32 GEMMs
  near kernel peak. Future speed work belongs in `bd-2mo` kernel epics or a new
  measured corpus bead, not in another artifact-load-tax retry.
  At/after `3f2f97e`, `scripts/gauntlet_timing.py` parses the cited SAM/CLIP
  drill-down labels; `ae7b8f2` records the bead-scoped evidence bundle; and
  `c248e6d` lands the `bd-2mo.26` head-to-head rows: page_0009 e2e `3.41x`
  (`vision 5.38x`, `prefill 4.62x`, `decode/tok 1.85x`, CER `0.00943`) and
  page_0014 e2e `2.81x` (`vision 5.62x`, `prefill 4.91x`, `decode/tok 1.92x`,
  CER `0.03529`) against pinned HF bf16. `c4c1684` closes `bd-2mo.26` with
  rounds 9-10. Use raw `[focr-timing]` stderr only for newer unknown stage
  names.
- Verification infrastructure claims are OP-VG: current `main` has closed
  `bd-zc1o` robot schema contract tests, `bd-n68o` structured test logging,
  `bd-29wv` model-gated skip-with-SUCCESS e2e discipline, and `bd-re8.7` L5
  OCR parity fixtures. Treat those as scoped gates, not proof that every
  corpus, model, or perf row is complete. Source at/after `adb4ee6` refreshes
  `tests/fixtures/robot_schema_v1.json` and the advertised-events assertion for
  additive TrOMR `staff` events; source at/after `0b74af0` also advertises
  additive schema-v1 `music_warning` in the frozen fixture and hard-coded event
  inventory. The older `bd-wp8.2.2` tracker item may still appear open or stale,
  so verify the focused schema tests before making a tracker-closure claim. Do
  not require a schema v2 for `staff` or `music_warning`, and do not let stale
  Beads wording override current fixture/source evidence.
- Determinism and fixture-provenance claims are OP-DG: committed `3e85c7d`
  closes `bd-3kge` and `bd-2pgf` with shared
  `assert_deterministic` / `assert_outputs_deterministic` helpers, real-model
  e2e adoption, `tests/fixtures/PROVENANCE.md`,
  `tests/fixtures/MANIFEST.toml`, and `scripts/check_fixture_manifest.py`
  wired into `scripts/check.sh`. Keep this separate from oracle
  nondeterminism-floor work.
- Conformance-accounting claims are OP-CM: committed `fb52843` (format follow-up
  `c685818`) closes `bd-re8.12` with `ConformanceTest`,
  `RequirementLevel`, `ConformanceCategory`, `conformance_registry()`, and
  `tests/conformance_matrix.rs`. The matrix enumerates `[SPEC-NNN]` clauses
  from `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`, requires MUST
  coverage >= 0.95, logs per-clause NDJSON, checks XFAIL sites against
  `DISC-NNN` or stated phase gaps, and runs registry entries in-process. Treat
  it as release/conformance accounting, not a universal OCR-quality proof.
- Differential, metamorphic, and golden-artifact claims are OP-DF/OP-MR/OP-GA:
  current `main` has closed `bd-re8.9`, `bd-re8.10`, and `bd-re8.11`.
  Differential compares a subject implementation against a bf16 oracle through
  per-op and L3-L5 ULP/tolerance rows; metamorphic checks oracle-free
  self-consistency under documented transforms and explicitly rejects the false
  multi-page concat/sum relation; golden tests
  freeze CLI/robot/schema/numeric artifacts with exact, fuzzy, scrubbed, or
  canonicalized comparison rules. Keep those three proof families separate from
  conformance accounting and from model-gated skip behavior.
- Ordered L0-L5 ladder scorecard claims are OP-LS: committed `1b84428` adds
  `scripts/ladder_scorecard.sh`, and `1112cf8` plus the `bd-re8.19` close
  evidence make the ladder runner tracker-closed. The runner folds
  `event=parity` rows and `event=result` outcomes from the serial
  `parity_ladder` integration test into one `focr-ladder-scorecard/v1` artifact
  with `gates`, `all_green`, `skipped_no_model`, `receipt`, and
  `not_meaningful` annotations above the first hard failure. Armed evidence on
  July 6, 2026 reported all six gates green; unarmed no-model runs still report
  `skipped_no_model=true` and `all_green=false`, so do not call skips green.
- Benchmark guardrail claims are OP-GB: `bd-1a6h` / `60d8af4` add
  `scripts/bench_guardrail.py` plus `benches/.bench-history/baseline.json`.
  The guardrail compares `gauntlet_focr.sh` stage records against frozen
  per-regime baselines, exits 1 on >10% regressions by default, logs one NDJSON
  row per stage, refuses perf reporting without an all-green L0-L5 parity
  receipt, treats `cv_pct > 5` and fairness-posture mismatches as ineligible
  rather than wins/losses, and skips-green when required fixtures, baselines, or
  receipts are absent. Baselines move only under explicit reviewed `--ratchet`,
  never automatically or in CI. Roofline floors remain `gauntlet_row.py`
  ledger columns; the guardrail is the frozen-baseline ratchet half.
- If output is for automation, prefer robot/JSON commands and keep stdout/stderr
  contracts explicit.

## Claim Taxonomy

| User claim | Use | Minimum honest answer |
|------------|-----|-----------------------|
| "Can focr do X?" | OP-LC + [CLI.md](references/CLI.md) | source/help/Beads classification plus exact command |
| "Can this model run?" | OP-MD/OP-MA + model operator | registry status, manifest packaging, artifact/tokenizer requirements |
| "Is output correct?" | OP-DF/OP-MR/OP-VG/model-specific proof | exact oracle, relation, fixture, or corpus boundary |
| "Is it conformant/release-ready?" | OP-CM plus OP-LS/OP-GA/OP-DG as needed | accounting result and every separate proof family still needed |
| "Is it deterministic?" | OP-DG, then OP-MR for transform consistency | same-input byte identity vs oracle quality kept separate |
| "Did the whole ladder pass?" | OP-LS | scorecard schema, `all_green`, `skipped_no_model`, and first-failure boundary |
| "Is it fast?" | OP-BG/OP-GB | architecture/selftest vs PERF_LEDGER/gauntlet rows, benchmark guardrails, and negative evidence kept separate |
| "Why does my binary differ?" | OP-SQ | exact binary path/version and release-lag/stale-local classification |

## Model Surface Map

Use this as the first-pass model router. For non-trivial claims, load the
matching operator/reference before answering.

- **Unlimited-OCR**: default `focr ocr` lane. Setup is `focr pull`. Do not
  claim generic arbitrary-checkpoint runtime.
- **GOT-OCR2**: `focr ocr --task formula|tables|chart|molecular|geometry|music`
  or `--format`. Setup is `focr pull got-ocr2`; keep `qwen.tiktoken` beside the
  artifact. Do not call it a default replacement for plain OCR.
- **SmolVLM2**: `focr ocr --task describe [--question ...]`. Setup is
  `focr pull smolvlm2`; it installs `smolvlm2.int8.focrq` plus
  `tokenizer.json`. Do not claim public VQA benchmark or human-label quality
  proof. Source at/after `f1ac972` routes `vision_rows` through
  `vision_siglip::forward_frames_batched` by default, with `FOCR_SIGLIP_SEQ=1`
  as the kill switch back to the per-frame loop; the current proof is
  byte-identical source/fixture evidence plus modest loaded-host timing wins,
  not a formal quiet-host PERF_LEDGER row.
- **OneChart**: `focr ocr --task chart-data`. Setup is `focr pull onechart`;
  it installs `onechart.int8.focrq` plus OPT tokenizer sidecars. Do not invent
  `focr chart` or broad chart quality.
- **TrOMR**: `focr ocr --task music --model tromr.int8.focrq` after the default
  pull, or `--model tromr.focrq` after `focr pull tromr --quant f32`, to
  partwise MusicXML. Setup is `focr pull tromr`; default pull installs
  `tromr.int8.focrq` plus four tokenizer tables, while
  `focr pull tromr --quant f32` installs the bit-exact `tromr.focrq`.
  Current evidence includes `bd-av64.2` per-staff
  skip resilience, robot `staff` events, music-run JSON `staves`, closed
  `bd-av64.6` Spohr real-scan corpus v1 measuring-device evidence, closed
  `bd-av64.14` fit-first geometry/p169 evidence, closed `bd-av64.5`
  annotate-only musical-sanity warnings, closed `bd-av64.13` residual-skew plus
  measured-negative/reverted TTA and one-crop levers, closed `bd-av64.12`
  TrOMR int8 storage publication, and the `bd-2sez` losing f32 PERF_LEDGER
  baseline. Do not claim `focr music`, int8 compute, a perf win,
  camera dewarp, default/lossless barline quality, `**kern`,
  `staff_detection`/`staff_result` event names, a schema-version bump for
  `staff` or `music_warning`, every-page PDF music `staves`, automatic
  correction from sanity warnings, or broad real-scan SER completion.
  `FOCR_TROMR_SPLIT=1` is an experimental over-budget-staff recognition-count
  rescue path only; it is not default behavior or a broad barline-quality proof.
  The int8 artifact is a published quantized-storage artifact, not a current
  int8-kernel runtime.

## Current Reality Snapshot

Public release/install facts and source facts can still differ, and the July 8,
2026 check now has a published `v0.6.0` release plus newer post-release source
commits. `f0a538b` created the `v0.4.0` public release boundary.
`bf28fd7` bumps `Cargo.toml` / `Cargo.lock` and the pushed annotated tag to
`v0.5.0`; the `v0.5.0` tag points at `bf28fd7`, and the GitHub release named
"cross-page parsing, hardened music lane, faster vision tower" was published
2026-07-07T23:49:41Z with binary plus `.sha256` assets for Apple Silicon,
macOS x86-64, Linux x86-64, Linux ARM64, and Windows x86-64. `8de3674` tags
the source/package as `v0.5.1`; the GitHub release named "v0.5.1 — the
vision-pipeline efficiency patch" was published 2026-07-08T05:02:37Z with the
same five platform binary families plus `.sha256` assets. `4cedacd` tags the
source/package as `v0.5.2`; the GitHub release named "v0.5.2 — the cold-start
release" was published 2026-07-08T06:05:23Z with the same five platform binary
families plus `.sha256` assets. `29516b9` tags the source/package as `v0.6.0`;
the GitHub release named "v0.6.0 — certified faster than PyTorch, end to end"
was published 2026-07-08T14:47:48Z with the same five platform binary families
plus `.sha256` assets. `install.sh` / `install.ps1` still keep
`v0.4.0` as the fallback constant for failed release lookup, and installed
binaries can still lag; normal online installers resolve the latest release
object first. The `592426c` README has been refreshed for `v0.6.0`, but the
standing support rule remains: when README/manual prose and the release API
disagree, verify the actual release object/assets before making support claims.
The July 8, 2026 source probe saw public `origin/main` at `592426c`
(`v0.6.0-4-g592426c` for clean source; the inspected checkout had no tracked
source diff). The post-`v0.5.2` commits are `0924479`, a README-only
clarification, `efd83e8`, the formal `bd-av64.10` G2 re-measurement / Beads
closeout, `3f2f97e`, the `bd-2mo.26` gauntlet harness bring-up fix, `4291807`
SmolVLM2 untied `lm_head` int8+refine certification, `91c12fd` CPU selftest
README refresh, `ae7b8f2` perf evidence bundle, `c248e6d` head-to-head
PERF_LEDGER rows, `c4c1684` `bd-2mo.26` plus rounds 9-10 closeout, `c29a78b`
release certification achieved, `7c7bd00` Beads closure for the release
certification trio, `29516b9` `v0.6.0` tag/version bump, `beaed7c` CI/dist
supplement notes, `db02421` README release-readiness evidence refresh,
`5df6395` committed post-certification fuzz corpus growth, and `592426c`
README release-identity refresh. Treat
`c960b77`, `29aa40a`, `7777e34`,
`2e5801b`, `18712cc`, `ab6e083`, `50d5dad`, `ab6fa6c`, `3f3d9d0`, `8cacf52`,
`ff22366`, `f1ac972`, `2665750`, `48a9896`, `c5e535a`, `8bd4037`, `b757bc0`,
`9bc715e`, pass-6 `3f2878d`, source version bump `8de3674`, mmap-loader half
`507cebe`, Beads note `0401df2`, README/release-readiness commits
`a391793`/`c8682a3`, pass-7 OneChart statics `38ab806`/`a9a406e`, pass-8
SmolVLM2 statics `9b2a03b`, README docs `8d6601d`, and source/package tag
`4cedacd`, README clarification `0924479`, G2 closeout `efd83e8`, and harness
fixes `3f2f97e`, SmolVLM2 head certification `4291807`, selftest README refresh
`91c12fd`, perf evidence bundle `ae7b8f2`, head-to-head rows `c248e6d`,
closeout `c4c1684`, release certification `c29a78b`, Beads closeout
`7c7bd00`, `v0.6.0` tag `29516b9`, CI/dist supplement `beaed7c`, README
release-readiness refresh `db02421`, fuzz corpus commit `5df6395`, and README
release-identity refresh `592426c` as current committed source evidence unless
a newer release asset, installed binary, or live checkout proves otherwise. If
the checkout is dirty after `592426c`, classify only those uncommitted diffs as
live-WIP;
uncommitted docs are not stronger evidence than committed source, tests, and
Beads.
Always verify the exact installed `focr --version`, help, tag, GitHub release
object/assets, and installer path before support claims; stale README/manual
prose is not stronger evidence than the release API or the installed binary.
The `v0.4.0` public
source/release boundary includes `ocr <image-or-pdf>`, `-o/--output`, JSON
layout boxes, `--extract-figures`, `ocr-batch`, `pull`, int8 `convert`, robot
commands, scanned PDFs via `PdfPages`, and layout APIs.
Post-`v0.4.0` additions covered by this skill include
multi-page cross-page parsing (`ocr-batch --multi-page` and PDF
`ocr --multi-page`), multi-page streaming `page` events, the 2-page and
10-page L5 multi-oracle rungs plus the frozen 20-page reference-collapse
fixture, additive TrOMR `staff` robot/JSON observability, fit-first
TrOMR staff crop geometry, experimental `FOCR_TROMR_SPLIT=1` barline rescue,
annotate-only TrOMR `music_warning` / JSON `warnings` sanity telemetry,
per-band TrOMR residual-skew refinement,
model-aware `robot selftest` rollups, and offline `aarch64-smmla` prepacked
`.focrq` layouts, default TrOMR `tromr.int8.focrq` storage publication with
`tromr.focrq` as the f32 reference artifact, and the closed `bd-av64.13`
negative-evidence ledger for TTA/one-crop routing. Treat those as
release-or-current-source claims only after the exact binary/help, tag, release
assets, and Beads closure agree.
Closed-current verification additions after the `v0.4.0` boundary also include
`bd-10sb.1` property/fuzz plumbing, the decode decompression-bomb guard, and
`bd-av64.10` pass-4 shared SAM/SigLIP/TrOMR/projector pre-transposed-linear
evidence (`f65fded` plus Beads evidence in `3c1b1ea`), pass-5 SmolVLM2 SigLIP
frame batching (`f1ac972`), pass-6 GOT statics caching (`3f2878d`), pass-7
OneChart statics caching (`38ab806` / `a9a406e`), pass-8 SmolVLM2 statics
caching (`9b2a03b`), README docs `8d6601d`, source/package tag `4cedacd`,
README split-clarification `0924479`, formal `bd-av64.10` G2 closeout
`efd83e8`, `bd-2mo.26` gauntlet harness hardening `3f2f97e`, and
SmolVLM2 untied-head certification `4291807`, `bd-2mo.26` head-to-head rows
`c248e6d`, release certification `c29a78b`, `7c7bd00` Beads closeout,
`v0.6.0` tag `29516b9`, CI/dist supplement `beaed7c`, README release-readiness
refresh `db02421`, fuzz corpus commit `5df6395`, README release-identity
refresh `592426c`, and source-current `bd-4yks`
CI/gate closure.
`e80360b` / `cc79d70` are in the `v0.5.0` ancestry; post-tag follow-ons now
extend through public `origin/main` `592426c` in this checkout. The
important new boundaries are:

- `ab6e083` / `50d5dad`: SIMD/polynomial-exp softmax in SAM attention measured
  dead and was reverted. Unit-level exp/softmax accuracy did not preserve token
  output; use the negative-evidence ledger rather than retrying it as an easy
  speed lever.
- `ab6fa6c` / `3f3d9d0`: round 8 is CLEAN after 4 fuzz targets x 5 minutes
  (4.7M total runs, zero crashes), `PROPTEST_CASES=2048` 8/8 green, and a 6/6
  advisory matrix, but this is bounded deep evidence, not exhaustive fuzzing or
  release readiness.
- `5df6395`: the post-certification fuzz corpus is committed, not dirty local
  WIP. It adds 3,271 seed files across `focrq_parse`, `image_decode`,
  `pretok_split`, and `safetensors_parse`. The later convergence sweep records
  four fuzz targets x four minutes, about 3.65M total zero-crash runs, plus
  `PROPTEST_CASES=2048` 8/8 green. Keep the old round-8 4.7M sweep as bounded
  historical evidence and the round-11 3.65M sweep plus committed corpus as the
  current convergence/corpus evidence.
- `592426c`: README-only current-source refresh for the public `v0.6.0`
  identity. It replaces stale badge/install/manual-download references,
  documents that source tag and binary assets are both `v0.6.0`, updates the
  binary-size claim from the old 5 MB-era language to about 13-17 MB, and
  clarifies CPU backend wording: Apple Silicon prefers SDOT, non-Apple ARM64
  can dispatch SMMLA/i8mm, and Intel/AMD select AVX-512-VNNI, AVX-VNNI, AVX2,
  or scalar by runtime feature detection. This is README/public-support
  evidence, not a new runtime feature.
- `8cacf52` / `ff22366` / `2665750` / `9bc715e`: these are now historical
  bundle bring-up commits. They made `scripts/gauntlet_cert.py --bundle` real
  and converted `certification_bundle` into a live `release_certificate.json`
  cell rather than a hard-coded red/self-referential predicate. The old
  certified-false state is superseded by the later `c29a78b`/`7c7bd00`
  certification closure; keep the bring-up details only when debugging how the
  bundle gate works.
- `48a9896`: README and fuzz metadata now name `v0.5.0`; installer fallback
  constants and installed binaries are still separate checks.
- `f1ac972`: SmolVLM2 SigLIP frame batching is current source. It proves
  byte-identical `forward_frames_batched` behavior with `FOCR_SIGLIP_SEQ=1` as a
  sequential kill switch and loaded-host self-relative wins. The later
  `efd83e8` G2 rows are the formal closeout state; this individual pass remains
  pass evidence, not a standalone release-readiness or quality claim.
- `c5e535a` / `8bd4037` / `b757bc0`: row-tiled SAM global-attention score
  buffers were a public experiment, then a public baseline restore and negative
  evidence ledger entry. Tiling was byte-identical but slower on Apple Silicon
  because many small GEMM dispatches outweighed cache locality. Do not retry row
  tiling as the next obvious win unless a fresh profile on a different hardware
  regime shows the score matrix is actually DRAM-bound and the replacement uses
  a fused blocked kernel rather than a loop of small matmul calls.
- `3f2878d`: GOT pass 6 makes `GotStatics` public current-main source rather
  than checkout-only evidence. The model-level `OnceLock` caches the GOT SAM
  tower, projector, and widened embed table for both sequential and batch paths.
  Keep its claim scoped to the committed evidence above; the formal G2 closeout
  is the later `efd83e8` row bundle, not the pass-6 commit by itself.
- `8de3674` / `4cedacd` / historical release `v0.5.2`: source/package tags
  `v0.5.1` and `v0.5.2` are committed; `8de3674` bumps Cargo to `0.5.1` and
  adds `memmap2`, while `4cedacd` bumps/tagged source/package `v0.5.2`. The
  `v0.5.2` GitHub release object and platform assets are real, published
  2026-07-08T06:05:23Z, but no longer latest after public `v0.6.0`. Keep
  release publication separate from post-tag source commits and from the
  installed binary on a user's PATH.
- `507cebe` / `0401df2`: the mmap half of `bd-2mo.22` is now current committed
  source. The default loader uses read-only mmap, `FOCR_NO_MMAP=1` and mapping
  failures force the owned-buffer fallback, and the byte-identity/is-mapped test
  proves mapped-vs-owned parser equivalence. Do not close or overclaim the full
  bead: 64B scratch alignment, decode-loop buffer reuse, and mimalloc
  measurement remain open.
- `38ab806` / `a9a406e`: OneChart pass 7 is now committed source. `OnechartStatics`
  owns the hydrated SAM tower, pre-transposed `mm_projector`, and widened OPT
  embed table; `OcrModel::onechart_statics` caches it with a `OnceLock`; and
  sequential, dense batch, and number-head finish paths use it. Evidence:
  chart-data output byte-identical to pre-statics reference, full lib 960,
  fmt/clippy clean, and Beads comment 90 reports `onechart.hydrate(cached)`
  about 0.10s once per model. Do not turn this individual pass into broad
  OneChart quality, release readiness, or a standalone G2/PERF_LEDGER row.
- `9b2a03b`: SmolVLM2 pass 8 is now committed source. `SmolStatics` owns the
  SigLIP tower, `modality_projection`, and widened untied text embed;
  `OcrModel::smol_statics` caches it with a `OnceLock`; and sequential describe,
  VQA, and batch paths reuse it. Evidence: describe output byte-identical to
  pre-statics, lib green, and Beads comment 91 reports `smolvlm2.hydrate(cached)`
  about 0.14s once per model. Do not turn this individual pass into public VQA
  quality, release readiness, or a standalone G2/PERF_LEDGER row.
- `4291807`: the SmolVLM2 untied `lm_head` int8+top-K-refine path is now
  certified and default-on. Evidence: paired describe outputs byte-identical
  f32-head vs int8+refine, armed L5 VQA green under the lever, full
  describe/VQA e2e suite pass, `lm_head` about `6.99 -> ~1.2 ms/tok`, decode
  `40.4 -> 55-57 tok/s`, default-path output re-proven byte-identical, and
  full lib 960 green. `FOCR_GOT_INT8_LMHEAD=0` is the f32-head kill switch.
  Keep conversion/storage separate: the `.focrq` still records the untied head
  as F32; the default-on lever is a runtime head path.
- `0924479` / `efd83e8`: `0924479` is README-only source-vs-binary
  clarification after the `v0.5.2` tag. `efd83e8` is the first post-release
  source commit that changes the project evidence surface: it lands the formal
  `bd-av64.10` G2 rows and closes the bead with the measured final state
  described above. It is current source evidence, not part of the `v0.5.2`
  release asset.
- `3f2f97e`: committed `bd-2mo.26` gauntlet harness bring-up, not a new model
  feature. It skips AppleDouble `._*.rs` junk in `check_ledgers.py`, teaches
  `scripts/gauntlet_timing.py` the new SAM/CLIP timing vocabulary
  (`sam.hydrate`, `sam.blocks`, `sam.forward`, `sam.block ...`,
  `clip.hydrate(cached)`, `clip.blocks`), and lets `scripts/gauntlet_runbook.sh`
  write to a fresh evidence home with `OUT_DIR`. It is now harness bring-up
  only; `c248e6d` and `c4c1684` are the later perf-row and closeout commits.
- `ae7b8f2` / `c248e6d` / `c4c1684`: `bd-2mo.26` is closed. Evidence bundles
  live under `artifacts/perf/bd-2mo.26/` and
  `artifacts/perf/bd-re8.17/G2-*-20260708/`; page_0009 records e2e `3.41x`
  (`vision 5.38x`, `prefill 4.62x`, `decode/tok 1.85x`, CER `0.00943`), and
  page_0014 records e2e `2.81x` (`vision 5.62x`, `prefill 4.91x`,
  `decode/tok 1.92x`, CER `0.03529`) against pinned HF bf16 with best-of-5
  warm runs, N=8 both sides, stdout identical on both sides, and cv% under 5.
- `c29a78b` / `7c7bd00` / `29516b9`: release certification is now current and
  publicly tagged as `v0.6.0`. The readiness
  scorecard reports `ship:true`, `green:13`, `red:0`; the bundle certificate is
  `certified:true`; convergence is `rounds=11/10, tail_clean=True`; and Beads
  closes `bd-wp8.8`, `bd-wp8.9`, and `bd-wp8.10`. `beaed7c` supplies the
  CI/dist supplement and `db02421` refreshes README evidence. Do not inflate
  this into closure of parent `bd-wp8`, parent `bd-2mo`, int4 (`bd-3gaa`),
  ARM64 Windows (`bd-3u97`), or proof that installed release binaries match a
  user's local PATH without a version check. Native Windows x86_64 is supported;
  only native ARM64 Windows hardware/package proof remains with `bd-3u97`.

Dirty-worktree-only surfaces to quarantine, not promote, now mean new
uncommitted deltas after public `592426c`, not the
already-landed `Linear::from_row_major`, `--bundle`, `forward_frames_batched`,
row-tile-negative-evidence/baseline-restore, `GotStatics`, or mmap-loader facts
or OneChart/SmolVLM2 statics facts, the `v0.5.2` source tag, or the README
source/binary split, G2 closeout, gauntlet harness fixes, SmolVLM2 head
certification, head-to-head perf rows, or release certification described
above. In the inspected checkout, committed fuzz corpus growth belongs to
`5df6395` and the current README release-identity refresh belongs to `592426c`;
only new tracked diffs or untracked files after that boundary should be treated
as live-WIP.
`focr pull [MODEL] --manifest <path-or-url>` resolves manifest source as
explicit `--manifest`, then `FOCR_MANIFEST_URL`, then the built-in repo
manifest (`BUILTIN_MANIFEST_JSON = include_str!("../models/manifest.json")`).
Closed `bd-av64.7` / commit `ece14f9` makes that manifest the distribution
source for all five ready models: the top-level default `unlimited-ocr`
artifact, and named `models.got-ocr2`, `models.smolvlm2`, `models.onechart`,
and `models.tromr` entries. Non-primary models install under
`~/.cache/franken_ocr/models/<model-id>/`, with `ModelEntry.sidecars` carrying
tokenizers and auxiliary files. `focr models --json` exposes each row's `pull`
object with `in_manifest` and `quants`; use that instead of guessing from the
registry row. Current `main` also has `focr models`, `.focrq` `model_id`,
implemented GOT-OCR2 (`focr pull got-ocr2`, `got-ocr2.int8.focrq`), implemented
SmolVLM2 (`focr pull smolvlm2`, `smolvlm2.int8.focrq`), implemented OneChart
(`focr pull onechart`, `onechart.int8.focrq`), implemented TrOMR
(`focr pull tromr`, default `tromr.int8.focrq` plus
`focr pull tromr --quant f32` for `tromr.focrq` and tokenizer sidecars), GOT
`--format` / `FOCR_GOT_FORMAT`, and `focr ocr --task` routing for
`formula|tables|chart|molecular|geometry|music|describe|chart-data`. GOT tasks
imply GOT format mode and need `--model got-ocr2.int8.focrq`; `describe` routes
through SmolVLM2 and needs `smolvlm2.int8.focrq`, plus optional `--question`;
`chart-data` routes through OneChart and needs `onechart.int8.focrq` plus the
OPT tokenizer files beside it; `music` is dual-lane: with `got-ocr2` it is GOT
sheet-music format mode, and with `tromr` it is native Polyphonic-TrOMR OMR to
partwise MusicXML. `focr pull tromr` with no explicit quant now selects the
published `int8` storage artifact; use `focr pull tromr --quant f32` or a
preverified local `tromr.focrq` when byte-exact reference weights matter, and
do not turn either artifact into an int8 compute claim.

Closed `bd-av64.8` and `bd-av64.9` move the zoo story beyond "manifest
prepared": GitHub releases `models-smolvlm2-v1`, `models-onechart-v1`, and
`models-tromr-v1` were published and then clean-cache verified. Evidence covers
exact release sizes/hashes, sidecar sets, per-model cache subdirectories,
idempotent repull, then-current TrOMR f32 fallback behavior now superseded by
the `bd-av64.12` default storage-int8 artifact, and real inference for TrOMR
MusicXML, SmolVLM2 description, and OneChart chart-data. The known
distribution gap is HF mirror auth/resilience: spot-checks returned 401 across
the weights mirrors, while GitHub-first production pulls work. Do not describe
HF mirroring as verified, and do not claim the deferred one-command pull-e2e
script exists until source/Beads show it.

Latest `main` also wires preprocessing controls into the engine:
`--base-size`, `--image-size`, and `--crop-mode` now reach
`PreprocessOverrides`; default crop mode is `base`. `--crop-mode gundam` is live
reference dynamic tiling, with first e2e evidence in `bd-1e9n`
(`rc=0`, 7 views, CER 0.0179 / WER 0.0138 on page_0107), but do not round that
up to a full parity sweep. SmolVLM2 conversion is now implemented and
model-gated-proven on real weights (`bd-3jo6.3.2`): `focr convert
--model-id smolvlm2` produced 489 tensors = 224 int8 decoder GEMMs + 265 F32
high-precision tensors with an untied high-precision `lm_head`; C5 then
certified the text-only SmolVLM2 decoder seam (`bd-3jo6.3.5`: f32 cos
1.000000 + 24-token L4 exact, int8 cos 0.998301 argmax-exact, DISC-002 near-tie
flip recorded). C6 tokenizer conformance is also closed (`bd-3jo6.3.6`):
`PretokScheme::SmolLm2` is selected from tokenizer JSON, the GPT-2
ByteLevel/Digits pretokenizer path is id-exact, and the real-tokenizer gate is
128/128 token-id and decode exact against pinned HF tokenizers. Current `main`
has C3/C4/A8/A9 closed for SmolVLM2 vision and connector seams: SigLIP post-LN
parity on 13 real frames, NaViT bucketized position ids, bit-exact
`pixel_shuffle`, and connector projection within measured tolerance.
`bd-3jo6.3.7` and `bd-3jo6.3.9` are now closed: `preprocess_smolvlm2` uses
Pillow-bit-exact LANCZOS (`resample: 1`) with longest-side 2048, row-major
512-frame tiling plus a final global 512 frame; `src/native_engine/smolvlm2.rs`
assembles prompt ids, vision rows, `<image>` splices, and SmolLM2 KV-cache
greedy decode; `--question` / `FOCR_SMOLVLM2_QUESTION` set the VQA question;
`model_arch` reports SmolVLM2 `implemented=true`; and the int8 route was proven
live with "What is in the sky..." -> "There is a sun in the sky."

SmolVLM2 sub-epic C is now closed in live Beads/source. `bd-3jo6.3.8` closed
the C8 parity/e2e quality/perf gate with: L0b preprocess exact, L0c prompt
876/876 id-exact, L2 SigLIP cos 1.0, L3 prefill drift `<5e-5` and
argmax-exact, L4 opt-in `FOCR_SMOLVLM2_CERT_FULL=1` O(n^2) re-prefill greedy
64/64 id-exact, ledgered DISC-003 KV-cache near-tie behavior, and L5
informational VQA 7/7 on both f32 and int8 with int8 answers identical to f32.
Measured release-int8 informational times on Apple M4 were describe e2e 51.8s
and VQA 13.0s; use A11/PERF_LEDGER rows for fairness-controlled performance
claims. `bd-3jo6.3.10` closed C10 with per-module tests plus
`scripts/smolvlm2_describe_e2e.sh`; `bd-3jo6.3` closed the sub-epic. So support
`focr ocr --task describe --model smolvlm2.int8.focrq [--question "..."]`, and
support `focr pull smolvlm2` after closed `bd-av64.7` when the running binary
embeds or resolves the updated manifest. Still avoid claiming public benchmark
quality or concurrency-safe per-request question handling unless
source/evidence specifically says so.
Latest C8 VQA work adds an informational L5 guard with exact evidence:
`scripts/gen_smolvlm2_vqa_fixtures.py` writes
`tests/fixtures/smolvlm2/vqa_fixtures.json` from torch-oracle greedy answers on
the committed sample photo, and `vqa_quality_matches_oracle_l5` scores focr's
answers by normalized exact match or symmetric content-word containment >=0.5.
The guard fails below 70% f32 or 50% int8 matches when `FOCR_SMOLVLM2_DIR` has
`model.safetensors` and/or `smolvlm2.int8.focrq`; live close evidence reports
7/7 on both legs. Treat skips as missing artifacts, not green evidence.
Current source also contains `scripts/smolvlm2_describe_e2e.sh`, a C10
CLI gate that emits `smolvlm2_describe_e2e/v1` NDJSON, checks missing-model and
wrong-family negative paths, then runs describe and VQA through the real int8
artifact. Treat it as model-gated e2e evidence and cite C10 closure only when
live Beads/source agree.
If `src/cli.rs` still has an older doc comment saying `OcrTask::Describe` is
planned, treat it as stale inline prose; the usage guards, `forward_smolvlm2`
dispatch, Beads, README examples, and DISC-003 are the current route evidence.
OneChart is now a ready runtime and pull route for chart-to-data extraction
when the running binary embeds or resolves the `bd-av64.7` manifest.
`bd-3jo6.4.2` closed the D2 arch-aware conversion half: `focr convert
--model-id onechart` classifies OPT decoder GEMMs under
`model.decoder.layers.`, verifies the tied `lm_head` against
`model.decoder.embed_tokens.weight`, dedups the tied head, writes
`model_id=onechart` plus the Apache-2.0 notice, converts the real 500 MB
checkpoint from 384 source records to 383 `.focrq` records, produces 72 int8
GEMMs (12 layers x q/k/v/out/fc1/fc2), keeps vision/projector/number-head/norms
high precision, and records a 346 MB artifact. `bd-av64.7` / `ece14f9` later
publishes that artifact through the committed pull manifest, so `focr pull
onechart` is current when the running binary embeds or resolves that manifest.
`bd-3jo6.4.9`
closed D9 tokenizer conformance: `PretokScheme::Gpt2` (plain GPT-2 regex, no
Digits stage), `Tokenizer::from_opt_dir/from_opt_files`, `vocab.json` +
`merges.txt` + `added_tokens.json`, base control tokens `<s>/<pad>/</s>/<unk>`
as splittable specials, pinned ids `<imgpad>` 50265 / `<img>` 50266 /
`</img>` 50267 / `<Number>` 50268, bos=eos 2, pad 1, and 29/29 token-id-exact
against the slow HF `GPT2Tokenizer`. `bd-3jo6.4.3` closed D3
vision/projector certification: `preprocess::onechart_view_tensor` builds the
OneChart image tensor by squash-resizing to 1024x1024 and using raw RGB
`[0,1]` pixels, with no CLIP mean/std normalization; `FOCR_RESAMPLE=pil-bicubic`
is the exactness comparison knob. `src/native_engine/onechart.rs::vision_features`
runs the certified SAM-ViT-B tower at `model.vision_tower` and the
`model.mm_projector` `Linear(1024->768,bias)` projection to `[256,768]`.
`scripts/gen_reference_fixtures_onechart.py` is now the committed oracle
fixture generator; it writes or references `onechart_preproc.bin`,
`onechart_proj_out.bin`, `onechart_final_logits.bin`, and
`tests/fixtures/onechart/oracle_fixtures.json`. The D3 armed close evidence
reports `proj_out cos 1.00000000` and maxabs `6.5e-4`. The live fixture's
`prompt_n` is 308; older prose may say 309, but fixture/source evidence wins.
`20ac599` then landed the D4 prefill half:
`DecoderFamily::Opt` and `DecoderConfig::onechart()` now express the OPT-125M
prefill path with pre-LN `LayerNorm` plus bias, learned absolute positions
offset 2, no RoPE, biased q/k/v/out/fc1/fc2 linears, plain ReLU `fc1`/`fc2`,
model-level `LayerNorm` plus tied head, and q scaling kept inside the shared
attention kernel. `src/native_engine/onechart.rs::build_inputs_embeds` embeds
the prompt, then scatters the D3 `[256,768]` projector rows into the 256
`<imgpad>` 50265 slots. The armed D4-prefill cert uses oracle projector rows
and `onechart_final_logits.bin`; it reports last-position argmax 50268
(`<Number>`), cos `1.00000000`, maxabs `6.1e-5`, and prompt length 308.
`2c77d21` added D4 cached decode support, and `2769d21` closed
`bd-3jo6.4.4`: `generate_greedy_kvcache` now covers the OneChart OPT family
path through the shared dense decoder, including `MlpW::ReluFc`, `family_norm`,
OPT layer norms and biases, learned absolute positions in seed and step, no
RoPE, OPT output-proj bias, final norm bias, centralized `lm_head`, and the
D4-decode oracle test `opt_kvcache_matches_greedy_and_oracle`. That test starts
from the same oracle-vision embeds as D4 prefill, prefers
`onechart.int8.focrq` when present so the B9 leg compares same-quantization
int8 paths, compares a 24-token KV-cache greedy stream against O(n^2)
re-prefill greedy on the same weights, records a measured 13-step exact prefix
at about 320 positions, gates prefix >=12, asserts first id 50268 (`<Number>`),
and checks the decoded output opens the chart dict (`{`). Full text-vs-oracle
chat comparison is informational because that path crosses precision.
`0145419` added the OneChart recognize pipeline and decoder hidden-state tap;
`2a56c96` closed `bd-3jo6.4.5` for D5 native assembly. Current committed source
has `src/native_engine/onechart.rs::ChartResult` with `json_text`, optional
`pred_locs`, optional `reliable_distance`, and `reliable`; `recognize` uses the
fixed 308-id prompt, `<imgpad>` splice, bounded 4096 OPT KV-cache decode,
`<Number>` tap through `prefill_final_hidden`, `number_head`, string-aware
`complete_json_string`, chart-value extraction/normalization, and
`reliable_distance` self-verify. D5 anchors are
`chart_prompt_ids_match_oracle_l0c`, `recognize_reads_the_committed_chart`,
`reliable_check_matches_upstream_goldens`, and `number_head_matches_golden`.
Then `e926c46` closed D6/D7/D8 and sub-epic D: `src/cli.rs` now has
`OcrTask::ChartData`, `model_spec_is_knowably_not_onechart`, and display value
`chart-data`; `src/native_engine/mod.rs` dispatches `onechart` artifacts through
`forward_onechart`; `model_arch` marks `onechart` `implemented=true`; and
`scripts/onechart_chart_e2e.sh` emits `onechart_chart_e2e/v1` NDJSON with
model-gated skip, missing-model exit 3, wrong-family exit 2, and a real
`--task chart-data` run. `bd-2lje` / `9cb91f9` added an in-distribution
SCRM-proxy corpus: six matplotlib-style charts, number head fires 6/6, mean
`reliable_distance` is about 0.015 int8 / 0.014 f32, decoded text is
byte-identical f32-vs-int8 on all six, but valid JSON is only 1/6 in both
precisions. Treat that as scoped product evidence, not as proof that OneChart's
text decoder is generally high-quality. The current boundary is no longer
manifest distribution; it is product surface and quality: `focr pull onechart`
is current after `bd-av64.7`, but a separate `focr chart` subcommand, broad
chart quality, and general JSON reliability claims still need their own
evidence.

Polyphonic-TrOMR is now a packaged runtime lane for sheet-music OMR. Current
`franken_ocr` `main` has the E5 v1
staff-detection front end, `recognize_page`, and multi-staff partwise MusicXML
committed: `752f3cd` closes `bd-3jo6.5.5`, and `ab0bae0` closes E10 plus
sub-epic `bd-3jo6.5`. The v1 detector scope is printed/scanned pages with
global deskew; camera dewarp, default/lossless barline quality, `**kern`, and
int8 compute/perf are separate follow-ups. The later
`FOCR_TROMR_SPLIT=1` barline route is experimental/off-by-default rescue for
over-budget staff bands, not part of the v1 detector proof. `bd-2sez` /
`5430e2c` adds the
f32 TrOMR PERF_LEDGER baseline row, but it is an honest losing row: focr f32 is
about 2.3-3.2x slower than pinned upstream torch on the measured staff example,
with exact token-stream agreement as the correctness anchor. `bd-av64.7` /
`ece14f9` publishes `focr pull tromr`; `efccce9` / closed `bd-av64.12` now
installs `tromr.int8.focrq` and the four WordLevel tokenizer tables by default,
while `focr pull tromr --quant f32` installs the bit-exact `tromr.focrq`
reference.
`c22b047` / `bd-3jo6.5.2` closed E2: `scripts/gen_tromr_safetensors.py`
exports the WS-folded checkpoint, the reference `tromr.focrq` self-declares
`model_id=tromr`, and the real artifact round-trips 260 tensors byte-exact with
`0 int8`. In current source, `focr convert --model-id tromr` has no f32 quant
mode; use it for `--quant int8` storage conversion, or use
`focr pull tromr --quant f32` to get the published reference artifact. The later
`tromr.int8.focrq` artifact quantizes exactly 40 decoder GEMMs for storage and
dequants through f32 accessors.
`7464590` / `bd-3jo6.5.6` closed E6: `src/tokenizer/music.rs` implements four
decode-only WordLevel tables (`tokenizer_rhythm.json`, `tokenizer_pitch.json`,
`tokenizer_lift.json`, `tokenizer_note.json`) with dense-size validation,
specials only on the rhythm stream, and upstream `staff2score.py`
detokenization semantics. `6403d4c` first landed E3 helper kernels in
`src/native_engine/nn.rs`: `tf_same_pad_amounts`, `tf_same_pad`, `max_pool2d`,
and in-place `group_norm` with optional `fuse_relu`, all with torch/timm golden
tests. `45da3a3` then closed the committed E3 encoder:
`src/native_engine/tromr.rs` hydrates `TromrEncoderW`, runs the WS-prefolded
ResNetV2 stages `[2,3,7]`, adds crop-indexed learned positions over the
80-wide table, executes four pre-LN ViT blocks, and verifies
`tromr_encoder_matches_torch_oracle` at `encoder_out cos 1.00000000` / maxabs
`3.8e-6` against `tromr_oracle_fixtures.json` with oracle floor 0.0.
`3472c1b` / `bd-3jo6.5.4` then closed E4: `TromrDecoderW` hydrates the
self-contained x-transformers decoder (not `decoder_qwen2`), runs four layers
of causal self-attn + cross-attn over encoder context + GEGLU FF, uses pre-LN
eps 1e-5, inner 512 (`8 heads x 64`) over dim 256, bias-free q/k/v and
GLU-gated attention out, rhythm/pitch/lift embeddings plus `pos/16`, final LN,
and four parallel heads. `generate` defaults to deterministic per-head argmax;
`FOCR_TROMR_SAMPLE=1` enables upstream top-k/T=0.2 sampling arithmetic with
`FOCR_TROMR_SEED` as the deterministic seed;
the cert reports all four step-0 heads at cos `1.00000000` and maxabs <=
`7.6e-6`, plus 42-step x 3-stream token-exact argmax generation ending
`[barline, EOS]`. `79d715c` / `bd-3jo6.5.7` closed E7:
`merge_semantic` ports upstream merge semantics over aligned rhythm/pitch/lift
streams with fail-loud control-id rules, and `semantic_to_musicxml` emits
partwise MusicXML 4.0 with clefs, 15-major key signatures, numeric/common/cut
time, chords, dotted durations, rests, multirests, and accidentals. `78a2de3`
/ `bd-3jo6.5.9` closed E9: `preprocess::tromr_staff_tensor` ports
`staff2score.py::readimg`, `tromr::recognize` assembles
preprocess -> encode -> generate -> merge -> MusicXML, `native_engine/mod.rs`
adds `forward_tromr`, `src/cli.rs` adds the dual-lane `OcrTask::Music` guard
with `model_spec_is_knowably_not_tromr`, `model_arch` marks `tromr`
`implemented=true`, and `scripts/tromr_music_e2e.sh` emits
`tromr_music_e2e/v1` NDJSON with skip, exit-3 missing-model, exit-2
wrong-family, and real MusicXML legs. `2cbded9` / `bd-3jo6.5.8` then closed the
single-staff E8 ladder: L0b output gate, L1/L2 encoder cos 1.0, L3 four-head
decoder cos 1.0, L4 token-exact argmax, L5 SER mean 0.211 with per-example max
0.375, OQ-T1/T2/T3/T4/T6 resolved, DISC-004 ledgered, and e2e 4/4. `bd-2sez`
then closed the f32 baseline perf row with exact token-stream agreement but
slower focr f32 timings than pinned upstream torch. `fc9d88a` adds the E5 v1
pure-Rust `preprocess::staff_detect` module: DISC-004 ink plane, Otsu
thresholding, global projection-profile deskew, five-line grouping, ordered
staff crops, and synthetic tests. `752f3cd` closes E5 by wiring `recognize_page`,
`staves_to_musicxml`, and `forward_tromr` full-page behavior: >=2 detected
staves are read sequentially top-to-bottom and serialized as one MusicXML
`<part>` per staff; 0 or 1 detected staves preserve the certified whole-image
single-staff path. The armed page cert
`tromr_page_detects_and_reads_stacked_examples` stacks examples 1 and 2, proves
order by cross-SER, and pins detector-lossless SER 0.125 / 0.040, identical to
direct crops. `9127676` adds
`tromr_alpha_ink_path_fires_only_when_alpha_varies`, protecting DISC-004's
varying-alpha-as-ink and opaque-RGBA-luma paths. `ab0bae0` closes E10 and the
TrOMR sub-epic with about 25 unit tests, 6 armed certs, 2 NDJSON e2e scripts,
and a full `scripts/check.sh` gate reporting 891 library tests plus integration
tests, doctests, clippy, fmt, and UBS. `eb0c70e` then adds the committed
real-scan crop-geometry mechanics; `40ee875` tightens the boundary to
fit-first behavior: already-fitting full-width bands keep historic geometry,
while over-budget bands use ink-extent horizontal trim and neighbor-bounded
extend-to-fit toward the 1280-column position budget. It also makes the
page-resilience test deterministic with one wide-but-fittable staff and one
genuinely unfittable staff, and promotes p055/p100 real-scan truth floors to
5 and 1 recognized staves. `bd-av64.14` is closed for the fit-first geometry
lane and p169 acceptance only; do not expand it into broad corpus SER, camera
dewarp, TrOMR int8, or perf evidence. `64edce3` / `bd-av64.4` later adds
experimental `FOCR_TROMR_SPLIT=1` barline segmentation for over-budget staff
recognition-count rescue, not a default or lossless quality route. `**kern`
export, camera dewarp, default/broad barline quality, and TrOMR int8
compute/perf rows remain gated. The TrOMR `tromr.int8.focrq` distribution
artifact itself is current storage-int8 publication after `bd-av64.12`.
The TrOMR A11/PERF_LEDGER row exists after `bd-2sez`, but it is a f32 baseline
loss row, not a performance win.
`FOCR_SPEC_DECODE` has a closed linear ON==OFF gate (`bd-1azu.36`, 20/20 pages
byte-identical in f32 and int8), but it is still an opt-in presence switch and
the current proof is output identity, not per-run engagement telemetry.
The head-to-head gauntlet bead `bd-re8.17` is closed with first pinned
HF baseline bf16 PERF_LEDGER rows. Decode-per-token is about 1.62x faster,
prefill about 4.8-5.0x, vision about 3.6-3.7x, end-to-end 2.34-2.71x, with
CER 0.00943 on `page_0009` and CER 0.03529 on `page_0014`. Cite rows exactly:
`page_0009` preprocess is 0.916, cv% is 0.1-2.2, and decode still sits about
3.66x above the memory floor, so do not claim a universal stage win. `int4`
remains phase-gated. Top-level task-specific subcommands such as `focr music`,
`focr chart`, and `focr describe` are absent; use `focr ocr --task ...` or
`--format` with the right model instead. `doctor` is implemented
current-source behavior after `25eadc5` / `bd-wp8.4` with detect-only, dry-run,
fix, undo, capabilities, robot-docs, and doctor-specific robot triage. `runs`
and `sync` are now current-source behavior after closed `bd-223.4` /
`03eadd2`: they use the fsqlite-backed `RunStore` for queries and JSONL audit
sync. Older installed binaries can still truthfully scaffold or omit them, so
verify source, help, tests, and Beads before answering which side a user is on.
If broad inline prose in `src/cli.rs` still says
subcommands are skeletons, do not treat that comment as more authoritative than
the concrete `run_runs`, `run_sync`, `run_doctor`, help, tests, and Beads
evidence.

Committed `59d376b` closes `bd-223.2`: cooperative Ctrl+C /
`request_shutdown()` cancellation, `cancel_checkpoint()` calls in seven decode
loops, `FOCR_THREADS` -> physical-core `thread_budget()`, `robot health` /
`robot backends` `threads`, and bounded sequential `stream_pages()` output
streaming with backpressure. Treat those as current-source behavior, while
still quarantining stale installed binaries that lack the fields.

Committed `03eadd2` closes `bd-223.4`, and `d52d344` closes the parent Phase 0:
durable run telemetry on frankensqlite is current-source behavior.
`src/storage.rs` exposes `SCHEMA_VERSION = 1`, `_meta`, `RunRecord`,
`RunStore`, `FOCR_RUN_STORE`, default `~/.cache/franken_ocr/runs.db`,
best-effort OCR run recording, `focr runs --format plain|json|ndjson`, and
locked/atomic `focr sync export-jsonl [--file ...]` /
`focr sync import-jsonl --file ...`. JSON records include `schema_version`,
`run_id`, timestamps, `input_path`, `mode`, `quant`, `model_version_tag`,
`exit_code`, and `status`. Export defaults to the store path with its extension
replaced by `.jsonl` (for `runs.db`, `runs.jsonl`), writes a same-directory
`.jsonl.tmp` file, fsyncs, and renames under a `.jsonl.lock` sentinel; import
takes the same lock and replays records idempotently by `run_id`.

Committed `3e85c7d` closes `bd-3kge` / `bd-2pgf`: shared determinism helpers
now live in `tests/support/parity_harness.rs`, emit structured
`parity` / `token_exact` lines, and fail loudly on injected HashMap-order
nondeterminism; the model-gated e2e real-model leg calls `recognize()` twice and
requires byte-identical output. Fixture policy is also explicit:
`tests/fixtures/PROVENANCE.md` is the prose catalogue,
`tests/fixtures/MANIFEST.toml` is the machine-readable committed vs
regenerated-committed policy, and `scripts/check_fixture_manifest.py` is
bidirectional plus wired into `scripts/check.sh`. This proves shared test
infrastructure, not model quality by itself.

Committed `fb52843` closes `bd-re8.12`, with `c685818` as a format-only
follow-up: `src/conformance.rs` defines the `ConformanceTest` trait, RFC-2119
`RequirementLevel`, `ConformanceCategory`, `RegisteredConformance`, and
`conformance_registry()`; `tests/conformance_matrix.rs` computes coverage from
the spec side, not from the test list. It parses
`docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`, saw 83 MUST clauses in
the close evidence, requires MUST coverage >= 0.95, emits one NDJSON clause row
plus a summary, rejects bare XFAIL emissions unless a real `DISC-NNN` or phase
gap is nearby, and asserts every registry entry runs green in-process. This
supports release conformance accounting; it does not mean every differential,
golden, metamorphic, model-corpus, or performance gate is closed.

Committed `390d05c` closes `bd-re8.9`: the oracle-differential comparator is
current source in `tests/parity_ladder.rs`. Its always-on contract prevents
oracle-vs-oracle false greens with `EngineIdentity`, emits `differential_row`
fields (`scope`, `oracle`, `module`, `max_diff`, `within_tol`, `xfail`,
`disc`), compares per-op outputs against the primary bf16 oracle through ULP
tables and L3-L5 tolerances, treats intentional divergences as `DISC-NNN`
XFAILs rather than skips, and model-gates full e2e evidence with
skip-with-SUCCESS only when artifacts are absent.

Committed `5f2d7ce` closes `bd-re8.10`: `tests/metamorphic.rs` now carries
oracle-free relations for identity resize, rotation bbox mapping,
mean-gray-padding strict text via `preprocess::PAD_FILL`, deterministic output
across repeat runs and `FOCR_THREADS=1` vs `4`, a documented white-pad
SHOULD/existential observation, and a gated cross-page-dependence relation. The
critical negative guard is part of the proof: R-SWA makes multi-page parsing
cross-page dependent, so do not add or cite a relation that a multi-page result
must equal the concatenation or sum of individually parsed pages.

Committed `f879211` closes `bd-re8.11`: the golden-artifact suite in
`tests/cli_robot_golden.rs`, `tests/fixtures/golden/PROVENANCE.md`, and
`docs/conformance/GOLDEN.md` covers exact CLI/help/schema JSON, fuzzy numeric
ULP artifacts, scrubbed robot NDJSON, and canonicalized cross-platform output.
`UPDATE_GOLDENS=1` is a manual human-reviewed refresh path only; CI must not set
it, mismatches write `.actual` files for review, and `.actual` / `.snap.new`
artifacts must stay gitignored.

Committed `1b84428` adds `scripts/ladder_scorecard.sh`, the ordered L0-L5
scorecard runner for `bd-re8.19`; `1112cf8` plus the tracker close evidence make
that bead closed-current. It runs
`cargo test --release --test parity_ladder -- --test-threads=1 --nocapture`,
folds the rungs' structured NDJSON into a single
`focr-ladder-scorecard/v1` JSON artifact, marks rungs above the first hard
failure as `not_meaningful`, keeps `skipped_no_model` separate from
`all_green`, and has a `--self-test` fold check that needs no model weights.
Armed close evidence reported all six gates green and a receipt with L0
`max_abs_diff=0.0078432`, L1 cosine 1, L2 maxabs `8.76e-05`, L3 cosine 1, L4
token exact fraction 1, and L5 CER 0. Unarmed no-model runs remain honest
skips (`skipped_no_model=true`, `all_green=false`), not green proof.

## Fast Probe

### Binary-only machines (no source checkout)

On an install-and-use machine, `~/projects/franken_ocr` does not exist and the
source-first Truth Stack is unreachable; the remedy for a stale binary is to
**upgrade the binary**, not "rebuild from source". Probe the installed binary
directly:

```bash
command -v focr && focr --version
focr models --json | jq '.models[] | {id, status}'   # ready vs planned
focr robot triage | jq '.recommendations'            # state-aware next steps
focr robot selftest | jq '.verdict'                  # kernel health on THIS CPU
focr robot health | jq .                             # model present? threads?
```

If facts here conflict with this skill's prose, trust the live binary.

### Source checkouts

Run the cheap probe first. It classifies the checkout, the obvious command
surface, and the freshest tracker boundaries without building anything:

```bash
cd ~/projects/franken_ocr
git status --short --branch
git log --oneline -16
rg -n "enum Commands|fn run_models|model_id|pub struct OcrEngine|pub fn recognize|recognize_multi_page|recognize_multi_page_dynamic_streaming|preprocess_dynamic_squash|page_decoded_event|FOCR_TROMR_SPLIT|barline_columns|recognize_split|arch_target|SmmlaPanels|robot_selftest|staff_event" \
  src
for id in \
  bd-av64.6 bd-av64.14 bd-av64.4 bd-1gv.25 bd-1gv.26 bd-1465 bd-2z0y bd-1a6h bd-2mo.24 \
  bd-1es bd-2mo.1 bd-2mo.3 bd-2mo.3.1 bd-3jo6.1.12 bd-2mo
do
  br show "$id" --json | jq -r '.[] | "\(.id)\t\(.status)\t\(.title)"'
done
```

Then run only the targeted probe for the surface in question:

```bash
# Model-zoo manifest and named pulls
rg -n "BUILTIN_MANIFEST_JSON|ModelEntry|sidecars|select_quant|pull\\.in_manifest|pull\\.quants|models-smolvlm2-v1|models-onechart-v1|models-tromr-v1|bd-av64\\.7|ece14f9" \
  src models README.md .beads/issues.jsonl

# SmolVLM2 describe/VQA and SigLIP frame batching
rg -n "forward_smolvlm2|set_smolvlm2_question|OcrTask::Describe|vision_rows|forward_frames_batched|FOCR_SIGLIP_SEQ|batched_frames_match_sequential_byte_for_byte" \
  src/cli.rs src/native_engine

# OneChart chart-data
rg -n "onechart|OneChart|ChartData|chart-data|forward_onechart|ChartResult|complete_json_string|prefill_final_hidden|recognize_reads_the_committed_chart|reliable_check_matches_upstream_goldens|number_head_matches_golden|chart_prompt_ids_match_oracle_l0c|corpus_quality_scrm_proxy|onechart_chart_e2e|DecoderConfig::onechart|DecoderFamily::Opt|build_inputs_embeds|opt_prefill|generate_greedy_kvcache|opt_kvcache|onechart_view_tensor|vision_features|FOCR_ONECHART_DIR" \
  src tests scripts docs/zoo README.md

# TrOMR music/real-scan
rg -n "tromr|TrOMR|MusicVocab|MusicTokenizer|WordLevel|TromrEncoderW|TromrDecoderW|decoder_forward|merge_semantic|semantic_to_musicxml|MusicResult|forward_tromr|tromr_staff_tensor|tromr_music_e2e|tromr_encoder_matches_torch_oracle|tromr_decoder_matches_argmax_oracle|tromr_ser_vs_committed_ground_truth|DISC-004|FOCR_TROMR_DIR|bd-3jo6\\.5|realscan_music_gate|bd-av64\\.6" \
  src tests scripts docs/zoo README.md docs/DISCREPANCIES.md

# Doctor, robot triage, release readiness, PDF/spreads, runtime, run-store
rg -n "enum Doctor|run_doctor|doctor::|DoctorExit|capabilities|robot-docs|--robot-triage|actions\\.jsonl|bd-wp8\\.4" src tests README.md .beads/issues.jsonl
rg -n "robot_triage_payload|robot triage|quick_ref|recommendations|agent_ergonomics|bd-wp8\\.7" src tests docs README.md .beads/issues.jsonl
rg -n "release-readiness|RELEASE_READINESS|franken_ocr\\.release_readiness\\.v1|--bundle|FINAL_GAUNTLET_REPORT|certification_bundle|release_certificate|gauntlet_convergence|bd-wp8\\.9|bd-wp8\\.10" scripts docs .beads/issues.jsonl
rg -n "split_spread|split-spreads|--split-spreads|--pages|content_rotation|bd-av64\\.11" src tests README.md .beads/issues.jsonl
rg -n "request_shutdown|shutdown_requested|reset_shutdown|cancel_checkpoint|thread_budget|stream_pages|FOCR_THREADS|ctrlc|bd-223\\.2|\\\"threads\\\"" src tests README.md .beads/issues.jsonl
rg -n "RunStore|RunRecord|FOCR_RUN_STORE|SCHEMA_VERSION|export-jsonl|import-jsonl|fsqlite|bd-223\\.4|runs\\.db|jsonl\\.lock|jsonl\\.tmp" src tests README.md .beads/issues.jsonl

# Verification families
rg -n "assert_deterministic|assert_outputs_deterministic|check_fixture_manifest|MANIFEST\\.toml|PROVENANCE\\.md|bd-3kge|bd-2pgf" tests scripts .beads/issues.jsonl
rg -n "ConformanceTest|RequirementLevel|ConformanceCategory|conformance_registry|conformance_matrix|SPEC-[0-9]{3}|MUST coverage|DISC-NNN|bd-re8\\.12" src tests docs .beads/issues.jsonl
rg -n "differential_per_op_vs_bf16_oracle|differential_row|EngineIdentity|bd-re8\\.9" tests/parity_ladder.rs docs .beads/issues.jsonl
rg -n "MR-1|MR-2|MR-3a|MR-4|MR-5|sum-of-parts|cross-page DEPENDENT|bd-re8\\.10" tests/metamorphic.rs docs/conformance/METAMORPHIC.md .beads/issues.jsonl
rg -n "UPDATE_GOLDENS|assert_golden|canonical_json|scrub_volatile|ROBOT_SCHEMA_VERSION|bd-re8\\.11" tests/cli_robot_golden.rs docs/conformance/GOLDEN.md tests/fixtures/golden/PROVENANCE.md .beads/issues.jsonl
rg -n "ladder_scorecard|focr-ladder-scorecard|not_meaningful|skipped_no_model|bd-re8\\.19" scripts docs/conformance tests .beads/issues.jsonl
```

Use live binary/source commands only when the claim depends on current CLI
behavior and the build cost is acceptable:

```bash
cargo run --bin focr -- pull --help | rg -- 'MODEL|quant|manifest'
cargo run --bin focr -- models --json | jq '.models[] | {id, implemented, pull}'
cargo run --bin focr -- robot triage | jq '.quick_ref, .recommendations[0], .exit_codes'
cargo run --bin focr -- doctor capabilities --json | jq .
python3 scripts/gauntlet_cert.py --release-readiness
python3 scripts/gauntlet_cert.py --bundle
cargo run --bin focr -- --help   # if live help is needed and build cost is acceptable
command -v focr && focr --version && focr --help
```

## Common Workflows

### Install the focr CLI

Use the README-supported prebuilt binary path first:

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
focr --version
focr pull
focr pull got-ocr2   # optional specialized model
focr pull smolvlm2   # optional describe/VQA model
focr pull onechart   # optional chart-data model
focr pull tromr      # optional OMR model; default quantized-storage artifact
focr pull tromr --quant f32  # bit-exact TrOMR reference artifact
focr pull --manifest ./manifest.json   # custom/airgapped manifest only
```

On native Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
focr --version
focr pull
```

The installer resolves the latest **binary** release (as of 2026-08-18 the
installers on `main` enumerate `/releases` and take the newest semver `v*` tag,
skipping the `models-*` weights releases), verifies SHA256, and puts `focr` on
`PATH`. There is **no baked-in fallback version**: a failed release lookup
exits 1 and a non-semver tag exits 2 — the installer is not self-healing. If a
bare one-liner install fails at "Resolving the latest release", pin explicitly:
`curl -fsSL .../install.sh | bash -s -- --version v0.7.2` (PowerShell:
`-Version v0.7.2`). Known trap (GH #12, fixed on `main` 2026-08-18): this repo
also publishes model-weight releases (`models-*`), and older installer copies
trusted `/releases/latest`, which returns whichever release was published most
recently regardless of tag shape.
It installs only the CLI binary; run `focr pull` afterward for the default
Unlimited-OCR weights/tokenizer — as of `v0.7.0+` that artifact is **4.16 GB,
fetched as 3 verified parts, installed as the versioned filename
`unlimited-ocr.v0.7.0.int8.focrq`** (not the older `unlimited-ocr.int8.focrq`); run
`focr pull got-ocr2`, `focr pull smolvlm2`, `focr pull onechart`, or
`focr pull tromr` for specialized models. Named models install into
`~/.cache/franken_ocr/models/<model-id>/`; the manifest's sidecars install the
tokenizers next to the `.focrq`. `focr pull tromr` now reports and installs the
published `int8` storage quant by default; use `focr pull tromr --quant f32`
when you need byte-exact reference weights. Use
`--manifest <path-or-url>` or `FOCR_MANIFEST_URL` only for an alternate
distribution manifest; a manifest override works only when that resolved
manifest actually lists the target model and quant.
If help lacks `--extract-figures`, treat that as release lag or a stale binary
and confirm the exact release/source boundary before support claims. Do not
recommend `cargo install`: sibling path deps are unpublished.
Before recommending a named pull, verify the running binary's manifest with
`focr models --json` and inspect `pull.in_manifest` / `pull.quants`. If the
installed binary is older than `bd-av64.7`, rebuild from source or pass a
manifest containing the named model instead of pretending the old binary can
pull it.

### OCR one image or PDF

```bash
focr ocr invoice.png --json
focr ocr invoice.png -o invoice.md
focr ocr invoice.png -o invoice.json
focr ocr invoice.png -o invoice.md --extract-figures
focr ocr scan.pdf --json
focr ocr book.pdf --pages 3,5-9 --split-spreads -o excerpt.md
focr ocr invoice.png --model ~/.cache/franken_ocr/models/unlimited-ocr.v0.7.0.int8.focrq
focr ocr formula.png --task formula --model got-ocr2.int8.focrq
focr ocr table.png --format --model got-ocr2.int8.focrq
focr ocr photo.jpg --task describe --model smolvlm2.int8.focrq --question "What is on the table?"
focr ocr chart.png --task chart-data --model /opt/models/onechart/onechart.int8.focrq
focr ocr staff.png --task music --model /opt/models/tromr/tromr.int8.focrq
focr ocr staff.png --task music --model /opt/models/tromr/tromr.focrq  # after --quant f32 pull
```

Plain human mode writes markdown/text. `--json` and `.json` output files write
machine-readable JSON with `schema_version`, `markdown`, layout boxes, and when
figures are extracted a `figures` array. PDFs nest per-page layout under `pages`.
Use `--robot` or `focr robot run
<image-or-pdf>` when the caller consumes NDJSON lifecycle events. The default
crop mode is `base`, the certified single 1024-pixel global view; `--crop-mode
gundam` selects reference dynamic tiling. There is first Gundam e2e evidence
from `bd-1e9n`, but require fresh target-corpus proof before parity claims.
PDFs are routed natively by `focr ocr`/`focr robot run`: scanned pages are
rasterized in
pure Rust, OCRed one at a time, and concatenated with blank lines.
**"Scanned" is load-bearing: only PDFs whose pages are image XObjects work on
the native fast path.** An ordinary born-digital (vector/text) PDF — the common
case for a PDF on disk — fails with exit 4:
`input decode error: PDF page 1 has no image XObject (vector/text PDFs are not
supported by the native fast path; rasterize the PDF out of band ...)`. The
same applies to the two codecs with no pure-Rust decoder, `JPXDecode`
(JPEG 2000) and `JBIG2Decode`. The fix is exactly what the error names —
rasterize first, then OCR the images:

```bash
pdftoppm -png -r 200 report.pdf page   # page-1.png, page-2.png, ...
focr ocr-batch page-*.png --json
```

**Resident warm-model daemon (source newer than `v0.7.2`).** Eligible
single-image `focr ocr` runs are served by a per-model background process that
keeps the loaded weights in RAM; back-to-back invocations skip the multi-GB
artifact load, and the daemon exits after 10 idle minutes. The output contract
is identical (any daemon problem silently falls back to the in-process load).
Opt out with `--no-resident` or `FOCR_NO_RESIDENT=1`; tune with
`FOCR_RESIDENT_IDLE_SECS`; `FOCR_RESIDENT_LOG=<file>` captures daemon
diagnostics. Released `v0.7.2` binaries do NOT have this — verify with
`focr ocr --help | grep no-resident` before claiming it.
`--pages` is PDF-only, 1-based, and accepts comma/range syntax such as
`3,5-9`; duplicates are deduped in source order. `--split-spreads` is also
PDF-only and heuristically splits wide scanned book spreads into left/right
logical pages with JSON/robot page metadata carrying the half. It is useful for
bound scans but can false-split non-book wide pages; treat it as an extraction
policy, not a correctness proof. It currently refuses to compose with
`--extract-figures`, because figure naming/placement across split halves needs
a separate contract.

### OCR a batch

```bash
focr ocr-batch page-1.png page-2.png page-3.png --json
focr ocr-batch page-1.png page-2.png --f32
FOCR_BATCH_SPINE=1 FOCR_BATCH_SIZE=64 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=1 FOCR_BATCH_SIZE=128 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=1 FOCR_BATCH_PACK=1 focr ocr-batch page-*.png --json
```

The library result shape is `Result<Vec<Result<String>>>`: outer errors mean
setup/model failure; inner errors are per-image failures. Batch mode takes image
paths; use single-document `focr ocr file.pdf` for native PDF handling, or
rasterize PDFs to images before batching.
Use an explicit `FOCR_BATCH_SIZE` when benchmarking. Earlier July 7 notes
treated this as a caveat; in `v0.4.0`-and-later source it is committed for the
shared `batch_scheduler` path: unset, blank, unparsable, or `0` uses `128`, and
larger values clamp at `256`. The older MoE
decoder-internal helper still has its own historical default constant in
`decoder.rs`; when answering a user-facing `FOCR_BATCH_SIZE` question, prefer
the README/source contract for `ocr-batch` and cite the exact module if
discussing internals.
For GOT-OCR2 timing on source at or after `3f2878d`, first check the
model-level cache: SAM weights, `mm_projector_vary`, and the widened embedding
table should hydrate once on `OcrModel`, both sequential and batch page paths
should reuse `GotStatics`, and `FOCR_TIMING` should show
`got.hydrate(cached)`. On older `d25dbd7`-only binaries, the narrower batch
hoist logs `got.hydrate(batch)` and `got.vision+splice(batch of N)`.

### Robot integration

```bash
focr robot schema | jq .
focr robot health | jq .
focr robot backends | jq .
focr robot selftest | jq .
focr robot triage | jq '.quick_ref, .recommendations[0], .commands'
FOCR_FORCE_ARCH=scalar focr robot selftest | jq .
set -o pipefail
focr robot run scan.pdf | jq -c .
```

`robot backends` is the cheap CPU/SIMD capability probe; `robot selftest` is the
kernel parity proof, not a performance row. Current source at/after `ad3ad20`
also returns a `models` rollup with stable ids for the registered int8 decoder
families: `unlimited-ocr`, `got-ocr2`, `smolvlm2`, and `onechart`. Use it to
answer "which model kernels passed on this host"; use case rows for details.
TrOMR is absent from that rollup because its current runtime consumes the int8
storage artifact through f32 dequant-on-access rather than an int8 decoder
kernel.
In current `bd-223.2` source, `robot health` and
`robot backends` also expose the resolved `threads` budget from `FOCR_THREADS`
or physical cores; if that field is missing, classify the binary/source as
pre-`bd-223.2` or stale before changing docs. Robot output must stay
line-oriented, versioned, and free of human decoration. `focr robot triage`
is the preferred one-shot automation entry point for agents: it returns
schema/version data, quick status, recommendations, copyable commands, and exit
code meanings without requiring a sequence of human probes.

### Diagnose and Repair Local State

```bash
focr doctor --json | jq .
focr doctor --dry-run --fix --json | jq .
focr doctor --fix --json | jq .
focr doctor undo <run-id> --json | jq .
focr doctor capabilities --json | jq .
focr doctor robot-docs
focr doctor --robot-triage | jq .
```

`focr doctor` is current, not scaffolded. Detect-only mode should be the first
move unless the user explicitly wants repair. `--dry-run --fix` reports the
planned mutation set without changing the cache. `--fix` writes backups under
`.doctor/runs/<run-id>/backups/`, logs `actions.jsonl`, and uses `.doctor/lock`
for concurrency. Use `undo` instead of hand-editing doctor-managed repairs.
Doctor exit codes are a doctor sub-contract: 0 healthy, 1 findings, 2 partial,
3 failed and rolled back, 4 refused unsafe, 5 concurrency lost, 6 online
required.

### Check Release Readiness

```bash
python3 scripts/gauntlet_cert.py --release-readiness
jq '{artifact, ship, green, red, blocking_cells, cells: [.cells[] | {cell, status, detail}]}' docs/gauntlet/RELEASE_READINESS.json
python3 scripts/gauntlet_cert.py --bundle
jq '{artifact, certified, git_describe, git_head, convergence, refusal_reasons}' docs/gauntlet/bundle/release_certificate.json
```

Release-readiness is a scorecard artifact, not an intuition. Current evidence
from `c29a78b` / `7c7bd00` plus the public `v0.6.0` tag `29516b9` has the
Phase-5 ship gate closed:
`docs/gauntlet/RELEASE_READINESS.json` reports `ship:true`, `green:13`,
`red:0`, no `blocking_cells`, and green `certification_bundle` plus
`gauntlet_convergence` cells. The convergence detail is
`rounds=11/10, tail_clean=True`, and Beads closes `bd-wp8.8`, `bd-wp8.9`, and
`bd-wp8.10`. `--bundle` writes
`docs/gauntlet/bundle/FINAL_GAUNTLET_REPORT.md`,
`certification_bundle.json`, `release_certificate.json`, `scorecards.json`, and
`benchmark_summary.json`; the current certificate says `certified:true` at
`v0.5.2-8-gc4c1684`. That is not a contradiction with `v0.6.0`: the bundle was
generated at `c4c1684`, `c29a78b` folded it into the all-green readiness
scorecard, and `29516b9` published/tagged the certified state. `beaed7c`
records CI/dist supplement notes, `db02421` refreshes README evidence,
`5df6395` commits post-certification fuzz corpus growth, and `592426c` refreshes
README `v0.6.0` public-release identity/asset-size/backend prose. The historical
pre-`c29a78b` red state
(old `ship:false`, `green:11`, `red:2`, `rounds=8/10, tail_clean=False`) is useful
only for debugging old artifacts. Say "ship gate green" only when the
scorecard, bundle certificate, convergence artifact, and Beads agree; still
separate that from installed-binary version and unrelated open epics.

### Query local run state and sync audit JSONL

```bash
FOCR_RUN_STORE=/tmp/focr-runs.db focr runs --format json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr runs --format ndjson
FOCR_RUN_STORE=/tmp/focr-runs.db focr sync export-jsonl --file /tmp/focr-runs.jsonl --json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr sync import-jsonl --file /tmp/focr-runs.jsonl --json | jq .
```

Current source has `RunStore` support after closed `bd-223.4`, but support
answers for installed binaries must still confirm exact help/source. The store
is fsqlite-backed, opens at `FOCR_RUN_STORE` or
`~/.cache/franken_ocr/runs.db`, refuses too-new `_meta.schema_version` with
exit 7, records OCR outcomes best-effort, and syncs canonical JSONL under a
lock. Store-write failures should not fail an OCR run; they produce a stderr
note. Automated tests should always set `FOCR_RUN_STORE` to a temp path so they
do not mutate the user's real run history.

Latest source docs also sharpen CPU backend guardrails. `robot backends` reports
the selected tier and available tiers; `FOCR_FORCE_ARCH` can force
`scalar|sdot|smmla|avx2|avxvnni|avx512vnni` when that tier exists. Apple
Silicon prefers SDOT over SMMLA unless a non-Apple aarch64 host justifies the
opposite, x86 reports AVX2 / AVX-VNNI / AVX-512-VNNI only when dispatchable, and
AMX is not advertised as current. `robot selftest` proves the selected int8 GEMM
matches the scalar oracle on the host; the `models` rollup is parity summary,
not a benchmark or TrOMR int8 evidence.

### Convert source weights

```bash
focr convert \
  /models/Unlimited-OCR/model.safetensors \
  -o ~/.cache/franken_ocr/models/unlimited-ocr-int8.focrq \
  --quant int8 --model-id unlimited-ocr --arch generic

focr convert \
  /models/SmolVLM2-500M/model.safetensors \
  -o ~/.cache/franken_ocr/models/smolvlm2.int8.focrq \
  --quant int8 --model-id smolvlm2 --arch generic --json

focr convert \
  /models/OneChart/model.safetensors \
  -o ~/.cache/franken_ocr/models/onechart.int8.focrq \
  --quant int8 --model-id onechart --arch generic --json

focr convert \
  /models/tromr/model.safetensors \
  -o ~/.cache/franken_ocr/models/tromr.int8.focrq \
  --quant int8 --model-id tromr --arch generic --json
```

For TrOMR, `--quant f32` is a `focr pull` manifest selection, not a converter
mode. Use `focr pull tromr --quant f32` when you need the high-precision
reference, and use `focr convert --quant int8 --model-id tromr` or default
`focr pull tromr` when you want the published storage-int8 artifact. Use
`--model-id got-ocr2` only with GOT-shaped weights; `focr pull got-ocr2`
installs the packaged GOT `.focrq` plus `qwen.tiktoken` automatically.
Self-converted GOT artifacts still need `qwen.tiktoken` beside the model. Use
`--model-id smolvlm2` only with SmolVLM2-shaped weights. `smolvlm2.int8.focrq`
routes through `--task describe` in current source; keep the C8/C10 closure
evidence and DISC-003 near-tie ledger with the artifact. Prefer
`focr pull smolvlm2` for the committed packaged artifact when
`focr models --json` reports `pull.in_manifest=true`; use conversion for local
or repinned weights. Use `--model-id onechart` only with OneChart/OPT-shaped
weights; conversion and tokenizer D9 are current, D3 vision/projector is
certified, D4 prefill/cached decode are certified, D5 native recognition
assembly is closed, and D6/D7/D8 route it through `focr ocr --task chart-data
--model onechart.int8.focrq`. Prefer `focr pull onechart` for the committed
packaged artifact; do not invent `focr chart`. Use `--model-id tromr` only with
WS-folded Polyphonic-TrOMR weights; E2 conversion is byte-exact but currently
all high precision (`0 int8`), E6 tokenizer support is decode-only, E3 encoder
support is committed and oracle-certified, and E4 deterministic decoder support
is committed and oracle-certified. The MusicXML assembly layer, single-staff
runtime, E8 single-staff quality ladder, E5 v1 detector, `recognize_page`
full-page runtime, and E10 test sweep are committed: use `focr pull tromr`,
`focr pull tromr --quant f32`, or a local TrOMR `.focrq` with tokenizer tables
beside the artifact for single-staff crops or v1 printed/scanned full-page
scores. Keep the remaining gates explicit: no standalone `focr music`, no int8
compute proof, no camera dewarp, no default/lossless barline quality, no TrOMR
perf win or int8 perf row yet, and no `**kern` export. The experimental
`FOCR_TROMR_SPLIT=1` barline path is a measured-not-lossless recognition-count
rescue, not a reason to relax those gates. Use `bd-2sez` as the f32 baseline
loss row; use `bd-av64.12` as storage-publication evidence, not speed evidence.
`int4` is intentionally phase-gated.

## Library Integration

```rust
use std::path::Path;
use franken_ocr::OcrEngine;

fn main() -> franken_ocr::FocrResult<()> {
    let engine = OcrEngine::new()?;
    let markdown = engine.recognize(Path::new("invoice.png"))?;
    println!("{markdown}");
    Ok(())
}
```

Create one long-lived `OcrEngine` per process. It is synchronous and blocking;
wrap it at the service boundary if your application is async. Do not create one
engine per request, do not nest runtimes around it, and do not run multiple live
forwards against one model for throughput experiments until the project exposes
an explicit safe policy. For structured boxes, use `recognize_with_layout` or
`recognize_dynamic_with_layout`; the returned `RecognizedDocument` contains
`markdown` plus `Vec<LayoutSpan>`. For cropped figures, use
`recognize_with_figures` / `recognize_dynamic_with_figures` for
`(RecognizedDocument, Vec<ExtractedFigure>)`. For one cross-page Unlimited-OCR
document pass over page images, use `recognize_multi_page` /
`recognize_multi_page_with_model` or the dynamic-image variants
`recognize_multi_page_dynamic` / `recognize_multi_page_dynamic_with_model`.
Those APIs return one markdown document with `<PAGE>` separators and do not
return per-page layout boxes. For PDFs, render pages with
`franken_ocr::pdf::PdfPages`; the CLI's PDF `--multi-page` route rasterizes
selected pages first and then calls the same multi-page engine. Model metadata
lives in `franken_ocr::model_arch`; non-default `.focrq` `model_id` drives
dispatch.

Stable exit-code meanings live in [ROBOT.md](references/ROBOT.md) and
[CLI.md](references/CLI.md); keep parsers tied to source/tests, not memory.

## Development Rules in `franken_ocr`

1. Read `AGENTS.md`, `README.md`, and for kernel/model work
   `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`.
2. Use `br`/`bv` in robot or JSON modes only: `br ready --json`,
   `br show <id> --json`, `bv --robot-triage`.
3. Never run bare `bv`; never assume `br` commits anything.
4. For substantive source changes, run `scripts/check.sh` or the equivalent
   `cargo fmt --check`, `cargo check --all-targets`,
   `cargo clippy --all-targets -- -D warnings`, `cargo test`, then `ubs`.
5. Respect unresolved `[OPEN]`/OQ gates before kernels or lossy optimizations.

## Do Not

| Anti-pattern | Correct move |
|--------------|--------------|
| Trust an old installed binary | Probe source and rebuild/run from source |
| Parse robot output as one JSON document | Consume NDJSON line by line |
| Enable lossy experimental env vars casually | Require parity/CER evidence and a kill switch |
| Use `focr pull` during inference | Pull once, then run offline |
| Create one `OcrEngine` per OCR request | Reuse one engine and batch where appropriate |
| Feed PDFs to `ocr-batch` expecting native routing | Use `focr ocr file.pdf` or rasterize pages first |
| Treat int4 as available | Keep it phase-gated until source says otherwise |
| Treat every `focr models` row as runnable | Check `status`/`implemented` and artifacts |
| Treat a ready registry row as manifest-packaged | Inspect `models/manifest.json`; `focr pull` can fetch only manifest entries |
| Invent `focr music` or `focr chart` subcommands | Use `focr ocr --task ... --model ...`; TrOMR music uses `--model tromr.int8.focrq` after default pull or `--model tromr.focrq` after `focr pull tromr --quant f32`, not a top-level subcommand |
| Invent `focr describe` | Use `focr ocr --task describe --model smolvlm2.int8.focrq` after `focr pull smolvlm2` or local conversion |
| Treat OneChart pull/route support as broad chart quality | Use OP-OC; D1-D9/sub-epic D and `focr pull onechart` are current, but broad quality and JSON reliability need corpus/A11 evidence |
| Treat TrOMR pull/runtime/crop-shaping/split support as int8 compute, camera-dewarp support, or unconstrained quality/perf proof | Use OP-TM/OP-ZM; default `focr pull tromr` installs the published `tromr.int8.focrq` storage artifact plus tokenizers, `focr pull tromr --quant f32` installs the bit-exact `tromr.focrq`, `bd-av64.6` is closed for corpus-v1 measurement, `bd-av64.14` is closed for fit-first geometry/p169 acceptance only, `bd-av64.12` is storage publication not int8 compute/perf, and `bd-av64.4` is experimental `FOCR_TROMR_SPLIT=1` recognition-count rescue only. A standalone `focr music` subcommand, int8 compute, perf wins/int8 perf rows, camera dewarp, default/broad barline quality, broad note-level SER, and `**kern` export remain gated. `bd-2sez` is a f32 baseline loss row, not a win |
| Treat doctor as scaffolded or safe to auto-fix without user intent | Use OP-DR; start detect-only or dry-run, then use `--fix`/`undo` through the doctor mutation ledger |
| Treat `robot triage` as decorative help | Use OP-RT; it is the preferred one-shot JSON state/recommendation contract for agents |
| Treat `v0.5.2` as the latest release after July 8 afternoon | Use OP-LC/OP-SG; latest observed release is `v0.6.0` at tag `29516b9`, with `origin/main` at `592426c` / `v0.6.0-4-g592426c` |
| Treat `v0.6.0-3-g5df6395` as the current clean source boundary | Use OP-LC/OP-SG; it is the historical post-certification fuzz-corpus boundary. Current clean source is `v0.6.0-4-g592426c` at `592426c` |
| Treat old red release-readiness artifacts as current | Use OP-SG; current `c29a78b`/`7c7bd00`/`29516b9` evidence is `ship:true`, `green:13`, `red:0`, `rounds=11/10`, `tail_clean=True`, and closed `bd-wp8.8`/`bd-wp8.9`/`bd-wp8.10` |
| Treat release certification as proof of every adjacent epic | Use OP-SG; it proves the Phase-5 ship gate and bundle only, while parent `bd-wp8`, parent `bd-2mo`, int4 (`bd-3gaa`), ARM64 Windows (`bd-3u97`), and installed-binary checks remain separate. Native Windows x86_64 is supported |
| Treat `certification_bundle` as hard-coded red or self-referential after `9bc715e` | Use OP-SG; it reads live `release_certificate.json`, `--bundle` excludes that cell from its own predicate, and after `c29a78b` the certificate is green |
| Treat round-8 deep fuzz as the final convergence proof by itself | Cite it as bounded evidence: 4.7M fuzz runs, `PROPTEST_CASES=2048`, 6/6 advisory matrix, zero new findings; the current convergence/corpus story also includes round-11 3.65M zero-crash runs and `5df6395` committing 3,271 fuzz seeds |
| Retry SIMD-exp/polynomial softmax as the next easy win | Use `docs/NEGATIVE_EVIDENCE.md` and `artifacts/perf/bd-av64.10-simd-exp/`; it was measured dead, reverted, and token-output fragile |
| Retry row-tiled SAM global attention without fresh target-specific profiling | Use OP-GB and the `c5e535a` / `8bd4037` / `b757bc0` boundary; it was byte-identical but publicly reverted and ledgered after slower same-regime measurements on Apple Silicon |
| Treat `3f2878d` GOT `GotStatics` caching as release readiness, `bd-av64.10` closure, or a formal A11/PERF_LEDGER row | Use OP-GB; it is committed pass-6 source evidence with byte-identity/full-lib proof and scoped timing, not final ship or fairness-ledger proof |
| Treat frame-batched SmolVLM2 SigLIP or the untied `lm_head` lever as broad VQA/product quality | Use OP-BG/OP-GB; `forward_frames_batched` is byte-identical source evidence, and `4291807` certifies the head lever with `FOCR_GOT_INT8_LMHEAD=0` kill switch, but public VQA benchmark/general quality remain separate |
| Treat `9b2a03b` SmolVLM2 `SmolStatics` caching as `bd-av64.10` closure, public VQA quality, release readiness, or a formal PERF_LEDGER row | Use OP-GB; it is committed pass-8 source evidence with byte-identical describe proof, lib-green proof, and scoped hydrate timing only |
| Treat A11 zoo decode-per-token ratios as end-to-end speedups, current final rows, or dense-batch proof | Use OP-GB; `3.37x` GOT-OCR2, `2.58x` OneChart, and `1.67x` SmolVLM2 are historical matched-thread Apple SDOT decode-per-token PERF_LEDGER rows, while e2e rows may include slower totals; prefer `efd83e8` final rows for `bd-av64.10` closeout claims |
| Treat conformal ratchet, Ville e-process monitors, or the capacity certificate as release approval | Use OP-SG/OP-CM; they are separate release-evidence instruments and must still be folded through the scorecard and capstone status |
| Treat `--split-spreads` as a truth guarantee | Use OP-PS; it is a PDF extraction heuristic for wide scanned spreads and can false-split |
| Treat old `staff_detection` / `staff_result` acceptance wording as emitted TrOMR event names | Use OP-TM; current `bd-av64.2` source emits additive schema-v1 `staff` events and JSON `staves` after `8af3887`; the old names are stale bead wording |
| Treat `FOCR_QKV_FUSED` as opt-in/default-off | Use OP-BG/OP-GB; after `98cc790` / `5474ae0` / closed `bd-241s`, fused q/k/v decode is default-on and `FOCR_QKV_FUSED=0` is the compatibility/profiling kill switch |
| Treat stale `bd-wp8.2.2` wording as stronger than current robot-schema source/tests | Use OP-VG; source at/after `adb4ee6` includes additive `staff` in the frozen schema fixture and advertised-events assertion, but tracker closure still needs a focused source/test/Beads check |
| Treat missing robot `threads` as intended current behavior | Cite OP-OE/OP-RP; `bd-223.2` is closed in current source, so missing `threads` usually means pre-`bd-223.2` or stale binary/source |
| Treat old `runs` / `sync` stubs as current-source failure | Cite OP-RS plus OP-SQ; `bd-223.4` is closed in current `main`, but installed binaries may still be pre-feature or stale |
| Treat `robot selftest` as a speed claim | Use OP-BG/OP-GB: selftest is parity; throughput needs gauntlet/PERF_LEDGER evidence |
| Treat `robot selftest.models` as a model-quality or TrOMR int8-compute claim | Use OP-BG; it is per-model int8 GEMM parity for registered int8 decoder families. TrOMR is absent because its current int8 artifact is consumed through f32 dequant-on-access, not an int8 decoder kernel |
| Treat dense batch closure as universal speed proof | Use OP-BS/OP-GB; `bd-3jo6.1.7.5` is closed for lossless GOT/SmolVLM2/OneChart dense batching, but broad `lm_head`, fairness rows, and decode-heavy B>=8 throughput are still follow-ups |
| Treat a closed harness bead as universal model quality | Use OP-VG: record the exact model, fixture, skip/native-path status, and remaining corpus/perf gates |
| Treat deterministic e2e output as an oracle-quality proof | Use OP-DG: determinism proves same input -> byte-identical output, not correctness against the HF/Python oracle |
| Add fixtures without manifest/provenance | Update `tests/fixtures/MANIFEST.toml` and `tests/fixtures/PROVENANCE.md`; `scripts/check_fixture_manifest.py` must pass |
| Treat conformance-matrix coverage as a universal release pass | Use OP-CM: `bd-re8.12` accounts SPEC clause coverage and XFAIL discipline; differential/golden/metamorphic/release-cert beads still need their own evidence |
| Treat differential rows, metamorphic relations, and goldens as interchangeable | Use OP-DF, OP-MR, or OP-GA according to the claim; each has different comparators, skip rules, and failure modes |
| Compare an oracle to itself in a differential gate | Use OP-DF; require `EngineIdentity` subject != oracle plus row fields for scope/oracle/module/max_diff/within_tol/xfail/disc |
| Assert a multi-page concat/sum equality relation | Use OP-MR; R-SWA makes multi-page parsing cross-page dependent, and MR-5 is gated rather than replaced by a false sum relation |
| Treat `--multi-page` as independent page parsing or per-page layout JSON | Use OP-MP; current `ocr-batch --multi-page` and PDF `ocr --multi-page` produce one cross-page markdown document with `<PAGE>` separators; streaming `page` events are now additive progress events for robot PDF multi-page, not per-page layout boxes |
| Treat `focr convert --arch aarch64-smmla` as performance proof or x86 packed-kernel proof | Use OP-AP/OP-BG; `bd-2mo.3` closes real offline SMMLA panel packing and loader fallback, not performance or x86 packed-consuming kernels |
| Treat model-gated skip-with-SUCCESS as green model evidence | Record it as unarmed/missing-artifact evidence; cite only always-on contract pieces unless the real artifact path ran |
| Auto-bless goldens or set `UPDATE_GOLDENS=1` in CI | Use OP-GA; review `.actual` diffs manually, keep transient outputs gitignored, and restamp provenance only after review |
| Treat a ladder scorecard with `skipped_no_model=true` as all-green proof | Use OP-LS; skipped rungs are visible missing-artifact evidence, not a green ladder |
| Treat an unarmed ladder scorecard as all-green proof | Use OP-LS; `bd-re8.19` is closed-current, but `skipped_no_model=true` / `all_green=false` still means missing model evidence |
| Let a broad stale source comment override concrete handlers/tests | Use OP-LC; classify from command handlers, help, tests, and Beads before editing docs |

## Reference Index

Open only what the task needs:

| Need | Reference |
|------|-----------|
| CLI commands and examples | [CLI.md](references/CLI.md) |
| Live source/help/Beads contract classification | [OPERATORS.md](references/OPERATORS.md#op-lc-live-contract-probe) |
| Embedding `franken_ocr` in Rust | [LIBRARY.md](references/LIBRARY.md) |
| Robot NDJSON and automation | [ROBOT.md](references/ROBOT.md) |
| Models, `.focrq`, env vars | [ARTIFACTS-AND-ENV.md](references/ARTIFACTS-AND-ENV.md) |
| Architecture and module map | [ARCHITECTURE.md](references/ARCHITECTURE.md) |
| Operator runbooks for tricky model claims | [OPERATORS.md](references/OPERATORS.md) |
| Model-zoo pull manifest and named model distribution | [OPERATORS.md](references/OPERATORS.md#op-zm-zoo-manifest-pullability) and [ARTIFACTS-AND-ENV.md](references/ARTIFACTS-AND-ENV.md) |
| Doctor diagnostics/repair contract | [OPERATORS.md](references/OPERATORS.md#op-dr-doctor-repair-contract) and [CLI.md](references/CLI.md#doctor) |
| Robot triage and agent ergonomics | [OPERATORS.md](references/OPERATORS.md#op-rt-robot-triage-and-agent-ergonomics) and [ROBOT.md](references/ROBOT.md) |
| Release-readiness scorecard and ship gate | [OPERATORS.md](references/OPERATORS.md#op-sg-release-ship-gate) and [VERIFICATION.md](references/VERIFICATION.md#release-readiness-scorecard) |
| PDF page selection and spread splitting | [OPERATORS.md](references/OPERATORS.md#op-ps-pdf-page-selection-and-spread-splitting) and [CLI.md](references/CLI.md#ocr-command) |
| Multi-page cross-page parsing | [OPERATORS.md](references/OPERATORS.md#op-mp-multi-page-cross-page-parsing) and [CLI.md](references/CLI.md#multi-page-cross-page-ocr) |
| Backend/SIMD/perf claim triage | [OPERATORS.md](references/OPERATORS.md#op-bg-backend-and-simd-claim-guard) and [MODEL-AND-KERNELS.md](references/MODEL-AND-KERNELS.md#simd-and-backend-dispatch) |
| Offline arch-specific prepacking | [OPERATORS.md](references/OPERATORS.md#op-ap-arch-specific-prepack-boundary) and [MODEL-AND-KERNELS.md](references/MODEL-AND-KERNELS.md#simd-and-backend-dispatch) |
| Verification infrastructure and model-gated gates | [OPERATORS.md](references/OPERATORS.md#op-vg-verification-gate-reality) and [VERIFICATION.md](references/VERIFICATION.md) |
| Determinism gates and fixture provenance | [OPERATORS.md](references/OPERATORS.md#op-dg-determinism-and-fixture-governance) and [VERIFICATION.md](references/VERIFICATION.md) |
| Conformance matrix, SPEC coverage, and XFAIL discipline | [OPERATORS.md](references/OPERATORS.md#op-cm-conformance-matrix-and-xfail-discipline) and [VERIFICATION.md](references/VERIFICATION.md) |
| Oracle-differential comparator, ULP rows, and XFAIL divergences | [OPERATORS.md](references/OPERATORS.md#op-df-differential-oracle-comparator) and [VERIFICATION.md](references/VERIFICATION.md#differential-metamorphic-and-golden-suites) |
| Metamorphic relation checks and false-relation avoidance | [OPERATORS.md](references/OPERATORS.md#op-mr-metamorphic-relations) and [VERIFICATION.md](references/VERIFICATION.md#differential-metamorphic-and-golden-suites) |
| Golden artifact update discipline | [OPERATORS.md](references/OPERATORS.md#op-ga-golden-artifact-discipline) and [VERIFICATION.md](references/VERIFICATION.md#differential-metamorphic-and-golden-suites) |
| Ordered L0-L5 ladder scorecard receipt | [OPERATORS.md](references/OPERATORS.md#op-ls-ladder-scorecard-runner) and [VERIFICATION.md](references/VERIFICATION.md#ladder-scorecard-runner) |
| Run history, `FOCR_RUN_STORE`, and JSONL sync claims | [OPERATORS.md](references/OPERATORS.md#op-rs-run-store-and-sync) and [CLI.md](references/CLI.md#run-state-and-sync) |
| Model architecture, kernels, and quant policy | [MODEL-AND-KERNELS.md](references/MODEL-AND-KERNELS.md) |
| `.focrq` format details | [FOCRQ.md](references/FOCRQ.md) |
| Verification, parity, gauntlet, and tests | [VERIFICATION.md](references/VERIFICATION.md) |
| Source-development workflow | [DEVELOPMENT.md](references/DEVELOPMENT.md) |
| Current Beads/BV reality | [BEADS-REALITY.md](references/BEADS-REALITY.md) |
| Failure diagnosis | [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) |
| Research notes behind this skill | [RESEARCH.md](references/RESEARCH.md) |

## Validate This Skill

```bash
.claude/skills/focr/scripts/validate.py .claude/skills/focr
```
