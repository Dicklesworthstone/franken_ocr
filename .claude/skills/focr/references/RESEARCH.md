# Research Notes for the focr Skill

This file is append-only historical context. Do not use it as the first source
for current capability answers. Resolve current truth through `SKILL.md`,
source/help/tests, and live Beads first; use this file only to understand why
older wording existed or what evidence a prior refresh inspected.

## Table of Contents

- [Sources Inspected](#sources-inspected)
- [Skill Design Decisions](#skill-design-decisions)
- [Ground Truth Findings](#ground-truth-findings)
- [CASS Findings](#cass-findings)
- [Known Caveats](#known-caveats)
- [Update Discipline](#update-discipline)

## Sources Inspected

Skill models:

- `ntm`
- `cass`
- `beads-br`
- `beads-bv`
- `sc`
- `sw`
- `operationalizing-expertise`

Project sources:

- `~/projects/franken_ocr/AGENTS.md`
- `~/projects/franken_ocr/README.md`
- `~/projects/franken_ocr/CHANGELOG.md`
- `~/projects/franken_ocr/Cargo.toml`
- `~/projects/franken_ocr/install.sh`
- `~/projects/franken_ocr/install.ps1`
- `~/projects/franken_ocr/models/manifest.json`
- `~/projects/franken_ocr/src/cli.rs`
- `~/projects/franken_ocr/src/lib.rs`
- `~/projects/franken_ocr/src/error.rs`
- `~/projects/franken_ocr/src/robot.rs`
- `~/projects/franken_ocr/src/pdf.rs`
- `~/projects/franken_ocr/src/dist.rs`
- `~/projects/franken_ocr/src/native_engine/`
- `~/projects/franken_ocr/src/native_engine/model_arch.rs`
- `~/projects/franken_ocr/src/native_engine/got.rs`
- `~/projects/franken_ocr/src/native_engine/decoder_qwen2.rs`
- `~/projects/franken_ocr/src/native_engine/batch_scheduler.rs`
- `~/projects/franken_ocr/src/quant/`
- `~/projects/franken_ocr/src/quant/focrq.rs`
- `~/projects/franken_ocr/src/simd/`
- `~/projects/franken_ocr/src/preprocess/`
- `~/projects/franken_ocr/src/tokenizer/`
- `~/projects/franken_ocr/src/tokenizer/ops.rs`
- `~/projects/franken_ocr/src/quant/convert.rs`
- `~/projects/franken_ocr/docs/focrq-format.md`
- `~/projects/franken_ocr/docs/zoo/got-ocr2-spec.md`
- `~/projects/franken_ocr/docs/zoo/smolvlm2-spec.md`
- `~/projects/franken_ocr/docs/zoo/onechart-spec.md`
- `~/projects/franken_ocr/docs/zoo/tromr-spec.md`
- `~/projects/franken_ocr/docs/DISCREPANCIES.md`
- `~/projects/franken_ocr/docs/PERF_LEDGER.md`
- `~/projects/franken_ocr/tests/fixtures/tokenizer_got/expected.json`
- `~/projects/franken_ocr/tests/fixtures/got/cer/README.md`
- `~/projects/franken_ocr/scripts/got_cer.py`
- `~/projects/franken_ocr/artifacts/perf/bd-3bom/README.md`
- `~/projects/franken_ocr/artifacts/perf/bd-2dlz/README.md`
- `~/projects/franken_ocr/tests/cli_robot_golden.rs`
- `~/projects/franken_ocr/tests/e2e_recognize.rs`
- `~/projects/franken_ocr/tests/installer_e2e.sh`
- `~/projects/franken_ocr/.claude/skills/focr/`
- `~/projects/franken_ocr/scripts/smolvlm2_convert_e2e.sh`
- `~/projects/franken_ocr/scripts/spec_gate_e2e.sh`
- `~/projects/franken_ocr/scripts/gauntlet_runbook.sh`
- `~/projects/franken_ocr/scripts/gen_smolvlm2_vqa_fixtures.py`
- `~/projects/franken_ocr/scripts/smolvlm2_describe_e2e.sh`
- `~/projects/franken_ocr/tests/fixtures/smolvlm2/vqa_fixtures.json`
- `~/projects/franken_ocr/artifacts/perf/bd-1e9n/validation_summary.txt`
- `~/projects/franken_ocr/artifacts/perf/bd-1azu.36/`
- `~/projects/franken_ocr/artifacts/perf/bd-re8.17/arch.json`

Live update note: `v0.3.0` contains the output/layout/resolver work; `f0a538b`
promotes the model-zoo, figure-extraction, PDF, doctor, run/sync, and
SAM-speedup wave into the `v0.4.0` public release boundary. `bf28fd7` later
bumps the Rust package and pushed tag to `v0.5.0`; the `v0.5.0` GitHub release
is now live with five platform binaries plus SHA256 files, while README/manual
examples and installer fallback constants can still mention `v0.4.0`. The
`v0.6.0` release is now live at tag `29516b9`, published
2026-07-08T14:47:48Z with platform assets. The July 8 source probe now sees
`origin/main` at `592426c`, with clean source describing as
`v0.6.0-4-g592426c` and no tracked source diff. Separate current-source
evidence through `592426c` from the installed binary actually on PATH and from
release assets. Dirty diffs after that revision are live-WIP only until
committed. `efd83e8` remains the formal `bd-av64.10` G2 closeout; `3f2f97e`
adds `bd-2mo.26` gauntlet harness fixes for AppleDouble source scans, current
SAM/CLIP timing labels, and `OUT_DIR` evidence homes; `4291807` certifies the
SmolVLM2 untied-head int8+refine runtime path; `c248e6d`/`c4c1684` close the
head-to-head gauntlet rows; `c29a78b`/`7c7bd00` close release certification;
`beaed7c` records CI/dist supplement notes; `db02421` refreshes README
release-readiness evidence; and `5df6395` commits post-certification fuzz
corpus growth. `592426c` then refreshes README public `v0.6.0` release identity,
binary-size, manual-download, and CPU-backend prose without changing runtime
behavior. Verify the
installed binary, release object/assets, and installer path before treating a
feature as present or absent. Commits
`9995d07` and `dfd5a38` add the Model Zoo roadmap.
Later `main` commits add
`focr models`,
`.focrq` `model_id`, GOT-OCR2 conversion/tokenizer/vision/decoder/KV-cache/e2e
source pieces. Newer commits make GOT-OCR2 ready/implemented, add
`focr pull got-ocr2` via `models/manifest.json`, add B8 L5 CER evidence, land
the bd-3bom lm-head pretranspose win, and then land bd-2dlz as the current GOT
decode default: int8+top-K-refine head, parallel attention, and last-row seed.
That was later superseded for formatted output by the B10 `--format` CLI lane;
task aliases later landed as `focr ocr --task` convenience routing while
task-specific subcommands remain absent.
2026-07-01 refresh: `origin/main` had no new code commits beyond `4c072ce`
after fetch; only model release tags were new. The refresh re-read README,
`src/cli.rs`, `src/lib.rs`, `src/dist.rs`, `src/native_engine/`, and env
censuses, sampled `br` JSON output, and searched CASS for `focr --json` and
`focr error` with no useful hits. It also re-applied `sc`, `sw`, and
`operationalizing-expertise` to strengthen the skill as an executable operator
manual instead of only a feature inventory.
2026-07-01 B10 refresh: source advanced to `cbe82bf` after `fcaaa30`, shipping
`focr ocr --format` for GOT-OCR2 structured Mathpix-Markdown output. The flag is
a boolean on `OcrRequestArgs`, also available through `robot run` and env
`FOCR_GOT_FORMAT`; it sets a process-global `native_engine::force_got_format`
read by `forward_got`. It is a no-op for default `unlimited-ocr`. At that
snapshot, task routing and specialized accuracy were follow-ups; later July 2
source closed the `--task`, no-repeat, and phase-1 corpus gaps described below.
2026-07-02 refresh: source advanced to `0418b03` on `main`. At that historical
snapshot, `focr ocr --task` had landed (bd-3jo6.1.5), specialized tasks implied
GOT format mode and required `got-ocr2`, and `--task describe` still returned
SmolVLM2 NotImplemented. The July 3 C7/C9 note below supersedes that describe
status.
`bd-ff4i` closed with GOT global `no_repeat_ngram=20` plus
`FOCR_GOT_NO_REPEAT_NGRAM`, and `bd-3j3p` closed the decode-tuning flag wiring
including `--max-length`/`FOCR_MAX_NEW_TOKENS`. `bd-30me` added
`FOCR_RESAMPLE=pil-bicubic` as a Pillow-bit-exact reference resampler while the
default remains CatmullRom under DISC-001. `bd-31vc` resolved the f32 oracle
ladder red rungs: armed L0-L5 rerun went 44/44 green, L4 token-exact 1.0, L5
CER 0.0. `bd-1azu.10` and `bd-1azu.14` closed batched vision and the spine
one-live-forward watchdog. `bd-3jo6.1.6` added `TokenizerOps`, and
`bd-3jo6.1.7` added shared dense decoder GQA support. At that snapshot,
SmolVLM2/OneChart/TrOMR census/spec docs had landed, but their registry rows
still remained planned.
2026-07-02 fresh-eyes follow-up: source commits `11e1d45`, `db9f32b`, and
`aa05851` tightened several operator-critical edges. `FOCR_BATCH_SPINE` is now
value-parsed (`0`/`off`/`false`/`no` disables), non-default archs route
sequentially even when the env is armed, and `CONTROL_CORRECTION.md` records the
fixed control. Present-but-broken parity artifacts now fail instead of green
xfail. GOT no-repeat precedence is CLI/`FOCR_NO_REPEAT_NGRAM` >
`FOCR_GOT_NO_REPEAT_NGRAM` > config, and `ocr-batch` reads
`FOCR_NO_REPEAT_NGRAM`. Convert now verifies both tied and untied lm-head claims;
GQA divisibility is enforced at forward entry. The gauntlet refuses stale raw
dirs and same-stage rows across different pages. `bd-2dui` tracks the minor
remaining caveats, especially spec-loop engagement telemetry.
No useful CASS hits were available during skill creation/update. Source, tests,
README, AGENTS, plan, and docs were treated as authority.

## Skill Design Decisions

From `sw` and `sc`:

- Keep `SKILL.md` as a concise entrypoint.
- Put depth in first-level `references/`.
- Include validation.
- Make trigger language explicit in frontmatter.

From `cass`:

- Include health/staleness handling instead of blindly requiring a rebuild.
- Treat historical search as useful but secondary to live source.

From `beads-br` and `beads-bv`:

- Use JSON/robot-safe command forms.
- Never run bare `bv`.
- Explicitly sync Beads after tracker mutation.

From `ntm`:

- Prefer action cards and recovery operators that an agent can execute under
  pressure.

From `operationalizing-expertise`:

- Encode expert moves as named operators.
- Require evidence ledgers for adaptive/lossy decisions.
- Separate deterministic fallback from experimental policy.
- Each operator should expose explicit triggers, failure modes, a reusable
  prompt module, canonical tags, and source/evidence anchors so a fresh coding
  agent can choose the right move without rereading the whole skill.

## Ground Truth Findings

2026-07-07 refresh at `franken_ocr` source head `e3d2a71`:

- `bd-2z0y` is closed. PDF `focr ocr doc.pdf --multi-page` now has both the
  shipped PDF route and the streaming half: robot-mode PDF multi-page emits
  additive schema-v1 `page` decoded progress events through
  `robot::page_decoded_event` at `<PAGE>` boundaries with `status:"decoded"`,
  `page`, `chars`, and raw `text`. These events are not layout boxes, figure
  extraction, or split-spread support.
- Multi-page preprocessing is reference-faithful 640x640 squash, not
  aspect-preserving Base padding. `preprocess_dynamic_squash` uses
  PIL-faithful bicubic at this multi-page site; the L5 oracle work caught the
  older pad-vs-squash and resize-kernel mismatch.
- `bd-1gv.26` is closed for the 2-page L5 multi-page oracle rung:
  `727701b` adds `l5_multi_page_matches_infer_multi_oracle`, with
  `ngram_window=1024`, deterministic-plate exactness, and CER budget evidence.
- `bd-1465` is now closed for scoped long-horizon evidence: `3201e8c` adds the
  10-page `l5_multi_page_10p_long_horizon` rung with fixture `p10`, subject cap
  7600, plate exactness, markers 8-vs-9, and CER 0.4045 <= 0.50; `e1332a7`
  freezes `p20` and records the forced conclusion that the reference itself
  collapses at 20 pages, so a 40-page CER gate is not meaningful.
- `bd-av64.4` is closed as experimental TrOMR barline split rescue:
  `64edce3` adds `staff_detect::barline_columns`, `recognize_split`, and
  `FOCR_TROMR_SPLIT=1` for over-budget staff bands. The route is off by
  default and measured-not-lossless: isolated segments are OOD, continuation
  pitch registration can drift, rhythm agreement was about `0.2`, and pixel
  clef prepend made the target case worse. Use it as recognition-count rescue,
  not as default barline quality, camera dewarp, broad SER, int8, or perf proof.

2026-07-07 refresh at `franken_ocr` source head `750a69a`:

- `bd-1gv.25` is closed for true Unlimited-OCR multi-page cross-page parsing.
  `4afcaca` adds `recognize_multi_page`, `f115403` adds
  `focr ocr-batch --multi-page`, and `b9cc16c` proves the real-model e2e gate
  when armed. The route is not a loop over independent pages: in current source
  it uses `preprocess_dynamic_squash` 640x640 page tensors, 111 placeholders per
  page, one cross-page prompt/decode, `ngram_window=1024`, and `<PAGE>`
  separators.
- `bd-2z0y` has the PDF half shipped at `a2dd1c9` and documented by
  `750a69a`: `focr ocr doc.pdf --multi-page` composes with `--pages` and
  refuses `--split-spreads` / `--extract-figures`. This historical note was
  superseded later the same day by `828ea4c` / closed `bd-2z0y`, which shipped
  streaming decoded page progress events.
- `bd-2mo.3` / `bd-2mo.3.1` are closed for offline SMMLA prepacking.
  `focr convert --arch aarch64-smmla` emits real `[2x8]` panels through
  `src/simd/pack.rs`, the artifact records `arch_target`, SMMLA dispatch can
  consume panels without runtime shuffle, and non-SMMLA dispatch un-permutes
  with a warning/fallback. VNNI/AMX remain tag-only and the evidence is not a
  speed claim.
- `bd-3jo6.1.12` is closed after `ad3ad20`/`adb4ee6`: robot selftest has
  per-model rollups for `unlimited-ocr`, `got-ocr2`, `smolvlm2`, and
  `onechart`; TrOMR stays absent because its published int8 artifact is
  storage-only and runtime dequants through f32 accessors. The e2e golden
  proves 44/44 cases green on scalar, sdot, and smmla.
- The `staff` robot-schema fixture drift described in older notes is
  source-fixed at `adb4ee6`: `tests/fixtures/robot_schema_v1.json` and the
  advertised-events assertion include additive `staff`. `bd-wp8.2.2` may still
  read as stale tracker state, so source/tests and tracker closure must be
  checked separately.
- `bd-av64.6` is closed for real-scan corpus-v1 measurement, not for broad
  note-level SER, 10-20 item expansion, GOT cross-reference, ladder scorecard
  row, or aggregate SER.
- `bd-av64.14` is closed for fit-first geometry and Cadwallader p169 5/5
  acceptance, not for camera dewarp, default/lossless barline quality, TrOMR
  int8, perf, or broad note-level SER. Later `bd-av64.4` adds experimental
  split rescue only.

Current source showed:

- two binaries, `focr` and `franken_ocr`, both through `cli_main()`,
- synchronous/blocking `OcrEngine`,
- model cache and runtime ownership inside the library,
- CLI surfaces for `ocr`, `ocr-batch`, `convert`, `pull`, `robot`, `runs`,
  `sync`, and `doctor`,
- current `focr pull` for packaged int8 `.focrq` + tokenizer acquisition,
- `focr pull` installs `unlimited-ocr.int8.focrq`, and the resolver probes
  quant-suffixed names so a fresh pull resolves without `--model`,
- `focr pull got-ocr2` selects `models.got-ocr2` from the manifest and installs
  `got-ocr2.int8.focrq` plus `qwen.tiktoken`,
- `focr ocr -o/--output FILE` writes markdown or structured JSON; `.json`
  selects JSON and `--json` forces JSON,
- structured JSON carries `schema_version`, `markdown`, and `layout` for images
  or `pages` for PDFs; boxes are `[x1,y1,x2,y2]` source-image pixels,
- layout-aware library APIs export `RecognizedDocument`, `LayoutSpan`,
  `recognize_with_layout`, and `recognize_dynamic_with_layout`,
- figure extraction exports `ExtractedFigure`, `recognize_with_figures`, and
  `recognize_dynamic_with_figures`; the CLI writes figures only after a successful
  recognition and rewrites markdown/JSON references,
- current `robot selftest` for SIMD/int8 kernel parity,
- current `focr models` discovery over the static model registry; no weights
  required,
- `unlimited-ocr` and `got-ocr2` are ready/implemented registry rows in that
  source; at this older snapshot SmolVLM2, OneChart, Polyphonic-TrOMR, TrOCR,
  and pix2tex still appeared as planned descriptors, but later SmolVLM2 findings
  below supersede that with C2/C5/C6 proof and then implemented describe routing,
- `focr ocr --task` is current CLI: `formula`, `tables`, `chart`,
  `molecular`, `geometry`, and `music` imply GOT format mode and require a GOT
  model; in that older snapshot `describe` still returned NotImplemented,
- `--crop-mode base` is the default certified single-view path; `gundam` is
  reference dynamic tiling but needs fresh e2e certification before parity
  claims,
- `focr ocr-batch --f32` is the current f32 decode flag; single-page f32 work
  uses safetensors through `FOCR_MODEL_PATH`, not an `ocr --f32` flag,
- `.focrq` format v1 metadata can carry `model_id`; absent/empty defaults to
  `unlimited-ocr`, unknown ids are refused, and license notice is validated per
  registered model,
- `focr convert --model-id got-ocr2` is source-supported for GOT-shaped weights;
  it writes `model_id`, the GOT Apache notice, and omits tied `lm_head.weight`,
- Before GOT-OCR2 was promoted to ready, its pieces included Qwen
  `qwen.tiktoken` loading, prompt building, GOT preprocess, SAM-prefix vision,
  `mm_projector_vary`, Qwen2 dense decode, O(n) KV-cache greedy decode, and an
  env-gated e2e golden test,
- the normal GOT engine path defaults to `format=false`; `--format`,
  `FOCR_GOT_FORMAT`, and `native_engine::force_got_format(true)` select the
  `.mmd`/layout-style `OCR with format: ` prompt for GOT-OCR2,
- GOT-OCR2 now uses the upstream-style global no-repeat n-gram guard by
  default (`20`), with `FOCR_GOT_NO_REPEAT_NGRAM` as the override,
- B11 GOT e2e runs the full preprocess -> vision -> splice -> KV-cache decode ->
  tiktoken path on `tests/fixtures/got/sample_text.png` under `FOCR_GOT_MODEL`
  and `FOCR_GOT_TIKTOKEN`,
- B8 GOT L5 evidence reports 2.5% CER and 5.4% WER on a real scanned book page
  via `scripts/got_cer.py`,
- bd-3bom performance evidence reports GOT page time 487 s -> 41 s and decode
  throughput 1.4 -> 23.9 tok/s with output identical, by precomputing the tied
  `lm_head` transpose once rather than quantizing it,
- bd-2dlz performance evidence makes the follow-on GOT decode lane current:
  `FOCR_GOT_INT8_LMHEAD` is default-on int8 lm_head + top-256 f32 refine,
  `FOCR_GOT_SEQ_ATTN=1` forces serial attention, and the default parallel
  per-head attention plus last-row seed moved page_0107 from 35.93 s -> 16.6 s
  and decode from 23.9 -> 64.7 tok/s with 688/688 byte-identical tokens,
- bd-2dlz left two explicit follow-ups: `bd-e4yr` for x86-prefill/qkv SIMD
  routing and `bd-34zu` for decode Mat/Vec scratch-buffer reuse,
- `FOCR_RESAMPLE=pil-bicubic` is implemented as the reference-exact Pillow
  BICUBIC comparison mode (370/370 randomized differential cases); default
  CatmullRom remains an accepted/current product divergence under DISC-001,
- `FOCR_BATCH_SPINE` is opt-in for int8 `ocr-batch` and value-parsed; `0`/`off`
  disables, `FOCR_BATCH_VISION` defaults on inside the armed spine and has a
  `0`/`off` kill-switch; `bd-1azu.14` reports the one-live-forward/no-lock-held
  sweep green with a later `CONTROL_CORRECTION.md`,
- `8497080` added dense batch primitives for `bd-3jo6.1.7.5`; `cf0b037` routed
  GOT-OCR2 `ocr-batch` through `recognize_batch_dense_got` /
  `got::recognize_batch`; `4ca1577` broadened the dense route through
  `OcrModel::recognize_batch_dense` for GOT-OCR2, SmolVLM2, and OneChart; and
  `fdd1d64` closed the bead for `v0.4.0`. Current source has
  `smolvlm2::recognize_batch`, `onechart::recognize_batch`,
  `generate_greedy_batched` with per-stream `caps: &[usize]`,
  `PageStream::with_max_emit`, `FOCR_BATCH_PACK`, and `FOCR_BATCH_SIZE` using
  128 as the default with a 256 cap. The proof is lossless at four levels, with
  scoped throughput evidence: SmolVLM2 1.32x, OneChart 1.27x, and GOT still
  vision-dominated. Broad batched `lm_head`, final A11/PERF_LEDGER rows, and
  decode-heavy B>=8 rows remain follow-ups,
- `bd-av64.10` now has measurement plus eight optimization/triage passes, three
  measured negatives, and a formal `efd83e8` G2 closeout:
  artifact hydration is not the GOT e2e tax, `01f07fe` landed the first SAM
  attention speedups, `f3d3215` added head-parallel global attention, and
  `0298651` pre-transposed CLIP linears plus cached hydrated `ClipWeights`
  (`vision.clip` 2.49s -> 0.77s steady-state on page_0009). `f65fded` later lands pass 4
  by moving shared SAM-style linears to validating `Linear::from_row_major`
  hydration with cached GEMM-ready matrices across SAM/GOT/OneChart/SmolVLM2/
  SigLIP/TrOMR consumers; `3c1b1ea` records the pass-3/pass-4 Beads evidence
  while `efd83e8` is the later formal closeout. `ab6e083` measures and reverts the
  SIMD/polynomial-exp softmax idea as negative evidence; `50d5dad` adds the
  `artifacts/perf/bd-av64.10-simd-exp/` pointer. `f1ac972` lands SmolVLM2
  SigLIP frame batching via `forward_frames_batched` with `FOCR_SIGLIP_SEQ=1`
  as the sequential kill switch and byte-identical test coverage; `3f2878d`,
  `38ab806`/`a9a406e`, and `9b2a03b` add GOT/OneChart/SmolVLM2 statics caches.
  `efd83e8` lands the final rows: GOT e2e `0.624 -> 0.885`, OneChart
  `0.546 -> 0.755`, SmolVLM2 `0.878 -> 0.890`, and decode-per-token
  `3.046x / 2.249x / 1.499x`. The bead is closed as measured final state, not
  as an e2e `>=1.0x` win,
  while public `c5e535a` / `8bd4037` / `b757bc0` record row-tiled SAM global
  attention as byte-identical but slower on the measured Apple-Silicon regime,
  with durable evidence in `artifacts/perf/bd-av64.10-rowtile/`,
- July 7/8 pass 6: `3f2878d` promotes GOT `GotStatics` from the earlier
  checkout-only follow-on to public current-main source. The cache lives on
  `OcrModel`, covers the SAM tower, `mm_projector_vary`, and widened embed table
  for sequential and batch GOT paths, and reports about 0.8s/page saved on the
  cited 2-page loop; keep it self-relative until formal G2/PERF_LEDGER rows
  land,
- July 8 `v0.5.1` tag/release split: `8de3674` bumps Cargo/tag to `v0.5.1` and
  commits the `memmap2` dependency; the GitHub release object and platform
  assets now exist, while later source commits still need their own evidence.
  `507cebe` then ships the mmap-backed `Weights::load` half with
  `FOCR_NO_MMAP`, mapped/owned fallback, and byte-identity proof; `0401df2`
  records that `bd-2mo.22` still has alignment/buffer-reuse/mimalloc work.
  `38ab806` / `a9a406e` add committed OneChart `OnechartStatics` /
  `onechart.hydrate(cached)` pass-7 evidence. `9b2a03b` adds committed
  SmolVLM2 `SmolStatics` / `smolvlm2.hydrate(cached)` pass-8 evidence, with
  byte-identical describe output, lib green, and Beads comment 91. `8d6601d`
  documents that reuse in README, `4cedacd` bumps/tags source to `v0.5.2`,
  the `v0.5.2` GitHub release object is real historical release-asset evidence
  with platform assets, and `0924479` clarifies the README split between source
  tag and binary release without changing source/runtime behavior,
- July 7 refresh: `98cc790` / `5474ae0` make `FOCR_QKV_FUSED` the default-on
  int8 decode path under closed `bd-241s`; use `FOCR_QKV_FUSED=0` only as the
  old three-call kill switch. The accepted evidence is the `bd-1waa` kept-win
  ledger, `bd-3pg7` attention-GEMM rollback/resolution, page_0590 SHA/CER
  identity, 20-page CER equality, and page_0009 controlled decode speed
  `0.072 -> 0.052 s/tok`. That same older refresh observed `bd-wp8.2.2`
  schema-fixture drift after additive robot `staff` events; the later
  `adb4ee6` source refresh fixed the `staff` fixture/assertion and `0b74af0`
  added `music_warning` to the same frozen schema-v1 inventory, so treat this
  as superseded historical context unless the live focused tests fail again,
- `TokenizerOps` is the shared tokenizer trait over byte-level BPE and Qwen
  tiktoken; shared dense decoder GQA support is present for SmolVLM2-style
  key/value head grouping,
- GOT runtime requires `qwen.tiktoken` adjacent to the `got-ocr2` `.focrq`; the
  packaged `focr pull got-ocr2` path installs both, while self-converted
  artifacts need manual tokenizer placement,
- native scanned-PDF routing through `src/pdf.rs`,
- exported dynamic-image APIs and `franken_ocr::pdf::PdfPages`,
- robot schema version 1 event names,
- stable user-facing exit codes,
- `.focrq` format version 1 and `FOCRQ\0` magic,
- the live runtime quant preference env is `FOCR_QUANT`; trust the source
  constant over older prose when they disagree,
- `convert --quant int8` implemented; `convert --quant int4` remains
  `NotImplemented`,
- historical snapshots had `doctor`, `runs`, and `sync` scaffolded. Current
  source supersedes that: `bd-wp8.4` / `bd-wp8.4.1` implement doctor, while
  closed `bd-223.4` / `bd-wp8.11` add fsqlite `RunStore` query/sync behavior
  that must be checked against source/help/Beads for the exact binary,
- tokenizer/preprocess/vision/connector/decoder/postprocess modules exist and
  the engine forwards through them when model accessors are satisfied,
- batch scheduler, int8/int4 quant modules, SIMD dispatch, adaptive artifacts,
  and gauntlet/parity docs exist and must be handled as phase/evidence-gated.
- current installer hardening covers `gum style --` dynamic text, fresh-account
  disk-space checks, checksum/download edge cases, and `tests/installer_e2e.sh`.
- the installer resolves the latest release and still falls back to `v0.4.0`;
  raw `main` installer text, a source Cargo version, or a pushed tag is not
  proof that the corresponding binary is installed locally. Earlier `v0.5.2`
  correction notes said the release object existed at tag `4cedacd`; the active
  July 8, 2026 boundary is now `v0.6.0` at tag `29516b9`, with public
  `origin/main` at `592426c` / `v0.6.0-4-g592426c`. Post-release source claims
  still need source/help/test evidence and should not be inferred from release
  assets alone.
- Model Zoo work (`bd-3jo6`) has promoted GOT-OCR2, SmolVLM2, OneChart, and
  TrOMR to implemented/ready runtime rows. A later distribution refresh
  (`bd-av64.7` / `ece14f9`) supersedes older manifest notes and makes
  `focr pull smolvlm2`, `focr pull onechart`, and `focr pull tromr` current
  when the running binary embeds or resolves that manifest. `efccce9` / closed
  `bd-av64.12` later makes default `focr pull tromr` install
  `tromr.int8.focrq` storage plus tokenizer sidecars, with
  `focr pull tromr --quant f32` for the `tromr.focrq` reference; this is not
  int8 compute or perf evidence. TrOCR
  and pix2tex remain planned. `focr models`, named pulls, GOT `--format`, and
  `focr ocr --task` are current, but task subcommands (`focr music`,
  `focr chart`, `focr describe`) are not current CLI features yet.
- Newer C8 work adds `scripts/gen_smolvlm2_vqa_fixtures.py`,
  `tests/fixtures/smolvlm2/vqa_fixtures.json`, and
  `vqa_quality_matches_oracle_l5` as an OQ-6/C8 informational guard: questions
  over the committed sample photo are scored against the fixture oracle's own
  greedy answers by normalized exact match or symmetric content-word containment
  >= 0.5; armed f32 needs >=70% and armed int8 needs >=50%, live close evidence
  reports 7/7 on both, and this is not a public benchmark. Current source also has
  `scripts/smolvlm2_describe_e2e.sh` as the C10 CLI e2e lane with
  `smolvlm2_describe_e2e/v1` NDJSON, model-gated skip-with-success, exit 3/2
  negative-path checks, and real int8 describe/VQA success checks.
- current env census includes development levers such as
  `FOCR_FUSE_NGRAM_LMHEAD`, `FOCR_SPEC_DECODE`, `FOCR_SPEC_VERIFY`,
  `FOCR_BATCH_PACK`, `FOCR_ATTN_GEMM`, and `FOCR_INT8_KV`; they need parity,
  token-identity, or negative-evidence treatment before being recommended.

2026-07-07 dense-batch / TrOMR observability refresh: dense zoo batching is no
longer a local-only claim after `4ca1577` and `fdd1d64`; update agents should
treat GOT-OCR2, SmolVLM2, and OneChart dense batching as current but keep speed
claims scoped to the recorded rows. TrOMR `bd-av64.2` staff observability is
also current after `8af3887` plus bead close `4e881d7`: use
`MusicPageMeta`, `OcrModel::take_music_meta`,
`OcrEngine::take_music_page_meta()`, `robot::staff_event`, robot event kind
`staff`, and `music_meta_to_json` as committed source anchors. The robot event
name is `staff`, not the older acceptance names `staff_detection` /
`staff_result`; schema remains v1 because the event kind is additive.

2026-07-07 real-scan and perf-guardrail refresh: source advanced through
`c420c28`, `af13d3e`, `60d8af4`, and `a0ad299`. `bd-av64.6` is still
in-progress but corpus v1 landed six public-domain Spohr fixtures under
`tests/fixtures/realscan_music/` and `scripts/realscan_music_gate.sh` with
`realscan_music/v1` NDJSON, tier-1 human attributes, tier-2 frozen MusicXML
anchors, and tier-3 robot-staff robustness floors. `bd-1a6h` closed with
`scripts/bench_guardrail.py` and `benches/.bench-history/baseline.json`:
frozen baselines move only via reviewed `--ratchet`, `cv_pct > 5` and posture
mismatch are ineligible, and perf reporting requires an all-green parity
receipt. `bd-2mo.24` closed as negative evidence for
`FOCR_FUSE_NGRAM_LMHEAD`: byte-identical but 16.43s -> 16.40s inside noise on
page_0023, so it stays opt-in/off unless multi-image `ngram_window=1024` ban
sets or much faster decode change the arithmetic. Phase sweep commits close
`bd-1es` and `bd-2mo.1` style status, but parent `bd-2mo` remains open with
prepacking, memory/allocator, NUMA, int8 attention, vectorized exp, and fusion
levers still separate.

2026-07-02 late refresh: source advanced to `acf45c5` on `main`. New current
facts: a project-local `.claude/skills/focr` was added; `bd-1e9n` closed the
dead preprocess-flag surface, wiring `--base-size`, `--image-size`, and
`--crop-mode` through `PreprocessOverrides` while correcting the default to
`base`; `artifacts/perf/bd-1e9n/validation_summary.txt` records default
byte-identity, a live `--base-size 512` diff, and first Gundam e2e on page_0107
(`rc=0`, 7 views, CER 0.0179 / WER 0.0138). `bd-3jo6.3.2` closed SmolVLM2 C2:
`focr convert --model-id smolvlm2` is arch-aware, verified on real weights, and
produces a 489-tensor `.focrq` census with 224 int8 decoder GEMMs and 265 F32
high-precision tensors including an untied `lm_head`; at that snapshot
SmolVLM2 forward still remained unimplemented. `bd-1azu.36` closed the LINEAR
`FOCR_SPEC_DECODE` e2e gate
with 20/20 ON==OFF byte identity in both f32 and int8 composition, while tree
verify remains parked under follow-up work. `bd-re8.17` gained a pinned HF
baseline quiet-host runbook and `artifacts/perf/bd-re8.17/arch.json` selftest
evidence (24/24 on selected `aarch64+neon+dotprod`); later same-day source
closed the bead with first PERF_LEDGER rows. CASS searches for SmolVLM2,
Gundam, and `FOCR_SPEC_DECODE` returned bounded `checkpoint_incomplete` robot
errors and were not rebuilt because source/artifacts/Beads were direct
evidence.

2026-07-02 fresh-eyes refresh: source advanced to `907653b` on `main`.
`bd-3jo6.3.5` closed C5: SmolVLM2 f32 decoder seam cos 1.000000 with all
24 generated ids token-exact; int8 seam cos 0.998301 and argmax-exact, with
kvcache==re-prefill but a known later near-tie flip recorded as DISC-002 and
the f32 path as kill-switch. This was a text-decoder seam proof, not image
describe/VQA; a later checkout note below supersedes its then-current
`model_arch` status.
`bd-re8.17` closed with first pinned HF bf16 rows for `page_0009` and
`page_0014`: decode-per-token ratios 1.614/1.619, prefill 4.787/4.990, vision
3.577/3.721, e2e 2.709/2.340, and CER 0.00943/0.03529. Do not repeat the broad
close-reason wording that focr beat every stage: the ledger row for
`page_0009` preprocess is 0.916, and a stale "No performance numbers" footer
still trails the populated table.

An intervening skill revision claimed there was no current `pull`, no native
PDF routing, no dynamic-image API, no selftest, and no working int8 conversion.
The current source and README contradict that, so this refresh restores those
capabilities and validates against stale image-only/no-pull regressions.

## CASS Findings

During the original creation/update passes, CASS did not produce useful `focr`,
`OcrEngine`, or layout-output hits. During the 2026-07-01 B10 refresh,
`cass search "FOCR_GOT_FORMAT" --robot --limit 20` produced one useful
Claude-Code session summary corroborating the Model Zoo arc: GOT-OCR2
sub-epic B reached `focr pull got-ocr2` plus `--format` structured mode, while
`bd-ff4i` and `bd-3kix` remained the immediate follow-ups. Treat CASS as
historical corroboration; source, tests, README, and Beads remain authority.
During the 2026-07-02 refresh, CASS searches for `focr --task`/GOT format,
`FOCR_BATCH_SPINE`/`FOCR_BATCH_VISION`, and `FOCR_RESAMPLE` returned bounded
`checkpoint_incomplete` robot errors. The index was not rebuilt during this
skill update because source, Beads, README, and docs already gave direct current
evidence.
During the 2026-07-02 late refresh, CASS again returned
`checkpoint_incomplete` for SmolVLM2 conversion, Gundam preprocess, and
`FOCR_SPEC_DECODE` searches. This is recorded as a historical-search miss, not
as evidence against the features.
During the 2026-07-03 SmolVLM2 feature refresh, CASS searches for tokenizer and
SigLIP history again returned bounded `checkpoint_incomplete` robot errors; the
global index was not repaired during this narrow skill commit because source,
Beads, README, and fixtures gave direct evidence.
During the later 2026-07-03 C8 VQA guard refresh, CASS searches for
`SmolVLM2 VQA fixtures focr` and `focr --task describe FOCR_SMOLVLM2_QUESTION`
also returned bounded `checkpoint_incomplete` robot errors. Source, Beads, the
new generator, and the fixture were treated as authority.

2026-07-03 SmolVLM2 feature refresh: source inspection plus Beads showed C6
tokenizer conformance closed as `bd-3jo6.3.6`. `PretokScheme` is now selected
from `tokenizer.json`; `Digits(individual_digits=true)` plus
`ByteLevel(use_regex=true)` selects `PretokScheme::SmolLm2`; special ids are
bos 1, eos 49279, pad 2, image 49190; and the real-tokenizer gate is 128/128
token-id-exact plus decode-exact over a pinned tokenizer JSON sha prefix
`5ece781d`. The same refresh found worktree source for the SmolVLM2 image
path: `vision_siglip.rs`, `token_compress.rs`, `gelu_tanh`, shared
`vision_sam` conv leaves, dev-profile `ft-kernel-cpu` opt3, and
`scripts/gen_reference_fixtures_smolvlm2_vision.py` with `sample_photo.png` and
`vision_oracle_fixtures.json`. At that earlier snapshot, live Beads had not yet
closed the SigLIP/A8/A9 proof chain, so that snapshot treated it as early seam
evidence, not shipped SmolVLM2 describe/VQA. The later note below supersedes
the live status.

2026-07-03 later refresh: source head `317ab91` and Beads superseded the
earlier C3/C4 status. `bd-3jo6.3.3`, `bd-3jo6.3.4`, `bd-3jo6.1.8`, and
`bd-3jo6.1.9` are closed. C3 certifies SigLIP post-LN parity on 13 real frames
with worst cos 1.00000000 and maxabs 4.4e-4; the fix also corrected the spec to
NaViT bucketized position ids `[0,0,1,...,30]` rather than identity. C4
certifies `pixel_shuffle` bit-exactness on the oracle post-LN seam and connector
projection cos 1.00000000, maxabs 2.59e-4 inside the 1.1e-3 measured budget.
At that intermediate seam-only point, C8 remained open and describe/VQA was not
yet a shipped route; the C7/C9 note below supersedes the route status while
keeping the broader quality/perf caveat. The 2026-07-04 C8/C10 refresh below
supersedes that open-tail status.

2026-07-03 C7 refresh: source inspection found the SmolVLM2 preprocess/prompt
lane before it closed; the later C7/C9 note supersedes this intermediate state.
`src/preprocess/mod.rs` now contains `preprocess_smolvlm2` and
`preprocess_smolvlm2_path`; they implement `SmolVLMImageProcessor` input
assembly with longest side 2048, 512-frame local tiles, row-major frame order,
and a global 512 frame. `src/preprocess/pil_resample.rs` now has
`resize_lanczos`, a Pillow-bit-exact LANCZOS path matching `resample: 1`, plus
Pillow 12.3.0 goldens generated with seed 301466 for downscale, upscale, and
squash cases. This supersedes any generic "PIL resampler" wording for
SmolVLM2: `FOCR_RESAMPLE=pil-bicubic` is still the Baidu/GOT BICUBIC reference
knob, not the SmolVLM2 LANCZOS selector. CASS searches for `focr --json`,
`focr error`, and `SmolVLM2 LANCZOS FOCR_RESAMPLE` returned bounded
`checkpoint_incomplete` robot errors in this refresh, so source/Beads/README
inspection remained the authoritative evidence path.

2026-07-03 C7/C9 refresh: source head `01c9784` closed `bd-3jo6.3.7` and
`bd-3jo6.3.9`. The checkout has `src/native_engine/smolvlm2.rs`,
`native_engine::set_smolvlm2_question`, `FOCR_SMOLVLM2_QUESTION`, CLI
`--question`, `--task describe` routing for `--model smolvlm2.int8.focrq`, and
`model_arch` `implemented=true`. The route is real and was proven live on the
int8 artifact. DISC-003 is now the C8 L4 near-tie ledger: prefill logits are
within `<5e-5`, the opt-in `FOCR_SMOLVLM2_CERT_FULL=1` O(n^2) greedy leg is
64/64 id-exact, and KV-cache first divergence must be the oracle rank-2 token
at top-2 gap <= 0.5. At this intermediate snapshot, C8 and C10 were not yet
closed for the broader caption/VQA L5 quality budget, perf, and detailed e2e
logging; the 2026-07-04 C8/C10 refresh below supersedes this.

2026-07-03 C8 VQA guard refresh: the live checkout at source head `01c9784`
plus staged work adds `scripts/gen_smolvlm2_vqa_fixtures.py`,
`tests/fixtures/smolvlm2/vqa_fixtures.json`, and the Rust
`vqa_quality_matches_oracle_l5` test in `src/native_engine/smolvlm2.rs`. The
generator uses `FOCR_SMOLVLM2_DIR`, the committed `sample_photo.png`, torch/PIL,
and transformers to store oracle greedy ids and answers for seven sky, sun,
tree, building, ocean, daytime, and scene questions. The Rust test preprocesses
the sample photo once, runs vision once per armed weight leg, asks each question
with max-new 24 and EOS 49279, and scores normalized exact match or symmetric
containment of content words >= 0.5 against the oracle answer. It requires >=70% for
f32 when `model.safetensors` is present and >=50% for int8 when
`smolvlm2.int8.focrq` is present; missing artifacts skip their leg, while no
artifact at all is not proof. Live `bd-3jo6.3.8` now reports the armed C8 gate
at 7/7 for both f32 and int8. The same current source also has
`scripts/smolvlm2_describe_e2e.sh`: a POSIX-sh C10 gate that emits
`smolvlm2_describe_e2e/v1` NDJSON, requires `FOCR_SMOLVLM2_DIR` with
`smolvlm2.int8.focrq` and tokenizer or skips-with-success, optionally uses
`FOCR_BIN`, proves missing-model exit 3 and wrong-family exit 2, then checks a
scene-ish describe answer and an affirmative sun VQA answer. Live
`bd-3jo6.3.10` now reports that the armed release-int8 C10 run passed.

2026-07-04 OneChart D2/D9 refresh: source head `8099ef0` closed
`bd-3jo6.4.2` and `bd-3jo6.4.9`. D2 added arch-aware OneChart conversion for
OPT-shaped weights: 384 source records became 383 `.focrq` records after
tied-head dedup, 72 decoder GEMMs are int8, high-precision tensors include
vision/projector/number-head/norms/biases/embeddings, `model_id=onechart` and
license are correct, and overflow proof covers K=768/K=3072. D9 added
`PretokScheme::Gpt2`, `Tokenizer::from_opt_dir/from_opt_files`, and the OPT
`vocab.json`/`merges.txt`/`added_tokens.json` loader; the armed gate is 29/29
token-id exact against slow HF `GPT2Tokenizer` fixtures with `<imgpad>` 50265,
`<img>` 50266, `</img>` 50267, `<Number>` 50268, bos=eos 2, pad 1. At that
snapshot the OneChart sub-epic still had the remaining forward/runtime gates
open, and the live checkout also had oracle-fixture WIP that should not be
cited as committed truth without checking a newer revision.
CASS searches for `focr OneChart tokenizer D9` and
`focr SmolVLM2 C8 C10 closed` returned zero hits in this refresh, so the live
source, Beads, README, and committed tests were the authoritative evidence.

2026-07-05 OneChart D3 refresh: source head `74471c1` closed `bd-3jo6.4.3`.
The close reason and source show `preprocess::onechart_view_tensor` as a single
squash-resized 1024x1024 raw `[0,1]` RGB tensor with no CLIP constants, and
`src/native_engine/onechart.rs::vision_features` as the certified SAM-ViT-B
`model.vision_tower` path plus `model.mm_projector`
`Linear(1024->768,bias)` to `[256,768]`. The committed oracle generator is
`scripts/gen_reference_fixtures_onechart.py`; compact metadata lives in
`tests/fixtures/onechart/oracle_fixtures.json`, while the armed blobs include
`onechart_preproc.bin`, `onechart_proj_out.bin`, and
`onechart_final_logits.bin` under `FOCR_ONECHART_DIR`. The D3 cert reports
`proj_out cos 1.00000000`, maxabs `6.5e-4`; the fixture's current `prompt_n`
is 308, despite older 309-token prose. At this D3-only snapshot, public
routing, structured output, parity/perf, and e2e support were still ahead; the
later D6-D8 refresh below supersedes the route status.

2026-07-05 OneChart D4-prefill refresh: source head `20ac599` committed D4
half 1 before the later `2769d21` D4 close. This
supersedes the D3-refresh observation that OPT prefill code was only
live-checkout WIP. The committed seam adds `DecoderFamily::Opt`,
`DecoderConfig::onechart()`, `nn::relu`, and
`onechart::build_inputs_embeds`; the OPT path uses learned absolute positions
with offset 2, no RoPE, pre-LN `LayerNorm` with bias, biased q/k/v/out/fc1/fc2
linears, ReLU MLP, tied head, and q pre-scaling inside the shared attention
kernel. The armed `opt_prefill_matches_torch_oracle` proof uses
`FOCR_ONECHART_DIR`, `onechart_proj_out.bin`, `onechart_final_logits.bin`,
`model.safetensors`, and `tests/fixtures/onechart/oracle_fixtures.json`;
fixture prompt length is 308 and the last-position logits match the oracle with
argmax 50268 (`<Number>`), cos `1.00000000`, and maxabs `6.1e-5`. At that
pre-`2c77d21` snapshot, the next D4 gates were OPT KV-cache decode/generate plus
then-future D5 numeric self-verify/product assembly, with D6-D8 still covering
parity/perf, CLI/product routing, and e2e. Do not translate D4-prefill into
`focr pull onechart`, `focr chart`, `--task chart-data`, chart
JSON/CSV extraction, or OneChart product readiness.
CASS searches for `focr OneChart D4 prefill` and `DecoderConfig::onechart`
found no newer authoritative evidence than live source/Beads.

2026-07-05 OneChart D4 cached decode refresh: source head `2c77d21` committed
the OPT cached decode seam before the later `2769d21` D4 close. This
supersedes the earlier D4-prefill note that the remaining
D4 half was only future KV-cache/generate work, but it does not close D4 or
make OneChart a product route. The committed seam extends the shared dense
decoder with OPT-family cached decode weights, `GotDecodeWeights`,
`family_norm`, learned absolute positions, no RoPE, output-proj bias, final
norm bias, OPT ReLU `fc1`/`fc2`, centralized `lm_head`, and
`generate_greedy_kvcache` support for `DecoderFamily::Opt`. The armed
`opt_kvcache_matches_greedy_and_oracle` proof uses `FOCR_ONECHART_DIR`, oracle
projector rows, tokenizer files, and weights; it compares a 24-token KV-cache
greedy stream with O(n^2) re-prefill greedy on the same weights, requires a
>=12-token exact prefix at the measured near-tie horizon, asserts first id
50268 (`<Number>`), and used a text-prefix check that `2769d21` later replaced
with the dict-open structural anchor. At the live checkout observed after
`2c77d21`, `src/native_engine/onechart.rs` also had an uncommitted
refinement that prefers `onechart.int8.focrq` when present so the B9 identity
leg uses same-quantization weights; that caveat was superseded when `2769d21`
committed the preference. At that snapshot, the registry flag for the row was
still false; D5 and the D6-D8 product gates had work ahead. CASS search for
`OneChart OPT cached decode` found no prior-session hits, so live source/Beads
are the authoritative evidence.

2026-07-05 OneChart D4 close refresh: source head `2769d21` closed
`bd-3jo6.4.4` after `2c77d21`. This supersedes the immediately preceding
`2c77d21` note's `in_progress` status and dirty/int8 caveat. D4 is now closed
for the decoder/prefill/cached-decode seam: `GotLayerW` uses `MlpW` for
SwiGLU-vs-ReluFc, `family_norm` switches LayerNorm/RMSNorm by bias presence,
out-proj and final-norm biases thread through, and learned positions are added
in both seed prefill and decode step. The D4 cert now pairs same-precision paths
by preferring `onechart.int8.focrq` when present, records a measured 13-step
exact prefix at about 320 positions before the DISC-003-style whitespace/quote
JSON near-tie, gates prefix >=12 plus `<Number>`-first and dict-open structural
anchors, and no longer treats full text-vs-oracle `chat()` prefix as the exact
gate. Close evidence says suite `1023/0` and clippy `-D warnings` clean.
At the `2769d21` snapshot, the registry flag for the row was still false, and a
stale test message still said "sub-epic D forward has not landed yet"; trust
newer Beads/source behavior rather than that older message text. At that
snapshot, `bd-3jo6.4` still had D5 structured assembly, D6 parity/perf, D7 CLI,
and D8 e2e ahead. The later D6-D8 refresh below supersedes runtime-route
status; distribution remains a separate manifest claim.

2026-07-05 OneChart D5 assembly refresh: source head `2a56c96` closed
`bd-3jo6.4.5` after `0145419` assembled the recognition path. This supersedes
the D4-close note's "D5 still open" status. Current committed
`src/native_engine/onechart.rs` has `ChartResult` with `json_text`, optional
`pred_locs`, optional `reliable_distance`, and `reliable`; `recognize` builds
the fixed 308-id prompt, splices D3 vision rows into the OPT prompt, decodes
within the 4096 cap, taps `<Number>` through `prefill_final_hidden`, runs
`number_head`, repairs/strips JSON through string-aware `complete_json_string`,
extracts and normalizes chart values, and computes `reliable_distance`. D5
anchors include `chart_prompt_ids_match_oracle_l0c`,
`recognize_reads_the_committed_chart`, `reliable_check_matches_upstream_goldens`,
and `number_head_matches_golden`; close evidence says text 2/4 was a measured
floor and suite `1028/0` was clean. This D5 snapshot was native-module
assembly before public routing: the CLI task enum, engine dispatch, and registry
flag had not landed yet. The later D6-D8 refresh below supersedes that route
boundary; do not use this D5 note as current route status.

2026-07-06 OneChart D6/D7/D8 and TrOMR partial refresh: source head `e926c46`
closed `bd-3jo6.4.6`, `bd-3jo6.4.7`, `bd-3jo6.4.8`, and the OneChart
sub-epic `bd-3jo6.4`. Current source has `OcrTask::ChartData`,
`forward_onechart`, `model_arch` `ONECHART` with `implemented=true`, and
`scripts/onechart_chart_e2e.sh` emitting `onechart_chart_e2e/v1` NDJSON. Use
`focr ocr --task chart-data --model /path/to/onechart.int8.focrq chart.png`
when a supplied, converted, or pulled artifact exists. The earlier
"local-artifact-only" distribution boundary is superseded by `bd-av64.7`,
`bd-av64.8`, and `bd-av64.9`: the checked-in manifest lists `onechart`, the
GitHub release `models-onechart-v1` is published, and clean-cache pull plus
real inference were verified. HF mirror availability is the remaining
resilience gap, not default `focr pull onechart`. Source head `9cb91f9` adds
`bd-2lje`, an
in-distribution SCRM-proxy OneChart corpus: six synthetic-style charts, number
head fires 6/6, mean reliable distance is about 0.015 int8 / 0.014 f32, f32 and
int8 decoded text are byte-identical, and only 1/6 outputs are valid JSON in
both precisions. Treat that as scoped regression evidence, not broad chart
quality.

For Polyphonic-TrOMR, source heads `c22b047`, `7464590`, `6403d4c`, `45da3a3`,
and `3472c1b` add E2 checkpoint export / WS fold / conversion census, E6
decode-only WordLevel music tokenizers, E3 NN kernels (`tf_same_pad`,
`max_pool2d`, `group_norm`, fused ReLU option), the committed E3 hybrid
ResNetV2+ViT encoder, and the committed E4 deterministic four-head AR decoder.
`45da3a3` hydrates `TromrEncoderW`, runs WS-prefolded stages `[2,3,7]`, adds
crop-indexed learned positions over the 80-wide table, executes four pre-LN ViT
blocks, and certifies `tromr_encoder_matches_torch_oracle` at `encoder_out cos
1.00000000` / maxabs `3.8e-6` with oracle floor 0.0. `3472c1b` closes
`bd-3jo6.5.4`: `TromrDecoderW`, `decoder_forward`, `generate`, `MusicStreams`,
and `tromr_decoder_matches_argmax_oracle` prove all four step-0 heads at cos
1.0 and 42-step x 3-stream token-exact argmax generation. This paragraph is
superseded for current runtime status by the 2026-07-06 E7/E9 refresh below;
keep it only as the pre-E7/E9 evidence trail.

2026-07-05 manifest/distribution refresh: source head `2769d21` still has
`src/cli.rs::PullArgs` with optional positional `model`, `--quant`,
`--manifest`, and `--json`; `src/dist.rs::resolve_manifest_source` resolves
explicit `--manifest` first, then `FOCR_MANIFEST_URL`, then the built-in
manifest. At the time of that note, the manifest packaged only the primary
`unlimited-ocr` artifact and `models.got-ocr2`; `bd-av64.7` / `ece14f9`
supersedes it with named SmolVLM2, OneChart, and TrOMR entries. Therefore
`focr models` registry readiness, runtime support from a supplied/converted
`.focrq`, and `focr pull <id>` distribution support remain three separate
claims, but current source answers pullability with `pull.in_manifest` and
`pull.quants`.

2026-07-06 TrOMR E7/E9/E8/E5 refresh: source head `fc9d88a` sat above
`2cbded9`, `78a2de3`, and `79d715c`. E7 adds
`merge_semantic` and `semantic_to_musicxml`: upstream-style aligned
rhythm/pitch/lift merge with fail-loud control-id rules, then partwise
MusicXML 4.0 over clef/key/time/chords/durations/rests/multirests/accidentals.
E9 adds `preprocess::tromr_staff_tensor`, `tromr::recognize`, `MusicResult`,
`forward_tromr`, `OcrTask::Music`, `model_spec_is_knowably_not_tromr`,
`model_arch implemented=true`, and `scripts/tromr_music_e2e.sh` /
`tromr_music_e2e/v1`. Beads report missing-model exit 3, wrong-family exit 2,
and a real MusicXML run in about 1s on upstream example 1. The current surface
is `focr ocr --task music --model tromr.focrq` for pulled or supplied artifacts
with tokenizer tables; after `752f3cd`, that covers single-staff images and v1
printed/scanned full-page scores. `bd-av64.7` / `ece14f9` later makes
`focr pull tromr` current for distribution; `efccce9` / closed `bd-av64.12`
later makes the default artifact `tromr.int8.focrq` storage plus tokenizer
sidecars, with `focr pull tromr --quant f32` for `tromr.focrq`. `2cbded9` closes
`bd-3jo6.5.8`: the single-staff L0b-L5 ladder is green, DISC-004 is ledgered,
the pinned four-example SER is 0.125 / 0.040 / 0.375 / 0.304 (mean 0.211),
OQ-T1/T2/T3/T4/T6 are resolved, and the music e2e is 4/4. `bd-2sez` /
`5430e2c` later adds the f32 baseline perf row: exact token-stream agreement,
but focr f32 loses to pinned upstream torch. `fc9d88a` adds the E5 v1 detector module
`src/preprocess/staff_detect.rs`: DISC-004 ink gray plane, Otsu thresholding,
global projection-profile deskew, five-line grouping, ordered full-width crops,
and focused synthetic tests. Superseding refresh: `752f3cd` closes
`bd-3jo6.5.5` by wiring `recognize_page`, `staves_to_musicxml`, and full-page
`forward_tromr`; the armed stacked-page cert proves two-staff order by
cross-SER and pins detector-lossless SER 0.125 / 0.040. `9127676` adds
`tromr_alpha_ink_path_fires_only_when_alpha_varies`, and `ab0bae0` closes E10
plus sub-epic `bd-3jo6.5` with about 25 unit tests, 6 armed certs, 2 NDJSON e2e
scripts, and a full 891-test `scripts/check.sh` gate. `**kern`, camera dewarp,
default/lossless barline quality, int8 publication, and perf wins remain
outside this runtime proof; experimental `FOCR_TROMR_SPLIT=1` is a later rescue
switch, not default quality. The `bd-2sez` TrOMR f32 baseline row exists but is
a loss row.

2026-07-06 determinism/fixture-governance refresh: committed source head
`3e85c7d` closes `bd-3kge` and `bd-2pgf`. The shared determinism gate lives in
`tests/support/parity_harness.rs` as `assert_deterministic` /
`assert_outputs_deterministic`, emits `parity` / `token_exact` evidence, and
has an injected HashMap-order nondeterminism self-test that must fail. The
model-gated e2e real-model leg in `tests/e2e_recognize.rs` adopts it by running
`recognize()` twice and requiring byte-identical markdown. Fixture governance
now has `tests/fixtures/PROVENANCE.md` as the prose catalogue,
`tests/fixtures/MANIFEST.toml` as the machine-readable committed vs
regenerated-committed policy, and `scripts/check_fixture_manifest.py` wired into
`scripts/check.sh`. This is infrastructure proof only; it does not replace
oracle-quality, model-corpus, or PERF_LEDGER evidence.

2026-07-06 conformance-matrix refresh: current `franken_ocr` head is `c685818`,
a format-only follow-up to `fb52843`, which closes `bd-re8.12`. The new source
surface is `src/conformance.rs`: `RequirementLevel`, `ConformanceCategory`,
`ConformanceTest`, `RegisteredConformance`, and `conformance_registry()`.
`tests/conformance_matrix.rs` computes coverage from the spec, not from the test
list: it parses `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`, scans
`src/**` and `tests/**` for `[SPEC-NNN]` and range references, gates MUST
coverage >= 0.95, emits per-clause NDJSON plus a summary, verifies every XFAIL
emission is attached to a `DISC-NNN` or explicit phase-gap reason, and runs the
registry entries in-process. The Bead close reason records 83 MUST clauses, a
first measured 0.9398 ratio before five accounting citations
(`SPEC-001/002/017/034/036`) were added at genuinely covering sites, 18 XFAIL
emission sites with 0 violations, and green registry runs. This is
conformance-accounting infrastructure only; differential, metamorphic, golden,
release-certification, conformal-ratchet, and e-process gates remain
independent proof surfaces even when closed.

Subsequent 2026-07-06 verification-suite refresh: current `franken_ocr` source
now closes `bd-re8.9`, `bd-re8.10`, and `bd-re8.11`. `bd-re8.9` / `390d05c`
pins the oracle-differential comparator in `tests/parity_ladder.rs`, including
`differential_per_op_vs_bf16_oracle`, `differential_row`, the `EngineIdentity`
guard against oracle-vs-oracle false greens, ULP/L3-L5 tolerance rows, and
`DISC-NNN` XFAIL discipline. `bd-re8.10` / `5f2d7ce` pins
`tests/metamorphic.rs`: identity resize, rotation bbox mapping, mean-gray
padding through `preprocess::PAD_FILL`, same-thread plus `FOCR_THREADS=1` vs
`4` determinism, white-pad SHOULD logging, and a gated cross-page dependence
relation. Its negative guard matters: R-SWA makes multi-page parsing
cross-page dependent, so the concat/sum page relation is not valid.
`bd-re8.11` / `f879211` pins
`tests/cli_robot_golden.rs`, `tests/fixtures/golden/PROVENANCE.md`, and
`docs/conformance/GOLDEN.md` for exact, fuzzy, scrubbed, and canonicalized
golden artifacts with manual `UPDATE_GOLDENS=1` review and no CI auto-blessing.

Same-day ladder-scorecard refresh: committed `1b84428` adds
`scripts/ladder_scorecard.sh` for `bd-re8.19`; `1112cf8` plus the close
evidence later make the bead closed-current. The runner executes
`parity_ladder` serially, folds `event=parity` and `event=result` lines into
`focr-ladder-scorecard/v1`, records `all_green`, `skipped_no_model`, per-gate
worst rows, and `not_meaningful` downstream gates after the first hard failure,
and includes `--self-test` for no-model fold validation. Armed close evidence
reported all six gates green; unarmed no-model runs still report
`skipped_no_model=true` and `all_green=false`.

2026-07-07 focr/franken_ocr refresh: current inspected `franken_ocr` head was
`7ad67a7`, which documents the two recent committed feature boundaries. The
GOT-OCR2 batch-path follow-up is `d25dbd7`: `got::recognize_batch` hydrates
`SamWeights`, `mm_projector_vary`, and `model.embed_tokens.weight` once per
batch, logs `got.hydrate(batch)` and `got.vision+splice(batch of N)`, preserves
byte identity through `recognize_batch_matches_sequential_e2e`, and reports
14.47s sequential vs 13.53s batch (~6.5%) on a same-binary 3-page run. Treat it
as narrow setup/splice amortization, not as the larger SAM-attention campaign,
closed `bd-av64.10`, or A11/PERF_LEDGER evidence. The TrOMR crop-geometry
follow-up is `eb0c70e`: `src/preprocess/staff_detect.rs` trims horizontal page
margins to ink extent with a line-spacing pad, grows wide bands vertically
toward neighbor midlines where that can fit the 1280-column budget, preserves
non-overlap, and leaves truly unfittable bands to the skip/error path. Focused
anchors include `trim_cuts_page_margins_but_keeps_ink`,
`wide_staff_with_room_fits_the_positional_budget`,
`packed_staves_stop_at_the_midline`,
`unpressured_band_keeps_the_generous_margins`,
`tromr_page_skips_overwide_staff_and_keeps_the_rest`, and
`tromr_page_all_staves_failing_is_a_named_error`. `bd-av64.14` still remained
open in Beads, so do not infer broad real-scan SER, geometry-record NDJSON, or
acceptance closure from these focused source tests. The checkout also had live
WIP in `src/native_engine/tromr.rs`; only committed anchors were used as skill
facts. Four `cass search ... --robot` probes for focr/GOT/crop-geometry context
were interrupted after hanging for more than 35s each, so no CASS-derived
evidence was incorporated. The next note supersedes the crop-mechanics
description: current source is fit-first and preserves already-fitting
full-width staff crops.

2026-07-07 fresh-eyes correction pass: current inspected `franken_ocr` head had
advanced to `ad3ad20`. Three follow-up commits changed the right skill
emphasis. `40ee875` makes TrOMR crop shaping fit-first: already-fitting
full-width staff bands keep the historic geometry byte-for-byte, while only
over-budget bands get ink-extent trim plus neighbor-bounded extend-to-fit
toward the 1280-column position budget; it also adds
`fitting_bands_keep_the_classic_full_width_geometry` and promotes Spohr p055 /
p100 out of page-level dense-XFAIL framing into conservative page-floor checks.
`91d552f` moves those full-page `min_recognized` floors into
`tests/fixtures/realscan_music/truth/attributes.json` so
`scripts/realscan_music_gate.sh` reads p055=5 and p100=1 from truth data rather
than shell literals. `ad3ad20` adds `robot selftest.models`, a per-decoder int8
parity rollup for unlimited-ocr, got-ocr2, smolvlm2, and onechart, including
model-specific overflow rows; TrOMR is intentionally absent because its
published int8 artifact dequants through f32 accessors rather than int8 decoder
kernels. Later source closes `bd-av64.14`, `bd-av64.12`, and `bd-av64.13`; use
the current Beads reality section instead of this older snapshot for those
lanes.

## Known Caveats

- The project is moving quickly; re-run source probes before strong claims.
- Installed or target-dir binaries can be stale.
- Installed release binaries can lag `main` features such as `--extract-figures`;
  check help/version before relying on Unreleased surfaces.
- Some commands may exist as scaffolds before full implementation.
- Model rows, source runtime, manifest pullability, and product-quality proof
  are separate claims. GOT-OCR2, SmolVLM2, OneChart, and TrOMR all have current
  manifest pull entries after `bd-av64.7`, but TrOMR's pulled artifact is f32,
  not int8. Do not round manifest pullability or local runtime support up to
  camera dewarp, `**kern`, perf-win/int8-perf support, or full model-quality
  certification.
- Platform support claims require live release/source/tests/Beads evidence.
- Experimental env vars are not production advice.

## Update Discipline

When updating this skill:

1. Re-read current `src/cli.rs`, `src/lib.rs`, `src/error.rs`, `src/robot.rs`,
   `src/native_engine/`, `src/quant/`, and relevant tests.
2. Re-run `br`/`bv` robot-safe probes if making tracker-backed claims.
3. Check whether CASS now has useful focr sessions.
4. Update references first, then tighten `SKILL.md`.
5. Preserve useful existing content; add stronger operator cards or references
   instead of replacing prior hard-won caveats.
6. Run the local validator.
