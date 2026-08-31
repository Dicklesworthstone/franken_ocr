# Troubleshooting focr

## Table of Contents

- [Stale Binary](#stale-binary)
- [Dirty Worktree Feature Claims](#dirty-worktree-feature-claims)
- [Installation Failure](#installation-failure)
- [Model Not Found](#model-not-found)
- [Zoo Release Pull Confusion](#zoo-release-pull-confusion)
- [Model ID and GOT-OCR2](#model-id-and-got-ocr2)
- [SmolVLM2 Tokenizer, Vision, Conversion, and Decoder Seams](#smolvlm2-tokenizer-vision-conversion-and-decoder-seams)
- [SmolVLM2 VQA Guard](#smolvlm2-vqa-guard)
- [OneChart Chart-Data Route and Distribution Boundary](#onechart-chart-data-route-and-distribution-boundary)
- [TrOMR Local Runtime Lane](#tromr-local-runtime-lane)
- [TrOMR Staff Skip Notes](#tromr-staff-skip-notes)
- [TrOMR Crop Geometry Boundary](#tromr-crop-geometry-boundary)
- [TrOMR Real-Scan Music Gate](#tromr-real-scan-music-gate)
- [Robot Schema Golden Drift](#robot-schema-golden-drift)
- [Output File Issues](#output-file-issues)
- [Figure Extraction Issues](#figure-extraction-issues)
- [Crop Mode and Preprocess Flags](#crop-mode-and-preprocess-flags)
- [Format Mismatch](#format-mismatch)
- [Robot Output Is Not JSON](#robot-output-is-not-json)
- [Cancellation and Thread Budget](#cancellation-and-thread-budget)
- [PDF InputDecode](#pdf-inputdecode)
- [Command Missing](#command-missing)
- [Selftest Fails](#selftest-fails)
- [Dense Repetition in Output](#dense-repetition-in-output)
- [Speculative Decode Drift](#speculative-decode-drift)
- [Reference Resampling Differences](#reference-resampling-differences)
- [Gauntlet or Perf Claim Confusion](#gauntlet-or-perf-claim-confusion)
- [A11 Zoo Ratio Confusion](#a11-zoo-ratio-confusion)
- [Release Evidence Instrument Confusion](#release-evidence-instrument-confusion)
- [Benchmark Guardrail Confusion](#benchmark-guardrail-confusion)
- [GOT SAM Timing Confusion](#got-sam-timing-confusion)
- [Batch Spine Confusion](#batch-spine-confusion)
- [QKV Fused Decode Confusion](#qkv-fused-decode-confusion)
- [Ngram-Lmhead Fusion Confusion](#ngram-lmhead-fusion-confusion)
- [Windows Notes](#windows-notes)
- [Build Takes Too Long](#build-takes-too-long)
- [No CASS Hits](#no-cass-hits)

## Stale Binary

Symptoms:

- `focr --help` lacks commands present in `src/cli.rs`.
- A target-dir binary behaves like an older checkout.
- `CHANGELOG.md` says a feature is under `Unreleased`, but an installed release
  binary does not show it.

Fix:

```bash
cd ~/projects/franken_ocr
rg -n "enum Commands|enum RobotCommands" src/cli.rs
cargo run --bin focr -- --help
```

If building is too heavy, say the installed binary is stale and base conclusions
on source only.

Specific current pitfall: README/source may document `--extract-figures`,
`--figures-dir`, and `ExtractedFigure` APIs while a curl-installed binary still
lags. The `v0.4.0` public release boundary includes SmolVLM2 C7-C10, OneChart
D2/D9, `--task`, `FOCR_RESAMPLE`, doctor, run/sync, and batch-spine vision
controls, but an installed binary can still be pre-`v0.4.0` or a release mirror
can lag. The `bf28fd7` / `v0.5.0` release is now live with platform binaries.
`48a9896` aligns README release prose with `v0.5.0`, but installer fallback
constants can still mention `v0.4.0`; normal online installers resolve the
latest release before using the fallback. The `v0.6.0` release is now live,
published 2026-07-08T14:47:48Z with platform assets, and points at source tag
`29516b9`. The July 8 source probe saw public `origin/main` at `592426c`, with
clean source describing as `v0.6.0-4-g592426c` and no tracked source diff.
Post-tag gate, bundle, fuzz, SigLIP, README-alignment, public SAM row-tile
negative evidence, live certification-bundle-cell fixes, GOT pass-6 statics
caching, the mmap-loader half, OneChart pass-7 statics caching, and SmolVLM2
pass-8 statics caching, SmolVLM2 untied-head certification, head-to-head perf
rows, and release certification are source evidence when the running
source/help/tests include them; `0924479` makes README source-vs-binary
labeling explicit, `efd83e8` lands the formal `bd-av64.10` G2 closeout,
`3f2f97e` lands the gauntlet harness fixes, `4291807` certifies the SmolVLM2
head lever, `c248e6d`/`c4c1684` close `bd-2mo.26`, and
`c29a78b`/`7c7bd00` close the release-certification gate, `beaed7c` adds the
CI/dist supplement, `db02421` refreshes README evidence, `5df6395` commits the
post-certification fuzz corpus, and `592426c` refreshes README public `v0.6.0`
identity/asset-size/backend prose. Dirty diffs after `592426c` remain live-WIP
until committed. If a
binary lacks a feature, prove whether this is stale-local,
source-ahead-of-release, fallback/manual-prose lag, post-tag-main lag, or
release-publication lag before changing docs or callers.

## Dirty Worktree Feature Claims

Symptoms:

- `README.md` or source mentions a feature that Beads still reports open.
- `git diff --stat` shows large, surprising, or mechanically generated-looking
  changes.
- A dirty checkout advertises new behavior beyond committed `d25dbd7`,
  `eb0c70e`, `3f2878d`, `8de3674`, `507cebe`, `0401df2`, `38ab806`,
  `a9a406e`, `9b2a03b`, `8d6601d`, `4cedacd`, `0924479`, `efd83e8`,
  `3f2f97e`, `4291807`, `c248e6d`, `c4c1684`, `c29a78b`, `7c7bd00`,
  `29516b9`, `beaed7c`, `db02421`, `5df6395`, or `592426c`.

Rules:

- Dirty source is useful reconnaissance, not release truth. Label it
  `live-WIP` until committed source, tests, and Beads agree.
- Run `git diff --stat` first. If a file has an absurd-sized diff or malformed
  repeated code, quarantine that path as likely corrupted WIP and do not derive
  capability claims from it.
- `d25dbd7`, `eb0c70e`, `3f2878d`, `8de3674`, `507cebe`, `0401df2`,
  `a391793`, `c8682a3`, `38ab806`, `a9a406e`, `9b2a03b`, `8d6601d`,
  `4cedacd`, `0924479`, `efd83e8`, `3f2f97e`, `4291807`, `c248e6d`,
  `c4c1684`, `c29a78b`, `7c7bd00`, `29516b9`, `beaed7c`, `db02421`,
  `5df6395`, and `592426c` are now committed source
  facts. Treat them as source-current only
  after confirming the target source/binary includes those commits. Do not
  confuse post-tag source commits with a user's installed binary; verify the
  binary and asset instead of inferring from source.
- For TrOMR geometry, check `br show bd-av64.14 --json`: the fit-first
  geometry/p169 lane is closed in current source, but it is not camera dewarp,
  default/lossless barline quality, TrOMR int8, perf, or broad note-level SER
  proof. If `FOCR_TROMR_SPLIT=1` appears, treat it as separate experimental
  over-budget-staff recognition-count rescue.
- For GOT hydration amortization, check `3f2878d` first: `OcrModel` owns
  `got::GotStatics`, sequential and batch page paths reuse it, `FOCR_TIMING`
  logs `got.hydrate(cached)`, and `recognize_batch_matches_sequential_e2e`
  remains the e2e contract. Use `d25dbd7` / `got.hydrate(batch)` only when
  diagnosing older batch-hoist binaries. Treat uncommitted changes beyond those
  anchors as candidate optimizations, not closed facts.
- `507cebe` ships the mmap-loader half: `FOCR_NO_MMAP=1` forces owned-buffer
  loading, `Backing::Mapped` is the default when mapping works, mapping failure
  falls back to owned bytes, and `mmap_load_is_byte_identical_to_owned_read`
  proves the mapped/owned parser contract. `bd-2mo.22` still remains open for
  64B scratch alignment, decode-loop buffer reuse, and mimalloc measurement.
- `38ab806` / `a9a406e` make `onechart::OnechartStatics`,
  `OcrModel::onechart_statics`, and `onechart.hydrate(cached)` committed
  pass-7 source evidence. Keep that claim scoped to byte-identical chart-data
  source evidence and do not promote the pass alone to formal PERF_LEDGER proof.
- `9b2a03b` makes `smolvlm2::SmolStatics`, `OcrModel::smol_statics`, and
  `smolvlm2.hydrate(cached)` committed pass-8 source evidence. Keep that claim
  scoped to byte-identical describe source evidence and do not promote it to
  public VQA quality or standalone formal PERF_LEDGER proof.

## Installation Failure

Preferred install path:

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
focr --version
```

Native Windows:

```powershell
irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
focr --version
```

Symptoms and fixes:

- `focr: command not found`: installer did not update the current shell's
  `PATH`, or the manual binary was not placed on `PATH`. Open a new shell or
  invoke the installed absolute path.
- `focr --version` works but OCR says model missing: install only placed the
  binary. Run `focr pull` once.
- Installer summary mentions a command that help lacks: the raw installer script
  may be newer than the release binary it installed. Trust `focr ocr --help` for
  the exact binary and source/tests for current `main`.
- `cargo install --git` fails: this is expected. The README says unpublished
  sibling path dependencies make prebuilt release binaries the supported path.
- Manual download checksum fails: re-download both the binary and the
  `<asset>.sha256` sidecar from the same release and verify with `shasum -a 256
  -c` on macOS or `sha256sum -c` on Linux.
- `gum: error: unknown flag ->`: current `install.sh` guards dynamic text with
  `gum style -- ...`; likely causes are an old raw installer, cached branch, or
  stale mirror. Re-fetch from `main` or use `FOCR_INSTALL_BASE_URL` only after
  confirming the mirrored script is current.

## Model Not Found

Exit code: 3.

Fix order:

```bash
focr pull
focr pull --manifest ./manifest.json # only when using a custom manifest
FOCR_MODEL_PATH=/path/to/model.focrq focr ocr page.png --json
focr ocr page.png --model /path/to/model.focrq
```

For library users, call `recognize_with_model` or set `FOCR_MODEL_PATH` in the
process environment before startup.

Fresh `focr pull` writes `unlimited-ocr.int8.focrq`; current resolution should
find that file from the default cache or `FOCR_MODEL_DIR` without `--model`.
If a fresh pull followed by bare `focr ocr page.png` still exits 3, suspect a
stale binary or a pre-fix resolver. Probe `focr --version`, source revision, and
whether help/tests mention the quant-suffixed resolver.

If `focr pull <model>` says the manifest has no such model, separate three
facts before diagnosing runtime support:

- `focr models` is the compiled registry and can mark a model ready.
- the resolved manifest is what `pull` can actually download.
- manifest source precedence is `--manifest`, then `FOCR_MANIFEST_URL`, then the
  built-in repo manifest.

Current checked-in manifest after `bd-av64.7` packages all five ready models:
`unlimited-ocr`, `got-ocr2`, `smolvlm2`, `onechart`, and `tromr`. If a named
pull fails, inspect the exact binary's embedded manifest with
`focr models --json` before assuming the model is not distributed.

## Zoo Release Pull Confusion

Use:

```bash
focr models --json | jq '.models[] | {id, implemented, pull}'
focr pull smolvlm2 --json
focr pull onechart --json
focr pull tromr --json
```

Rules:

- `bd-av64.8` and `bd-av64.9` are the current GitHub release and clean-cache
  proof for `models-smolvlm2-v1`, `models-onechart-v1`, and
  `models-tromr-v1`: exact sizes/hashes, sidecars, cache subdirectories,
  idempotent repull, and one real inference smoke per model.
- The GitHub-first path is the proven distribution path. The secondary mirror
  is not proven available: the known spot-check returned 401, so do not debug a
  GitHub pull failure by treating that fallback as already usable.
- TrOMR now defaults to the published `tromr.int8.focrq` storage artifact after
  `efccce9` / closed `bd-av64.12`; use `focr pull tromr --quant f32` when you
  need the bit-exact `tromr.focrq` reference. This is quantized storage with
  f32 dequant-on-access, not an int8 decoder-kernel or speed claim.
- A separate one-command pull-e2e script is not current evidence unless
  source/Beads show it landed.

## Model ID and GOT-OCR2

Use:

```bash
focr models --json | jq .
focr pull got-ocr2
focr ocr page.png --model got-ocr2.int8.focrq
focr ocr formula.png --task formula --model got-ocr2.int8.focrq
focr convert got.safetensors -o got-ocr2.int8.focrq --quant int8 --model-id got-ocr2
ls -l "$(dirname got-ocr2.int8.focrq)/qwen.tiktoken"
```

Rules:

- A `.focrq` with no `model_id` is `unlimited-ocr`.
- Unknown `model_id` is a format/compatibility problem, not a missing file.
- GOT artifacts must carry `model_id = "got-ocr2"` and the registered license.
- `focr pull got-ocr2` installs the packaged GOT `.focrq` and `qwen.tiktoken`.
- Self-converted GOT runtime needs `qwen.tiktoken` beside the `.focrq`; missing
  tokenizer surfaces as `ModelNotFound`.
- If `focr models` marks GOT `planned`, the binary is stale or not at the
  current source boundary and release-publication state.
- If a GOT run returns plain OCR when the user expected tables, LaTeX, charts,
  music, or `.mmd` formatting, check whether the caller passed `--format`, set
  `FOCR_GOT_FORMAT`, used `focr ocr --task ...`, or used
  `native_engine::force_got_format(true)`. Plain `--model
  got-ocr2.int8.focrq` still uses `format=false`.
- If a specialized `--task` errors before weights load, check whether the model
  path is missing or knowably `unlimited-ocr`; the fix is `focr pull got-ocr2`
  and `--model got-ocr2.int8.focrq`.
- If `--task describe` errors, first classify binary/source age. Older binaries
  can still return `NotImplemented`; current source expects a SmolVLM2 artifact
  (`--model smolvlm2.int8.focrq`) and accepts `--question` for VQA. If it loads
  but produces weak or divergent output, check DISC-003 first: ledgered
  near-tie KV-cache flips are known; then check C8/C10 evidence and A11/perf
  rows before calling it a regression.
- If source inspection finds a stale `OcrTask::Describe` doc comment saying the
  route is planned, verify the actual `validate_task_selection`,
  `set_smolvlm2_question`, and `forward_smolvlm2` path before reporting status.
- If a GOT e2e proof is requested, require `FOCR_GOT_MODEL` and
  `FOCR_GOT_TIKTOKEN`; `FOCR_ORACLE_IMAGE` and `FOCR_ORACLE_HIDDEN0` are needed
  for the vision/splice oracle gate.
- If `--format` or `--task` produces weak LaTeX/table/chart/music/molecular
  output, do not treat that as disproving the plumbing: `bd-3kix` phase 1 proves
  five real-model smoke fixtures, but exact per-modality budgets are still a
  follow-up. Capture the image, expected formalism, model hash, and output.

## SmolVLM2 Tokenizer, Vision, Conversion, and Decoder Seams

Use:

```bash
FOCR_SMOLVLM2_TOKENIZER_JSON=/path/to/tokenizer.json cargo test smolvlm2_token_id_conformance_gate
SMOLVLM2_SAFETENSORS=/path/to/model.safetensors scripts/smolvlm2_convert_e2e.sh --no-build
focr convert /path/to/model.safetensors -o smolvlm2.int8.focrq --quant int8 --model-id smolvlm2 --json
FOCR_SMOLVLM2_DIR=/path/to/smolvlm2 cargo test smolvlm2_ -- --nocapture
```

Rules:

- `--model-id smolvlm2` is current for conversion. Current source routes
  `smolvlm2.int8.focrq` through `--task describe`; use `--question` or
  `FOCR_SMOLVLM2_QUESTION` for VQA. That is an implemented, sub-epic-C-closed
  route; quality/perf claims still need exact C8/C10/A11 evidence.
- A correct C2 conversion has `model_id=smolvlm2`, 489 tensors, 224 int8
  decoder GEMMs, 265 F32 high-precision tensors, and an untied high-precision
  `lm_head`.
- C5 decoder seam tests are current: f32 reached cos 1.000000 and 24-token L4
  exactness; int8 reached cos 0.998301 and argmax exactness with
  kvcache==re-prefill. DISC-002 records the known later near-tie flip.
- C6 tokenizer tests are current: `PretokScheme::SmolLm2` is selected from
  tokenizer JSON, the SmolLM2 GPT-2 ByteLevel/Digits path is 128/128
  token-id-exact and decode-exact against the pinned HF tokenizer JSON
  (`5ece781d...`), and bos/eos/pad/image ids are 1/49279/2/49190.
- If tokenizer conformance is skipped, set `FOCR_SMOLVLM2_TOKENIZER_JSON` or
  `FOCR_SMOLVLM2_DIR`. If it fails after a tokenizer update, regenerate with
  `scripts/gen_smolvlm2_token_id_fixtures.py` only after repinning the source
  tokenizer and documenting the changed ids.
- If `vision_siglip.rs`, `token_compress.rs`, or `smolvlm2.rs` exists, separate
  implemented route status from full describe/VQA quality/perf certification.
  C3/C4/A8/A9/C7/C9/C8/C10 are closed in current source, but each layer still
  needs its own evidence. The oracle lane uses
  `scripts/gen_reference_fixtures_smolvlm2_vision.py`,
  `tests/fixtures/smolvlm2/sample_photo.png`, `vision_oracle_fixtures.json`,
  and off-repo blobs under `FOCR_SMOLVLM2_DIR`; missing blobs should skip
  model-gated seams, while present-but-broken blobs should fail loudly.
- SigLIP drift often means the wrong activation/mask/position path:
  SmolVLM2 uses tanh GELU (`gelu_tanh`), bidirectional attention, NaViT
  bucketized learned 1-D position ids (`[0,0,1,...,30]`, not identity), 1024
  patch tokens per 512 frame, and final post-layernorm.
  Pixel-shuffle drift should be bit-exact; check scale 4, square-grid shape, and
  `[1024,768] -> [64,12288]` row order before tolerancing it.
- For C7 preprocess drift, inspect `preprocess_smolvlm2` before the vision
  tower. SmolVLM2 uses `SmolVLMImageProcessor` semantics: Pillow LANCZOS
  (`resample: 1`) through `pil_resample::resize_lanczos`, longest-side 2048,
  512-frame tiling, row-major local frames, and a final global 512 frame.
  `FOCR_RESAMPLE=pil-bicubic` is only the Baidu/GOT reference BICUBIC knob; it
  is not a SmolVLM2 LANCZOS selector. If L0 fails, compare against the
  Pillow 12.3.0 LANCZOS goldens in `pil_resample.rs` and the
  `FOCR_SMOLVLM2_DIR` pixel-value oracle, then classify as resampler, frame
  order, normalization, prompt-image expansion, or downstream vision drift.
- If an old binary reports only one int8 tensor or quantizes `lm_head`, it lacks
  the arch-aware classifier. Treat the binary as stale.
- If conversion rejects a tied `lm_head`, believe the error: that checkpoint or
  descriptor does not match the SmolVLM2 census.
- You may recommend `focr ocr --task describe --model smolvlm2.int8.focrq` only
  after verifying the exact source/binary exposes that route and the artifact
  exists. Pair it with current C8/C10 evidence and boundaries: DISC-003
  near-tie behavior, L5 is an oracle-answer guard rather than a public
  benchmark, and A11/perf/manifest caveats still matter.

## SmolVLM2 VQA Guard

Use:

```bash
FOCR_SMOLVLM2_DIR=/path/to/smolvlm2 cargo test vqa_quality_matches_oracle_l5 -- --nocapture
```

Rules:

- This is the C8 L5 oracle-answer guard, not a public VQA benchmark. It scores
  focr answers against the fixture oracle's own greedy output for one committed
  sample photo by normalized exact match or symmetric content-word containment;
  live C8 closure reports 7/7 f32 and 7/7 int8.
- If `vqa_quality_matches_oracle_l5` skips, check three things before treating
  it as meaningful: `FOCR_SMOLVLM2_DIR` must exist, the directory must contain
  `tokenizer.json` plus `model.safetensors` and/or `smolvlm2.int8.focrq`, and
  `tests/fixtures/smolvlm2/vqa_fixtures.json` must be present.
- If the fixture is missing, regenerate it only from the pinned model directory
  with `scripts/gen_smolvlm2_vqa_fixtures.py`, then record the model path,
  script revision, and resulting oracle answers.
- If the test fails below 70% f32 or 50% int8, preserve the question/output
  transcript and treat it as a regression or OQ-6 quality collapse until source
  proof says otherwise. Do not "fix" it by lowering floors or broadening answer
  matching.
- Missing artifacts skip their specific weight leg; present-but-broken artifacts
  should fail. If only one leg ran, say exactly which artifact was armed.
- If the task is CLI end-to-end behavior rather than oracle-answer fixture
  scoring, use `sh scripts/smolvlm2_describe_e2e.sh` when present. It emits
  `smolvlm2_describe_e2e/v1` NDJSON on stdout and `SVLM ` telemetry on stderr;
  it should skip-with-success when weights are absent, fail if negative paths do
  not produce exit 3/2, and pass only after real describe and VQA runs succeed.

## OneChart Chart-Data Route and Distribution Boundary

Use:

```bash
focr convert /path/to/onechart/model.safetensors -o onechart.int8.focrq --quant int8 --model-id onechart --json
FOCR_ONECHART_DIR=/path/to/onechart cargo test onechart_token_id_conformance_gate -- --nocapture
FOCR_ONECHART_DIR=/path/to/onechart cargo test vision_features_match_torch_oracle -- --nocapture
FOCR_ONECHART_DIR=/path/to/onechart cargo test opt_prefill_matches_torch_oracle -- --nocapture
FOCR_ONECHART_DIR=/path/to/onechart cargo test opt_kvcache_matches_greedy_and_oracle -- --nocapture
FOCR_ONECHART_DIR=/path/to/onechart cargo test recognize_reads_the_committed_chart -- --nocapture
FOCR_ONECHART_DIR=/path/to/onechart sh scripts/onechart_chart_e2e.sh
```

Rules:

- `--model-id onechart` is current for conversion of OneChart-shaped weights,
  and current source marks `onechart` `implemented=true`. A converted artifact
  is a chart-data runtime route when used as `focr ocr --task chart-data --model
  onechart.int8.focrq` with tokenizer files beside it.
- A correct D2 conversion has `model_id=onechart`, OneChart Apache-2.0 license,
  tied-head dedup (`lm_head.weight` omitted after byte-verifying equality with
  `model.decoder.embed_tokens.weight`), 72 OPT int8 decoder GEMMs, and high
  precision vision/projector/number-head/norms/biases/embeddings.
- If conversion keeps `lm_head.weight` for OneChart, quantizes `num_decoder.*`,
  or uses Qwen/SmolLM2 tensor suffixes, suspect a stale converter or wrong
  `--model-id`.
- The D9 tokenizer is OPT GPT-2 BPE over `vocab.json`, `merges.txt`, and
  `added_tokens.json`. It is not Qwen tiktoken, not SentencePiece, and not the
  SmolLM2 Digits pretokenizer.
- A correct D9 gate pins `<imgpad>` 50265, `<img>` 50266, `</img>` 50267,
  `<Number>` 50268, bos=eos 2, pad 1, and 29/29 token-id exact fixtures.
- D3 is current vision/projector proof only. A correct D3 setup has
  `scripts/gen_reference_fixtures_onechart.py`,
  `tests/fixtures/onechart/oracle_fixtures.json`, `onechart_preproc.bin`,
  `onechart_proj_out.bin`, and, for D4 oracle context,
  `onechart_final_logits.bin` under `FOCR_ONECHART_DIR`.
- If D3 vision fails, first check that `onechart_view_tensor` is using a single
  squash-resized 1024x1024 RGB tensor with raw `[0,1]` values. Do not introduce
  CLIP mean/std constants. Then check that `vision_features` uses
  `model.vision_tower` and `model.mm_projector`
  `Linear(1024->768,bias)` to produce `[256,768]` rows.
- The D3 armed close metric is `proj_out cos 1.00000000`, maxabs `6.5e-4`.
  Treat looser or missing metrics as unproven until regenerated from the same
  source revision and model directory.
- The live OneChart oracle fixture reports `prompt_n` 308. If older docs say
  309, use the fixture/source value unless live source has changed again.
- D4 prefill half 1 is committed source/test proof. A correct prefill setup has
  `DecoderFamily::Opt`, `DecoderConfig::onechart`, `nn::relu`,
  `build_inputs_embeds`, learned offset-2 positions, LayerNorm-with-bias, biased
  q/k/v/out/fc1/fc2 linears, tied head, `onechart_proj_out.bin`, and
  `onechart_final_logits.bin`.
- If D4 prefill fails, first confirm `FOCR_ONECHART_DIR` points at the real
  OneChart model directory with oracle blobs, then check prompt length 308 and
  the `<imgpad>` 50265 mask count. The expected last-position argmax is 50268
  (`<Number>`), with cos `1.00000000` and maxabs around `6.1e-5` in the armed
  close evidence.
- D4 cached decode support is committed source/test proof. A correct decode
  setup has `generate_greedy_kvcache`, `GotDecodeWeights`, `family_norm`,
  learned positions, no RoPE, output-proj bias, final norm bias,
  `opt_kvcache_matches_greedy_and_oracle`, a 24-token KV-cache vs O(n^2)
  re-prefill comparison, measured 13-step exact prefix, gate >=12-token exact
  prefix, first id 50268, and dict-open decoded output.
- If D4 cached decode fails, inspect the current source before deciding which
  artifact is authoritative. Current source can run from `model.safetensors`
  but prefers `onechart.int8.focrq` when present so the B9 identity leg compares
  same-quantization weights. A later near-tie around the measured horizon is not
  the same as a step-0/1 structural decode bug.
- Do not diagnose a missing `focr chart` as a D4-prefill or D4 cached-decode
  failure; `focr chart` is still not a current subcommand. Diagnose missing
  `--task chart-data` as stale binary/revision first, because D7 is closed in
  current source.
- D5 native recognition assembly is closed in `bd-3jo6.4.5`: a correct source
  snapshot has `ChartResult`, `recognize`, a fixed 308-id prompt,
  `complete_json_string`, `<Number>`/`prefill_final_hidden`, `number_head`,
  `reliable_distance`, `recognize_reads_the_committed_chart`,
  `reliable_check_matches_upstream_goldens`, `number_head_matches_golden`, and
  `chart_prompt_ids_match_oracle_l0c`.
- D6/D7/D8 are closed in current source: a correct source snapshot has
  `OcrTask::ChartData`, `model_spec_is_knowably_not_onechart`,
  `forward_onechart`, `model_arch implemented=true`, and
  `scripts/onechart_chart_e2e.sh`. If those are absent, the binary/source is
  stale relative to the current skill.
- If `--task chart-data` fails with a GOT/Smol/default-named model, that is the
  intended usage guard (exit 2). Re-run after `focr pull onechart` or with a
  local `onechart.int8.focrq` plus tokenizer sidecars.
- If `--task chart-data` runs but chart text quality looks poor, inspect the
  scoped corpus facts before calling it a port bug: `bd-2lje` records number
  head 6/6 with mean distance about 0.015 int8 / 0.014 f32, f32-vs-int8 decoded
  text byte-identical, and valid JSON only 1/6 in both precisions.
- `focr pull onechart` is current after `bd-av64.7`; if it fails, inspect
  `focr models --json` `pull.in_manifest` and classify the binary/manifest as
  stale before filing a route bug. `focr chart` remains unavailable; that is
  CLI-sugar scope, not a runtime-forward bug.

## TrOMR Local Runtime Lane

Use:

```bash
FOCR_TROMR_DIR=/path/to/tromr sh scripts/tromr_convert_e2e.sh
FOCR_TROMR_DIR=/path/to/tromr cargo test tromr_real_artifact_roundtrips_byte_exact -- --nocapture
cargo test group_norm_matches_torch_golden tf_same_pad_amounts_match_timm max_pool2d_same_matches_timm_golden -- --nocapture
FOCR_TROMR_DIR=/path/to/tromr cargo test tromr_encoder_matches_torch_oracle -- --nocapture
FOCR_TROMR_DIR=/path/to/tromr cargo test tromr_decoder_matches_argmax_oracle -- --nocapture
FOCR_TROMR_DIR=/path/to/tromr cargo test tromr_preprocess_envelope_and_output_gate -- --nocapture
FOCR_TROMR_DIR=/path/to/tromr sh scripts/tromr_music_e2e.sh
```

Rules:

- `tromr.focrq` is both reference artifact proof and local runtime input, but only when
  paired with the four tokenizer tables and a single-staff image or v1
  printed/scanned full-page score. Expected E2 output is `model_id=tromr`, 260
  tensors, `0 int8`, and byte-exact roundtrip against the WS-folded export.
- TrOMR tokenizer failures usually mean missing or malformed
  `tokenizer_rhythm.json`, `tokenizer_pitch.json`, `tokenizer_lift.json`, or
  `tokenizer_note.json` beside the artifact. The loader requires WordLevel,
  dense id spaces, and sizes 260/71/7/2.
- `group_norm`, `tf_same_pad`, and `max_pool2d` are E3 shared helper kernels.
  `TromrEncoderW` / `tromr_encoder_matches_torch_oracle` are the committed E3
  encoder proof. Failures there are kernel/encoder parity problems; they need
  E7/E9 evidence before claiming a callable route.
- `TromrDecoderW`, `decoder_forward`, `generate`, `FOCR_TROMR_SAMPLE`, and
  `tromr_decoder_matches_argmax_oracle` are committed E4 decoder-conformance
  evidence.
- `merge_semantic` and `semantic_to_musicxml` are E7 assembly evidence.
- `tromr_staff_tensor`, `tromr::recognize`, `MusicResult`, `forward_tromr`,
  `model_spec_is_knowably_not_tromr`, `model_arch implemented=true`, and
  `tromr_music_e2e/v1` are E9 local-runtime evidence.
- If `focr pull tromr` fails, first inspect `focr models --json`
  `pull.in_manifest` / `pull.quants`; current source after `efccce9` lists
  TrOMR with `int8` and `f32` quants. Older binaries or custom manifests may
  still expose only the historical f32 entry or lack TrOMR entirely. Default
  `focr pull tromr` should install `tromr.int8.focrq`; use
  `focr pull tromr --quant f32` or an explicit `--model tromr.focrq` when
  debugging against the f32 reference.
- If a full-page score fails, first classify whether it is inside the E5 v1
  detector scope. Current source handles printed/scanned pages with global
  deskew and five-line staff groups; it does not promise camera dewarp, warped
  phone captures, or default/lossless barline quality. If an over-budget staff
  band is the problem, `FOCR_TROMR_SPLIT=1` can be tested as experimental rescue
  after closed `bd-av64.4`, but do not make it the default support answer. For
  full-page regressions, inspect
  `staff_detect::detect_staves`, `recognize_page`, `staves_to_musicxml`, and
  `tromr_page_detects_and_reads_stacked_examples`.
- If exact clef expectations differ between G2 and F4, check `DISC-004` and
  `tromr_alpha_ink_path_fires_only_when_alpha_varies`. Current source keeps the
  inverted-alpha path only for varying-alpha rendered inputs and sends fully
  opaque RGBA through RGB luma so pages do not blank.

## TrOMR Staff Skip Notes

Symptoms:

- Full-page TrOMR exits 0 but stderr says one or more staff crops were skipped.
- Output MusicXML is present but shorter than expected.
- `FOCR_TIMING=1` shows per-staff dims/outcome lines.

Rules:

- `bd-av64.2` changed `recognize_page` from page-abort to skip-on-bad-staff
  when at least one staff succeeds. The current source records
  `PageRecognition { staves, skips }` and
  `StaffSkip { index, bbox, reason }`, then keeps stdout as data and prints the
  human skip note on stderr.
- If every staff fails, the error should name every staff reason. That is
  different from a successful partial page with warnings.
- Do not stop at the human skip note when the caller needs machine-readable
  observability. Current `bd-av64.2` source also emits robot `staff` events and
  music-run JSON `staves` arrays using `MusicPageMeta`,
  `OcrModel::take_music_meta`, `OcrEngine::take_music_page_meta()`,
  `robot::staff_event`, and `music_meta_to_json`.
- If robot consumers see no `staff` events, first suspect a stale binary, a
  non-music route, missing TrOMR artifacts, or a run that never entered the
  full-page music path. Recheck `focr robot schema`, the exact binary path, and
  `src/robot.rs::EVENT_KINDS`. The emitted event name is `staff`; the older
  `staff_detection` / `staff_result` names are stale bead acceptance wording.
  Schema version remains v1 because `staff` is additive.
- If `--json` output lacks `staves`, check that the run is a music run and that
  it used the TrOMR model path. Non-music shapes intentionally have no
  `staves` key. PDF+music multi-page runs currently expose only the last page's
  staves through the side channel; single-image music is the documented path.
- Do not treat the skip note as the geometry fix. Skip recovery is `bd-av64.2`;
  crop shaping is the later `eb0c70e`/`40ee875` fit-first mechanics, and broad
  note-level SER/camera-dewarp/default-barline-quality/int8/perf claims need
  their own evidence.

## TrOMR Crop Geometry Boundary

Symptoms:

- Wide staff systems no longer fail purely because page margins made the crop
  resize past the 1280-column position table.
- `FOCR_TIMING=1` or staff skip reasons still mention an unfittable staff.
- `br show bd-av64.14 --json` should show the scoped fit-first geometry closure
  in current source; if it does not, treat the tracker as stale until source and
  tests are checked.

Rules:

- Source at or after `40ee875` is fit-first. Bands that already fit the
  1280-column position budget keep the historic full-width geometry. Only
  over-budget bands trim horizontal page margins to the detected ink extent,
  pad by line spacing, and grow vertically toward neighbor midlines until they
  fit when possible.
- Truly unfittable bands are still emitted to the existing per-staff recovery
  path; a named skip/error is correct.
- Focused proof is unit/page-level:
  `fitting_bands_keep_the_classic_full_width_geometry`,
  `trim_cuts_page_margins_but_keeps_ink`,
  `wide_staff_with_room_fits_the_positional_budget`,
  `packed_staves_stop_at_the_midline`,
  `unpressured_band_keeps_the_generous_margins`,
  `tromr_page_skips_overwide_staff_and_keeps_the_rest`, and
  `tromr_page_all_staves_failing_is_a_named_error`.
- Keep `bd-av64.14` scoped in your wording: do not claim broad real-scan SER,
  dewarp, default/lossless barline quality, int8, or performance from crop
  tests alone. `FOCR_TROMR_SPLIT=1` is separate experimental rescue evidence.

## TrOMR Real-Scan Music Gate

Use:

```bash
FOCR_TROMR_DIR=/path/to/tromr sh scripts/realscan_music_gate.sh
br show bd-av64.6 --json
```

Symptoms:

- The runner exits 0 and says `SKIP` because local TrOMR artifacts are absent.
- A case reports `XFAIL` for the double-dotted system.
- p055 or p100 reports fewer recognized staves than its truth-data floor.
- A frozen MusicXML golden differs.
- A GOT cross-reference over a cropped staff returns molecule/SMILES-like text.

Rules:

- Skip-with-SUCCESS means no local weights; it is not real-scan quality proof.
- The gate emits `realscan_music/v1` NDJSON. Preserve stdout as data and stderr
  as human `RSMU` telemetry.
- Tier 1 truth is `truth/attributes.json`: human-verified attributes, spot
  notes, and current full-page `min_recognized` floors. Tier 2
  `goldens/*.musicxml` are frozen model output anchors, not human truth. Tier 3
  page checks count robot `staff` events against those truth-data floors.
- XFAIL is deliberate and counted. XPASS is a failure because the fixture should
  be promoted.
- p055 and p100 are promoted floor checks after `40ee875`, not page-level
  XFAILs. The page checker reads their floors from `truth/attributes.json`
  after `91d552f`: p055 floor 5, p100 floor 1. The double-dotted system XFAIL
  still remains after closed `bd-av64.13`; lever 1 residual-skew refinement was
  corpus-safe, while the later `FOCR_TROMR_TTA=3` vote and single-staff
  refined-crop routing were measured negative and reverted.
- `music_warning` events and JSON `warnings` are current after closed
  `bd-av64.5`. If absent, distinguish stale binary/pre-bd-av64.5 source from
  genuinely warning-free output. If present, treat them as annotate-only
  recognition telemetry, not MusicXML correction. If robot schema/golden docs
  omit `music_warning` on source at/after `0b74af0`, suspect a stale fixture or
  stale installed binary before inventing schema v2.
- Per-band residual-skew refinement is current after `39651e6`, but only for
  detected page bands. Closed `bd-av64.13` specifically measured single-staff
  refined-crop routing as negative on no17_top, so keep the certified
  whole-image path for pre-cropped staves unless a later held-out-corpus gate
  supersedes it.
- If local source or an installed binary produces `tromr.int8.focrq`, verify
  that it also has the matching `efccce9` / closed `bd-av64.12` manifest and
  source behavior before relying on it. The current artifact is published and
  pullable, but it is a storage artifact: 40 decoder GEMMs are stored as
  `QInt8PerChan`, then dequantized through `Weights::mat()` / `Weights::vec()`.
  Do not debug it as an int8 compute path or cite a perf win.
- `bd-10sb.1` is closed-current property/fuzz verification plumbing. If local
  source has `tests/property_suite.rs`, `proptest`, or `fuzz/` targets, do not
  describe them as new CLI/runtime behavior. Current committed anchors include
  `focrq_parse`, `safetensors_parse`, `image_decode`, `pretok_split`, public
  `tokenizer::pretok`, `cargo test --test property_suite`, and a
  decompression-bomb guard in `decode_reader`
  (`decompression_bomb_png_is_rejected_before_allocation`) after fuzz found a
  tiny PNG that declared a huge image and tried to allocate tens of GB.
  `bd-4yks` is now closed for CI/gate hardening after `e80360b`, `cc79d70`,
  `c960b77`, `29aa40a`, `7777e34`, `2e5801b`, `18712cc`, and round-8 follow-on
  `3f3d9d0`: macOS+Ubuntu run
  the full `scripts/check.sh` gate, gate logs upload, `bench-guardrail` is
  advisory, dist is green, and the advisory matrix was noted 6/6 after the
  SMMLA layout-aware compare fix. Round 8 also records 4.7M fuzz runs,
  `PROPTEST_CASES=2048`, and zero new findings. Round 11 later records 3.65M
  additional zero-crash fuzz runs, `5df6395` commits 3,271 fuzz seed files
  from the post-certification sweep, and `592426c` refreshes README public
  `v0.6.0` identity/asset-size/backend prose. Do not treat `bd-10sb.1` or
  `bd-4yks` as exhaustive fuzz coverage, a `TEST_LOG_DIR` capture layer, a scheduled
  full-model self-hosted run, ARM64 Windows completion (`bd-3u97`), or release
  approval by itself. Native Windows x86_64 is supported/proven separately.
  Current release approval is OP-SG evidence from `c29a78b` / `7c7bd00` / `29516b9` (`v0.6.0`).
- `bd-av64.6` is closed for corpus-v1 measurement. Expansion to 10-20 items,
  GOT cross-reference outputs, ladder-scorecard row wiring, and aggregate SER
  remain. Do not call broad real-scan note-level SER complete.
- For GOT cross-reference, use full systems instead of narrow staff strips;
  auto-format can classify narrow strips as SMILES/molecules.

## Robot Schema Golden Drift

Symptom:

- `cargo test --test cli_robot_golden robot_schema_matches_frozen_contract_fixture -- --exact`
  or `robot_schema_advertises_version_and_all_events` fails after a `staff`
  event appears in `robot schema`.

Rules:

- Check source before inventing a schema break. At/after `adb4ee6`, TrOMR
  `staff` is included in `tests/fixtures/robot_schema_v1.json` and the
  hard-coded advertised-events assertion in `tests/cli_robot_golden.rs`. If
  `br show bd-wp8.2.2 --json` still reports open, treat it as possible stale
  tracker state until focused schema tests prove otherwise.
- This does not require a schema bump. The `staff` event is additive schema-v1
  observability from closed `bd-av64.2`.
- If the fixture/assertion are stale, update them and rerun the focused tests
  plus full `cargo test`. If they already include `staff`, debug the concrete
  diff rather than assuming the old drift still applies.

## Multi-Page OCR Confusion

Symptoms:

- `focr ocr image.png --multi-page` fails.
- `focr ocr doc.pdf --multi-page --split-spreads` or `--extract-figures`
  returns a usage error.
- JSON consumers expect page-level layout boxes or streaming `page` events from
  a multi-page run.

Rules:

- Use `focr ocr-batch page1.png page2.png --multi-page` for image lists.
- Use `focr ocr doc.pdf --multi-page [--pages ...]` for scanned PDFs.
- Current output is one cross-page markdown document with `<PAGE>` separators;
  PDF multi-page JSON does not include per-page layout boxes.
- `bd-2z0y` is closed for robot-mode streaming decoded progress events:
  additive schema-v1 `page` events appear at `<PAGE>` boundaries with
  `status:"decoded"`, `page`, `chars`, and raw `text`. They are progress/text
  events, not per-page layout boxes, split-spread extraction, or figure output.

## Output File Issues

Use:

```bash
focr ocr page.png -o page.md
focr ocr page.png -o page.json
focr ocr page.png --json -o page.md
```

Rules:

- `.json` output paths select structured JSON with `markdown` and layout boxes.
- Other extensions select markdown unless `--json` forces JSON.
- Single-image JSON has top-level `layout`; PDF JSON has per-page `pages`.
- On model load or recognition failure, no empty or partial output file should
  remain. If one does, capture the exact exit code and file size.
- Human mode with `-o` keeps stdout empty and writes a short confirmation to
  stderr.

## Figure Extraction Issues

Use:

```bash
focr ocr page.png -o page.md --extract-figures
focr ocr page.png --json --figures-dir figs
```

Rules:

- `--extract-figures` needs `-o` to derive `<output-stem>_figures/`, unless
  `--figures-dir DIR` is provided.
- Missing destination is a usage error, exit 2, before model load.
- JSON includes `figures` only when figures were written.
- A clean run may write zero figures if the model grounds no image spans.
- On model load failure, no output file or derived figures directory should be
  left behind.

## Crop Mode and Preprocess Flags

Use:

```bash
focr ocr page.png --base-size 512
focr ocr page.png --crop-mode gundam
```

Rules:

- `--base-size`, `--image-size`, and `--crop-mode` are no longer dead flags.
  They are mapped into `PreprocessOverrides`.
- Default `base` means the certified single 1024-pixel global view.
- `gundam` means reference dynamic tiling. `bd-1e9n` records first e2e evidence
  on page_0107 (`rc=0`, 7 views, CER 0.0179 / WER 0.0138), but not a full
  corpus parity certificate.
- If docs say the flag is parsed but ignored, or that Gundam has no e2e evidence
  at all, check current source, `artifacts/perf/bd-1e9n/`, and `br show
  bd-1e9n --json`.
- If a run changes output after `--base-size` or `--crop-mode`, that can be the
  expected proof that the flag is alive; compare against target-corpus parity
  before calling it a regression.

## Format Mismatch

Exit code: 7.

Likely causes:

- `.focrq` generated by an incompatible converter.
- Corrupt or partial artifact.
- Runtime binary older/newer than artifact format.

Fix:

1. Verify artifact size/hash if available.
2. Re-run `focr convert model.safetensors -o model.focrq --quant int8` with current source.
3. Re-run `focr robot selftest`.
4. Do not retry the same artifact in a loop.

## Robot Output Is Not JSON

Check:

```bash
focr robot schema | jq .
focr robot run page.png | head -5
```

Likely causes:

- stale binary,
- human-mode command used by mistake,
- diagnostics leaked to stdout,
- parser expected JSON array/object instead of NDJSON.

Fix parser first if it expects one JSON document.

## Cancellation and Thread Budget

Current `bd-223.2` source adds cooperative cancellation and a single thread budget:

- first Ctrl+C requests shutdown and should surface `Cancelled` / exit 6 at the
  next `cancel_checkpoint()` boundary,
- a second Ctrl+C hard-exits 130 for wedged work,
- embedders can call `request_shutdown()` and `reset_shutdown()` around a
  process-global shutdown flag,
- `FOCR_THREADS` sets the process-wide `thread_budget()` when positive;
  otherwise physical cores win,
- `robot health` and `robot backends` report `threads` when that source is
  present.

If cancellation does not fire, inspect `cancel_checkpoint` call sites in
`src/native_engine/mod.rs`, `decoder_qwen2.rs`, and `tromr.rs`. If `threads` is
missing from robot output, first classify the binary as pre-bd-223.2,
stale-local, or release-lagged before changing docs.

## PDF InputDecode

Exit code: 4.

Native PDF support is a scanned-image fast path. It handles common scan codecs
such as JPEG (`DCTDecode`), CCITT Group 4 fax, and `FlateDecode`/LZW raw rasters.
It intentionally rejects unsupported PDF inputs with precise messages:

- `JPXDecode` / JPEG 2000,
- `JBIG2Decode`,
- unsupported color spaces or bit depths,
- born-digital/vector/text pages with no image XObject.

Fix:

1. Read the `InputDecode` message; it names the unsupported page/codec.
2. Rasterize that PDF out of band, for example with `pdftoppm`.
3. Retry with the generated page images.
4. For library integrations, use `pdf::PdfPages` + `OcrEngine::recognize_dynamic`
   only when the PDF is a scanned-image PDF.

## Command Missing

If a command is in this skill but not live help:

1. Inspect `src/cli.rs`.
2. Check `git status --short`.
3. Run from source if feasible.
4. Check Beads for whether the command is scaffolded.
5. Report stale or unimplemented state clearly.

Do not update downstream docs to match an old binary.

## Selftest Fails

Use `selftest` as a fast machine-readable smoke test, not as full OCR parity.

First moves:

```bash
focr robot selftest | jq .
focr robot health | jq .
focr robot backends | jq .
```

`robot selftest` checks SIMD/int8 kernel parity and does not need OCR weights.
At/after `ad3ad20`, inspect `.models[]` for per-model verdicts
(`unlimited-ocr`, `got-ocr2`, `smolvlm2`, `onechart`) and `.cases[]` for the
underlying shape failures. TrOMR is absent from `.models[]` because its current
runtime dequants the published storage-int8 artifact to f32 rather than using an
int8 decoder kernel. If selftest fails, treat it as a backend
correctness problem: capture selected tier, `FOCR_FORCE_ARCH`, model verdicts,
case failures, source revision, and host CPU before claiming support.

## Dense Repetition in Output

OCR text repeating phrases may involve decode controls. Check
`FOCR_NO_REPEAT_NGRAM` for Unlimited-OCR and `FOCR_GOT_NO_REPEAT_NGRAM` for
GOT-OCR2. GOT now defaults to a global no-repeat n-gram size of 20, but
`--no-repeat-ngram` / `FOCR_NO_REPEAT_NGRAM` wins over `FOCR_GOT_NO_REPEAT_NGRAM`
and `ocr-batch` reads `FOCR_NO_REPEAT_NGRAM` too. Setting either guard to `0` is
a diagnostic kill-switch, not a production recommendation. Do not "fix" by
silently changing decoding defaults in an integration; record a discrepancy or
issue with image, model hash, revision, and output.

## Speculative Decode Drift

Use:

```bash
FOCR_SPEC_E2E_IMAGES=/path/to/pages FOCR_MODEL_PATH=/path/to/model.focrq scripts/spec_gate_e2e.sh --no-build
```

Rules:

- `FOCR_SPEC_DECODE` is presence-armed. `FOCR_SPEC_DECODE=0` still turns it on;
  scripts use `env -u FOCR_SPEC_DECODE` for the OFF arm.
- `bd-1azu.36` proves the LINEAR gate ON==OFF over 20 pages in f32 and int8.
  A new drift means the gate regressed or the environment composed in another
  lever.
- That proof is output identity, not proof that the spec loop engaged on every
  ON run. If engagement matters, first add/read telemetry such as spec-round
  timing or acceptance counters.
- Do not certify with `FOCR_ATTN_GEMM` or `FOCR_INT8_KV` present; those rejected
  key-batch levers are explicitly barred from the spec path.
- Preserve the failing workdir from `scripts/spec_gate_e2e.sh` and compare the
  `.off.md` and `.on.md` outputs before changing tolerances.

## Reference Resampling Differences

If an oracle comparison shows L0/preprocess drift, rerun with:

```bash
FOCR_RESAMPLE=pil-bicubic focr ocr page.png --json
```

`pil-bicubic` is the Pillow-bit-exact reference resampler from `bd-30me`; it is
for parity investigation and oracle reproduction. The product default remains
CatmullRom under `DISC-001`, so do not present `FOCR_RESAMPLE=pil-bicubic` as a
universal quality fix unless a measured A/B on the target corpus supports that.

## Gauntlet or Perf Claim Confusion

Use:

```bash
bash scripts/gauntlet_runbook.sh preflight
cat artifacts/perf/bd-re8.17/arch.json | jq .
br show bd-re8.17 --json
```

Rules:

- `scripts/gauntlet_runbook.sh` is the current quiet-host path for the pinned HF
  baseline, thread parity, page hashes, CER, roofline, and PERF_LEDGER row draft.
- `artifacts/perf/bd-re8.17/arch.json` is a `robot selftest` proof of selected
  SIMD tier and 24/24 kernel cases. It is architecture/kernel evidence, not by
  itself a performance row.
- `bd-re8.17` is closed on current source with first PERF_LEDGER rows:
  decode-per-token about 1.62x faster, prefill about 4.8-5.0x, vision about
  3.6-3.7x, and end-to-end 2.34-2.71x against pinned HF bf16 @8 on
  `page_0009`/`page_0014`, with CER 0.00943 / 0.03529. Cite the table exactly:
  `page_0009` preprocess is 0.916, and a stale "No performance numbers" footer
  may still trail the populated rows.
- If a gauntlet rerun lands suspicious data, check whether `raw/run_*.meta.json`
  came from a previous session and whether focr/reference docs name the same
  page. Current scripts fail closed on both conditions.

## A11 Zoo Ratio Confusion

Symptoms:

- A README or release note says "the zoo is faster" without naming the stage.
- Someone cites `3.37x`, `2.58x`, or `1.67x` as end-to-end speedup.
- Dense batching rows are mixed with matched HF CPU reference rows.

Rules:

- The v0.4.0 A11 zoo summary is stage-specific: GOT-OCR2 `3.37x`, OneChart
  `2.58x`, and SmolVLM2 `1.67x` are matched-thread Apple SDOT
  decode-per-token ratios over pinned Hugging Face CPU references.
- `docs/PERF_LEDGER.md` also keeps end-to-end rows, including slower totals
  when artifact loading or preprocessing dominates. Do not hide those rows.
- Dense batch rows answer a different question: they are same-engine
  batch-vs-sequential rows, not HF-reference rows.
- TrOMR's `bd-2sez` row is a f32 baseline loss row and should not be grouped
  with the int8 zoo speedups.

## Release Evidence Instrument Confusion

Use:

```bash
python3 scripts/gauntlet_cert.py --release-readiness
python3 scripts/gauntlet_cert.py --bundle || true
python3 scripts/gauntlet_cert.py --eprocess-fold test-log.ndjson --eprocess-state /tmp/focr-eprocess-state.json
cargo test --test many_pages_without_deadlock capacity_certificate_bounded_stream_soak -- --nocapture
```

Rules:

- The conformal ratchet (`docs/conformance/RATCHET.md`), Ville e-process
  monitors, and capacity certificate are release-evidence instruments. They do
  not mean "ship-ready" until the release-readiness scorecard and capstone bead
  agree.
- If `FOCR_FIXTURES_DIR` or model fixture roots are missing, the run is unarmed
  or partial evidence. Report that state explicitly.
- A red `certification_bundle` or `gauntlet_convergence` cell blocks a ship
  claim even when an individual instrument passes. Current `c29a78b` evidence
  has both cells green, `ship:true`, `green:13`, `red:0`, and
  `rounds=11/10, tail_clean=True`.
- `--bundle` writes bundle artifacts even in an unconverged state; read
  `docs/gauntlet/bundle/release_certificate.json` and `br show bd-wp8.9 --json`.
  Older bundle machinery can honestly report `certified:false` and exit
  non-zero. After `9bc715e`, `certification_bundle` is live, not a static red
  cell: it reads `release_certificate.json`, and `--bundle` excludes that cell
  from its own certification predicate. Current `c29a78b` evidence reports
  `certified:true`.

## Benchmark Guardrail Confusion

Use:

```bash
python3 scripts/bench_guardrail.py --self-test
python3 scripts/bench_guardrail.py --stages artifacts/.../focr_stages.json \
  --baseline benches/.bench-history/baseline.json \
  --parity-receipt tests/fixtures/ladder_scorecard/scorecard_armed.json
br show bd-1a6h --json
```

Rules:

- The guardrail catches regressions against a frozen baseline. It is not a
  head-to-head proof that focr beats the reference.
- A >10% slower current stage fails by default.
- `cv_pct > 5` on either side is `noise_ineligible`; it must not be reported as
  a fake win/loss.
- Thread/allocator/precision mismatch is `posture_mismatch`; compare only
  matching fairness posture.
- Missing stages/baseline/parity receipt is skip-green and logged because CI may
  lack multi-GB fixtures.
- Perf reporting is refused unless the parity receipt is all-green.
- Baselines move only with reviewed `--ratchet`, never automatically in CI.

## GOT SAM Timing Confusion

Use:

```bash
FOCR_TIMING=1 focr ocr tests/fixtures/got/sample_text.png --model got-ocr2.int8.focrq
br show bd-av64.10 --json
```

Rules:

- Current `bd-av64.10` measurement says the old artifact-hydration hypothesis is
  wrong for the observed GOT e2e gap: `sam.hydrate` is negligible and
  `Weights::load` is not the bottleneck.
- `01f07fe` landed a bit-identical SAM attention pass, `f3d3215` landed
  the second bit-identical pass with head-parallel global attention across
  disjoint output spans, `0298651` landed the committed CLIP pass with
  pre-transposed `LinearParams` and model-level `ClipWeights` cache, and
  `f65fded` landed the shared `vision_sam::Linear::from_row_major` pass across
  SAM/GOT/OneChart/SmolVLM2/SigLIP/TrOMR linear consumers. `f1ac972` adds
  SmolVLM2 SigLIP frame batching through
  `vision_siglip::forward_frames_batched`, with `FOCR_SIGLIP_SEQ=1` as the
  sequential kill switch and `batched_frames_match_sequential_byte_for_byte` as
  the byte-identity proof. Current
  self-relative evidence says `attn(win)` 1.88s to 0.72s; `attn(GLOBAL)` 2.10s
  to 1.66s; `sam.forward` 5.55s to 4.24s to 3.4-3.6s; GOT forward 6.7s to
  5.7s to 4.6-4.8s; unlimited-OCR real page 19.3s to 13.5s (-30%); and
  `vision.clip` 2.49s to 0.77s steady-state, with byte-identical output, armed
  GOT/L2 certs, `vision_sam` 37/37, `vision_clip` 41/41, full-lib 957 green,
  and `clippy -D` clean where cited by the matching pass. `3c1b1ea` records
  the pass-3/pass-4 Beads evidence; the later `efd83e8` row bundle is the
  formal `bd-av64.10` closeout.
- Public `c5e535a` row-tiles the SAM global-attention score buffers, public
  `8bd4037` restores the untiled baseline, and public `b757bc0` records
  `artifacts/perf/bd-av64.10-rowtile/` plus the negative-evidence row after
  same-regime measurement found the tiled path byte-identical but slower. Treat
  row tiling as a negative/reverted current-main lever, not a current suspect or
  closure.
- At/after `3f2f97e`, `scripts/gauntlet_timing.py` parses the current SAM/CLIP
  drill-down labels, including `sam.hydrate`, `sam.forward`, `sam.blocks`,
  `sam.block attn(GLOBAL)`, `sam.block attn(win)`, `sam.block mlp`,
  `clip.hydrate(cached)`, and `clip.blocks`. Inspect raw `[focr-timing]` stderr
  only for newer unknown labels. `c248e6d`/`c4c1684` later close `bd-2mo.26`.
- Do not keep SAM SIMD-exp softmax on the fresh-suspect list. `ab6e083`
  measured polynomial-exp softmax dead, reverted it, and recorded the negative
  ledger; `50d5dad` adds `artifacts/perf/bd-av64.10-simd-exp/`. The experiment
  had tiny softmax drift but forked greedy OCR text, so numerics substitutions
  need token/output fixture gates, not just unit error bounds.
- Do not keep row-tiled SAM attention on the fresh-suspect list either now that
  `b757bc0` made the negative-evidence row public.
- `3f2878d` makes `GotStatics`, `got_statics`, and `got.hydrate(cached)`
  current source for sequential/batch GOT statics caching. Diagnose missing
  labels as stale binary/source or a real regression; do not call the cache dirty
  WIP anymore, and do not promote pass 6 alone to formal A11/PERF_LEDGER proof.
- `8de3674` makes the source version `v0.5.1` and adds `memmap2` to Cargo, and
  `507cebe` ships the mmap loader. `4cedacd` makes source/tag `v0.5.2`, and
  the `v0.5.2` GitHub release is a historical live release with platform
  assets; latest observed release is `v0.6.0` at `29516b9`. `0924479` clarifies
  README source-tag/binary-release wording after the tag; check `focr --version`
  and the asset actually installed before support claims.
- `38ab806` / `a9a406e` make OneChart statics caching committed pass-7 source
  evidence, and `9b2a03b` makes SmolVLM2 statics caching committed pass-8
  source evidence. Trust `onechart.hydrate(cached)` /
  `smolvlm2.hydrate(cached)` only within those scoped source boundaries.
- `efd83e8` closes `bd-av64.10` as measured final state: nine rows under
  `artifacts/perf/bd-av64.10-g2r/`, GOT e2e `0.624 -> 0.885`, OneChart
  `0.546 -> 0.755`, SmolVLM2 `0.878 -> 0.890`, and decode-per-token
  `3.046x / 2.249x / 1.499x`. This is not an e2e `>=1.0x` win; the remaining
  gap belongs to load-inclusion bias plus vision f32 GEMMs and future `bd-2mo`
  kernel work.
- Remaining suspects are whatever fresh timings show. Do not spend the first
  pass optimizing artifact load unless fresh timings contradict this, and do not
  turn self-relative pass evidence into a new final A11/PERF_LEDGER claim.
- `d25dbd7` is a separate committed batch-path hoist: GOT-OCR2 batch runs
  hydrate SAM/projector/embed state once per batch and log `got.hydrate(batch)`.
  It measured 14.47s sequential vs 13.53s batch on a same-binary 3-page run
  with byte identity. Do not attribute the larger SAM-attention speedup to this
  hoist.
- `3f2878d` supersedes that as the current public GOT statics boundary:
  `OcrModel` caches `got::GotStatics` once, sequential and batch page paths
  reuse the SAM/projector/embed state, `FOCR_TIMING` logs
  `got.hydrate(cached)`, and the measured claim is about 0.8s/page saved on the
  sequential page loop. Keep it scoped to committed pass-6 evidence, not
  release readiness or the formal G2 closeout by itself.

## Batch Spine Confusion

Use:

```bash
FOCR_BATCH_SPINE=1 FOCR_BATCH_SIZE=64 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=1 FOCR_BATCH_PACK=1 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=0 focr ocr-batch page-*.png --json
br show bd-3jo6.1.7.5 --json
```

Rules:

- `FOCR_BATCH_SPINE` is value-parsed. Unset, empty, `0`, `off`, `false`, and
  `no` disable; any other present value arms.
- Default Unlimited-OCR int8 and dense zoo models have separate spine routes.
  Default Unlimited-OCR uses the original R-SWA batch scheduler and batched
  vision path. Closed `bd-3jo6.1.7.5` covers dense GOT-OCR2, SmolVLM2, and
  OneChart decode batching under `FOCR_BATCH_SPINE=1`.
- For dense zoo debugging, confirm `OcrModel::recognize_batch_dense`,
  `matches!(self.arch().id(), "got-ocr2" | "smolvlm2" | "onechart")`,
  `smolvlm2::recognize_batch`, `onechart::recognize_batch`,
  `generate_greedy_batched(..., caps: &[usize], ...)`,
  `PageStream::with_max_emit`, `DEFAULT_BATCH_SIZE = 128`,
  `MAX_BATCH_SIZE = 256`, and `FOCR_BATCH_PACK`.
- Dense zoo proof is lossless/current after `4ca1577` and `fdd1d64`: per-stream
  step gates, scheduler mixed-cap/EOS identity, real binary byte-identical
  markdown for GOT/SmolVLM2/OneChart, and the model-gated GOT e2e gate
  `recognize_batch_matches_sequential_e2e`.
- Source at or after `3f2878d` adds the current GOT-specific statics cache on
  top of the dense-zoo batch proof: `OcrModel` owns `got::GotStatics` for the
  SAM tower, `mm_projector_vary`, and widened embed table, and both sequential
  and batch GOT page paths reuse it. Probe `got.hydrate(cached)` when
  diagnosing stale binaries or missing setup amortization. For older
  `d25dbd7`-only builds, the narrower batch hoist appears as
  `got.hydrate(batch)` and `got.vision+splice(batch of N)`.
- If packed admission is involved, debug "wrong page got wrong text" by checking
  input-order restoration and stream ids before inspecting tokenizer/model
  quality. Packing is supposed to change only admission grouping, never emitted
  token streams.
- Current dense batch evidence is lossless and has scoped self-relative
  throughput rows: SmolVLM2 74.6s vs 98.3s (1.32x) for 700 decode tokens,
  OneChart 2.39s vs 3.04s (1.27x) for 438 tokens, and GOT remains
  vision-dominated at roughly +3% to +16% on the cited fixtures. Do not turn
  that into broad batched `lm_head`, A11/PERF_LEDGER, or decode-heavy B>=8 proof.

## QKV Fused Decode Confusion

Use:

```bash
FOCR_QKV_FUSED=0 focr ocr page.png --json
br show bd-241s --json
br show bd-1waa --json
br show bd-3pg7 --json
rg -n "qkv_fused_enabled|fuse_qkv|fused_qkv_gemv_is_byte_identical_to_three_calls|FOCR_QKV_FUSED" src/native_engine/decoder.rs docs/NEGATIVE_EVIDENCE.md
```

Rules:

- Current source makes fused q/k/v int8 decode the default. `FOCR_QKV_FUSED=0`,
  `off`, `false`, or `no` disables it and restores the older three-call path.
- Trust `qkv_fused_enabled()` and the `bd-241s` close evidence over stale
  adjacent comments that still sound like the fused path is opt-in.
- `bd-1waa` kept fused q/k/v as the lossless win; `bd-3pg7` resolved the R-SWA
  attention-GEMM idea as real-speed-but-not-bit-exact. Do not revive
  `FOCR_ATTN_GEMM` or broad `FOCR_INT8_KV` without a new bit-exact gate,
  page-level SHA/CER proof, and 20-page CER proof.
- The current win is decode projection only. Do not claim prefill q/k/v fusion.

## Ngram-Lmhead Fusion Confusion

Use:

```bash
FOCR_FUSE_NGRAM_LMHEAD=1 focr ocr page.png --json
br show bd-2mo.24 --json
rg -n "FOCR_FUSE_NGRAM_LMHEAD|fused_ngram_lmhead_is_byte_identical_to_separate_mask" \
  src/native_engine/decoder.rs docs/NEGATIVE_EVIDENCE.md
```

Rules:

- `FOCR_FUSE_NGRAM_LMHEAD` is implemented and bit-identity tested, but
  `bd-2mo.24` / `a0ad299` ledgered it as correct-but-does-not-pay.
- The measured ngram-heavy page_0023 A/B was decode 16.43s -> 16.40s, inside
  noise, with all outputs byte-identical.
- Keep it opt-in/off by default. Do not describe it as a speed win, a default,
  or a good next lever.
- Retry only when the workload changes the arithmetic: multi-image
  `ngram_window=1024` ban sets or a roughly 10x faster decode step. Rerun the
  same A/B first.
- Contrast with `FOCR_QKV_FUSED`: fused q/k/v measured 1.40x and became the
  default. The rule is measure-then-decide, not fuse-everything.

## Windows Notes

Current source/docs say native Windows x86_64 OCR and `focr pull` both work,
including the full multi-part model download, reassembly, and SHA-256 verify.
If Windows pull regresses, treat it as a new distribution bug and capture the
exact OS error, manifest source, binary revision, and whether `robot selftest`
still passes. Do not claim ARM64 Windows support without current evidence.

## Build Takes Too Long

The project is Rust nightly and can compile heavy CPU-kernel code.

Options:

```bash
rch exec -- cargo check --all-targets
cargo check --all-targets
```

For docs/skill updates outside franken_ocr, do not run franken_ocr's full cargo
gate unless the task changed source or requires live binary proof.

## No CASS Hits

When CASS returns no useful `focr` history:

1. Confirm `cass status --json`.
2. If stale but usable, launch a capped background refresh.
3. Search without overly narrow workspace filters.
4. Fall back to source, README, docs, and Beads.

No CASS hits is not evidence that a capability is absent.
