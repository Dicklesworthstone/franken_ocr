# Artifacts and Environment

## Table of Contents

- [Model Artifact Model](#model-artifact-model)
- [Cache and Paths](#cache-and-paths)
- [`.focrq` Format](#focrq-format)
- [Conversion Lane](#conversion-lane)
- [Runtime Env Vars](#runtime-env-vars)
- [Experimental Env Vars](#experimental-env-vars)
- [Offline Deployments](#offline-deployments)
- [Platform Notes](#platform-notes)
- [License Notes](#license-notes)

## Model Artifact Model

`franken_ocr` primarily targets Baidu Unlimited-OCR and now has a model
architecture registry for the zoo. Real inference needs either:

- source safetensors weights for conversion, or
- a packaged `.focrq` artifact and tokenizer/cache state for runtime use.

The project intends no general ML framework at inference time. Model artifacts
are local files after setup.

The normal setup path is:

```bash
focr pull          # default fast plain-text OCR model
focr pull got-ocr2 # optional specialized structured-output model
focr pull smolvlm2 # optional describe/VQA model
focr pull onechart # optional chart-data model
focr pull tromr    # optional OMR model; default quantized-storage artifact
focr pull tromr --quant f32 # bit-exact TrOMR reference artifact
focr pull --manifest ./manifest.json # custom/airgapped manifest only
```

`pull` downloads packaged `.focrq` models plus tokenizer/sidecar files into the
cache and verifies every byte by SHA256. The packaged default model filename is
`unlimited-ocr.int8.focrq`; current inference lookup also probes this
quant-suffixed name when the default `unlimited-ocr.focrq` basename is
requested. That resolver behavior shipped in `v0.3.0` and is included in the
`v0.4.0` public release boundary; confirm the installed binary before
diagnosing a fresh-pull ModelNotFound report, because a user's existing binary
can still be pre-`v0.3.0` even though the current installer fallback is
`v0.4.0`. `bf28fd7` / `v0.5.0` is now a real GitHub release with platform
binaries plus SHA256 files. `48a9896` updates README/fuzz metadata to `v0.5.0`,
but installer fallback constants still lag by design. `8de3674` / `v0.5.1` is
also now a real GitHub release, published 2026-07-08T05:02:37Z with platform
binaries plus SHA256 files; `a391793` updates README/manual-release prose to
`v0.5.1`. `4cedacd` tags the source/package as `v0.5.2`, and the `v0.5.2`
GitHub release object is live, published 2026-07-08T06:05:23Z with the same
five platform binary families plus SHA256 assets. `29516b9` tags the
source/package as `v0.6.0`, and the `v0.6.0` GitHub release object is live,
published 2026-07-08T14:47:48Z with the same five platform binary families plus
SHA256 assets. `0924479` clarifies README badges/prose after the `v0.5.2` tag
without changing runtime behavior. The July 8 source probe saw public
`origin/main` at `592426c` (`v0.6.0-4-g592426c` for clean source; no tracked
source diff): `efd83e8` adds the formal `bd-av64.10` G2 closeout, `3f2f97e`
adds `bd-2mo.26` gauntlet harness hardening, `4291807` certifies and defaults
on the SmolVLM2 untied `lm_head` int8+refine runtime path, `c248e6d` lands the
`bd-2mo.26` head-to-head rows, `c29a78b` certifies the release gate, `7c7bd00`
closes the release-certification Beads, `beaed7c` records CI/dist supplement
notes, `db02421` refreshes README release-readiness evidence, `5df6395` commits
post-certification fuzz corpus growth, and `592426c` refreshes public README
release identity, binary-size, manual-download, and CPU-backend wording for
`v0.6.0`. These commits are current-source/tracker evidence but still separate
from the installed binary actually on PATH. Dirty diffs after `592426c` are
live-WIP only until committed/tests/Beads agree.
Check the installed binary and release assets before diagnosing stale behavior,
and separate normal latest-release installer resolution from fallback-only
constants.

Current `main` ships all five ready models through the committed manifest after
`bd-av64.7` / `ece14f9`. The manifest has a `models` map and
`ModelEntry.sidecars`; non-primary models install under
`~/.cache/franken_ocr/models/<model-id>/`. `focr pull got-ocr2` installs
`got-ocr2.int8.focrq` plus `qwen.tiktoken`; `focr pull smolvlm2` installs
`smolvlm2.int8.focrq` plus `tokenizer.json`; `focr pull onechart` installs
`onechart.int8.focrq`, `vocab.json`, `merges.txt`, and `added_tokens.json`;
`focr pull tromr` installs `tromr.int8.focrq` plus the four music tokenizer
tables by default; `focr pull tromr --quant f32` installs `tromr.focrq`.
A self-converted GOT-OCR2 runtime artifact still needs `qwen.tiktoken` beside
the `.focrq`.

`bd-av64.8` and `bd-av64.9` are the current publication/clean-cache proof for
the non-primary zoo releases. GitHub releases `models-smolvlm2-v1`,
`models-onechart-v1`, and `models-tromr-v1` were checked by exact
size/hash/sidecar expectations, clean-cache `focr pull`, idempotent repull, and
one real inference smoke per model. Use that as GitHub-first distribution
evidence. Do not use it as HF mirror evidence: despite the current README's
broad "Hugging Face mirror" sentence, the known Beads close evidence says mirror
spot-checks returned 401 for the weights repos and remain an auth/resilience
follow-up.

Manifest selection is a separate contract from model discovery:
`focr pull --manifest <path-or-url>` has highest precedence, then
`FOCR_MANIFEST_URL`, then the embedded `BUILTIN_MANIFEST_JSON` from
`models/manifest.json`. `focr models` reads the compiled registry and augments
rows with `pull.in_manifest` plus `pull.quants`; use those fields when
diagnosing `ModelNotFound`, missing-tokenizer, or "manifest has no model"
reports. A stale binary may have an older embedded manifest, so verify the exact
binary before contradicting source.

SmolVLM2 is now both a packaged pull target and a current runtime route. The
`smolvlm2` registry descriptor,
conversion path, and describe/VQA forward route are current: C5 text-only
decoder seam is certified, C6 tokenizer conformance is closed, C7 prompt/IO and
preprocessing is closed, C9 CLI route support is closed, C8 parity/e2e
quality/perf is closed, C10 detailed tests/e2e are closed, and sub-epic C is
closed. Current source has
`preprocess_smolvlm2` / `preprocess_smolvlm2_path`,
Pillow-bit-exact `resize_lanczos` for the `SmolVLMImageProcessor` `resample: 1`
path, `src/native_engine/smolvlm2.rs`, `--task describe`, `--question`,
`FOCR_SMOLVLM2_QUESTION`, and `model_arch` `implemented=true`. DISC-003 is the
current L4 near-tie ledger; keep it with the artifact because the fast KV-cache
path is allowed to diverge only on the ledgered near-tie shape.
`scripts/smolvlm2_convert_e2e.sh`
proves `focr convert --model-id smolvlm2` on the real 500M safetensors when the
weights are present, and skips with success otherwise. Decoder seam tests use
`FOCR_SMOLVLM2_MODEL`, `FOCR_SMOLVLM2_ORACLE_HIDDEN0`, and
`FOCR_SMOLVLM2_ORACLE_LOGITS`; oracle fixture generation uses
`FOCR_SMOLVLM2_DIR`. C6 tokenizer conformance additionally uses
`FOCR_SMOLVLM2_TOKENIZER_JSON` or `$FOCR_SMOLVLM2_DIR/tokenizer.json`, with an
optional `FOCR_SMOLVLM2_CORPUS` for regenerating committed token-id fixtures.
The C3/C4/C7/C8 vision oracle lane writes its compact metadata to
`tests/fixtures/smolvlm2/vision_oracle_fixtures.json` and large seam blobs under
`FOCR_SMOLVLM2_DIR`. Use `focr pull smolvlm2` for the committed packaged
artifact or deploy a supplied/converted `smolvlm2.int8.focrq` for
`--task describe`. For quality/perf, cite the exact C8/C10 and A11 evidence
instead of extrapolating from route support.
The newer C8 VQA guard adds `scripts/gen_smolvlm2_vqa_fixtures.py` and
`tests/fixtures/smolvlm2/vqa_fixtures.json`. It is an oracle-answer fixture over
the committed sample photo: the Rust test loads `FOCR_SMOLVLM2_DIR/tokenizer.json`
plus `model.safetensors` and/or `smolvlm2.int8.focrq`, scores answers against
the oracle's greedy text by normalized exact match or symmetric content-word
containment >=0.5, and guards f32 at >=70% / int8 at >=50%. It is a regression
signal, not a public VQA benchmark; live C8 close evidence reports 7/7 on f32
and 7/7 on int8, with int8 answers identical to f32.
`scripts/smolvlm2_describe_e2e.sh` is the C10 model-gated CLI gate for the real
int8 artifact. It uses `FOCR_SMOLVLM2_DIR` and optional `FOCR_BIN`, emits
`smolvlm2_describe_e2e/v1` NDJSON on stdout, logs human telemetry as `SVLM `
stderr lines, checks missing-model and wrong-family negative paths, then runs
describe and VQA on the committed sample photo. Live C10 close evidence says
the release-int8 armed run passed; unarmed skips still mean missing artifacts.

OneChart has current packaged distribution, conversion/tokenizer support,
certified D3 vision/projector, certified D4 prefill/cached decode, D5 native
recognition assembly, and D6-D8 public runtime chart-data routing.
`bd-3jo6.4.2`
closed `focr convert --model-id onechart` for OneChart-shaped weights: the
converter targets `Decoder::OptDense`, keeps tied `lm_head` high precision by
byte-verifying and deduping it against `model.decoder.embed_tokens.weight`,
quantizes 72 OPT decoder GEMMs, records `model_id=onechart` and the OneChart
Apache-2.0 notice, and preserves high precision for vision, projector, number
head, norms, embeddings, and biases. `bd-3jo6.4.9` closed the D9 tokenizer gate
via `Tokenizer::from_opt_dir` / `from_opt_files` over `vocab.json`,
`merges.txt`, and `added_tokens.json` with plain GPT-2 pretokenization.
`bd-3jo6.4.3` closed D3: `onechart_view_tensor` preprocesses a single
squash-resized 1024x1024 RGB view as raw `[0,1]` pixels; `vision_features`
combines `model.vision_tower` with `model.mm_projector`
`Linear(1024->768,bias)` and was certified against `onechart_proj_out.bin`
with `proj_out cos 1.00000000`, maxabs `6.5e-4`. `FOCR_ONECHART_DIR` is the
directory used for tokenizer and oracle checks; large oracle blobs such as
`onechart_preproc.bin`, `onechart_proj_out.bin`, and
`onechart_final_logits.bin` live there, while
`tests/fixtures/onechart/oracle_fixtures.json` carries compact metadata.
`20ac599` adds the D4-prefill proof: `DecoderFamily::Opt`,
`DecoderConfig::onechart`, `nn::relu`, and
`onechart::build_inputs_embeds` now cover learned offset-2 positions,
LayerNorm-with-bias, biased OPT linears, ReLU `fc1`/`fc2`, tied head, and
`<imgpad>` splice. The armed prefill gate uses `FOCR_ONECHART_DIR`,
`model.safetensors`, `onechart_proj_out.bin`, and `onechart_final_logits.bin`,
and reports argmax 50268, cos `1.00000000`, maxabs `6.1e-5`, prompt length
308. `2c77d21` added committed D4 cached decode support, and `2769d21` closed
`bd-3jo6.4.4`:
`generate_greedy_kvcache` now covers the OneChart OPT family path, and
`opt_kvcache_matches_greedy_and_oracle` uses the same oracle-vision embeds to
compare a 24-token KV-cache greedy stream with O(n^2) re-prefill greedy,
preferring `onechart.int8.focrq` when present so the B9 leg uses
same-quantization int8 weights. The committed gate records a measured 13-step
exact prefix, requires prefix >=12, first id 50268 (`<Number>`), and dict-open
decoded output. `0145419` added the recognize pipeline and hidden-state
tap, and `2a56c96` closed `bd-3jo6.4.5`: `ChartResult` carries `json_text`,
optional `pred_locs`, optional `reliable_distance`, and `reliable`; `recognize`
uses `complete_json_string`, `<Number>`/`prefill_final_hidden`, `number_head`,
and `reliable_distance`; and D5 tests include
`recognize_reads_the_committed_chart`, `reliable_check_matches_upstream_goldens`,
`number_head_matches_golden`, and `chart_prompt_ids_match_oracle_l0c`.
`e926c46` closed D6/D7/D8 and sub-epic D: `model_arch` marks
`onechart.implemented()` true, `src/cli.rs` exposes `OcrTask::ChartData`,
`native_engine/mod.rs` dispatches via `forward_onechart`, and
`scripts/onechart_chart_e2e.sh` proves the model-gated CLI route. `bd-2lje`
adds scoped SCRM-proxy corpus evidence (six charts; head fires 6/6; mean
distance about 0.015 int8 / 0.014 f32; decoded text byte-identical f32-vs-int8;
valid JSON 1/6 in both precisions). `bd-av64.7` / `ece14f9` later publishes
`focr pull onechart`; deploy the pulled artifact or a supplied/converted
`onechart.int8.focrq` with the OPT tokenizer files beside it. Do not turn this
into a separate `focr chart` subcommand or broad chart-quality claim.

TrOMR currently has a pullable OMR path for single-staff images and v1
printed/scanned full-page scores, including closed `bd-av64.2` per-staff
skip resilience, robot `staff` events, and music-run JSON `staves`. Default
pull is now `tromr.int8.focrq` after `efccce9` / closed `bd-av64.12`, with
`tromr.focrq` retained behind `focr pull tromr --quant f32` for bit-exact
reference work. The
int8 artifact is quantized weight storage with f32 dequant-on-access, not an
int8 compute path. TrOMR still lacks camera dewarp, default/lossless barline
quality, `**kern`, or unconstrained parity/perf proof. Experimental
`FOCR_TROMR_SPLIT=1` barline splitting exists after closed `bd-av64.4` as
off-by-default over-budget-staff recognition-count rescue; do not treat it as a
default quality, perf, or camera-dewarp claim.
Closed `bd-av64.5` adds annotate-only musical-sanity telemetry (`focr-sanity`
comments, robot `music_warning`, JSON `warnings`), not auto-correction.
`bd-av64.13` is closed after `69039c3`: per-band residual-skew refinement
landed at `39651e6`, while `FOCR_TROMR_TTA=3` micro-rotation voting and
single-staff refined-crop routing were both measured negative and reverted.
Do not confuse `FOCR_TROMR_TTA` with shipped `FOCR_TROMR_SPLIT=1`; one is a
negative/reverted voting experiment, the other is an off-by-default split
rescue. `bd-av64.12` closes the TrOMR int8 storage publication:
`QInt8PerChan` dequant-on-access in `Weights::mat()` / `Weights::vec()` lets
the f32 TrOMR runtime read quantized-storage tensors, and the manifest
publishes `tromr.int8.focrq`. It is still not an int8 compute proof or perf
win.
`c22b047` / `bd-3jo6.5.2` closed E2 conversion for the WS-folded export:
`tromr.focrq` self-declares `model_id=tromr`, round-trips 260 tensors
byte-exact, and contains `0 int8` tensors. `efccce9` / `bd-av64.12` later
adds the published `tromr.int8.focrq` storage artifact with 40 decoder GEMMs
quantized and all high-precision tensors preserved. `7464590` /
`bd-3jo6.5.6` closed E6 decode-only music
tokenization via four WordLevel tables beside the artifact. `6403d4c` landed
the shared E3 helpers (`tf_same_pad`, `max_pool2d`, `group_norm`, `fuse_relu`),
and `45da3a3` committed the E3 hybrid ResNetV2+ViT encoder
(`TromrEncoderW`, `tromr_encoder_matches_torch_oracle`, `encoder_out cos
1.00000000`, maxabs `3.8e-6`). `3472c1b` / `bd-3jo6.5.4` committed the E4
deterministic decoder (`TromrDecoderW`, `decoder_forward`, `generate`,
`tromr_decoder_matches_argmax_oracle`) with step-0 head cos 1.0 and 42-step
token-exact argmax generation. `79d715c` / `bd-3jo6.5.7` add
`merge_semantic` and `semantic_to_musicxml`; `78a2de3` / `bd-3jo6.5.9` add
`tromr_staff_tensor`, `MusicResult`, `forward_tromr`, `OcrTask::Music`, and
`model_arch implemented=true`. `bd-av64.7` / `ece14f9` publishes
`focr pull tromr`, and `efccce9` / `bd-av64.12` adds the default
`tromr.int8.focrq` quant plus the retained `f32` reference quant. Pull installs
`tokenizer_rhythm.json`, `tokenizer_pitch.json`, `tokenizer_lift.json`, and
`tokenizer_note.json` beside whichever artifact is selected. `select_quant`
uses the requested quant exactly when present, so `focr pull tromr --quant f32`
reports and installs `tromr.focrq`.

## Cache and Paths

Common path controls:

| Env var | Use |
|---------|-----|
| `FOCR_MODEL_PATH` | Exact model file to load |
| `FOCR_MODEL_DIR` | Extra model search path for inference resolution |
| `FOCR_QUANT` | Prefer a quant variant such as `int8` during model lookup |
| `FOCR_MANIFEST_URL` | Override manifest URL/path for `focr pull` when no `--manifest` is passed |
| `FOCR_INSTALL_BASE_URL` | Override release binary base URL for installer mirror/airgap/e2e tests |

Resolution guidance:

1. Use explicit CLI `--model` or library `recognize_with_model` for tests.
2. Use `FOCR_MODEL_PATH` in deployments.
3. Use `FOCR_MODEL_DIR` or `~/.cache/franken_ocr/models` for local model
   search after `focr pull` or deployment artifact setup. Fresh-pull lookup
   should find `unlimited-ocr.int8.focrq` without `--model`. Named pulls install
   into `~/.cache/franken_ocr/models/<model-id>/`; keep sidecars in that
   directory next to the `.focrq`.
4. If both exact and quant-suffixed names exist, exact basename wins unless
   `FOCR_QUANT` intentionally prefers a quant variant. Treat this as the live
   runtime env name; if older prose lags source constants, the constant wins.
5. For GOT-OCR2, prefer an explicit model path such as
   `--model got-ocr2.int8.focrq` or `FOCR_MODEL_PATH=/.../got-ocr2.int8.focrq`.
6. For OneChart and TrOMR, prefer the model subdirectory as the explicit model
   path root because the tokenizer sidecars are part of runtime resolution.

Do not hide model resolution failures. Exit code 3 is actionable.

Run telemetry is a separate store from model artifacts. In current source after
closed `bd-223.4`, `RunStore::default_path()` uses `FOCR_RUN_STORE` when set,
otherwise `~/.cache/franken_ocr/runs.db`. This fsqlite database records
best-effort local OCR telemetry and backs `focr runs` plus `focr sync`; it is
not a model search directory and should not be copied into inference-only model
bundles unless the deployment intentionally wants run history.

## `.focrq` Format

Current format facts from source/docs:

- format version: 1
- magic: `FOCRQ\0`
- stores quantized model payload and metadata
- records source sha256 information
- carries required model-specific license metadata
- optional `model_id` selects the model architecture from `model_arch`
- absent or empty `model_id` means `unlimited-ocr`, preserving v1 artifacts
- unknown `model_id` is a forward-incompatible artifact and is refused
- non-default licenses must exactly match the registered model notice

Format mismatch maps to exit code 7. A format mismatch is not a retryable OCR
failure; update the artifact or binary.

## Conversion Lane

Implemented:

```bash
focr convert model.safetensors -o model.focrq --quant int8 --model-id unlimited-ocr
focr convert got.safetensors -o got-ocr2.int8.focrq --quant int8 --model-id got-ocr2
focr convert smolvlm2/model.safetensors -o smolvlm2.int8.focrq --quant int8 --model-id smolvlm2 --json
```

Current CLI reality: int8 conversion is implemented; int4 conversion returns
`NotImplemented` because the int4 group-quantized path is not validated.
`--arch` records the offline packing target (`generic`, `aarch64-smmla`,
`x86-vnni`, `x86-amx`). `aarch64-smmla` now emits real offline SMMLA panels
after `bd-2mo.3`; non-SMMLA hosts un-permute with a warning/fallback, and
VNNI/AMX remain tag-only until packed-consuming x86 kernels land. Treat
`--arch` as an artifact layout target, not timing evidence. `--model-id
got-ocr2` writes the GOT Apache notice and omits the tied `lm_head.weight`; do
not use it for Baidu-shaped weights.
`--model-id smolvlm2` is arch-aware: it quantizes the Idefics3-nested text
decoder under `model.text_model.layers.`, keeps SigLIP/connector/embeddings/all
norms high-precision, keeps the untied `lm_head` high-precision, and records
`model_id=smolvlm2`. At runtime after `4291807`, that stored F32 head can be
executed through the default-on int8+top-K-refine path; use
`FOCR_GOT_INT8_LMHEAD=0` for the f32-head kill switch, and do not describe the
artifact storage itself as int8. The C2 real-weights census (`bd-3jo6.3.2`) expects 489
tensors total, 224 `QInt8PerChan` decoder GEMMs, and 265 F32 high-precision
tensors. Normal GOT acquisition should use `focr pull got-ocr2`; conversion is
for source weights or reproducibility work. Current policy for implementing or
reviewing conversion:

- Decoder FFN/expert GEMMs are the validated quantization surface.
- Vision tower, projector, embeddings, router gate, and norms stay high
  precision unless a future gate proves otherwise.
- `int4` is not an available default; it requires separate parity evidence.

When documenting conversion results, include source sha256, output path, quant
mode, binary revision, and validation command.

## Runtime Env Vars

User-facing or operationally relevant env vars seen in source:

| Env var | Use |
|---------|-----|
| `FOCR_MODEL_PATH` | exact model artifact |
| `FOCR_MODEL_DIR` | extra model search path |
| `FOCR_QUANT` | prefer `unlimited-ocr.<quant>.focrq` lookup |
| `FOCR_NO_REPEAT_NGRAM` | decoding repetition control; CLI/env override also reaches GOT and `ocr-batch` |
| `FOCR_GOT_NO_REPEAT_NGRAM` | GOT-OCR2 global no-repeat n-gram guard; default 20, lower priority than CLI/`FOCR_NO_REPEAT_NGRAM`, `0` disables for diagnostics |
| `FOCR_MAX_NEW_TOKENS` | generated-token cap; explicit `--max-length` wins and capped output is a true prefix |
| `FOCR_FORCE_ARCH` | force architecture/backend probe path |
| `FOCR_THREADS` | current `bd-223.2` source: process-wide thread budget; parsed once, positive integer wins, otherwise physical cores; reported as robot `threads` |
| `FOCR_RUN_STORE` | closed `bd-223.4` source: exact fsqlite run-store path; default is `~/.cache/franken_ocr/runs.db`; tests should set this to a temp path |
| `FOCR_NO_MMAP` | current committed source after `507cebe`: kill switch for the default read-only mmap weight loader; forces owned-buffer loading, and mmap failures also fall back to owned bytes |
| `FOCR_STAGE_BUDGET_FORWARD_MS` | stage budget/timing control |
| `FOCR_FIXTURES_DIR` | armed release/conformance fixture root for native f32 receipt paths; missing means unarmed evidence, not green release proof |
| `FOCR_BATCH_SPINE` | arm the continuous-batch int8 `ocr-batch` spine; unset, empty, `0`, `off`, `false`, or `no` disable it; any other present value arms it |
| `FOCR_BATCH_SIZE` | spine in-flight stream count; current shared scheduler defaults to 128 for unset/blank/unparsable/0 and caps larger values at 256 |
| `FOCR_BATCH_VISION` | inside the armed spine, batch vision across pages by default; trim/case-folded `0`/`off`/`false`/`no` reverts to per-page vision |
| `FOCR_TIMING` | timing output/control |
| `FOCR_RESAMPLE` | Baidu/GOT preprocess resampler: unset CatmullRom default, `pil-bicubic` for Pillow-bit-exact BICUBIC reference comparison; it does not select SmolVLM2 LANCZOS |
| `FOCR_SPEC_E2E_IMAGES` | corpus path/list for `scripts/spec_gate_e2e.sh` ON/OFF speculative-decode e2e gate |
| `SMOLVLM2_SAFETENSORS` | real SmolVLM2 source shard for the model-gated conversion e2e script |
| `SMOLVLM2_FOCRQ_OUT` | optional destination for persisting the SmolVLM2 converted artifact |
| `FOCR_SMOLVLM2_DIR` | source/output directory used by SmolVLM2 decoder, tokenizer, and vision fixture generators |
| `FOCR_SMOLVLM2_TOKENIZER_JSON` | exact real SmolVLM2 `tokenizer.json` for C6 token-id conformance; falls back to `$FOCR_SMOLVLM2_DIR/tokenizer.json` |
| `FOCR_SMOLVLM2_CORPUS` | optional alternate corpus for `scripts/gen_smolvlm2_token_id_fixtures.py` |
| `FOCR_SMOLVLM2_MODEL` | SmolVLM2 `.focrq` or raw f32 safetensors used by decoder seam tests |
| `FOCR_SMOLVLM2_ORACLE_HIDDEN0` | SmolVLM2 oracle hidden-state binary for decoder seam tests |
| `FOCR_SMOLVLM2_ORACLE_LOGITS` | SmolVLM2 oracle last-position logits for decoder seam tests |
| `FOCR_SMOLVLM2_QUESTION` | SmolVLM2 describe/VQA question fallback; CLI `--question` wins |
| `FOCR_SMOLVLM2_DIR` + `tests/fixtures/smolvlm2/vqa_fixtures.json` | arms the C8 VQA informational guard when the directory contains tokenizer JSON and f32 or int8 SmolVLM2 weights |
| `FOCR_BIN` | optional prebuilt `focr` binary for `scripts/smolvlm2_describe_e2e.sh`; otherwise the script builds release `focr` |
| `FOCR_ONECHART_DIR` | arms OneChart tokenizer/oracle/native-assembly/runtime checks from a directory containing `vocab.json`, `merges.txt`, `added_tokens.json`, `model.safetensors`, and when applicable `onechart_preproc.bin`, `onechart_proj_out.bin`, `onechart_final_logits.bin`; current source also uses `onechart.int8.focrq` when present for same-quantization KV-cache, recognition, and chart-data e2e checks |
| `FOCR_TROMR_DIR` | arms TrOMR conversion, tokenizer, E3 encoder, E4 decoder, E7 merge/MusicXML, E9 music e2e, E8/DISC-004 checks, E5 page-detection checks, and the `bd-av64.6` real-scan music gate when source has them, from a directory containing `tromr.focrq`, tokenizer tables, upstream examples when present, `tromr_oracle_fixtures.json`, `tromr_preproc.bin`, and seam binaries such as `tromr_seam_encoder_out.bin` |
| `FOCR_TROMR_SAMPLE` | enables TrOMR upstream top-k/T=0.2 sampling arithmetic; unset uses deterministic per-head argmax |
| `FOCR_TROMR_SEED` | deterministic PCG32 seed for `FOCR_TROMR_SAMPLE`; same seed means same TrOMR decode stream |
| `FOCR_TROMR_SPLIT` | experimental TrOMR over-budget-staff barline split rescue after closed `bd-av64.4`; only `1` arms it, default is off, and it is not a default quality/perf/dewarp proof |
| `FOCR_GAUNTLET_WORK` / `FOCR_GAUNTLET_MODEL_DIR` / `FOCR_GAUNTLET_PAGES_DIR` / `FOCR_GAUNTLET_VENV_PY` | quiet-host gauntlet runbook paths |
| `FOCR_DECODE_STATELESS` | force stateless re-prefill decode oracle |
| `FOCR_DECODE_INT8` | opt into int8 decode path |
| `FOCR_QKV_FUSED` | default-on fused q/k/v int8 decode projection after `98cc790` / closed `bd-241s`; set `0`, `off`, `false`, or `no` to restore the older three-call path for parity/profiling |
| `FOCR_PROFILE_DECODE` | emit decode profiler detail |
| `FOCR_GOT_FORMAT` | request GOT-OCR2 `OCR with format:` Mathpix-Markdown mode |
| `FOCR_GOT_INT8_LMHEAD` | GOT and SmolVLM2 lm_head int8 GEMV plus top-K f32 refine; default on, set `0`/`f32`/`off` for f32-head kill-switch |
| `FOCR_GOT_SEQ_ATTN` | force serial GOT decode attention; unset default parallelizes independent heads |
| `FOCR_GOT_MODEL` / `FOCR_GOT_TIKTOKEN` | test-only env-gated GOT tokenizer/e2e fixtures |
| `FOCR_ORACLE_IMAGE` / `FOCR_ORACLE_HIDDEN0` | GOT vision/splice oracle fixture inputs |

Always verify current names in `src/cli.rs`, `src/native_engine/`,
`src/quant/`, `src/simd/`, and related modules.

Run-store details from closed `bd-223.4` to preserve when documenting or testing:

- `SCHEMA_VERSION = 1` with `_meta.schema_version`,
  `_meta.franken_ocr_version`, and `_meta.model_version_tag`.
- A too-new store is refused as `FormatMismatch` / exit 7.
- `RunRecord` JSON carries `schema_version`, `run_id`, `started_at`,
  `finished_at`, `input_path`, `mode`, `quant`, `model_version_tag`,
  `exit_code`, and `status`.
- `focr sync export-jsonl` writes canonical JSONL with a same-directory temp
  file and `jsonl.lock` / `jsonl.tmp` discipline before rename.
- `focr sync import-jsonl --file FILE` is idempotent by `run_id`.
- OCR recording is best-effort: a store failure should print a note and leave
  the OCR result/exit code governed by the actual OCR outcome.

## Experimental Env Vars

Treat these as gated development levers, not production knobs:

Some env vars appear both in the operational table above and in this caution
table. That is intentional: the first entry tells an agent how the variable is
used; this section tells the agent what proof or kill-switch discipline is
needed before recommending it.

| Env var family | Caution |
|----------------|---------|
| `FOCR_INT8_KV` | attention/KV quantization can affect OCR output; the `bd-1waa` ledger rejected the old broader lever and later parity-gated work must cite its own bead/evidence |
| `FOCR_ATTN_GEMM` | attention kernel substitution needs parity proof; `bd-1waa` / `bd-3pg7` ledger evidence showed real M4 attention speed but non-bit-exact runaway OCR, so do not revive it without a bit-exact gate and 20-page CER proof |
| `FOCR_INT8_ATTN` | quantizes attention surfaces only behind CER proof |
| `FOCR_INT8_LMHEAD` / `FOCR_LMHEAD_INT4` | high-risk output-head quantization |
| `FOCR_GOT_INT8_LMHEAD` | default-on for GOT after bd-2dlz and for SmolVLM2 untied head after `4291807`; keep the f32 kill-switch and distinguish runtime int8+refine from artifact storage |
| `FOCR_GOT_SEQ_ATTN` | measurement/safety switch for serial vs parallel GOT attention |
| `FOCR_BATCH_PACK` | batch scheduler packing; sorts admission by similar prefill length, must restore output order, and needs pack-on/pack-off byte identity before throughput claims |
| `FOCR_RESAMPLE=pil-bicubic` | not experimental in code, but use it deliberately as a reference-exact comparison mode; default remains CatmullRom under DISC-001 |
| `FOCR_FUSE_NGRAM_LMHEAD` | opt-in ngram-lmhead fusion; `bd-2mo.24` / `a0ad299` says correct-but-does-not-pay, with `fused_ngram_lmhead_is_byte_identical_to_separate_mask` and page_0023 A/B at 16.43s -> 16.40s inside noise; keep off by default and retry only for multi-image `ngram_window=1024` ban sets or a much faster decode step |
| `FOCR_TROMR_SPLIT=1` | experimental/off-by-default barline split rescue for over-budget TrOMR staff bands; measured as recognition-count rescue, not lossless quality, broad SER, camera dewarp, int8, or perf evidence |
| `FOCR_SPEC_DECODE` / `FOCR_SPEC_VERIFY` | speculative draft/verify levers; linear `FOCR_SPEC_DECODE` has bd-1azu.36 ON==OFF e2e output identity, but remains opt-in, presence-armed, and lacks per-run engagement telemetry |
| fusion/tile/internal kernel toggles | benchmark only with parity evidence |

If enabling an experimental path, record:

- exact env vars,
- model artifact hash,
- corpus/images,
- expected loss or CER/TEDS budget,
- fallback trigger,
- Beads issue or evidence ledger.

## Offline Deployments

Recommended:

1. Build/pin `focr` and the `franken_ocr` library revision.
2. Run `focr pull` during image/build/deploy setup, or install a preverified
   `.focrq` plus tokenizer through your deployment artifact system. Use named
   pulls for specialized models: `focr pull got-ocr2`, `focr pull smolvlm2`,
   `focr pull onechart`, or `focr pull tromr`.
3. For self-converted weights, use `focr convert ... --quant int8` and record
   source SHA256, output hash, converter revision, and command transcript.
   For SmolVLM2, preserve the C2 census, C5 decoder seam proof, C6 tokenizer
   conformance proof, C7 LANCZOS/preprocess/prompt proof, C9 route evidence,
   DISC-003 L4 near-tie ledger, the C8 VQA fixture/guard result, C10
   `scripts/smolvlm2_describe_e2e.sh` NDJSON, and any C3/C4 vision/connector
   oracle artifacts with the artifact. You may exercise `--task describe`; make
   quality/perf claims only at the level the C8/C10/A11 evidence actually
   supports.
   For OneChart, preserve the D2 conversion census, tied-head dedup proof, D3
   vision/projector certification, D4-prefill certification, D4 cached decode
   closure, D5 `ChartResult`/`recognize` native assembly, D6/D7/D8 public route
   evidence, D9 tokenizer conformance result, and the `bd-av64.7` manifest
   evidence if using the packaged pull.
   For TrOMR, preserve the E2 WS-folded export provenance, byte-exact
   `tromr.focrq` / `0 int8` reference artifact proof, E6 WordLevel tokenizer fixtures,
   E3 helper-kernel evidence, E3 encoder oracle evidence, E4 decoder oracle
   evidence, E7 MusicXML evidence, E9 `tromr_music_e2e/v1` evidence, the
   `bd-av64.7` manifest evidence if using the packaged pull, `bd-av64.12`
   storage-int8 evidence if using `tromr.int8.focrq`, and the caveat that int8
   storage still dequants into f32 compute.
4. Copy `.focrq` and tokenizer/cache artifacts into the target. Prefer the
   current pulled basename `unlimited-ocr.int8.focrq` unless deployment policy
   pins an explicit `FOCR_MODEL_PATH`. For GOT-OCR2, copy both
   `got-ocr2.int8.focrq` and `qwen.tiktoken`. For SmolVLM2, copy
   `smolvlm2.int8.focrq` and `tokenizer.json`. For OneChart, copy
   `onechart.int8.focrq`, `vocab.json`, `merges.txt`, and
   `added_tokens.json`. For TrOMR, copy `tromr.int8.focrq` (or
   `tromr.focrq` for reference mode) and all four tokenizer JSON files.
5. Set `FOCR_MODEL_PATH` or `FOCR_MODEL_DIR`.
6. Run `focr robot schema`, `focr robot health`, `focr robot backends`, and
   `focr robot selftest`. At/after `ad3ad20`, inspect
   `focr robot selftest | jq '.models'` for per-decoder int8 parity verdicts;
   TrOMR is absent from that rollup because its default int8 artifact is
   consumed through f32 dequant-on-access, not an int8 decoder kernel.
7. Disable network during inference tests.

Never make the first production OCR request responsible for downloading the
model.

## Platform Notes

The project prioritizes CPU:

- Apple Silicon / ARM64: NEON, dotprod, i8mm lanes.
- x86-64: AVX2, AVX-VNNI, and AVX-512-VNNI when dispatchable. AMX is not a
  current advertised backend.

Current README/source say native Windows x86_64 OCR and `focr pull` work,
including full multi-part model download and SHA-256 verification. Recent CI
work adds an `aarch64-pc-windows-msvc` matrix/smoke row, but do not round that
up to published ARM64 Windows support unless release artifacts and docs confirm
it. Verify current release artifacts, source, tests, and Beads before making
stronger platform claims.

## License Notes

Baidu Unlimited-OCR is MIT licensed. GOT-OCR2's registry notice is Apache-2.0.
Packaged derivative artifacts must preserve the registered model notice. Do not
strip license metadata from `.focrq` conversion or release workflows.
