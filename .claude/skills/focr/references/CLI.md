# focr CLI Reference

## Table of Contents

- [Contract](#contract)
- [Binary Selection](#binary-selection)
- [Installation](#installation)
- [Commands](#commands)
- [OCR Command](#ocr-command)
- [Batch OCR](#batch-ocr)
- [Model Pull](#model-pull)
- [Model Discovery](#model-discovery)
- [Conversion](#conversion)
- [Robot Commands](#robot-commands)
- [Run State, Sync, and Doctor](#run-state-sync-and-doctor)
  - [Run State and Sync](#run-state-and-sync)
  - [Doctor](#doctor)
- [Examples](#examples)
- [Validation](#validation)

## Contract

`focr` is the short binary for `franken_ocr`. The long binary
`franken_ocr` exists too. Both are thin shims over `franken_ocr::cli_main()`;
when behavior differs, the binary is stale or the source is dirty.

Inference is local after model acquisition. The normal runtime should not need
network access.

## Binary Selection

Prefer the exact binary path used by the system under test.

```bash
command -v focr
focr --version
focr --help
```

When working from source:

```bash
cd ~/projects/franken_ocr
cargo run --bin focr -- --help   # if live help is needed and build cost is acceptable
```

Stale-binary warning signs:

- Source contains `ocr-batch`, `pull`, or `robot selftest`, but help does not.
- Source contains `Models(ModelsArgs)` / `run_models`, but help lacks `models`.
- Source contains `thread_budget`, `FOCR_THREADS`, and robot `threads`, but
  `robot health` / `robot backends` do not expose the field.
- Source contains `RunStore`, `FOCR_RUN_STORE`, `export-jsonl`, and
  `import-jsonl`, but help still shows `runs` / `sync` as stubs or omits the
  `--file` shapes.
- Source contains `run_doctor`, `doctor capabilities`, `robot-docs`, or
  `DoctorExit`, but help still treats `doctor` as missing or scaffolded.
- Source contains `robot_triage_payload`, but help lacks `robot triage`.
- Installed `focr` is missing while repo builds.
- A target-dir binary prints an older command set after source changes.

Do not update this skill from stale help. Inspect `src/cli.rs` first.

## Installation

The README-supported path is to install a prebuilt release binary, then fetch
model artifacts with `focr pull`.

Unix/macOS/WSL:

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
focr --version
focr pull
```

The shell installer detects OS and CPU architecture, resolves the latest GitHub
release, falls back to its baked-in fallback when release lookup is unavailable,
downloads the matching binary, verifies the SHA256 sidecar, and installs `focr`
on `PATH`. Under WSL it proceeds as Linux. Under native Git-Bash, MSYS, or
Cygwin it exits cleanly and points at the PowerShell installer.

Release-boundary rule: features listed under `CHANGELOG.md` `Unreleased` are
source-on-`main` features, not guaranteed in the installed binary unless the
matching release/help confirms them. Committed `f0a538b` created the `v0.4.0`
public release boundary; the July 7, 2026 check found README badges/manual
examples and installer fallback constants still pointing at `v0.4.0`.
Committed/tagged `bf28fd7` bumps the Rust package/tag to `v0.5.0`, and the
`v0.5.0` GitHub release is now live with binary plus SHA256 assets for Apple
Silicon, macOS x86-64, Linux x86-64, Linux ARM64, and Windows x86-64. Normal
online installers resolve the latest GitHub release first; the `v0.4.0`
constants are fallback or historical manual prose, not proof that the latest
prebuilt binary is still v0.4.
The July 8 source probe saw public `origin/main` at `592426c`, with clean source
describing as `v0.6.0-4-g592426c` and no tracked source diff. The `v0.6.0`
GitHub release is live, published 2026-07-08T14:47:48Z with platform binaries
plus SHA256 assets, and points at source tag `29516b9`. `0924479` is a
README-only clarification after the `v0.5.2` tag; `efd83e8` is the formal
`bd-av64.10` G2 closeout; `3f2f97e` is `bd-2mo.26` gauntlet harness hardening;
`4291807` certifies SmolVLM2 untied `lm_head` int8+refine default-on;
`c248e6d` lands `bd-2mo.26` head-to-head rows; `c29a78b` certifies the release
gate; `7c7bd00` closes the release-certification Beads; `beaed7c` records
CI/dist supplement notes; `db02421` refreshes README release-readiness evidence;
`5df6395` commits post-certification fuzz corpus growth; and `592426c`
refreshes README `v0.6.0` release identity, asset-size, manual-download, and
CPU-backend prose. Treat post-tag
gate/bundle/fuzz/SigLIP/README alignment, public SAM row-tile negative
evidence, live certification-bundle-cell fixes, GOT pass-6 statics caching, the
mmap-loader half, OneChart pass-7 statics caching, SmolVLM2 pass-8 statics
caching, SmolVLM2 head certification, G2/head-to-head rows, and release
certification as current-source evidence when source/help/tests agree. Dirty
diffs after `592426c` are live-WIP only until committed/tests/Beads agree.
The older `v0.4.0` public source/release boundary includes `-o/--output`,
JSON layout boxes, fresh-pull `unlimited-ocr.int8.focrq` auto-resolution,
figure extraction,
`focr models`, `.focrq` `model_id`, `convert --model-id`, implemented GOT-OCR2,
`focr pull got-ocr2`, GOT `--format`, `focr ocr --task`, SmolVLM2
`--task describe` / `--question`, `focr pull smolvlm2`, `focr pull onechart`,
`focr pull tromr`, `focr robot triage`, implemented `focr doctor`, PDF
`--pages` / `--split-spreads`, reference `FOCR_RESAMPLE=pil-bicubic`,
batch-spine/vision controls, closed `bd-223.2` runtime cancellation/thread
budgeting, and closed `bd-223.4` / `bd-wp8.11` run-state/sync source. After any
installer run, prove
the exact
binary before relying on source-only surfaces:

```bash
focr --version
focr --help | rg -- 'models|convert|doctor|robot'
focr ocr --help | rg -- '--output|extract-figures|figures-dir|--format|--task|--question|crop-mode|--pages|--split-spreads'
focr models --json | jq '.models[] | {id, implemented, pull}'
focr robot triage | jq '.quick_ref'
focr doctor capabilities --json | jq .
```

Current installer work is covered by a true pty-style e2e script
(`tests/installer_e2e.sh`) and uses `FOCR_INSTALL_BASE_URL` for mirror/airgap
or file-url tests. If a user reports `gum: error: unknown flag ->`, suspect an
old raw installer or stale release branch; current `install.sh` passes `--`
before dynamic `gum style` text.

Latest-source boundary to verify before support claims: `main` now wires
preprocess flags into the engine (`--base-size`, `--image-size`, `--crop-mode`),
has first Gundam e2e evidence, has model-gated SmolVLM2 conversion via
`--model-id smolvlm2`, has C5 SmolVLM2 text-decoder seam evidence, and has
C7/C9 SmolVLM2 `--task describe` / `--question` routing through
`src/native_engine/smolvlm2.rs`. Current live source/Beads also close C8/C10
and sub-epic C: C8 reports L0b exact, L0c 876/876 id-exact, L2 cos 1.0, L3
`<5e-5`, L4 64/64 id-exact on the opt-in full-cert greedy leg, L5 7/7 f32 and
7/7 int8 oracle-answer guard, and DISC-003 near-tie KV-cache ledgering; C10
reports the detailed `smolvlm2_describe_e2e/v1` CLI gate green on the release
int8 artifact. It also has the
`FOCR_SPEC_DECODE` ON/OFF gate artifacts and first pinned HF bf16 PERF_LEDGER
rows under `bd-re8.17`. Those can be newer than a curl-installed release
binary; check exact help, source, Beads, and ledger rows before relying on them.
The latest C8 VQA work adds an oracle-answer fixture/guard:
`scripts/gen_smolvlm2_vqa_fixtures.py` generates
`tests/fixtures/smolvlm2/vqa_fixtures.json`, and
`vqa_quality_matches_oracle_l5` compares full-pipeline answers to the oracle's
greedy answers on a seven-question sample-photo set. Treat f32 >=70% and int8
>=50% as the floors and cite the live 7/7 + 7/7 close evidence exactly; do not
turn it into a public VQA benchmark.
`scripts/smolvlm2_describe_e2e.sh` is the C10 CLI e2e lane:
it emits `smolvlm2_describe_e2e/v1` NDJSON, proves `/nonexistent` model exit 3
and wrong-family usage exit 2, then runs `--task describe` and a sun VQA
question through the real int8 artifact. It is model-gated; a skip means missing
weights, while live C10 closure means the armed release-int8 run passed.
OneChart is current for arch-aware conversion, tokenizer conformance, D3
vision/projector, D4 prefill/cached decode, D5 recognition assembly, and D6-D8
public route wiring. `focr convert --model-id onechart` can produce
`onechart.int8.focrq` from OneChart-shaped weights, and `FOCR_ONECHART_DIR`
arms the D9 tokenizer gate over `vocab.json` + `merges.txt` +
`added_tokens.json`; D3 uses `onechart_view_tensor`, `vision_features`,
`model.vision_tower`, `model.mm_projector`, `onechart_preproc.bin`, and
`onechart_proj_out.bin`; D4 uses `DecoderFamily::Opt`,
`DecoderConfig::onechart`, `build_inputs_embeds`, `onechart_final_logits.bin`,
`generate_greedy_kvcache`, and `opt_kvcache_matches_greedy_and_oracle`; D5 uses
`ChartResult`, `recognize`, `complete_json_string`,
`<Number>`/`prefill_final_hidden`, `number_head`, and `reliable_distance`; D7
adds `OcrTask::ChartData` and `forward_onechart`. Use
`focr pull onechart`, then
`focr ocr chart.png --task chart-data --model onechart.int8.focrq`, or use an
explicit local artifact. A separate `focr chart` subcommand is still not
current, and broad chart quality remains corpus-gated.
TrOMR is now the fifth implemented runtime model and is pullable: default
`focr pull tromr` installs `tromr.int8.focrq` storage after `efccce9` / closed
`bd-av64.12`, while `focr pull tromr --quant f32` installs the bit-exact
`tromr.focrq` reference. E2 produced the `tromr.focrq` reference artifact with
`0 int8`; current `focr convert` has no f32 quant mode. E6 adds
`MusicTokenizer` WordLevel tables, E3 adds `TromrEncoderW` /
`tromr_encoder_matches_torch_oracle`, E4 adds `TromrDecoderW` /
`tromr_decoder_matches_argmax_oracle`, E7 adds `merge_semantic` /
`semantic_to_musicxml`, E9 adds `forward_tromr` plus
`focr ocr --task music --model tromr.int8.focrq` after default pull or
`--model tromr.focrq` after `focr pull tromr --quant f32`, E5 adds full-page
`recognize_page` with staff detection, and E10 closes the sweep. The committed
public route is partwise MusicXML for single-staff images and v1
printed/scanned full-page scores; camera dewarp, default/lossless barline
quality, int8 compute/perf, and `**kern`
export remain follow-ups. Experimental `FOCR_TROMR_SPLIT=1` is a separate
over-budget-staff recognition-count rescue path, not default quality.
`bd-2sez` adds a public f32 PERF_LEDGER baseline row, but it is an honest losing
row against pinned upstream torch, not a TrOMR speed win.
The real-scan TrOMR regression gate is not a top-level CLI subcommand: run
`FOCR_TROMR_DIR=/path/to/tromr sh scripts/realscan_music_gate.sh`. It uses the
committed `tests/fixtures/realscan_music/` Spohr corpus and emits
`realscan_music/v1` NDJSON. Treat `truth/attributes.json` as human-verified
attributes plus the authoritative full-page `min_recognized` floors,
`goldens/*.musicxml` as frozen model-output anchors, and XFAILs as
promote-on-XPASS expectations. Source at/after `91d552f` reads p055/p100 page
floors from that JSON (`5` and `1`) instead of duplicated shell literals.
`bd-av64.6` is closed for corpus-v1 measurement, but do not inflate it into the
10-20 item expansion, GOT cross-reference, ladder row, or aggregate SER.

Native Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
focr --version
focr pull
```

The PowerShell installer downloads `focr-x86_64-pc-windows-msvc.exe`, verifies
it by SHA256, and puts `focr` on `PATH`. Native Windows x86_64 is supported;
ARM64 Windows is not published.

Manual release download is also supported. Release assets are raw executables,
not archives. Download the platform asset and its `<asset>.sha256` sidecar from
the release, verify, mark executable if needed, and place it on `PATH`.

Current README asset names:

| Platform | Asset |
|----------|-------|
| macOS Apple Silicon | `focr-aarch64-apple-darwin-neon-sdot-i8mm` |
| macOS Intel | `focr-x86_64-apple-darwin` |
| Linux x86-64 glibc | `focr-x86_64-unknown-linux-gnu` |
| Linux ARM64 glibc | `focr-aarch64-unknown-linux-gnu` |
| Windows x86-64 | `focr-x86_64-pc-windows-msvc.exe` |

Apple Silicon manual example:

```bash
base=https://github.com/Dicklesworthstone/franken_ocr/releases/download/v0.5.0
asset=focr-aarch64-apple-darwin-neon-sdot-i8mm
curl -fsSLO "$base/$asset"
curl -fsSLO "$base/$asset.sha256"
shasum -a 256 -c "$asset.sha256"   # Linux: sha256sum -c "$asset.sha256"
chmod +x "$asset"
mv "$asset" /usr/local/bin/focr
```

From-source builds are advanced, not an install one-liner:

```bash
cd ~/projects/franken_ocr
cargo build --release
# produces target/release/focr and target/release/franken_ocr
```

The source tree requires the pinned nightly toolchain plus sibling repositories
laid out as expected (`asupersync`, `frankentorch`, `frankensqlite`). A fresh
clone or `cargo install --git` will not resolve those dependencies. There is no
working crates.io `cargo install` path in the README.

## Commands

Status below is current-source status. Installed release binaries can lag
`main`; if help disagrees, classify it as stale or release lag before changing
docs or downstream code.

| Command | Status | Purpose |
|---------|--------|---------|
| `ocr <image-or-pdf>` | implemented path | OCR one document image or scanned PDF |
| `ocr-batch <images...>` | implemented path | OCR multiple document image paths |
| `convert` | int8 implemented | Convert safetensors into `.focrq`; supports `--model-id` |
| `pull [model]` | implemented path | Download and verify default or named model artifacts |
| `models` | implemented discovery | List model registry rows and ready/planned status |
| `robot schema` | implemented path | Emit robot schema metadata |
| `robot health` | implemented path | Machine-readable health probe |
| `robot backends` | implemented path | Emit backend availability/capability data |
| `robot selftest` | implemented path | Run fast machine-readable self-check |
| `robot run <image-or-pdf>` | implemented path | OCR with NDJSON lifecycle events |
| `robot triage` | implemented path | One-shot JSON quick_ref/health/recommendations/commands for agents |
| `doctor` | implemented path | Detect, dry-run, fix, undo, and report capabilities for local state |
| `runs` | implemented in current source after closed `bd-223.4`; scaffolded in older binaries | Query fsqlite run history |
| `sync` | implemented in current source after closed `bd-223.4`; scaffolded in older binaries | Export/import run-state audit JSONL |

Treat `runs` / `sync` as a source/help/tracker boundary: older binaries may
still return `NotImplemented` exit 1 or omit the new flags, while current
`bd-223.4` source implements the store and JSONL sync. Treat scaffolded output
as release lag or stale-local evidence until source and tests prove otherwise.
Treat `doctor` the same way: current source is implemented after `bd-wp8.4` and
`bd-wp8.4.1`; a scaffolded doctor means stale binary/source, not the current
contract.

## OCR Command

Typical use:

```bash
focr ocr page.png
focr ocr page.png --json
focr ocr page.png -o page.md
focr ocr page.png -o page.json
focr ocr page.png -o page.md --extract-figures
focr ocr scan.pdf --json
focr ocr book.pdf --pages 3,5-9 --split-spreads -o excerpt.md
focr ocr paper.pdf --pages 1-12 --multi-page -o paper.md
focr ocr page.png --model /path/to/model.focrq
focr ocr formula.png --model got-ocr2.int8.focrq
focr ocr formula.png --task formula --model got-ocr2.int8.focrq
focr ocr chart.png --task chart-data --model /path/to/onechart.int8.focrq
```

Important flags:

| Flag | Use |
|------|-----|
| `--json` | Return structured JSON instead of human markdown/text |
| `-o, --output <file>` | Write result to a file; `.json` selects JSON, other extensions select markdown |
| `--extract-figures` | Crop grounded figure/photo/chart regions and rewrite output references |
| `--figures-dir <dir>` | Save extracted figures in a chosen directory; also enables figure extraction |
| `--robot` | Emit NDJSON robot events |
| `--model <path>` | Use explicit `.focrq` or model artifact path; e.g. a pulled `got-ocr2.int8.focrq` |
| `--pages <spec>` | PDF-only 1-based page selection, e.g. `3,5-9`; duplicates are deduped in source order |
| `--split-spreads` | PDF-only heuristic split for wide scanned book spreads; emits left/right logical pages |
| `--multi-page` | PDF-only Unlimited-OCR cross-page pass over selected rasterized pages; emits one markdown document with `<PAGE>` separators |
| `--base-size`, `--image-size`, `--crop-mode` | Match reference image preprocessing knobs; default `base` is the certified single 1024-pixel global view |
| `--max-length`, `--temperature`, `--no-repeat-ngram`, `--ngram-window` | Decode controls; `--temperature` and `--ngram-window` affect Unlimited-OCR only, while GOT stays greedy/global |
| `--format` | GOT-OCR2 `OCR with format:` Mathpix-Markdown mode |
| `--task <task>` | Convenience selector: `ocr`, `formula`, `tables`, `chart`, `molecular`, `geometry`, `music`, `describe`, or `chart-data` |

Crop-mode rule: `base` is the default and certified current e2e path.
`--crop-mode gundam` selects reference dynamic-resolution tiling. The flag now
reaches `PreprocessOverrides`; `bd-1e9n` records first live evidence (`rc=0`,
7 views, CER 0.0179 / WER 0.0138 on page_0107) plus default-output
byte-identity and a live `--base-size 512` probe. Treat that as first e2e
coverage, not a full parity sweep; require fresh target-corpus proof before
strong Gundam claims. If README prose still says certification is pending,
prefer source, artifacts, and Beads over the stale sentence.

Task rule: `--task ocr` is plain OCR. `--task formula|tables|chart|molecular|
geometry|music` implies `--format` and requires GOT-OCR2, usually
`--model got-ocr2.int8.focrq`; using one of those tasks with the default model
fails as a usage error before weights load. `--task describe` routes through
SmolVLM2 when the model spec is `smolvlm2`; add `--question` for VQA. With no
SmolVLM2 model spec, it fails early with usage guidance. `--task chart-data`
routes through OneChart when the model spec is `onechart`; with no OneChart
model spec, or a knowably wrong family, it fails early with usage guidance.
Older binaries can still return `NotImplemented` or lack `chart-data`;
classify that as release lag or stale local help.
Known source pitfall: if the `OcrTask::Describe` enum doc comment still says
"planned", do not stop there. Verify the actual request validation, `--question`
plumbing, and `forward_smolvlm2` dispatch before classifying the route.

Timeouts are controlled by stage-budget env vars such as
`FOCR_STAGE_BUDGET_FORWARD_MS`; current source has no timeout flag for the OCR
subcommand.

Structured output contract:

- single-image JSON contains `schema_version`, `markdown`, and top-level
  `layout`,
- PDF JSON contains `schema_version`, `markdown`, and `pages`, where each page
  has `page` and `layout`,
- PDF `--multi-page` JSON contains the single cross-page `markdown` result and
  does not carry per-page layout boxes for that route,
- each `layout` entry is `{label, boxes}`,
- each box is `[x1, y1, x2, y2]` in source-image pixels,
- `--json` forces JSON even with `-o out.md`,
- a failed recognition must not leave an empty or partial `-o` file behind.
- with `--robot`, `-o` writes the file first and the NDJSON `run_complete` still
  carries the recognized markdown.

Figure extraction contract:

- `--extract-figures` saves figure/image regions the model grounds but does not
  transcribe,
- default directory is `<output-stem>_figures/`; `--figures-dir DIR` overrides
  and enables extraction for stdout runs,
- without `-o` or `--figures-dir`, `--extract-figures` is a usage error (exit 2),
- markdown placeholders become real image-link references to saved figure paths,
- JSON gains `figures: [{label, page, bbox, path}]`,
- figure format is content-selected: JPG q85 for photos, PNG for line art,
  charts, and screenshots,
- failures before recognition must not leave output files or figure directories,
- `--split-spreads` plus `--extract-figures` is currently a clean usage error;
  do not recommend that combination until figure naming/placement across split
  halves has its own contract.

Input scope includes document images (PNG/JPG/etc.) and native scanned PDFs.
`focr ocr file.pdf` detects `.pdf` extension or `%PDF-` magic, renders pages
one at a time in pure Rust, OCRs each page through the normal image pipeline,
and joins page markdown with a blank line.
`--pages` applies only to native PDF input. It accepts comma-separated numbers
and inclusive ranges such as `3,5-9`, rejects invalid ranges as usage errors,
and preserves source order after deduplication. `--split-spreads` applies only
to PDF pages after rasterization. It uses the committed `split_spread` heuristic
for wide pages and emits left/right logical pages; treat it as extraction
ergonomics for scanned book spreads, not as a guarantee that every wide page was
or was not a spread. Before OCR and splitting, current source applies page
`/Rotate` and axis-aligned content-stream image-placement rotation through
`content_rotation`, fixing common scanned books that store portrait rasters but
display them upright through a rotated `cm` transform.

### Multi-Page Cross-Page OCR

Use `--multi-page` only when the document benefits from cross-page context and
fits the Unlimited-OCR 32K context budget. This is not independent per-page OCR:
current source rasterizes selected PDF pages, squash-resizes each page to
640x640 with `preprocess_dynamic_squash` using PIL-faithful bicubic at this
site, feeds all page image tokens into one prompt, decodes once with
`ngram_window=1024`, and returns markdown separated by `<PAGE>` markers.

Valid combinations:

```bash
focr ocr doc.pdf --multi-page
focr ocr doc.pdf --pages 3,5-9 --multi-page -o excerpt.md
focr ocr-batch page-001.png page-002.png --multi-page --json
```

Invalid/currently unsupported combinations:

- non-PDF `focr ocr image.png --multi-page`; use `ocr-batch --multi-page` for
  page image lists,
- PDF `--multi-page --split-spreads`,
- PDF `--multi-page --extract-figures`,
- non-Unlimited model-specific cross-page parsing unless source/tests add it.

`bd-1gv.25` is closed for the core and image-list CLI path. `bd-2z0y` is closed
for PDF `--multi-page` plus robot-mode streaming progress: PDF multi-page robot
mode emits additive schema-v1 `page` events at `<PAGE>` boundaries with
`status:"decoded"`, `page`, `chars`, and raw `text`, while final `run_complete`
still carries the assembled markdown. Do not confuse those decoded progress
events with per-page layout boxes, figures, or split-spread support. `bd-1gv.26`
closes the 2-page L5 multi-page oracle rung. `bd-1465` is now also closed:
`l5_multi_page_10p_long_horizon` adds the 10-page rung with fixture `p10`,
subject cap 7600 as a true-prefix comparison, plate byte-exactness, markers
8-vs-9, and CER 0.4045 <= 0.50; the uncapped 10-page subject terminates at the
32768 position cap while the oracle EOSes at 7117. The frozen 20-page `p20`
fixture shows the reference model itself collapsing at this horizon, so cite it
as degradation/upper-bound evidence and do not promise a meaningful 40-page CER
gate or arbitrary-long-document quality.

Native PDF is a scanned-image fast path. It covers common scan codecs:
`DCTDecode` JPEG, CCITT Group 4 fax, and `FlateDecode`/LZW raw rasters. It
returns `InputDecode` with a precise message for unsupported `JPXDecode`
(JPEG 2000), `JBIG2Decode`, unsupported color spaces, or born-digital/vector
pages with no image XObject. In those cases, rasterize out of band and retry
with page images.

## Batch OCR

```bash
focr ocr-batch page-1.png page-2.png page-3.png --json
focr ocr-batch page-1.png page-2.png --multi-page --json
focr ocr-batch page-1.png page-2.png --f32
FOCR_BATCH_SPINE=1 FOCR_BATCH_SIZE=64 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=1 FOCR_BATCH_PACK=1 focr ocr-batch page-*.png --json
FOCR_BATCH_SPINE=0 focr ocr-batch page-*.png --json
```

Use batch mode when the caller has multiple document images and wants one setup
cost. `ocr-batch` does not use the CLI's native PDF router; pass images, or
rasterize PDFs first. Batch behavior may preserve per-page failure information rather than
aborting the entire batch for every input failure. Confirm exact JSON shape from
the current source or a golden test before binding a production parser.

Useful batch flags/env vars seen in source:

| Surface | Meaning |
|---------|---------|
| `--multi-page` | Treat all image inputs as one Unlimited-OCR cross-page document pass; JSON uses `command: "batch.multi_page"` with `pages`, `seconds`, and `markdown` |
| `--f32` | Use the high-precision f32 decode path for `ocr-batch`; for single-page `ocr`, point `FOCR_MODEL_PATH` at bf16 safetensors when investigating f32 behavior |
| `FOCR_BATCH_SPINE` | Arm the continuous-batch decode spine for int8 `ocr-batch`; unset, empty, `0`, `off`, `false`, or `no` disable it |
| `FOCR_BATCH_SIZE` | In-flight stream count for the spine; current shared scheduler defaults to 128 and clamps at 256 |
| `FOCR_BATCH_VISION` | Inside the armed spine, batch the vision tower across pages by default; trim/case-folded `0`/`off`/`false`/`no` selects the per-page vision loop |
| `FOCR_BATCH_PACK` | Admission ordering by similar prefill length; output order must be restored and pack-on/pack-off must be byte-identical before speed claims |

Do not invent concurrency around batch mode. Let the engine own its runtime and
kernel fanout. Committed proof now has two related but distinct spine stories:
the default Unlimited-OCR int8 `ocr-batch` uses the original R-SWA scheduler and
batched-vision spine, while closed `bd-3jo6.1.7.5` adds the dense zoo route for
GOT-OCR2, SmolVLM2, and OneChart under `FOCR_BATCH_SPINE=1`. Dense-zoo source
truth after `4ca1577` is `OcrModel::recognize_batch_dense`,
`matches!(self.arch().id(), "got-ocr2" | "smolvlm2" | "onechart")`,
`smolvlm2::recognize_batch`, `onechart::recognize_batch`,
`generate_greedy_batched` taking `caps: &[usize]`,
`PageStream::with_max_emit`, `DEFAULT_BATCH_SIZE = 128`,
`MAX_BATCH_SIZE = 256`, and `FOCR_BATCH_PACK`. The close evidence in `fdd1d64`
is lossless at four levels and includes real binary byte-identical markdown for
all three dense zoo lanes. Keep the caveat ergonomic: this is current support for
lossless dense batching, not proof of broad batched `lm_head`,
fairness-controlled A11/PERF_LEDGER throughput, or every future zoo artifact.
For GOT-OCR2 specifically, `d25dbd7` first hydrates `SamWeights`,
`mm_projector_vary`, and `model.embed_tokens.weight` once per batch inside
`got::recognize_batch`; `3f2878d` then moves that state into model-level
`GotStatics` so ordinary sequential pages and dense batches share one
`OcrModel` cache. `FOCR_TIMING=1` should show `got.hydrate(cached)` once and
`got.vision+splice(batch of N)` for the batch. The measured pass-6 attribution
is narrow: about 0.8s/page saved in the cited 2-page sequential loop
(`got.vision+splice` 4.15 -> 3.31s/page, one 0.14s hydrate total), with GOT
sample byte identity and full-lib/fmt/clippy/ubs proof. Do not turn that into a
formal A11/PERF_LEDGER throughput row.
For OneChart specifically, source at/after `38ab806` / `a9a406e` has
`OcrModel::onechart_statics`, `onechart::OnechartStatics`,
`onechart::hydrate_statics`, and `onechart.hydrate(cached)`, with sequential
and batch calls routed through cached SAM/projector/embed state. Treat that as
committed pass-7 source evidence with byte-identical chart-data proof, full lib
960, and fmt/clippy clean, but not as standalone formal PERF_LEDGER evidence.
For SmolVLM2, source at/after `9b2a03b` has
`OcrModel::smol_statics`, `smolvlm2::SmolStatics`,
`smolvlm2::hydrate_statics`, and `smolvlm2.hydrate(cached)`, with sequential
describe/VQA and batch calls routed through cached SigLIP/projector/embed
state. Treat that as committed pass-8 source evidence with byte-identical
describe proof, lib green, and Beads comment 91, but not as `bd-av64.10`
closure, public VQA quality, or formal PERF_LEDGER evidence.
If batched vision fails in the default spine, current source falls back to the
per-page tower so typed per-page errors are preserved instead of being
stringified onto every page.

## Model Pull

```bash
focr pull
focr pull got-ocr2
focr pull smolvlm2
focr pull onechart
focr pull tromr
focr pull --quant int8 --json
focr pull --manifest ./manifest.json # custom/airgapped manifest only
```

`pull` downloads packaged `.focrq` weights and matching tokenizer/sidecar files,
validates hashes, and writes into the local cache. With no positional model it
fetches the manifest primary model (`unlimited-ocr`). With a positional id such
as `got-ocr2`, `smolvlm2`, `onechart`, or `tromr`, it selects that entry from
the manifest `models` map and installs the artifact plus `ModelEntry.sidecars`.
Non-primary models install under `~/.cache/franken_ocr/models/<model-id>/`.
It is the sanctioned networked setup step. Inference should use cached
artifacts.
Manifest source precedence is exact and worth preserving in integrations:
`--manifest <path-or-url>` wins, then `FOCR_MANIFEST_URL`, then the built-in repo
manifest embedded with `BUILTIN_MANIFEST_JSON`. The manifest may be a local JSON
file or `http(s)` URL. A model registry row being `implemented=true` is not
enough for `focr pull <id>`; the id must also appear in the resolved manifest.
Closed `bd-av64.7` / `ece14f9` updates the checked-in manifest with a top-level
default `unlimited-ocr` artifact plus named `models.got-ocr2`,
`models.smolvlm2`, `models.onechart`, and `models.tromr` entries.
Closed `bd-av64.8` and `bd-av64.9` then published and clean-cache verified the
new model releases: `models-smolvlm2-v1` contains
`smolvlm2.int8.focrq` plus `tokenizer.json`; `models-onechart-v1` contains
`onechart.int8.focrq` plus `vocab.json`, `merges.txt`, and
`added_tokens.json`; the then-current `models-tromr-v1` proof covered
`tromr.focrq` plus the four music tokenizer tables. `efccce9` / closed
`bd-av64.12` later updates the TrOMR distribution boundary: default
`focr pull tromr` installs `tromr.int8.focrq` storage plus the same tokenizer
tables, while `focr pull tromr --quant f32` installs the bit-exact
`tromr.focrq` reference. The clean-cache proof covers exact sizes/hashes,
GitHub release URLs, per-model cache subdirectories, idempotent repull, and real
inference smoke for each model. The older TrOMR int8-request-to-f32 fallback was
then-current historical evidence, not the current default. The HF mirror check
is not green: weights mirror URLs returned 401 and remain an auth/resilience
follow-up. Treat GitHub-first pulls as verified and HF mirroring as unverified
unless live source/Beads say otherwise.

Useful env vars:

| Env var | Meaning |
|---------|---------|
| `FOCR_MODEL_DIR` | Extra model search path for inference resolution |
| `FOCR_MODEL_PATH` | Exact model path for inference |
| `FOCR_QUANT` | Prefer a quant-suffixed artifact such as `int8` during default lookup |
| `FOCR_MANIFEST_URL` | Override manifest URL/path for pull when `--manifest` is absent |
| `FOCR_NO_MMAP` | Current committed source after `507cebe`: force owned-buffer weight loading instead of the default read-only mmap path; mmap failures also fall back to owned bytes |

`focr pull` installs `unlimited-ocr.int8.focrq`. Current default inference
resolution searches for the exact/default `unlimited-ocr.focrq` name and the
quant-suffixed names `unlimited-ocr.int8.focrq` / `unlimited-ocr.int4.focrq`,
so a fresh default pull should work without `--model`. `focr pull got-ocr2`
installs `got-ocr2.int8.focrq` and `qwen.tiktoken`; run it explicitly with
`--model got-ocr2.int8.focrq`. `focr pull smolvlm2` installs
`smolvlm2.int8.focrq` and `tokenizer.json`. `focr pull onechart` installs
`onechart.int8.focrq`, `vocab.json`, `merges.txt`, and `added_tokens.json`.
`focr pull tromr` installs `tromr.int8.focrq` plus `tokenizer_rhythm.json`,
`tokenizer_pitch.json`, `tokenizer_lift.json`, and `tokenizer_note.json` by
default; `focr pull tromr --quant f32` installs `tromr.focrq`. If
both exact and quant-suffixed default artifacts exist, the exact basename wins
unless `FOCR_QUANT` intentionally changes preference.

The default published TrOMR quant is now `int8` storage after `bd-av64.12`;
the `f32` quant remains for bit-exact reference work. `select_quant` uses an
exact quant when present, otherwise falls back only when a model entry has a
sole published quant. Unknown model ids or quant requests that cannot be
resolved are usage errors that name available manifest entries.

On Windows x86_64, OCR and `focr pull` are both proven: the native async
HTTP/TLS stack downloads, reassembles, and verifies the full multi-part model.
ARM64 Windows is not published.

## Model Discovery

```bash
focr models
focr models --json | jq .
```

`focr models` reads the static `model_arch` registry; it does not load weights.
Machine JSON contains `schema_version` and `models[]` with `id`,
`display_name`, `implemented`, `status` (`ready` or `planned`), `tasks`,
`vision_encoder`, `decoder`, `tokenizer`, `default_artifact`, `license`, and a
`pull` object. `pull.in_manifest` is the machine-readable pullability check for
the currently resolved built-in manifest, and `pull.quants` lists published
quants for that model.

Current source rule: `unlimited-ocr`, `got-ocr2`, `smolvlm2`, `onechart`, and
`tromr` have `implemented=true` / `status=ready` in the registry. SmolVLM2
sub-epic C is closed: conversion is model-gated-proven for the 500M checkpoint
(`bd-3jo6.3.2`), text-only decoder seam is certified (`bd-3jo6.3.5`), C6
tokenizer conformance is closed (`bd-3jo6.3.6`), C7/C9 route support is closed
(`bd-3jo6.3.7` / `.3.9`), C8 parity/e2e quality/perf is closed
(`bd-3jo6.3.8`), C10 detailed tests/e2e are closed (`bd-3jo6.3.10`), and
`bd-3jo6.3` is closed. `preprocess_smolvlm2` uses Pillow-exact LANCZOS
`resize_lanczos` (`resample: 1`); `src/native_engine/smolvlm2.rs` assembles
prompt/vision/splice/decode; `--task describe` requires
`--model smolvlm2.int8.focrq`; and `--question` /
`FOCR_SMOLVLM2_QUESTION` supply the VQA question. DISC-003 is the current
near-tie ledger, not an open-route blocker. OneChart sub-epic D is also closed:
D2 conversion, D3 vision/projector, D4 prefill/cached decode, D5 native
recognition, D6 parity/perf, D7 `OcrTask::ChartData` / `forward_onechart`, D8
`onechart_chart_e2e/v1`, D9 tokenizer, and `model_arch implemented=true` are
closed/current. Polyphonic-TrOMR is also implemented in current source:
E2/E6/E3/E4 are closed, E7 `merge_semantic` / `semantic_to_musicxml` is closed,
E8 closes the single-staff ladder, E5 closes v1 printed/scanned full-page
runtime, E9 adds `OcrTask::Music`, `model_spec_is_knowably_not_tromr`,
`forward_tromr`, `MusicResult`, `model_arch implemented=true`, and
`scripts/tromr_music_e2e.sh` / `tromr_music_e2e/v1`, and E10/sub-epic E are
closed. TrOCR and pix2tex remain `implemented=false`.
For SmolVLM2 VQA quality checks, use the C8 fixture guard only when the source
contains `tests/fixtures/smolvlm2/vqa_fixtures.json`. It scores against oracle
answers, not human labels or public VQA benchmarks, by normalized exact match or
symmetric content-word containment >=0.5. If `FOCR_SMOLVLM2_DIR` lacks both
`model.safetensors` and `smolvlm2.int8.focrq`, the Rust test is unarmed rather
than green evidence.
Distribution rule: trust `focr models --json` `pull.in_manifest` and
`pull.quants`, then the resolved manifest, not prose memory. Current committed
manifest entries make `focr pull smolvlm2`, `focr pull onechart`, and
`focr pull tromr` valid on source/binaries that include `bd-av64.7`. Older
installed binaries may still lack those entries; classify that as release lag
or stale binary.

Model selection guidance:

- `unlimited-ocr` is the default fast plain-text document OCR model.
- `got-ocr2` is implemented and pullable, but is heavier per page. Use it for
  specialized structured-output cases the default cannot target well: math
  (LaTeX), tables, charts, molecular SMILES, geometry, and sheet music.
- Default GOT dispatch is plain OCR (`format=false`). Add `--format` to request
  GOT's `OCR with format: ` prompt and Mathpix-Markdown `.mmd` output; it is a
  no-op for the default `unlimited-ocr` model.
- `--format` is a boolean GOT mode switch. It auto-selects from the image.
- `focr ocr --task` is current as a convenience selector over this same
  model-zoo surface. GOT specialized tasks imply `--format`; SmolVLM2
  `describe` uses the question string instead of GOT instruction modes.
- `focr convert --model-id smolvlm2` is current for SmolVLM2-shaped
  safetensors. To exercise describe/VQA, use the pulled or converted artifact
  with
  `focr ocr photo.jpg --task describe --model smolvlm2.int8.focrq --question
  "..."`; cite C8/C10 close evidence and DISC-003 before quality/perf claims.
- `focr convert --model-id onechart` is current for OneChart-shaped
  safetensors, the OPT tokenizer proof is current, D3 vision/projector is
  certified, D4 prefill is certified, and D4 cached decode source/test support
  is committed with `bd-3jo6.4.4` closed. D5 native recognition assembly is also
  closed in `bd-3jo6.4.5`: `ChartResult`/`recognize` can produce repaired JSON
  text plus optional numeric self-verify inside `src/native_engine/onechart.rs`.
  D6/D7/D8 then made it a public route via `--task chart-data`,
  `OcrTask::ChartData`, `forward_onechart`, and `onechart_chart_e2e/v1`.
  `focr pull onechart` is current after `bd-av64.7`; there is still no separate
  `focr chart`.
- `focr convert --model-id tromr` is current only for WS-folded TrOMR exports
  and current CLI conversion supports `--quant int8` only (`--quant int4`
  returns NotImplemented; there is no `--quant f32`). It writes a `.focrq`
  that self-declares `model_id=tromr`, and after `bd-av64.12` quantizes exactly
  40 Seq2SeqDense decoder GEMM tensors for storage while keeping
  encoder/embeddings/norms/heads high precision. Get the f32 reference through
  `focr pull tromr --quant f32`, not `focr convert`. `MusicTokenizer` / `MusicVocab`
  support is decode-only; `group_norm`, `tf_same_pad`, `max_pool2d`, and
  `TromrEncoderW` are E3 encoder evidence; `TromrDecoderW` and
  `decoder_forward` are E4 decoder evidence; `merge_semantic` and
  `semantic_to_musicxml` are E7 output evidence; `tromr::recognize`,
  `forward_tromr`, and `scripts/tromr_music_e2e.sh` are E9 runtime evidence.
  Use `focr pull tromr`, `focr pull tromr --quant f32`, or a supplied local
  artifact plus tokenizer tables for single-staff images or v1 printed/scanned
  full-page scores; do not infer int8 compute, camera dewarp/default-barline-
  quality support, unconstrained quality/perf, a perf win/int8 perf row, or
  `**kern`.
  `FOCR_TROMR_SPLIT=1` is experimental rescue only. `bd-2sez` is the f32
  baseline row for future int8 compute/perf work.
- Task-specific subcommands such as `focr music`, `focr chart`, or
  `focr describe` are not current.
- Specialized-output plumbing now has `bd-3kix` phase-1 real-model smoke
  coverage over five modalities. Fine-grained CER/TEDS/Formula-CDM budgets are
  still phase-2 follow-up work.
- GOT no-repeat-ngram hardening (`bd-ff4i`) is implemented: the default global
  guard is 20, CLI/`FOCR_NO_REPEAT_NGRAM` overrides win over
  `FOCR_GOT_NO_REPEAT_NGRAM`, and `FOCR_GOT_NO_REPEAT_NGRAM=0` disables it only
  for controlled diagnostics. `ocr-batch` also reads the README-documented
  `FOCR_NO_REPEAT_NGRAM`.

## Conversion

```bash
focr convert \
  /models/Unlimited-OCR/model.safetensors \
  -o /models/franken_ocr/unlimited-ocr-int8.focrq \
  --quant int8 --model-id unlimited-ocr --arch generic

focr convert \
  /models/GOT-OCR2/model.safetensors \
  -o /models/franken_ocr/got-ocr2.int8.focrq \
  --quant int8 --model-id got-ocr2 --arch generic

focr convert \
  /models/SmolVLM2-500M/model.safetensors \
  -o /models/franken_ocr/smolvlm2.int8.focrq \
  --quant int8 --model-id smolvlm2 --arch generic --json

focr convert \
  /models/OneChart/model.safetensors \
  -o /models/franken_ocr/onechart.int8.focrq \
  --quant int8 --model-id onechart --arch generic --json

focr convert \
  /models/tromr/model.safetensors \
  -o /models/franken_ocr/tromr.int8.focrq \
  --quant int8 --model-id tromr --arch generic --json
```

Conversion facts:

- `focr convert` currently accepts `--quant int8` and `--quant int4`; int4 is a
  scaffolded refusal and f32 is not a converter mode. For TrOMR f32 reference
  weights, use `focr pull tromr --quant f32` or a preverified local
  `tromr.focrq`.
- `.focrq` format version 1 uses magic `FOCRQ\0`.
- The file carries source sha256 metadata and a model-specific license notice.
- Non-default artifacts write a `model_id` header, e.g. `got-ocr2`; absent or
  empty `model_id` defaults to `unlimited-ocr` for v1 back-compat.
- `--model-id got-ocr2` writes the Apache-2.0 StepFun notice and omits GOT's
  tied `lm_head.weight` only after byte-verifying it matches the embedding
  table; use it only with GOT-shaped weights.
- `--model-id smolvlm2` uses the Idefics3 nested decoder prefix
  `model.text_model.layers.`, keeps the untied `lm_head.weight`
  high-precision, writes the Apache-2.0 SmolVLM2 notice, and rejects a
  checkpoint whose `lm_head` is byte-tied despite the descriptor declaring it
  untied. The model-gated C2 census expects 489 tensors: 224 QInt8PerChan
  decoder GEMMs and 265 F32 high-precision tensors. C5 decoder tests can then
  use `FOCR_SMOLVLM2_MODEL`, `FOCR_SMOLVLM2_ORACLE_HIDDEN0`, and
  `FOCR_SMOLVLM2_ORACLE_LOGITS` for the real-weight seam.
- `--model-id onechart` uses the OPT decoder prefix
  `model.decoder.layers.`, byte-verifies and dedups the tied `lm_head.weight`
  against `model.decoder.embed_tokens.weight`, writes the OneChart Apache-2.0
  notice, and should produce the D2 census for real weights: 384 source records
  -> 383 `.focrq` records, 72 QInt8PerChan decoder GEMMs, high-precision
  vision/projector/number-head/norms/biases/embeddings, and K=768/K=3072
  overflow proof rows. The D6-D8 route makes the converted artifact usable with
  `focr ocr --task chart-data --model onechart.int8.focrq` when tokenizer files
  are present; `bd-av64.7` later makes `focr pull onechart` available for the
  committed packaged artifact.
- D3 OneChart vision proof: the current seam is
  `preprocess::onechart_view_tensor` plus
  `src/native_engine/onechart.rs::vision_features`, certified against
  `onechart_proj_out.bin` with `proj_out cos 1.00000000`, maxabs `6.5e-4`, and
  a live `prompt_n` fixture value of 308.
- D4 OneChart prefill proof: the current seam is
  `DecoderFamily::Opt` / `DecoderConfig::onechart`, `nn::relu`,
  `onechart::build_inputs_embeds`, and `forward_prefill`, certified against
  `onechart_final_logits.bin` with last-position argmax 50268 (`<Number>`), cos
  `1.00000000`, maxabs `6.1e-5`, and prompt length 308.
- D4 OneChart cached decode proof:
  `2c77d21` committed `generate_greedy_kvcache` support for the OPT family,
  and `2769d21` closed `bd-3jo6.4.4` with
  `opt_kvcache_matches_greedy_and_oracle`, which compares a 24-token KV-cache
  greedy stream to O(n^2) re-prefill greedy, requires a >=12-token exact prefix,
  records a measured 13-step exact prefix, asserts first id 50268 (`<Number>`),
  and checks dict-open decoded output. Current source prefers
  `onechart.int8.focrq` when present for same-quantization checking.
- D5 OneChart recognition assembly proof:
  `0145419`/`2a56c96` added `ChartResult`, `recognize`,
  `complete_json_string`, `prefill_final_hidden`, `number_head`,
  `reliable_distance`, and the tests `recognize_reads_the_committed_chart`,
  `reliable_check_matches_upstream_goldens`, `number_head_matches_golden`, and
  `chart_prompt_ids_match_oracle_l0c`.
- D6-D8 OneChart public route proof: `bd-3jo6.4.6`, `.4.7`, `.4.8`, and
  `e926c46` add `OcrTask::ChartData`, `model_spec_is_knowably_not_onechart`,
  `forward_onechart`, `implemented=true`, and `scripts/onechart_chart_e2e.sh`.
- TrOMR conversion/tokenizer/encoder facts: `bd-3jo6.5.2` writes byte-exact
  `tromr.focrq` with `0 int8`; `bd-3jo6.5.6` adds decode-only WordLevel
  `MusicTokenizer`; `bd-3jo6.5.3` / `45da3a3` commits `group_norm`,
  `tf_same_pad`, `max_pool2d`, `TromrEncoderW`, and
  `tromr_encoder_matches_torch_oracle`; `bd-3jo6.5.4` / `3472c1b` commits
  `TromrDecoderW`, `decoder_forward`, and `tromr_decoder_matches_argmax_oracle`.
  `bd-3jo6.5.7` / `79d715c` commits `merge_semantic` and
  `semantic_to_musicxml`; `bd-3jo6.5.9` / `78a2de3` commits
  `tromr_staff_tensor`, `MusicResult`, `forward_tromr`, `OcrTask::Music`,
  `model_spec_is_knowably_not_tromr`, `model_arch implemented=true`, and
  `tromr_music_e2e/v1`. `bd-3jo6.5.8` is closed for the single-staff ladder,
  `fc9d88a` adds the E5 v1 detector module, `752f3cd` closes E5 with
  `recognize_page`, `staves_to_musicxml`, and detector-lossless stacked-page
  SER 0.125 / 0.040, `9127676` pins DISC-004 alpha routing, and `ab0bae0`
  closes E10 plus sub-epic E. At/after `40ee875`, source-current crop shaping
  is fit-first: already-fitting page-detected staff crops keep classic
  full-width geometry, while over-budget crops get ink-extent trim and
  neighbor-bounded extend-to-fit. `bd-av64.14` is closed for the fit-first
  geometry lane and p169 acceptance only. `bd-av64.7` publishes
  `focr pull tromr`; after `efccce9` / `bd-av64.12` default pull is
  `tromr.int8.focrq`, with `focr pull tromr --quant f32` for `tromr.focrq`.
  `bd-2sez` / `5430e2c` adds the f32 TrOMR
  PERF_LEDGER baseline row with exact token-stream agreement but slower focr
  f32 timings than pinned upstream torch. `**kern`, camera dewarp,
  default/lossless barline quality, int8 compute, and perf wins
  remain separate. `64edce3` / closed `bd-av64.4` adds experimental
  `FOCR_TROMR_SPLIT=1` barline splitting through
  `staff_detect::barline_columns` and `recognize_split`, but it is off by
  default and should be treated as recognition-count rescue only, not as a
  broad quality/perf/camera-dewarp proof.
- TrOMR resilience and staff observability after closed `bd-av64.2` are current
  source behavior. `recognize_page` returns
  `PageRecognition { staves, skips }`, records
  `StaffSkip { index, bbox, reason }`, skips failed staff crops when at least
  one staff succeeds, and reports every staff reason on all-fail pages.
  `forward_tromr` keeps stdout as data and prints human skip notes on stderr;
  `FOCR_TIMING=1` can show per-staff dims/outcome. Robot mode emits additive
  schema-v1 `staff` events through `robot::staff_event`, and `--json` /
  `-o .json` music runs include a detection-ordered `staves` array produced by
  `music_meta_to_json` from `MusicPageMeta` / `OcrEngine::take_music_page_meta()`.
  The event name is `staff`; do not promise `staff_detection` /
  `staff_result`, and do not claim a schema bump. Non-music JSON shapes remain
  unchanged, and PDF+music multi-page runs currently surface only the last
  page's staves through this side channel. Source at or after `40ee875` also
  shapes page-detected staff crops fit-first: classic full-width geometry is
  preserved when it already fits, and only over-budget crops get ink-extent trim
  and neighbor-bounded extend-to-fit. Do not expand the scoped `bd-av64.14`
  closure into camera dewarp, default/lossless barline quality, TrOMR int8,
  perf, or broad note-level SER claims. If `FOCR_TROMR_SPLIT=1` is relevant,
  frame it separately as the closed `bd-av64.4` experimental over-budget-band
  rescue path.
- TrOMR musical-sanity telemetry after closed `bd-av64.5` is current source
  behavior. `tromr::sanity_warnings` checks recognized semantic streams for
  overfull bars, underfull non-final bars, impossible durations, and
  cross-staff key mismatches. It is annotate-only: XML output receives
  `<!--focr-sanity: ...-->` comments that can be stripped while preserving the
  pre-pass musical content, robot mode emits additive schema-v1
  `music_warning` events, music-run JSON gets a `warnings` array, human mode
  prints a one-line stderr count, and `FOCR_TIMING=1` can print per-warning
  detail. Do not describe this as auto-correction, quality repair, or a reason
  to accept bad MusicXML structure; it is recognition-quality telemetry and the
  deterministic fallback for later correction work.
- TrOMR residual-skew refinement after `39651e6` is `bd-av64.13` lever 1, and
  `69039c3` closes the bead with negative/reverted evidence for the later
  levers. `refine_band_skew()` runs inside staff detection for
  detected page bands, sweeps +/-1.5 degrees at 0.1-degree steps, engages only
  when the winner is at least 0.2 degrees from flat, and abandons the change if
  the five-line group cannot be re-found. Straight bands stay byte-stable and
  the corpus remained green; the no21 double-dotted XFAIL did not flip. The
  measured follow-ups are negative: `FOCR_TROMR_TTA=3` regressed no17_sys at
  2.8x cost, and single-staff refined-crop routing broke the committed golden.
  Do not reintroduce them without held-out calibration evidence.
- `bd-av64.12` TrOMR int8 is committed/current but storage-scoped.
  `src/quant/convert.rs` enables `Decoder::Seq2SeqDense` decoder GEMMs for
  `--quant int8 --model-id tromr`, and `src/native_engine/weights.rs` adds
  `QInt8PerChan` `dequant_qint8()` fallback so f32 accessors can read
  quantized-storage tensors. Do not claim TrOMR uses int8 compute, that
  `robot selftest.models` covers it, or that the storage publication is a perf
  win.
- After converting GOT weights, copy the matching `qwen.tiktoken` into the same
  directory as the `.focrq`; `focr pull got-ocr2` does this automatically for
  the packaged artifact.
- `--arch` records offline packing target: `generic`, `aarch64-smmla`,
  `x86-vnni`, or `x86-amx`. `aarch64-smmla` now emits real offline SMMLA
  panels; VNNI/AMX remain tag-only until source adds packed-consuming x86
  kernels, and none of this is timing evidence.
- `int8` is implemented for the validated conversion lane.
- `int4` is not a casual option; it is phase-gated behind evidence.
- Do not convert or redistribute artifacts without preserving required license
  metadata.

## Robot Commands

```bash
focr robot schema | jq .
focr robot health | jq .
focr robot backends | jq .
focr robot selftest | jq .
focr robot selftest | jq '.models'
focr robot triage | jq '.quick_ref, .recommendations[0], .commands'
set -o pipefail
focr robot run scan.pdf | jq -c .
```

Robot mode is for agents and automation. Consume stdout as line-oriented NDJSON
when using `run`. Never rely on human prose in robot mode.

In current `bd-223.2` source, `robot health` and `robot backends` include a
`threads` field from the single `thread_budget()` source (`FOCR_THREADS` or
physical cores). Golden tests scrub it like `logical_cpus` because it is
host-dependent. If a binary lacks `threads`, classify the binary/source boundary
before assuming the field is absent by design.

See ROBOT.md for event and parser rules.

## Run State, Sync, and Doctor

### Run State and Sync

Closed `bd-223.4` adds the first real run-state surface in current source:

```bash
FOCR_RUN_STORE=/tmp/focr-runs.db focr runs --format json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr runs --format ndjson
FOCR_RUN_STORE=/tmp/focr-runs.db focr sync export-jsonl --file /tmp/focr-runs.jsonl --json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr sync import-jsonl --file /tmp/focr-runs.jsonl --json | jq .
```

Current contract from source/tests:

- `src/storage.rs` uses `fsqlite`, not `rusqlite`.
- `RunStore::default_path()` uses `FOCR_RUN_STORE` else
  `~/.cache/franken_ocr/runs.db`.
- `_meta.schema_version` is `SCHEMA_VERSION = 1`; a too-new store maps to
  `FormatMismatch` / exit 7.
- `RunRecord` fields are `run_id`, `started_at`, `finished_at`, `input_path`,
  `mode`, `quant`, `model_version_tag`, `exit_code`, and `status`.
- OCR run recording is best-effort telemetry: store failures print a stderr
  note and do not fail the OCR request.
- `focr runs --format plain|json|ndjson` queries by `--id` or most-recent
  `--limit`; JSON wraps records in `{schema_version, command, store, count,
  runs}` and NDJSON emits one record per line.
- `focr sync export-jsonl` writes a canonical JSONL audit file; `--file` is
  optional and defaults to the store path with its extension replaced by
  `.jsonl` (`runs.db` -> `runs.jsonl`).
- `focr sync import-jsonl --file FILE` replays records idempotently; the input
  file is required.
- Export uses a `.jsonl.lock` sentinel and `.jsonl.tmp` temp file before
  atomic rename; import takes the same lock and does not create an output temp.
  Lock contention is a clear error, not a silent partial write.
- Tests isolate `FOCR_RUN_STORE` to temp paths; copy that pattern in automated
  tests so the user's real cache is not polluted.

Because `bd-223.4` is closed in current `main`, call this closed-current when
source/help/tests agree. On an older binary, `runs` / `sync` may still be
truthful `NotImplemented` stubs or absent flags; classify with OP-LC/OP-SQ
before filing bugs or docs patches.

### Doctor

Current doctor contract after `bd-wp8.4` and `bd-wp8.4.1`:

```bash
focr doctor --json | jq .
focr doctor --dry-run --fix --json | jq .
focr doctor --fix --json | jq .
focr doctor undo <run-id> --json | jq .
focr doctor capabilities --json | jq .
focr doctor robot-docs
focr doctor --robot-triage | jq .
```

- Detect-only mode is pure and should be the default diagnostic.
- `--dry-run --fix` reports planned repairs without mutating the cache.
- `--fix` writes backups under `.doctor/runs/<run-id>/backups/`, appends
  `actions.jsonl`, and uses `.doctor/lock` for concurrency.
- `undo <run-id>` is the repair rollback path; do not hand-edit
  doctor-managed backups.
- Doctor exit codes are local to doctor: 0 healthy, 1 findings, 2 partial,
  3 failed and rolled back, 4 refused unsafe, 5 concurrency lost, 6 online
  required.

If an installed binary still treats `doctor` as a stub, classify it as release
lag or stale binary/source before filing bugs or editing docs.

No task-specific zoo subcommands are current yet. Do not claim `focr music`,
`focr chart`, or `focr describe`. Use `focr ocr --model
got-ocr2.int8.focrq --format <image>` or `focr ocr --task
formula|tables|chart|molecular|geometry|music --model got-ocr2.int8.focrq
<image>` for the current structured `.mmd` mode; use `focr ocr --task describe
--model smolvlm2.int8.focrq [--question "..."] <image>` for SmolVLM2
caption/VQA.

## Examples

Human OCR:

```bash
focr pull
focr pull got-ocr2
focr pull smolvlm2
focr pull onechart
focr pull tromr
focr ocr receipt.jpg > receipt.md
focr ocr scan.pdf > scan.md
focr ocr book.pdf --pages 3,5-9 --split-spreads -o excerpt.md
focr ocr receipt.jpg -o receipt.md
focr ocr receipt.jpg -o receipt.json
focr ocr receipt.jpg -o receipt.md --extract-figures
focr ocr formula.png --model got-ocr2.int8.focrq
focr ocr --model got-ocr2.int8.focrq --format formula.png
focr ocr formula.png --task formula --model got-ocr2.int8.focrq
focr ocr photo.jpg --task describe --model smolvlm2.int8.focrq --question "What is in the sky?"
focr ocr-batch *.png --f32
focr ocr page_0107.png --crop-mode gundam
focr convert /models/SmolVLM2-500M/model.safetensors -o smolvlm2.int8.focrq --quant int8 --model-id smolvlm2 --json
FOCR_BATCH_SPINE=1 FOCR_BATCH_SIZE=64 focr ocr-batch *.png --json
```

Automation OCR:

```bash
set -o pipefail
focr robot run scan.pdf | tee receipt.ndjson | jq -c .
```

Explicit offline model:

```bash
FOCR_MODEL_PATH=/opt/models/unlimited-ocr-int8.focrq focr ocr receipt.jpg --json
focr ocr formula.png --model ~/.cache/franken_ocr/models/got-ocr2.int8.focrq
```

Source truth probe:

```bash
rg -n "struct PullArgs|fn run_pull|fn run_models|struct OcrArgs|struct ConvertArgs|model_id|run_doctor|robot_triage_payload|split_spread|recognize_multi_page|arch_target|SmmlaPanels|robot_selftest|staff_event" \
  src/cli.rs src/dist.rs src/native_engine src/pdf.rs src/doctor.rs src/lib.rs
```

## Validation

Before documenting a new CLI behavior, run:

```bash
focr --help
focr ocr --help | rg -- '--output'
focr ocr --help | rg -- 'extract-figures|figures-dir|--format|--task|crop-mode|--pages|--split-spreads|--multi-page'
focr models --json | jq .
focr pull --help | rg -- 'MODEL|quant|manifest'
focr models --json | jq '.models[] | {id, pull}'
focr convert --help | rg -- '--model-id|--arch'
focr ocr-batch --help | rg -- '--f32|--multi-page'
focr robot schema | jq .
focr robot health | jq .
focr robot backends | jq .
focr robot selftest | jq .
focr robot triage | jq '.quick_ref'
focr doctor capabilities --json | jq .
focr doctor --dry-run --fix --json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr runs --format json | jq .
FOCR_RUN_STORE=/tmp/focr-runs.db focr sync export-jsonl --file /tmp/focr-runs.jsonl --json | jq .
python3 scripts/gauntlet_cert.py --release-readiness
```

If the binary cannot be trusted, cite the source lines and say the live binary
was stale instead of pretending help output is current.
