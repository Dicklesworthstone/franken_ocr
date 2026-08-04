# Changelog

All notable changes to `franken_ocr` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Scope note.** Work began on 2026-06-24 as a planning and scaffolding effort and
reached a working, end-to-end engine for the first tagged release on 2026-06-29.
The pre-release kickoff (master plan, crate skeleton, agent conventions, oracle
harness) is no longer tracked as a separate `Unreleased` wave; it is folded into
the `0.1.0` history below where it belongs. Going forward, history lives in git and
in the `.beads/` issue graph, and capability waves are promoted into dated version
sections as they land.

## [Unreleased]

No changes yet.

## [0.7.2] - 2026-07-12

This patch release makes the exact pinned source closure portable, certifies the
default parallel R-SWA decode path on the current tree, and hardens the native
Windows installer gate without changing the immutable Unlimited-OCR model
payload.

### Changed

- Replace host-specific `/dp/asupersync` dependency paths with the portable
  pinned sibling layout used by CI, dist, fuzzing, and downstream source builds.
- Record a strict current-tree A/B for parallel R-SWA attention: the focused
  decode median improved by 11.94%, the capped 20-page decode median improved by
  8.53%, and both workloads produced byte-identical cross-arm output.
- Binary SemVer remains independent from model-artifact SemVer: the `v0.7.2`
  executables continue to consume the exact hash-pinned `v0.7.0` conservative
  Unlimited-OCR artifact.

### Fixed

- Remove release-time source rewriting for asupersync paths so fresh clones and
  all six native build targets consume the same immutable source closure.
- Give the Windows offline-installer lock holder up to 30 seconds to publish its
  deterministic readiness marker, and emit its captured logs on early exit or
  timeout.

### Verification

- The exact pinned dependency closure passed `scripts/check.sh`, including all
  repository validators, installer E2E, formatting, locked all-target check,
  clippy, 1,046 passing library tests with one ignored test, integration suites,
  doc tests, and UBS.
- The committed performance receipts bind the subject binary, 22,489-entry
  source manifest, model and tokenizer hashes, five measured runs per arm after
  warmup, quiet-host preflight, and byte-identical outputs for both workloads.

## [0.7.1] - 2026-07-11

This patch release restores reproducible Windows distribution and tightens the
current-tree performance evidence boundary without changing the immutable
Unlimited-OCR model payload.

### Changed

- Binary SemVer is independent from model-artifact SemVer: the `v0.7.1`
  executables continue to consume the exact hash-pinned `v0.7.0` conservative
  Unlimited-OCR artifact.
- CI and distribution workflows now pin the exact asupersync revision carrying
  the Windows socket-handle fix, and tag rebuilds bind their source and sibling
  dependency inputs to immutable Git objects.

### Fixed

- Preserve canonical LF manifest bytes on Windows, hash executable bytes in
  binary mode, and accept canonical multiline `focr --version` output in the
  native release and offline-installer path.
- Reject build-input drift during exact-tag rebuilds and require accuracy-risky
  attention/KV environment opt-ins to be truthy rather than merely present.
- Reject incomplete or failed batch results and incomplete decode-phase timing
  evidence in the current-tree gauntlet, with explicit writable cache and
  scratch roots for reproducible large-model runs.

### Verification

- The exact pinned dependency closure passed `scripts/check.sh`, including all
  validators, installer E2E, formatting, locked all-target check, clippy, 1,048
  runnable library tests, integration suites, doc tests, and UBS.
- CI run `29160016305` and dist run `29160016312` passed the Windows x86-64 and
  ARM64 check, build, native test, no-weights smoke, checksum, offline
  `install.ps1`, evidence-binding, and artifact-upload stages.

## [0.7.0] - 2026-07-10

This release hardens the native Unlimited-OCR path around exact reference
semantics, artifact identity, rollback safety, and reproducible distribution.

### Added

- An embedded 2,710-tensor Unlimited-OCR census with exact name, shape, dtype,
  source-size, recipe, and provenance validation for both full loads and bounded
  availability probes.
- A reproducible Torch 2.10 CPU MoE fixture generator covering 2,048 unsorted
  top-k cases and 256 production-width routed-reduction cases.
- An interactive stderr progress bar for long multi-page PDF and sequential
  `ocr-batch` runs, reporting pages, percent, elapsed time, and ETA. It is
  disabled for robot/JSON output, non-interactive stderr, `TERM=dumb`,
  `FOCR_TIMING`, and `FOCR_NO_PROGRESS=1`; end-to-end tests pin the
  machine-output boundary.
- Strict source-root and evidence-path binding for gauntlet rows and release
  certificates, plus committed evidence from a 14-pass optimization campaign.
- Native release matrices for Apple Silicon, Intel macOS, Linux x86-64/ARM64,
  and Windows x86-64/ARM64, with no-weights execution smoke tests for every
  staged binary.
- An embedded schema-v2 model manifest that pins the versioned 4.16 GB
  conservative Unlimited-OCR artifact and its three GitHub release parts while
  preserving the schema-v1 endpoint consumed by v0.6 binaries.
- Fail-closed CI, model-parity, performance, and distribution evidence
  producers plus a strict finalizer that refuses certification without fresh
  workflow receipts and three registry-pinned OpenPGP signers.

### Changed

- MoE routing now reproduces pinned Torch 2.10 CPU `topk(sorted=False)` slot
  order and the model's `[tokens, 6, 1280]` reduction order across f32, int8,
  batched prefill, and decode. `FOCR_MOE_SCORE_ORDER=1` retains the prior
  deterministic score-order policy as a truthy-only rollback.
- R-SWA checkpoints now bind cache identity and lineage epoch, rejecting
  cross-cache, cloned-cache, abandoned-branch, overflow, and ABA rollbacks
  before mutation.
- Model files load into owned bytes by default. `FOCR_MMAP=1` is an explicit
  capability opt-in, and both paths retain the same descriptor through
  validation and load.
- Installer version discovery fails closed, and post-install execution must
  report the exact requested semver. Unix and PowerShell installers now use
  destination-scoped locks, staged verification, atomic replacement, and crash
  recovery without clobbering an existing binary.

### Fixed

- Reject overlapping `.focrq` data/scale ranges and unknown architecture IDs in
  both full and bounded parsers.
- Pin CI and distribution actions to real commit objects and align clean-runner
  sibling revisions with `Cargo.lock` package versions.
- Restore clean termination on the adversarial `page_0590` table: the exact
  Torch route emits EOS at 4,033 tokens instead of exhausting the 12,000-token
  diagnostic cap.
- Reject source checkpoints that do not match the pinned Unlimited-OCR shard
  identity before conversion can emit an artifact the production loader would
  refuse.
- Make pull locks process-scoped and reusable after crashes, preserve the
  working model on failed replacement, and distinguish live staging downloads
  from orphaned partial files in `focr doctor`.
- Anchor strict certificate self-test round dates to the fixture clock so the
  freshness checks remain valid across UTC date changes.

### Verification

- The conservative exact-recipe artifact completed all 20 pinned pages with no
  missing results and normalized aggregate CER `0.19307925` against the `0.25`
  budget. The known `page_0590` tail remains documented at CER `0.61648860` and
  is not represented as individually exact.
- The full 6.67 GB source shard and 4.16 GB conservative artifact independently
  matched all 2,710 embedded tensor entries and their pinned whole-file hashes.
- SIMD self-test passed 44/44 model-shape and worst-case-overflow cases on Apple
  M4; exact MoE policy subprocess tests and batched parity passed on the final
  implementation.

## [0.6.0] - 2026-07-08

The first evidence-bundled performance release. It promoted only the rows that
survived the pinned-reference gauntlet and made the certification bundle itself a
release input.

### Added

- A reproducible release-certification bundle with scorecards, source evidence,
  convergence rounds, and a fail-closed 13-cell ship gate.
- A formal Unlimited-OCR head-to-head lane against pinned Torch BF16 CPU with
  thread, allocator, precision, and workload receipts.

### Performance

- Certified SmolVLM2's int8/refined `lm_head`, reducing head time by about 6x and
  decode time by about 40% on the measured fixture.
- Recorded the accepted Unlimited-OCR head-to-head rows: 3.41x and 2.81x
  end-to-end speedups on the two pinned workloads at matched thread counts.

### Fixed

- Hardened the first real gauntlet runbook against three evidence-linkage and
  aggregation defects before certification was allowed to pass.

## [0.5.2] - 2026-07-08

### Performance

- Added a read-only mmap weight-loading capability and measured its warm-cache
  startup benefit. This was the default in `0.5.2`; current main has since made
  owned bytes the safe default and retained mmap as an explicit immutable-inode
  opt-in.
- Cached OneChart and SmolVLM2 model statics across page runs, eliminating
  repeated vision/projector/embed hydration.

## [0.5.1] - 2026-07-07

### Added

- CI gate-log artifacts, advisory benchmark guardrails, deep fuzz/property
  coverage, and the first generated release-certification bundle.
- Frame-batched SmolVLM2 vision execution and model-level GOT-OCR2 statics reuse.

### Fixed

- Corrected layout-aware parity for offline SMMLA panels after the aarch64 CI
  lane caught a false comparison.
- Rejected and reverted SAM row tiling after every interleaved pair favored the
  untiled kernel; the loss remains recorded as negative evidence.

## [0.5.0] - 2026-07-07

The multi-page, hardening, and model-throughput release.

### Added

- Reference-faithful cross-page `infer_multi` orchestration for image batches and
  PDFs, including `--multi-page`, streamed page events, 10/20-page fixtures, and
  the 640px squash preprocessing contract.
- TrOMR staff-level robot observability, musical-sanity warnings, residual-skew
  refinement, a measured int8 artifact, and experimental barline splitting behind
  `FOCR_TROMR_SPLIT`.
- Offline SMMLA panel packing and zero-shuffle consumption, plus per-model SIMD
  self-test verdicts.
- Property-based and fuzz infrastructure; its first campaign found and fixed a
  parser denial-of-service class.

### Performance

- Enabled fused QKV GEMV by default after byte-identity proof.
- Hoisted GOT-OCR2 batch hydration and cached/pre-transposed CLIP and SAM weights.
- Extended the continuous-batch spine across the dense model zoo.

### Fixed

- Preserved fittable TrOMR staff crops and enforced the model position budget.
- Repaired CI provisioning after the `frankensqlite` asupersync dependency change.

## [0.4.0] - 2026-07-07

The model-zoo release. Four new runtime engines join Baidu Unlimited-OCR — all
pure-Rust, all CPU-only, all pullable with one command — plus real scanned-book
support (page selection, spread splitting, a critical sideways-scan fix) and a
bit-identical 30% speedup of the shared SAM vision tower.

### Added

- **Four new model engines.** `got-ocr2` (structured OCR: LaTeX formulas,
  tables, charts, molecular, geometry, sheet music via `--task`/`--format`),
  `smolvlm2` (photo description / VQA via `--task describe [--question]`),
  `onechart` (chart→data with the number-head reliability check via
  `--task chart-data`), and `tromr` (polyphonic sheet-music OMR to MusicXML via
  `--task music`, staff crops or full printed pages with native staff
  detection). Each was hand-ported and certified against its pinned reference
  (token-exact / cosine~1.0 oracle gates), runs through the shared int8 kernel
  layer with runtime ISA dispatch, and decodes 1.7-3.4x faster per token than
  the Hugging Face CPU reference at matched threads (`docs/PERF_LEDGER.md`).
- **The whole zoo is now `focr pull`-able** (`bd-av64.7/.8/.9`). Published GH
  releases `models-smolvlm2-v1` (1.09 GB int8 + tokenizer), `models-onechart-v1`
  (363 MB int8 + the OPT tokenizer triple), and `models-tromr-v1` (86 MB f32 +
  the four music tokenizer tables); `models/manifest.json` gained the entries
  plus a backward-compatible `sidecars` field. Non-primary models install into
  per-model cache subdirectories (two models can now both ship a
  `tokenizer.json` without clobbering), the resolver searches those subdirs,
  and a model publishing a single quant is pulled under the default `--quant`
  with a visible note (`focr pull tromr` just works despite being f32-only).
  `focr models` gained a truthful `PULL` column driven by the embedded
  manifest. Verified end-to-end on a clean cache: byte-exact downloads,
  idempotent re-pull, and one real inference per model — with the pulled
  OneChart artifact producing byte-identical output to the certified local one.
- **`focr ocr --pages` + `--split-spreads`** (`bd-av64.11`). Page selection for
  PDF inputs ("3", "3-7", "1,5-9,218"; 1-based, deduplicated, source order —
  a 218-page book no longer means 218 forwards), and opt-in two-page-spread
  splitting whose gutter detector accepts both a blank inter-page gap and a
  bound book's dark binding shadow; halves become logical pages
  (`{"page": N, "half": "left"|"right"}` in JSON). On the motivating scanned
  book, one spread went from a 10-minute forward-budget timeout to a 46-second
  correct two-half extraction.
- **`focr ocr --extract-figures`** (`bd-23s8`). Saves the figure/image regions the
  model grounds but does not transcribe to text (the `![](images/…)` placeholders)
  as real image files in a subfolder — default `<output-stem>_figures/`, or set
  `--figures-dir DIR` — and rewrites the Markdown to reference them
  (`![figure N](report_figures/page1_figure_1.jpg)`); the JSON gains a `figures`
  array of `{label, page, bbox, path}`. Each figure's format is chosen by content:
  JPG q85 for photographic regions, lossless PNG for line-art / charts /
  screenshots. PDFs name figures per page. The crop comes from a fresh EXIF-aligned
  decode of the source so it lands exactly on the grounded box. New library entry
  points `OcrEngine::recognize_with_figures` / `recognize_dynamic_with_figures`
  (and `_model` variants) return the document plus the cropped `ExtractedFigure`s.

### Performance

- **SAM vision tower: bit-identical 36% speedup** (`bd-av64.10`). Five exact
  transformations (parallel attention windows, parallel heads, hoisted rel-pos
  tables, a division-free bias add, memcpy QKV splits) took the shared SAM
  tower from 5.55s to 3.4s per 1024px view on an M4 — GOT-OCR2 full forward
  6.7s → 4.7s, a real unlimited-ocr book page 19.3s → 13.5s — with the CLI
  output proven byte-identical and every armed oracle cert green. Stage-level
  timing instrumentation now ships behind `FOCR_TIMING` for the whole tower.
- **Continuous-batch spine for the dense decoders** (`bd-3jo6.1.7.5`).
  `focr ocr-batch` can prefill and decode multiple pages together on the
  GOT-class dense engine, byte-identical per page to the sequential path.

### Fixed

- **Scanned books were OCR'd sideways** (`bd-av64.11`, critical). Many book
  scans store each page's raster in portrait orientation and rotate it into
  place with the page content stream's transformation matrix instead of a
  `/Rotate` entry; the native PDF fast path honored only `/Rotate`, so these
  documents reached the model rotated 90° and decoded hallucinated text until
  the forward budget expired. The renderer now classifies the CTM at the first
  image draw (axis-aligned rotations only; the direction was verified
  empirically against a reference render) and composes it with `/Rotate`.
- **TrOMR MusicXML emitter crashes and illegal output** (`bd-av64.1/.3`).
  Pitched thirty-second/sixty-fourth notes crashed the duration parser
  (multi-underscore names mis-split); `rest-256th`/`rest-512th` were missing
  from the duration table entirely; and mixed chord groups emitted
  importer-rejecting `<chord/><rest/>`. Every emitted document is now
  validated at emit time (structural lint — a violation is an engine bug and
  fails loudly rather than shipping broken XML).
- **One bad staff no longer aborts a TrOMR page** (`bd-av64.2`, partial). The
  full-page music path recognizes per staff and reports skipped staves (index,
  bbox, reason) on stderr while the page succeeds with the staves that worked;
  the page errors only when every staff fails, naming each reason.

### Direction (not yet shipped)

- **int4 expert quantization.** Group-quantized int4 expert weights at a
  Q4_K_M-class footprint, gated on a measured character-error-rate budget. A packed
  int4 s4s8 micro-kernel and exploratory `FOCR_EXPERTS_INT4` / `FOCR_LMHEAD_INT4`
  experiments already exist behind kill switches, but int4 is not validated and
  `focr convert` accepts only int8 as of `0.3.0`.
- **ARM64 Windows** (`bd-3u97`). `x86_64` Windows ships today; the aarch64-windows
  target is not yet published.

## [0.3.0] - 2026-06-30

The first-run-experience release. A clean-machine install report exposed three
papercuts on the path from "curl the installer" to "OCR a page", and all three are
now closed: the installer no longer aborts under `gum`, a freshly-pulled model is
found with no flags, and `focr ocr` can write its result straight to a file —
including structured JSON with bounding boxes. The native PDF path picked up two
robustness fixes on top: a decompression-bomb bound and a machine-visible per-page
skip event. Still pure, memory-safe Rust — no Python, no CUDA, no FFI at inference.

### Added

- **`focr ocr -o/--output FILE`** (`bd-sreb`). Writes the OCR result to a file
  instead of stdout; the format follows the extension — `.json` emits structured
  JSON, any other extension (e.g. `.md`) emits markdown, and `--json` forces JSON.
  The structured JSON carries the rendered `markdown` plus a `layout` array of
  `{label, boxes}` spans, each box `[x1, y1, x2, y2]` in source-image pixels (a PDF
  nests these under a per-page `pages` array). The same shape is what `--json`
  prints to stdout. New engine entry points `OcrEngine::recognize_with_layout` /
  `recognize_dynamic_with_layout` (and `_model` variants) return the markdown and
  the layout parsed from the **same** decode, so the two can never disagree
  ([e3c3e68](https://github.com/Dicklesworthstone/franken_ocr/commit/e3c3e68)).
- **Robot per-page skip event for PDFs** (`bd-fck1`). The resilient PDF document
  loop used to drop an undecodable page silently from the machine stream. It now
  emits a structured `page` NDJSON event (`status=skipped`, 1-based page number, a
  machine-classifiable `error_kind`) in robot mode, so a consumer can tell the
  document is missing pages; human mode keeps the stderr warning. `page` is an
  already-advertised event kind, so the robot schema stays v1
  ([ba71ff4](https://github.com/Dicklesworthstone/franken_ocr/commit/ba71ff4)).

### Fixed

- **Fresh-install OCR happy path** (`bd-3u6x`, critical). `focr pull` installs the
  model as `unlimited-ocr.int8.focrq`, but the default `focr ocr` lookup previously
  searched only the bare `unlimited-ocr.focrq` basename — so a freshly-pulled model
  was invisible without a manual `--model`, and the clean-machine happy path was
  broken. The resolver now also probes the quant-suffixed names `focr pull`
  installs (`.int8.focrq`, `.int4.focrq`); a pulled model resolves with no flag. An
  exact-basename match still wins over a quant variant
  ([e3c3e68](https://github.com/Dicklesworthstone/franken_ocr/commit/e3c3e68)).
- **Installer aborted instantly under `gum`** (`bd-1km0`). The first status line
  rendered `gum style … "-> …"`, so `gum` parsed the leading `->` as an unknown
  flag, printed its usage, and under `set -euo pipefail` aborted the whole install.
  It only triggered with `gum` installed AND an interactive TTY, so no
  non-interactive CI run ever exercised it. The whole `set -e`/arg-parse class is
  fixed: every `gum style` text arg gets a `--` flag terminator; `check_disk_space`
  guards a `df` that exits non-zero when the default `~/.local/bin` parent does not
  exist yet on a fresh account; the checksum pipelines and a failed `install`
  (ETXTBSY on reinstall-while-running) are guarded; the lock path is per-user and
  `$TMPDIR`-aware; `install.ps1` null-guards the top-level `LOCALAPPDATA` /
  `USERPROFILE` `Join-Path` so non-Windows `pwsh` reaches the friendly message, not
  a stack trace; and a `FOCR_INSTALL_BASE_URL` override lands for mirrors / airgap /
  tests. A new true end-to-end installer test (`tests/installer_e2e.sh`, wired into
  `scripts/check.sh` + CI with `gum` installed) drives the **real** installer
  through the `gum`/pty path against a fake `file://` release into a fresh-account
  dir — so this class of bug can never ship silently again
  ([7f25594](https://github.com/Dicklesworthstone/franken_ocr/commit/7f25594),
  [89c5b21](https://github.com/Dicklesworthstone/franken_ocr/commit/89c5b21)).

### Security

- **PDF decompression-bomb bound** (`bd-2zpu`, `bd-2yqe`). `lopdf`'s
  `Stream::decompressed_content` materializes the full inflated output before any
  length check, so a tiny highly-compressed sole-`FlateDecode` image stream (a "zip
  bomb") could inflate to gigabytes and OOM the process. `decompressed_stream` now
  inflates a sole `FlateDecode` (no PNG/TIFF predictor) itself via the pure-Rust
  `flate2` `ZlibDecoder` + `Read::take(cap + 1)` under an `expected_sample_cap`
  bound (4× the already-`MAX_PIXELS`-bounded declared sample bytes); a clean inflate
  that overruns the cap is the bomb signal and errors, while a non-zlib raw stream
  falls back to lopdf's framing-tolerant decoder. A fresh-eyes follow-up
  (`bd-2yqe`) clamped a hostile declared bit-depth that could otherwise saturate the
  cap to `u64::MAX` and reopen the hole. `flate2` is a direct dependency now
  (already in the lock graph via `lopdf`/`png`/`tiff`; pure-Rust `miniz_oxide`
  backend, no FFI)
  ([ba71ff4](https://github.com/Dicklesworthstone/franken_ocr/commit/ba71ff4),
  [89c5b21](https://github.com/Dicklesworthstone/franken_ocr/commit/89c5b21)).

## [0.2.0] - 2026-06-29

The document-input release. `focr` now reads **PDFs** natively, the agent-facing robot
stream finally carries the recognized text, and **native Windows (x86_64)** is proven
end to end — all in pure, memory-safe Rust, still with no Python, no CUDA, and no FFI at
inference. A two-pass fresh-eyes security and correctness review hardened the new PDF
codec path before it shipped.

### Added

**Native PDF page OCR — `focr ocr file.pdf`** (epic `bd-0a7`). Scanned / image-XObject
PDF pages are rasterized **in-process, in pure memory-safe Rust with no FFI**, then fed
through the identical preprocess → vision → decoder → postprocess pipeline a PNG takes —
no out-of-band `pdftoppm`. `lopdf` (with `default-features` off; 8 net-new pure-Rust
crates) is the container parser, and the project owns the image-codec dispatch:
`DCTDecode` (JPEG via `zune-jpeg`), `CCITTFaxDecode` Group 4 (the pure-Rust `fax` crate),
and `FlateDecode` / `LZWDecode` / `ASCII85Decode` raw samples in RGB / gray / CMYK /
1-bpc bilevel, with `/Rotate` applied and inheritable page attributes resolved. Pages
render lazily one at a time, so a 600-page book never materializes 600 rasters at once.
`JPXDecode` (JPEG 2000), `JBIG2Decode`, and born-digital vector / text pages — none of
which has a pure-Rust decoder — surface a precise, actionable error instead of a wrong
guess. The document loop is per-page resilient: an undecodable page is logged and skipped
so one bad page never discards the rest of the document, while whole-run conditions
(model-not-found, cooperative cancel, model format mismatch) abort immediately
([397f281](https://github.com/Dicklesworthstone/franken_ocr/commit/397f281)). In-memory
`OcrEngine::recognize_dynamic` / `recognize_dynamic_with_model` entry points back this
path and are reusable by library embedders.

**Native Windows (x86_64)** (epic `bd-3u97`). The `x86_64-pc-windows-msvc` target
compiles with zero errors and `focr.exe` runs full OCR end to end on Windows 10:
`robot backends` selects AVX2 via runtime ISA detection, `robot selftest` passes 24/24
(int8 GEMM bit-identical to the scalar oracle, including the K=6848 i32-accumulation
overflow case), and `focr ocr page.png` on a real scanned page produces the same markdown
as a Mac or Linux host. The model cache resolves to `%LOCALAPPDATA%\franken_ocr\models`
(falling back to `%USERPROFILE%\.cache\franken_ocr\models`), and an `install.ps1`
one-liner downloads and SHA256-verifies the Windows binary, then offers to run
`focr pull`. `focr pull` works on Windows too: the full 3.9 GB multi-part download,
reassembly, and SHA-256 verify complete over the native async HTTP/TLS stack — the
earlier send-path bug that surfaced as `WSAENOTCONN` / os error 10057 (`bd-15ow`) is
fixed in the asupersync runtime. ARM64 Windows is not yet published
([d8e40bd](https://github.com/Dicklesworthstone/franken_ocr/commit/d8e40bd),
[44d4949](https://github.com/Dicklesworthstone/franken_ocr/commit/44d4949)).

### Changed

- **Robot `run_complete` carries the recognized markdown** (`bd-3o5p`). `focr ocr --robot`
  and `focr robot run` previously emitted a payload-less `run_complete`, so a machine
  consumer received no OCR text at all; the terminal success event now carries the
  recognized `markdown`. `run_complete` was already an advertised event kind, so this
  finalizes its payload without a schema-version bump (`ROBOT_SCHEMA_VERSION` stays `1`)
  ([1351013](https://github.com/Dicklesworthstone/franken_ocr/commit/1351013)).
- Input is now **document images and PDFs**, not image-only; `AGENTS.md` and
  `focr ocr --help` document PDF as a supported input
  ([49b19c2](https://github.com/Dicklesworthstone/franken_ocr/commit/49b19c2)).

### Fixed and hardened (PDF codec review)

A two-pass fresh-eyes review of the new PDF path, before release:

- **Gigapixel-allocation DoS / overflow.** A crafted PDF declaring gigapixel image
  dimensions could overflow the CMYK `width*height` product or drive a raster `Vec`
  reserve into hundreds of TB. A 1-Gpx declared-dimension guard (computed in `u64`, so
  the check itself cannot overflow) now fires before any per-pixel allocation
  ([770a5e5](https://github.com/Dicklesworthstone/franken_ocr/commit/770a5e5)).
- **CCITT no longer pre-reserves from the attacker-controlled `/Height`** — the output
  grows as the Group 4 decoder emits lines.
- **Multi-filter codec chains** ending in `DCTDecode` / `CCITTFaxDecode` are rejected
  with an accurate message instead of feeding still-encoded bytes to the image codec.
- **Per-page PDF resilience with whole-run aborts** — cooperative cancel and model
  format-mismatch propagate instead of being swallowed per page, and a Ctrl+C mid-document
  aborts rather than logging a skip per remaining page
  ([49b19c2](https://github.com/Dicklesworthstone/franken_ocr/commit/49b19c2)).

### Performance

- The ~9.9 MB BPE tokenizer is cached on the model (`OnceLock`), so a multi-page PDF
  parses `tokenizer.json` once instead of twice per page; a load failure is not cached
  ([770a5e5](https://github.com/Dicklesworthstone/franken_ocr/commit/770a5e5)).

## [0.1.0] - 2026-06-29

First tagged release. `franken_ocr` is a pure-Rust, memory-safe, CPU-only runner for
the Baidu Unlimited-OCR document-parsing VLM: no Python, no CUDA, no FFI at inference,
no GPU. The library plus the `focr` CLI parse a document image into structured
markdown (or JSON) end to end. On a real page the int8 path matched the Baidu
reference to a measured character-error-rate of 0.0094, agreeing with the reference
oracle to within one token and beating it on one token; that is a measured result on
the 6.67 GB model, not a target. Earlier README and changelog copy that called this a
"pre-Phase-0 scaffold that does not run yet" is superseded by this release.

### Added

**Forward path: fp32 reference parity, then int8.** A framework-free forward built
from a thin substrate facade and nine engine modules. The vision tower
(SAM-ViT-B, 16x conv token compressor, CLIP-L/14) and the 2048 to 1280 projector
were proven against the Baidu reference at cosine 0.9996
([4a56122](https://github.com/Dicklesworthstone/franken_ocr/commit/4a56122)); the
DeepSeek-V2 MoE decoder (12 layers, hidden 1280, 10 heads, head_dim 128; layer 0
dense, layers 1 to 11 routing top-6 of 64 experts plus 2 shared experts) and the
`lm_head` were proven argmax-exact
([c864604](https://github.com/Dicklesworthstone/franken_ocr/commit/c864604)), then
the production `focr ocr` path was wired and proven argmax-exact against the reference
end to end
([ad798b4](https://github.com/Dicklesworthstone/franken_ocr/commit/ad798b4)).

**int8 decode engine.** A per-channel S8S8 decode engine on SDOT/VNNI delivers roughly
6.5x prefill and 2.5x decode with zero accuracy loss on easy pages
([d067900](https://github.com/Dicklesworthstone/franken_ocr/commit/d067900)), on top
of a bespoke parallel SIMD GEMV with a dequant-once weight cache
([a2e48ea](https://github.com/Dicklesworthstone/franken_ocr/commit/a2e48ea)),
per-expert parallelism
([c33be34](https://github.com/Dicklesworthstone/franken_ocr/commit/c33be34)), and a
fused q/k/v decode GEMV
([41d9a94](https://github.com/Dicklesworthstone/franken_ocr/commit/41d9a94)). The int8
path is byte-identical to f32 on easy pages; the vision tower stays high precision
because quantizing it wrecks OCR.

**R-SWA decode and KV cache.** Reference Sliding Window Attention is wired into decode:
generated-token KV is bounded to a window of 128 via a ring-buffer cache, while the
reference block (visual plus prompt tokens) is kept as a frozen, never-evicted global
KV. The O(n) ring-cache decode was proven bit-identical to the stateless reference
([c9e751a](https://github.com/Dicklesworthstone/franken_ocr/commit/c9e751a)), with a
batched `BatchedRingCache` for multi-stream decode
([a59dfdc](https://github.com/Dicklesworthstone/franken_ocr/commit/a59dfdc)).

**`focr convert`.** Offline weight transformation from bf16 safetensors into a custom
int8 `.focrq` artifact, with write-invariant validation and a single-source license
notice
([1ff638e](https://github.com/Dicklesworthstone/franken_ocr/commit/1ff638e),
[c86680e](https://github.com/Dicklesworthstone/franken_ocr/commit/c86680e)). Proven
byte-identical to the load-time int8 path on the real model (6,672,547,120 bytes of
bf16 down to 3,914,093,440 bytes of int8, about 3.9 GB), with the source SHA256
stamped into the artifact header.

**`focr pull`.** Auto-downloads the int8 weights (about 3.9 GB) over asupersync's
native HTTP stack (rustls plus webpki-roots, not reqwest or ureq), reassembles the
GitHub split parts, verifies every byte by SHA256 against a committed manifest, and
caches to `~/.cache/franken_ocr/models` so subsequent runs are fully offline
([4601ea3](https://github.com/Dicklesworthstone/franken_ocr/commit/4601ea3),
[cadfcdf](https://github.com/Dicklesworthstone/franken_ocr/commit/cadfcdf)).
`tokenizer.json` is fetched as a sidecar beside the weights. Production-verified
against this repo's `models-v1` GitHub release and a Hugging Face mirror. A failed
download leaves no `.partial` behind
([47c339c](https://github.com/Dicklesworthstone/franken_ocr/commit/47c339c)). On an
interactive TTY, `focr ocr` with no local model offers to download; robot mode and
non-TTY never auto-download and instead emit a model-not-found result with a
`focr pull` hint.

**Runtime ISA dispatch.** One binary per architecture detects CPU features at load and
selects the best int8 kernel tier: ARM SDOT and SMMLA (i8mm), x86 AVX2, AVX-VNNI, and
AVX-512-VNNI. SDOT is preferred over SMMLA on Apple Silicon after an A/B measurement,
and `FOCR_FORCE_ARCH` can pin a tier
([c387841](https://github.com/Dicklesworthstone/franken_ocr/commit/c387841)). On a
Threadripper 5995WX (a Zen 3 part with an AVX2 ceiling) dispatch correctly selects
AVX2.

**`focr robot selftest`.** Re-runs the dispatched int8 GEMM against a bit-identical
scalar oracle across a shape battery, including the worst-case K=6848 i32-accumulation
overflow, and emits a single JSON verdict (exit 1 on any divergence)
([59efd78](https://github.com/Dicklesworthstone/franken_ocr/commit/59efd78)). Proven
24/24 on Apple SDOT and on a real x86 AVX2 host.

**Throughput: batched and continuous decode.** A `focr ocr-batch` command loads the
model once and reuses the amortized int8 weight cache across many images
([61bf415](https://github.com/Dicklesworthstone/franken_ocr/commit/61bf415)). The
128-core throughput epic (`bd-1azu`) landed a continuous-batch decode scheduler driven
through `ocr-batch`, batched per-layer projection / grouped-MoE / R-SWA attention over
B streams, a batched K-token verify forward, and a lossless speculative-decode loop,
each parity-gated bit-identical to sequential decode
([f56dd0b](https://github.com/Dicklesworthstone/franken_ocr/commit/f56dd0b),
[f507ea7](https://github.com/Dicklesworthstone/franken_ocr/commit/f507ea7)).

**CLI surface.** The full command set: `ocr`, `ocr-batch`, `convert`, `pull`,
`robot run`, `robot schema`, `robot health`, `robot backends`, `robot selftest`, plus
the Phase-0/Phase-5 scaffolds `runs`, `sync export-jsonl`, `sync import-jsonl`, and
`doctor`
([5b26674](https://github.com/Dicklesworthstone/franken_ocr/commit/5b26674)). `ocr`
routes through the native model resolver with crop fusion and an n-gram fallback
([3614ba5](https://github.com/Dicklesworthstone/franken_ocr/commit/3614ba5)).
Inference tuning flags (`--base-size`, `--image-size`, `--crop-mode` with a default of
`gundam` dynamic-resolution tiling, `--max-length`, `--temperature`,
`--no-repeat-ngram`, `--ngram-window`) are shared by `ocr` and `robot run`. Recognized
environment variables: `FOCR_MODEL_PATH`, `FOCR_MANIFEST_URL`, `FOCR_NO_REPEAT_NGRAM`,
`FOCR_FORCE_ARCH`, `FOCR_STAGE_BUDGET_FORWARD_MS`, and `HOME` (for cache resolution).

**Robot mode.** A versioned NDJSON event stream (`run_start`, stage, page,
`run_complete`, `run_error`); `focr robot schema` self-describes the contract, and
the exit-code contract is wired into `run_error` events
([91015aa](https://github.com/Dicklesworthstone/franken_ocr/commit/91015aa)). Stable,
documented process exit codes: 0 success, 1 generic / not-yet-implemented, 2 usage,
3 model-not-found, 4 image-decode failure, 5 budget or timeout exceeded,
6 cooperative cancel, 7 format or version mismatch.

**Cross-platform binary release.** Four prebuilt single-file binaries, each a raw
executable (not a tar.gz) with a `<asset>.sha256` sidecar, sizes about 4.7 to 5.9 MB:
`focr-aarch64-apple-darwin-neon-sdot-i8mm` (Apple Silicon),
`focr-x86_64-apple-darwin` (Intel Mac), `focr-x86_64-unknown-linux-gnu`, and
`focr-aarch64-unknown-linux-gnu`. Linux binaries are gnu (glibc), not musl. There is
one binary per architecture that dispatches ISA at runtime, with no per-CPU-feature
variant; the Linux x86-64 binary is selftest-verified on real AVX2
([8c7bed4](https://github.com/Dicklesworthstone/franken_ocr/commit/8c7bed4)).

**Verification harness and oracle.** A Baidu reference oracle with a per-stage
activation dump and a CER scorer
([16af5cb](https://github.com/Dicklesworthstone/franken_ocr/commit/16af5cb)), real
engine subject-seam capture against the oracle (projector L1/L2, `lm_head` L3), and a
seeded parity-ladder contract that gates correctness before speed.

**Project foundation (folded from the pre-release kickoff).** The master engineering
plan, `AGENTS.md` doctrine, and the single-crate Rust 2024 (nightly) package with two
interchangeable binaries (`focr` and `franken_ocr`) over one shared `cli_main()`
entrypoint
([9b38939](https://github.com/Dicklesworthstone/franken_ocr/commit/9b38939)). A Source
and Oracle Truth Pack pinned and hashed the exact Baidu Unlimited-OCR source commits
and resolved all 18 open questions from pinned source before any forward code was
written
([3c18187](https://github.com/Dicklesworthstone/franken_ocr/commit/3c18187),
[2c337e3](https://github.com/Dicklesworthstone/franken_ocr/commit/2c337e3)). Phase 0
added the architecture and parity docs, the real oracle harness, the dual-binary fix,
tokenizer conformance, the `.focrq` container spec, and the artifact ledgers
([bfa982b](https://github.com/Dicklesworthstone/franken_ocr/commit/bfa982b)).

### Changed

- Source relicensed to the MIT License with an OpenAI/Anthropic Rider, Copyright (c)
  2026 Jeffrey Emanuel
  ([1e881a7](https://github.com/Dicklesworthstone/franken_ocr/commit/1e881a7)). The
  Baidu Unlimited-OCR weights, and any quantized derivative this project distributes,
  remain under the MIT License, Copyright (c) 2026 Baidu; that notice travels with any
  distributed weight artifact.
- Project status moved from "pre-Phase-0 scaffold that does not run yet" to a working,
  measured engine. The `ocr`, `convert`, and `pull` paths run for real; `runs`,
  `sync`, and `doctor` remain explicit scaffolds (see Known limitations).
- Version set to 0.1.0 for the first release
  ([a862cf3](https://github.com/Dicklesworthstone/franken_ocr/commit/a862cf3)).

### Fixed

- A hardening pass cleared 11 runtime test failures across five subsystems, taking the
  library suite to a green bar
  ([266a5fe](https://github.com/Dicklesworthstone/franken_ocr/commit/266a5fe)), and a
  `clippy -D warnings` sweep brought `--all-targets` to green
  ([a6438e6](https://github.com/Dicklesworthstone/franken_ocr/commit/a6438e6)).
- Forward kernels were hardened against malformed tensors: overflow-guarded igemm and
  int4/int8 unpack shapes, shape-arithmetic checks before allocation across the SAM,
  CLIP, MoE, and decoder paths, and rejection of non-finite MoE router and preprocess
  inputs
  ([a05ddaa](https://github.com/Dicklesworthstone/franken_ocr/commit/a05ddaa),
  [77ac01d](https://github.com/Dicklesworthstone/franken_ocr/commit/77ac01d),
  [34bc30a](https://github.com/Dicklesworthstone/franken_ocr/commit/34bc30a)).
- Robot streams now emit `run_start` before any `run_error`
  ([7a81aa3](https://github.com/Dicklesworthstone/franken_ocr/commit/7a81aa3)), the
  SAM/CLIP SIMD tiers are reported correctly by `robot backends`
  ([49156af](https://github.com/Dicklesworthstone/franken_ocr/commit/49156af)), and the
  i8mm SMMLA tier string was split from plain SDOT dotprod
  ([b8bae61](https://github.com/Dicklesworthstone/franken_ocr/commit/b8bae61)).
- Preprocessing matches Pillow's pad rounding and rejects grid overflow
  ([e1a91be](https://github.com/Dicklesworthstone/franken_ocr/commit/e1a91be)).

### Known limitations and negative evidence

- **int8 on hard dense tables.** int8 decode is byte-identical to f32 on easy pages,
  but on some dense tables it can trigger repetition runs (for example `page_0590`,
  tracked as `bd-ic8`) where f32 stays clean. The int8 decode path is behind a kill
  switch.
- **Vision int8 is not viable.** Quantizing the vision tower breaks OCR accuracy
  (measured CER about 0.37), so the tower is kept high precision. This is recorded
  negative evidence, not a retry candidate.
- **int4 is not shipped.** int8 is the only validated quant in 0.1.0. An int4-via-unpack
  `lm_head` path *regressed* by roughly 5.8x: the `s4s8` kernel unpacks nibbles to an int8
  buffer in memory before the dot product, adding memory traffic, so int4 was shelved on
  perf grounds; `focr convert` accepts only int8
  ([4370c15](https://github.com/Dicklesworthstone/franken_ocr/commit/4370c15)).
- **No native Windows binary.** Under WSL the Linux path applies; native Git-Bash, MSYS,
  or Cygwin are not supported targets (epic `bd-3u97`).
- **Image input only.** PDFs are rasterized out of band.
- **Scaffold surfaces.** `runs`, `sync export-jsonl`, `sync import-jsonl`, and `doctor`
  emit structured JSON but return not-implemented (exit code 1) in this release.

### Methodology and evidence

The forward path was built code-first against a Baidu reference oracle, then proven
stage by stage: vision and projector at cosine 0.9996, decoder and `lm_head`
argmax-exact, and the full `focr ocr` pipeline argmax-exact, with a final
character-error-rate of 0.0094 on a real page against the reference. Every perf and
quant lever carries a parity gate and an honest ledger entry, including the negative
results above. The `.focrq` byte-parity, the SHA256 manifest verification in
`focr pull`, and the 24/24 `robot selftest` (including the K=6848 overflow case) were
all verified on real hardware on Apple Silicon and on a real x86 AVX2 host.

[Unreleased]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.7.2...HEAD
[0.7.2]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Dicklesworthstone/franken_ocr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dicklesworthstone/franken_ocr/releases/tag/v0.1.0
