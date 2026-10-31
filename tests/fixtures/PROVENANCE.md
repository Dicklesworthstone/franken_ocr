# tests/fixtures — provenance + committed-vs-regenerated policy (bd-2pgf)

Every fixture under this tree is either **[C] COMMITTED-CANONICAL** (frozen in
git; changing it is a reviewed golden change — see `golden/PROVENANCE.md` §4
for the `UPDATE_GOLDENS=1` review loop) or **[R] REGENERATED** (produced by a
pinned script from pinned sources; committed only when small, otherwise living
beside the zoo weights and consumed by armed, skip-with-SUCCESS tests). The
machine-readable twin of this catalogue is `MANIFEST.toml` (validated by
`scripts/check_fixture_manifest.py`, wired into `scripts/check.sh`).

The provenance rule (docs/conformance/LADDER_HARNESS.md §5): every measured
result traces to the pinned model. Oracle-derived fixtures carry their
generator script, the pinned stack, and the checkpoint hash **inside the
fixture** (`_meta` blocks) — this file is the index, not the authority.

## Catalogue

| Path | Kind | Origin | Regenerate with |
|------|------|--------|-----------------|
| `golden/` | [C] | Frozen CLI/robot surfaces (scrubbed, canonicalized) | `UPDATE_GOLDENS=1 cargo test --test cli_robot_golden` (reviewed) |
| `robot_schema_v1.json` | [C] | The frozen robot NDJSON contract (bd-zc1o) | Never regenerated — versioned by hand with the schema |
| `runs_schema.json` | [C] | The frozen `focr runs`/`sync` record + one-way-audit contract (bd-wp8.11) | Never regenerated — versioned by hand |
| `test_log_schema.json` | [C] | The TestLog line contract (bd-n68o) | Never regenerated — versioned by hand |
| `moe_torch_2_10_cpu.json` | [R→C] | Pinned macOS arm64 Torch 2.10 CPU `topk(sorted=False)` and `[N,6,1280]` reduction oracle; file SHA256 `84fa230a314ee2722a256b07504f9b319c470e246cb50729d1f49db78a3d01a2` | `uv run --python 3.12 --with torch==2.10.0 python scripts/gen_moe_torch_2_10_cpu_fixture.py --check tests/fixtures/moe_torch_2_10_cpu.json`; use `--write` only for reviewed regeneration on the pinned host |
| `tokenizer/corpus.txt` | [C] | Hand-curated conformance corpus (OQ-16) | Hand-edited only |
| `tokenizer_baidu/expected.json` | [R→C] | `scripts/gen_token_id_fixtures.py` (pinned `LlamaTokenizerFast`, truth-pack commit) | Rerun the script; commit is a reviewed change |
| `tokenizer/corpus_uocr2.txt` | [C] | Hand-curated Unlimited-OCR-2 cases (U2-A4): `<PAGE>` prompts, the four new added tokens, near misses, page-joined output | Hand-edited only |
| `tokenizer_baidu/expected_uocr2.json` | [R→C] | `FOCR_BAIDU_MODEL=unlimited-ocr-2 scripts/gen_token_id_fixtures.py` over the pinned v2 `tokenizer.json` (`c55c165e…799c589`), in a minimal venv of Python 3.10.19 + `tokenizers==0.21.4` (the v2 reference stack's pin; transformers is not imported) | Rerun the script in that venv; commit is a reviewed change |
| `tokenizer_got/` | [R→C] | `scripts/gen_got_token_id_fixtures.py` (pinned Qwen tiktoken) | Same |
| `tokenizer_onechart/` | [R→C] | Part of `scripts/gen_reference_fixtures_onechart.py` (GPT-2 BPE ids) | Same |
| `tokenizer_smolvlm2/` | [R→C] | `scripts/gen_smolvlm2_token_id_fixtures.py` | Same |
| `got/sample_text.png`, `got/oracle_fixtures.json`, `got/l0c_prompt.json`, `got/format_corpus/`, `got/cer/` | [R→C] | `scripts/gen_reference_fixtures_got.py` + `gen_got_format_corpus.py` (torch 2.12.1 + transformers 4.45.2, checkpoint per `docs/zoo/got-ocr2-spec.md`) | Rerun scripts against the zoo dir |
| `smolvlm2/sample_photo.png`, `smolvlm2/{oracle,vision_oracle,vqa}_fixtures.json` | [R→C] | `scripts/gen_reference_fixtures_smolvlm2*.py`, `gen_smolvlm2_vqa_fixtures.py` (truth-pack stack; floors recorded in-file) | Same |
| `onechart/sample_chart.png`, `onechart/corpus/` (6 charts + `manifest.json` + `detok` goldens) | [R→C] | `scripts/gen_reference_fixtures_onechart.py` + the bd-2lje corpus generator (matplotlib, sha256s in `manifest.json`) | Same |
| `tromr/tokenizer_{rhythm,pitch,lift,note}.json` | [C] | Upstream NetEase Polyphonic-TrOMR tables, byte-copied (Apache-2.0; byte-equality vs the zoo copies is asserted by `scripts/tromr_convert_e2e.sh` step 6) | Never regenerated — upstream-frozen |
| `tromr/detokenize_goldens.json` | [R→C] | HF `tokenizers` WordLevel oracle over the committed tables (generated 2026-07-05; provenance in `_meta`) | Rerun per its `_meta.script` note |
| `realscan_music/` | [C] | Public-domain Louis Spohr violin-school scans from Internet Archive, with tier-1 attributes, a frozen MusicXML anchor, and page/staff fixtures (bd-av64.6) | Hand-reviewed additions only; gate with `scripts/realscan_music_gate.sh` |
| `lilypond_selectable_text.pdf` | [C] | A real LilyPond-generated PDF with selectable text and opaque music glyphs, folded onto main with the MTDT-audited TrOMR snapshot (eea10a5); asserted exactly by `checked_in_lilypond_pdf_has_exact_text_and_opaque_music_golden` in `src/pdf_text.rs` | Never regenerated — a changed file is a reviewed golden change |
| `ladder_scorecard/scorecard_{armed,unarmed}.json` | [R→C] | `scripts/ladder_scorecard.sh` folding the parity ladder's own NDJSON (armed: M4 host, 2026-07-06, f32 safetensors + `fixtures/native_f32` oracle set — all six gates green, L4 token-exact 1.0, L5 CER 0.0; unarmed: the skip-honest shape, `skipped_no_model:true`) | Rerun the script armed/unarmed; a changed scorecard is a reviewed parity change |
| `cjk_stream/page_{1,2}.png` | [R→C] | `scripts/gen_cjk_stream_fixture.py` (Pillow, `/System/Library/Fonts/Hiragino Sans GB.ttc`; the script refuses a font missing any glyph): two 1024² pages of Chinese text mixing single-token characters with ones the v1 byte-level BPE splits, for `multi_page_streaming_survives_split_cjk_characters_when_armed` (U2-X1). Re-rendered at 40 px on a 64 px pitch for bd-1jx0 (sha256 `3600a87d…` and `6b9fa651…`): the reference, the all-f32 decode and the shipping int8 decode all read it as the same two pages, where the int8 decode read the first layout as one (DISC-004) | `python3 -I scripts/gen_cjk_stream_fixture.py --font "/System/Library/Fonts/Hiragino Sans GB.ttc" --out tests/fixtures/cjk_stream` |
| `uocr2_v1_goldens/` | [R→C] | `tests/uocr2_goldens.rs` on the pinned v1 conservative int8 artifact (size + sha256, inputs, token counts, git revision and threads in `CAPTURE.json`): generated token ids, final markdown and projected robot `page` events for the fast cases (`page_0009`, `page_0014`, `p9_p14`, its streamed twin) and the long ones (`page_0590`, `p10`, `p20`) (U2-T3) | `UPDATE_GOLDENS=1 FOCR_E2E_LONG=1 scripts/token_golden_gate.sh`, only with a reviewed reason (`docs/conformance/GOLDEN.md` §4a) |
| `uocr2_v2_goldens/` | [R→C] | `tests/uocr2_goldens.rs` (`v2_lane_matches_goldens`) on `unlimited-ocr-2.v0.1.0.int8.focrq`, the artifact U2-C4 certified, opened beside v2's pinned `tokenizer.json` (size + sha256, inputs, token counts, git revision and threads in `CAPTURE.json`): generated token ids, final markdown and projected robot `page` events for `page_0009`, `page_0014` and U2-A6's S2 document (pages 9, 14 and 19 in one pass, plain and streamed) (U2-C4) | `UPDATE_GOLDENS=1 FOCR_GOLDEN_GATE=1 FOCR_MODEL_PATH=<dir with the v2 .focrq and tokenizer.json>/unlimited-ocr-2.v0.1.0.int8.focrq cargo test --profile release-perf --test uocr2_goldens -- --exact v2_lane_matches_goldens`, only with a reviewed reason (`docs/conformance/GOLDEN.md` §4a) |
| `uocr2_oracle/` | [R→C] | `scripts/gen_reference_fixtures_uocr2.py`: Baidu's HF Unlimited-OCR-2 (`5632329a`) in f32 on CPU, on the v2 reference stack (`docs/truth-pack/PINNED_SOURCES.md`), with explicit `no_repeat_ngram_size=35`, `ngram_window` 128 (single page) or 1024 (multi-page), `max_page_tokens=20000` and the page guard (U2-A6). Contents: `<fixture>.ids.txt` and `<fixture>.md` for S1 (`page_0590`, `page_0009`, `page_0014`, base), S2 (`page_0009`, `page_0014`, `page_0019`, base), S3 (the `multi_page/` p9_p14, p10 and p20 page lists, base) and S4 (`page_0590` and `page_0009`, gundam, 2×3 tiles); `ORACLE.json` (input sha256s, params, counters, provenance); `FLOOR.json` (the nondeterminism floor and the bf16 ledger; per layer it covers the hidden states, attention outputs and router stages (`iq`, `ik`, `kbar`, scores, prior), and counts the rows whose selected block set changes (`routing_set_rows_differing`) separately from rows whose slots are only reordered). `scripts/check_oracle_provenance.py` validates all of it | Each run below writes `<out>/<dtype>/<tag>/`; `--commit-dir` is passed only to the f32 t12 run. `P=<venv>/bin/python -I`, `G="scripts/gen_reference_fixtures_uocr2.py --model-dir <hf snapshot> --corpus ~/.cache/franken_ocr/corpus/pages --out $FOCR_FIXTURES_DIR/uocr2_oracle --sets S1,S2,S4,S3"`: `$P $G --dtype f32 --threads 12 --tag t12 --commit-dir tests/fixtures/uocr2_oracle`; `$P $G --dtype f32 --threads 4 --tag t4`. The repeat and the bf16 runs skip S3-p10/p20 (bf16 on CPU is several times slower), with `O="--only S1-page_0590,S1-page_0009,S1-page_0014,S2-p9_p14_p19,S4-page_0590-gundam,S4-page_0009-gundam,S3-p9_p14"`: `$P $G $O --dtype f32 --threads 12 --tag t12r`; `$P $G $O --dtype bf16 --threads 12 --tag t12`; `$P $G $O --dtype bf16 --threads 4 --tag t4`. Then `$P scripts/oracle_nondeterminism_floor.py --uocr2-group f32=<f32/t12>,<f32/t4> --uocr2-group f32_rerun=<f32/t12>,<f32/t12r> --uocr2-group bf16=<bf16/t12>,<bf16/t4> --uocr2-group ledger=<f32/t12>,<bf16/t12>` (the `f32` group must cover every committed fixture) |

## The weights-gated fixture families (NOT in this tree)

Large oracle dumps live **beside the zoo weights** (`$FOCR_<MODEL>_DIR` /
`FOCR_FIXTURES_DIR`), consumed by armed rungs that skip-with-SUCCESS when
absent — committed here would be multi-GB:

* `fixtures/native_f32` (Unlimited-OCR L0–L5 oracle set; bf16 twin is the
  cross-precision ledger only),
* `uocr2_oracle/<dtype>/<tag>/` (Unlimited-OCR-2, U2-A6): `<fixture>_reference.json`
  plus `activations/<fixture>/*.npy` in the `native_f32` layout. The L0 tensors, the
  page2d coordinates and cos/sin, per-layer router seams (`iq`, `ik`, `kbar`, scores,
  candidate mask, fallback flag, entmax top-M, topi/bias/valid), the layer inputs,
  attention outputs and logits, for the prefill and for decode steps 0–63 and ±8
  around every `<PAGE>`. `f32/t12` is the oracle; the other tags exist for the floor
  and the bf16 ledger. Verify with
  `scripts/check_oracle_provenance.py --uocr2-dumps $FOCR_FIXTURES_DIR/uocr2_oracle`,
* the S1-corpus references (U2-C4's corpus gate, `uocr2_l5_corpus_gate_admits_v2_against_the_v1_baseline`):
  * v2: `uocr2_oracle/f32/t12/S1C-page_*`, one base page per corpus page, from
    `python -I scripts/gen_reference_fixtures_uocr2.py --model-dir <v2 HF snapshot> --out $FOCR_FIXTURES_DIR/uocr2_oracle --sets S1C --dtype f32 --threads 12 --tag t12`
    in the v2 oracle venv. Not committed, so no FLOOR coverage is needed. `S1C-page_0009`, `_0014` and `_0590` repeat
    the S1 fixtures' token SHA-256s.
  * v1: `v1_oracle_s1c/f32/page_*_reference.json`, from
    `python -I scripts/gen_reference_fixtures.py --model-dir <v1 HF snapshot> --corpus <corpus pages> --out $FOCR_FIXTURES_DIR/v1_oracle_s1c/f32 --mode base --device cpu --cpu-patch --f32`
    on the pinned v1 stack (torch 2.10.0, transformers 4.57.1). The remote code also imports matplotlib, requests and
    tqdm. `HF_MODULES_CACHE` must name a writable directory.
* `<zoo>/{got,smolvlm2,onechart}_*.bin|npz` seams,
* `<zoo>/tromr_preproc.bin` + `tromr_seam_*.bin` + `tromr_oracle_fixtures.json`
  (floors 0.0 recorded in-file; `scripts/gen_reference_fixtures_tromr.py`).

## Policy

1. **[C] fixtures never change silently** — a byte change is a reviewed golden
   change (`UPDATE_GOLDENS` or an explicit commit citing the reason).
2. **[R→C] fixtures change only by rerunning their generator** against the
   pinned stack; the commit message cites the script and why.
3. A new fixture family MUST be added to `MANIFEST.toml` (the validator fails
   otherwise) and to this catalogue.

## Windowed SGLang oracle runs (U2-D2a; dumps NOT in this tree)

`scripts/gen_reference_fixtures_uocr2.py --windowed W` serves one base-mode document
the way the pinned SGLang wheel (`sglang-0.5.20.dev10+g8d62f5fb9c`, sha256
`a3f3e2c9…c2ba70d`) does with `SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=W` and the README
server settings (per-page cap 20,000, one lookahead page with KV reuse, ctx 131,072),
on Baidu's HF reference (`5632329a`) in f32 on CPU. It commits nothing, so there is no
`MANIFEST.toml` entry. Each run writes `<out>/<tag>/run.json` (plan, substitutions,
per-window summary, document text, checks, provenance) and
`<out>/<tag>/w<bi>/window.json` plus `rows.*.npy` (per KV row: ids, origin, source row,
query page, coordinate bucket, tok_pos, RoPE page, frame bucket, page-state slots).

The 2026-10-30 set used script sha256 `5edc7645…627780e92` on git `dab3fc6`+dirty (macOS arm64),
with `P=<venv>/bin/python -I` and
`G="scripts/gen_reference_fixtures_uocr2.py --model-dir <hf snapshot> --corpus ~/.cache/franken_ocr/corpus/pages --out <dump root>"`.
Wall times are `/usr/bin/time` real, model load included:

* `$P scripts/gen_reference_fixtures_uocr2.py --self-test --windowed 2` (model-free; under 1 s)
* `$P $G --windowed 3 --doc S2-p9_p14_p19 --windowed-check degenerate,extend --threads 12 --tag degenerate-s2-w3-t12` (249 s)
* `$P $G --windowed 2 --doc S3-p10:7 --threads 12 --tag w2-n7-t12` (291 s)
* `$P $G --windowed 3 --doc S3-p10:7 --threads 12 --tag w3-n7-t12` (215 s)
* `$P $G --windowed 2 --doc S3-p10:7 --threads 12 --force-fallback 1:pool_too_small --force-fallback 2:no_text --force-fallback 3:no_entry --windowed-check recompute --tag w2-n7-recompute-t12` (364 s)
* `$P $G --windowed 2 --doc S3-p10:7 --threads 12 --tag w2-n7-t12r` (264 s)
* `$P $G --windowed 2 --doc S3-p10:7 --threads 4 --tag w2-n7-t4` (234 s)
* `$P $G --windowed 2 --doc S3-p10:5 --threads 12 --force-fallback 1:splice_alloc_failed --tag w2-n5-markeronly-t12` (126 s)
* `$P scripts/gen_reference_fixtures_uocr2.py --windowed-compare <dump root>/w2-n7-t12,<dump root>/w2-n7-t12r,<dump root>/w2-n7-t4` (model-free; under 1 s)

The dump root was a session scratch directory, so rerun the commands to regenerate the dumps.

`tests/fixtures/uocr2_windowed/<tag>.json` keeps what the Rust gates read from five of those runs
(`w2-n7-t12`, `w3-n7-t12`, `w2-n7-recompute-t12`, `w2-n5-markeronly-t12`,
`degenerate-s2-w3-t12`; the `t12r` and `t4` repeats hold the same ids). That is the plan, the forced
fallbacks and the bodies, and per window the copy plan, forced and generated ids, retention and the
sha256 of each per-row coordinate array's int32 data. `python3 -I
scripts/extract_uocr2_windowed_fixtures.py <dump root> <tag>...` writes them from the dumps. It
checks every array against the dump's own manifest hash.
