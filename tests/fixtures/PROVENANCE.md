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
| `tokenizer_got/` | [R→C] | `scripts/gen_got_token_id_fixtures.py` (pinned Qwen tiktoken) | Same |
| `tokenizer_onechart/` | [R→C] | Part of `scripts/gen_reference_fixtures_onechart.py` (GPT-2 BPE ids) | Same |
| `tokenizer_smolvlm2/` | [R→C] | `scripts/gen_smolvlm2_token_id_fixtures.py` | Same |
| `got/sample_text.png`, `got/oracle_fixtures.json`, `got/l0c_prompt.json`, `got/format_corpus/`, `got/cer/` | [R→C] | `scripts/gen_reference_fixtures_got.py` + `gen_got_format_corpus.py` (torch 2.12.1 + transformers 4.45.2, checkpoint per `docs/zoo/got-ocr2-spec.md`) | Rerun scripts against the zoo dir |
| `smolvlm2/sample_photo.png`, `smolvlm2/{oracle,vision_oracle,vqa}_fixtures.json` | [R→C] | `scripts/gen_reference_fixtures_smolvlm2*.py`, `gen_smolvlm2_vqa_fixtures.py` (truth-pack stack; floors recorded in-file) | Same |
| `onechart/sample_chart.png`, `onechart/corpus/` (6 charts + `manifest.json` + `detok` goldens) | [R→C] | `scripts/gen_reference_fixtures_onechart.py` + the bd-2lje corpus generator (matplotlib, sha256s in `manifest.json`) | Same |
| `tromr/tokenizer_{rhythm,pitch,lift,note}.json` | [C] | Upstream NetEase Polyphonic-TrOMR tables, byte-copied (Apache-2.0; byte-equality vs the zoo copies is asserted by `scripts/tromr_convert_e2e.sh` step 6) | Never regenerated — upstream-frozen |
| `tromr/detokenize_goldens.json` | [R→C] | HF `tokenizers` WordLevel oracle over the committed tables (generated 2026-07-05; provenance in `_meta`) | Rerun per its `_meta.script` note |
| `realscan_music/` | [C] | Public-domain Louis Spohr violin-school scans from Internet Archive, with tier-1 attributes, a frozen MusicXML anchor, and page/staff fixtures (bd-av64.6) | Hand-reviewed additions only; gate with `scripts/realscan_music_gate.sh` |
| `ladder_scorecard/scorecard_{armed,unarmed}.json` | [R→C] | `scripts/ladder_scorecard.sh` folding the parity ladder's own NDJSON (armed: M4 host, 2026-07-06, f32 safetensors + `fixtures/native_f32` oracle set — all six gates green, L4 token-exact 1.0, L5 CER 0.0; unarmed: the skip-honest shape, `skipped_no_model:true`) | Rerun the script armed/unarmed; a changed scorecard is a reviewed parity change |

## The weights-gated fixture families (NOT in this tree)

Large oracle dumps live **beside the zoo weights** (`$FOCR_<MODEL>_DIR` /
`FOCR_FIXTURES_DIR`), consumed by armed rungs that skip-with-SUCCESS when
absent — committed here would be multi-GB:

* `fixtures/native_f32` (Unlimited-OCR L0–L5 oracle set; bf16 twin is the
  cross-precision ledger only),
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
