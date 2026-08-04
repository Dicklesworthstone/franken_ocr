# GOT `--format` validation corpus — phase 1 (model-free synthesis, bd-3kix)

Small, clean, fully synthetic images for each GOT-OCR2 specialized
`OCR with format:` modality — the corpus that did not exist anywhere on this
machine (scout inventory, bd-3kix), so `--format` (bd-3jo6.2.10) was
plumbing-tested but accuracy-unvalidated. Every image is rendered from a source
that is committed verbatim as its ground-truth sidecar:

| image | modality | ground truth (the render source) |
|---|---|---|
| `formula.png` (600×200) | math LaTeX | `formula.gt.txt` — the mathtext LaTeX string |
| `table.png` (520×220) | table | `table.gt.json` — headers + rows |
| `chart.png` (600×400) | chart | `chart.gt.json` — title/ylabel/labels/values |
| `molecule.png` (400×400) | molecular (SMILES) | `molecule.gt.txt` — aspirin SMILES |
| `music.png` (800×267) | sheet music (kern) | `music.gt.krn` — the 2-bar `**kern` source |

`manifest.json` is generator-written: library versions + per-PNG sha256
(the generator is byte-deterministic on pinned versions — verified by two
back-to-back runs producing identical hashes).

## Regenerate

```
uv venv /private/tmp/focr_corpus_venv --python 3.12
uv pip install --python /private/tmp/focr_corpus_venv/bin/python \
    matplotlib pillow rdkit verovio cairosvg
# macOS: cairosvg needs homebrew cairo (brew install cairo) for the verovio SVG->PNG step
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
    /private/tmp/focr_corpus_venv/bin/python scripts/gen_got_format_corpus.py
```

Versions used for the committed assets (2026-07-01): python 3.12.12,
matplotlib 3.11.0, pillow 12.3.0, rdkit 2026.03.3, verovio 6.2.1-8d42439,
cairosvg 2.9.0, homebrew cairo 1.18.4. `molecule.png`/`music.png` are optional
at generation time (the script SKIPs them with a note if rdkit/verovio are
unavailable); both installed cleanly on this machine, so none were skipped.

## Consumers

- **Phase 1 (this):** `src/native_engine/got.rs` `format_corpus_*_smoke_e2e`
  tests — env-gated (`FOCR_GOT_MODEL` + `FOCR_GOT_TIKTOKEN`, skip-with-success)
  format-mode smoke gates asserting non-empty output containing LENIENT
  structural markers (a table digit, a chart value, ...). Run:

  ```
  FOCR_GOT_MODEL=/path/to/got-ocr2.int8.focrq \
  FOCR_GOT_TIKTOKEN=/path/to/qwen.tiktoken \
      cargo test --release format_corpus -- --nocapture
  ```

- **Phase 2 (bd-3kix follow-up):** once the real-model outputs are eyeballed,
  freeze per-asset hypothesis files and exact CER budgets scored with
  `scripts/got_cer.py` against these sidecars (the B8 pattern,
  `tests/fixtures/got/cer/`).
