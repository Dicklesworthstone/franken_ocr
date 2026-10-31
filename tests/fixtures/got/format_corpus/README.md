# GOT `--format` validation corpus (bd-3kix, bd-fqn1)

Small, clean, fully synthetic images for each GOT-OCR2 specialized
`OCR with format:` modality. Every image is rendered from a source that is
committed verbatim as its ground-truth sidecar:

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

## Readings and scores (bd-fqn1)

Each asset has two frozen readings of the upstream `--format` pipeline, which stretches
the image to a square:

- `<asset>.got.txt`: the published int8 artifact, through the shipped KV-cache decode.
- `<asset>.f32.got.txt`: the upstream checkpoint
  (`stepfun-ai/GOT-OCR2_0@979938bf`), through the re-prefill decode. This is the only
  true f32 path: `got::recognize` keeps attention as the file stores it but quantizes
  the decoder FFN at build, whatever the file holds.

`chart.task.got.txt` and `chart.task.f32.got.txt` are the same two arms on what
`--task chart` feeds the model, the image padded to a white square. `--task formula`
pads too, but `formula.png` stays misread, so it has no reading of its own here.

A frozen reading records what the model says, and **frozen does not mean right**.
`scripts/got_cer.py --format-corpus` scores every reading against its sidecar, with
markup and whitespace removed, and fails if a score differs from `FORMAT_SCORES` in
either direction. `scripts/check.sh` runs it.

| asset | int8 CER | f32 CER | verdict |
|---|---:|---:|---|
| table | 0 | 0 | Read exactly. |
| molecule | 0 | 0 | Read as aspirin. RDKit canonicalizes the reading and the sidecar to one SMILES. |
| chart | 0.30 | 0.10 | **Misread.** Int8 gets three of four values wrong (A3 B5 C9 D10 for 3/7/5/9); f32 gets all four. Both say "Budget" for "Widget". See bd-kd75. |
| chart, `--task chart` | 0.55 | 0.55 | **Values read, title lost.** Every value right on both arms, byte-identical. The title reads `None`, and on a 20-character reference that alone costs more than three wrong digits: CER is a poor measure of a chart. |
| music | 1.3023 | 1.3023 | **Misread.** Both invent a second spine and `*M22`, and lose the second bar. |
| formula | 4.8485 | 4.8485 | **Misread as SMILES.** Both arms, byte-identical. |

Formula, table, molecule and music are byte-identical between the arms, so int8 costs
nothing on them; the squashed chart is the one reading where it does. Five images is one image per
format, so this table can show a lane failing but cannot show one working in general.
`artifacts/accuracy/bd-kd75/` holds the larger probe behind the chart and formula
findings.

Run the readings, armed by real weights:

```
FOCR_GOT_MODEL=/path/to/got-ocr2.int8.focrq \
FOCR_GOT_TIKTOKEN=/path/to/qwen.tiktoken \
    cargo ltest --release format_corpus -- --nocapture
```

`FOCR_GOT_MODEL=/path/to/model.safetensors` runs the f32 arm instead, at about five
minutes per asset. To freeze a new reading, set `UPDATE_GOLDENS=1`, review the diff,
and update its score in `scripts/got_cer.py` in the same change.
