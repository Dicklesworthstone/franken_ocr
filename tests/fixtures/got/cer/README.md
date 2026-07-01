# GOT-OCR2 L5 accuracy (CER) — real-document gate (bead B8)

The GOT-OCR2 forward is certified seam-by-seam against the bit-deterministic torch
oracle (L0a tokenizer, L0b preprocess, L0c prompt, vision cos 1.0, decoder cos 1.0,
generation == L4). This directory adds the **L5 end-to-end accuracy** check on a
*real* document page — proving the pipeline works beyond the synthetic fixture.

## Method

- **Page**: a dense body-text page from *The Royal Navy: A History* (Clowes, vol. 2,
  1897 — public domain), scanned at ~1024px. The page image is off-repo (large); the
  file here is the accurate ground-truth transcription.
- **Ground truth** (`page_0107.gt.txt`): transcribed by a vision model (no OCR
  involved) — the reading text, de-hyphenated at line breaks.
- **Hypothesis** (`page_0107.got.txt`): the verbatim output of
  `focr ocr --model got-ocr2.int8.focrq page_0107.png` (pure-Rust CPU int8).
- **Scoring**: `scripts/got_cer.py <hyp> <gt>` — normalized (lowercase, collapse
  whitespace, join line-end hyphens, strip the running header, quote/dash
  normalization) Levenshtein CER + WER.

## Result

| metric | value |
|---|---|
| ground-truth chars | 2531 |
| edit distance | 64 |
| **CER** | **0.0253 (2.5%)** |
| WER | 0.0536 (5.4%) |

2.5% CER on a 130-year-old scanned book page (with its period typography) — GOT-OCR2
running in franken_ocr reads real documents accurately. Reproduce with:

```
python3 scripts/got_cer.py tests/fixtures/got/cer/page_0107.got.txt \
                           tests/fixtures/got/cer/page_0107.gt.txt
```
