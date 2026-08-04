# bd-av64.10 SAM Row-Tile Negative Evidence

This directory anchors the negative-evidence ledger entry for the reverted
row-tiled SAM global-attention score-matrix experiment.

## Claim

The experiment tested whether processing the global SAM attention score matrix
in `[128, 4096]` query-row tiles would keep QK, relative-position bias,
softmax, and AV work cache-resident enough to beat the historic full `[4096,
4096]` score-matrix path.

## Result

The tiled path was byte-identical on the real Unlimited-OCR and GOT-OCR2
fixtures used for the paired run, but it regressed wall time on the measured
Apple Silicon host:

```text
FOCR_SAM_TILE=0   sam.forward: 3.62 / 3.62 / 3.74 / 3.71 s
FOCR_SAM_TILE=128 sam.forward: 3.76 / 3.84 / 3.86 / 3.76 s
```

The hypothesis failed because the row-tiled implementation multiplied GEMM
dispatch count and lost enough intra-GEMM parallelism to outweigh the cache
benefit. The source was reverted in `8bd4037`.

## Reproduction Shape

```bash
FOCR_TIMING=1 FOCR_SAM_TILE=0   focr ocr page_0009.png
FOCR_TIMING=1 FOCR_SAM_TILE=128 focr ocr page_0009.png
```

The corresponding ledger row in `docs/NEGATIVE_EVIDENCE.md` carries the model
artifact hashes, fixture names, measured pairs, and do-not-retry guidance.
