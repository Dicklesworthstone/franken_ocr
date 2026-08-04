# Real-scan music corpus (bd-av64.6)

Real scanned engravings for the TrOMR music lane — the measuring device for
the geometry (bd-av64.14), micro-rotation voting (bd-av64.13), barline-split
(bd-av64.4), and int8 (bd-av64.12) gates. Every prior TrOMR quality number
was measured on synthetic renders or the upstream demo staves; the first real
book (2026-07-06) immediately exposed failure classes none of those covered.
This corpus freezes those classes as committed fixtures.

## Provenance & license

All images are from **Louis Spohr's Celebrated Violin School**, London 1843
(translated by John Bishop), scanned by the University of Toronto — Internet
Archive item [`louisspohrsceleb00spohuoft`]
(https://archive.org/details/louisspohrsceleb00spohuoft). Published 1843;
author died 1859; translator died 1885 — **public domain worldwide**. The
scans are faithful reproductions of a public-domain work. Grayscale,
downscaled from the IA PDF (150/300 dpi renders).

## The three-tier design (read before adding items)

Hand-writing token-exact note ground truth from 1843 engravings proved
error-prone even with high-zoom reading (chunk boundaries cut noteheads;
staccato dots vs. paper foxing; ledger-line ambiguity). WRONG ground truth
poisons a corpus silently, so truth here is tiered by verifiability:

1. **`truth/attributes.json` — the truth tier.** Only facts robustly
   verifiable by eye: clef sequences, key signatures, time signatures, staff
   counts, bar-count LOWER bounds, and spot-verified opening notes (each
   additionally cross-checked against an independent model reading). The
   harness gates on these exactly.
2. **`goldens/*.musicxml` — regression anchors, NOT truth.** Frozen model
   output on a fixture, clearly labeled. A diff means *the output changed*
   (investigate + re-freeze deliberately); it does not certify correctness.
3. **Fixture classes with no note assertions** — wide systems, double-dotted
   content (outside the TrOMR duration vocab), mixed prose pages: these gate
   detection/robustness (staff counts, no aborts), which is exactly what
   bd-av64.14/.4 change.

Expansion path: each new item enters at tier 1; promote spot-verified bars
toward fuller note truth only via the cross-model adjudication protocol
(agreeing independent readings from TrOMR + a human + GOT-kern where GOT
cooperates — note GOT's auto-format misclassifies narrow staff strips as
molecules/SMILES, so give it full systems).

## Items

| fixture | class | gates |
|---|---|---|
| `staves/spohr_no17_top.png` | single staff, clean quarters + 16th runs | attributes + bar1 notes + golden |
| `staves/spohr_no17_sys.png` | two-staff system (detection) | attributes |
| `staves/spohr_no21_sys.png` | two-staff, DOUBLE-DOTTED (vocab-external) | attributes; must not crash |
| `staves/spohr_p116_sys29.png` | WIDE system, 2 flats, virtuosic | attributes; the bd-av64.14 class |
| `pages/spohr_p055.png` | full 12-staff exercise page | staff-count floor via robot staff events |
| `pages/spohr_p100.png` | staves embedded in prose | detector precision floor; no abort |

Runner: `scripts/realscan_music_gate.sh` (model-gated skip-with-SUCCESS;
NDJSON per case on stdout, human table on stderr).
