# bd-2sez — TrOMR (music) perf row: focr f32 vs pinned upstream torch f32

**Date:** 2026-07-06 · **Threads:** focr=ref=8 (fairness-pinned OMP/MKL/OpenBLAS/VecLib/NumExpr)
· **Page:** tromr-upstream examples/1.png (single staff, sha256 f6bdab10…)
· **focr:** tromr.focrq f32 (sha256 a9d41485…) · **ref:** upstream TrOMR
img2score_epoch47.pth (sha256 02925259…) on torch 2.12.1, multinomial
argmax-forced (matches focr's default per DISC-004).

| stage | focr best ms | ref best ms | ratio (ref/focr) |
|-------|-------------:|------------:|------------------:|
| vision_encode | 200.000 (cv 4.3%) | 61.685 (cv 50.6%) | 0.308 |
| decode_per_token (68 tok both) | 34.559 (cv 1.4%) | 14.793 (cv 13.8%) | 0.428 |
| end_to_end | 2586.465 (cv 1.7%) | 1097.444 (cv 13.6%) | 0.424 |

**Honest reading:** the f32 music lane LOSES to torch ~2.3–3.2× per stage —
this row is the baseline the gated int8 experiment (bd-av64.12) must beat
losslessly. Token streams agree EXACTLY (68 tokens both lanes, argmax): the
slowness is kernel maturity, not divergence. focr end_to_end is the process
wall INCLUDING binary start + 88 MB artifact load + MusicXML emission; ref
excludes load (bias favors ref). Roofline columns are n/a: the roofline
machinery models int8 kernels only; the int8 row will carry them.
