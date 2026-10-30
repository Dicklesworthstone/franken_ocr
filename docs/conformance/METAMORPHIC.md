# METAMORPHIC.md — the oracle-free invariant catalog

**Bead:** `bd-re8.10` (VERIFY-metamorphic-suite) — the metamorphic leg of the
conformance pillar (plan §8.3, [`/testing-metamorphic`]). This document is the
**catalog of metamorphic relations** the `franken_ocr` engine must satisfy
**without** consulting the PyTorch oracle. It is the design spec the test suite
(`tests/metamorphic.rs`) implements; the suite is gated on a working end-to-end
path (`bd-re8.7`, the L5 gate) and, for the multi-page relation, on **OQ-13**.

> **Why metamorphic at all.** Differential testing (`bd-re8.9`) proves "same
> answer as the bf16 reference" — but it can only run on the frozen golden
> corpus, and it *inherits any bug the oracle also has*. Metamorphic testing
> proves **self-consistency under input transformations**: it runs on arbitrary
> generated inputs (no oracle, no 6.67 GB weights needed for the relations that
> are pixel-domain), and it catches a class of bugs differential testing is blind
> to — e.g. a coordinate-system bug that the oracle shares, or a non-determinism
> leak that only shows up across runs. The two suites are complementary, not
> redundant. **A metamorphic relation that encodes a falsehood is worse than no
> test** (see §6, the multi-page trap); every relation below is derived from the
> model's *documented* semantics, line-backed to THE SPEC.

## Provenance (every relation traces to a pinned source line)

Pinned source @ HF
**`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** / GitHub
`7e98affeacba24e95562fbaa234ddb89b856874a`
([`../truth-pack/PINNED_SOURCES.md`](../truth-pack/PINNED_SOURCES.md); SHA-256s in
[`../truth-pack/SOURCE_HASHES.md`](../truth-pack/SOURCE_HASHES.md)). The relations
are derived from the model's documented semantics, each `[SPEC-NNN]` clause
line-backed in
[`../truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`](../truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md),
and the multi-page relation from **OQ-13** in
[`../truth-pack/oq/rswa-attention.md`](../truth-pack/oq/rswa-attention.md). Runtime
pin (for any relation compared against the oracle's own transform): `torch==2.10.0`,
`transformers==4.57.1`, `Pillow==12.1.1`.

## Scope boundary (what lives here vs. elsewhere)

This suite covers **oracle-free invariants under content-preserving (or
content-relating) transforms**: identity-resize, 90°-rotation/transpose,
whitespace padding, run-to-run determinism, and the cross-page-dependence
property. It does **NOT** cover:

- **Malformed / adversarial input** (corrupt or truncated images, mid-decode
  cancellation, budget-exceeded) — that is the **fault** leg of the conformance
  pillar (plan §8.5), owned by `VERIFY-r1-fault-suite`. Keep the boundary explicit
  so robustness coverage is not assumed to live here.
- **"Same as the reference"** numeric parity (the L0–L5 ladder) — that is the
  differential suite (`bd-re8.9`) and the parity gates (plan §8.2).
- **Surface / contract** snapshots (`focr ocr --json`, `--help`, robot NDJSON) —
  that is the golden-artifact suite
  ([`GOLDEN.md`](GOLDEN.md), `bd-re8.11`).

---

## 0. How a metamorphic relation is stated (the shape every entry follows)

A metamorphic relation is a pair **(input transform `T`, output relation `R`)**
such that for every input `x`, `R(focr(x), focr(T(x)))` must hold. We state each
relation as:

| Field | Meaning |
|-------|---------|
| **Transform `T`** | the deterministic input mutation (pixel-domain, page-order, etc.). |
| **Relation `R`** | the property that must hold between `focr(x)` and `focr(T(x))`. |
| **Strength** | `STRICT` (bit/string-exact equality) · `STRUCTURAL` (equal up to a documented coordinate map) · `EXISTENTIAL` (a dependence/change *may* exist — the relation asserts the *possibility*, never independence). |
| **Determinism precondition** | every relation runs under `temperature=0` greedy (`do_sample=False`) so the engine is a pure function of (image, args); the determinism relation (MR-4) *proves* this precondition. |
| **Generator** | how the transformed-input pairs are produced. |
| **Structured-log line** | the NDJSON the test emits: `{relation, n_cases, violations, ...}`. |
| **Source backing** | the `[SPEC-NNN]`/OQ clause the relation is derived from. |

**Determinism is the foundation of every other relation.** A relation of the form
`focr(x) == focr(T(x))` is only meaningful if `focr(x)` is itself reproducible;
otherwise an equality failure is ambiguous (transform bug vs. run-to-run noise).
So **MR-4 (Base-mode determinism) is asserted first** and is a precondition for
MR-1, MR-3, and the strict half of MR-2. The engine is deterministic by
construction under greedy decoding (plan §7.3, §1.1 G5: "same image+args →
byte-identical output"); MR-4 is the guard that keeps it so.

### Per-relation structured logging

Every relation emits exactly one NDJSON summary line on completion and one line
per violation (so a CI failure is self-describing and grep-able):

```jsonc
// summary (always)
{"suite":"metamorphic","relation":"MR-1","n_cases":64,"violations":0,
 "strength":"STRICT","gated_on":null,"elapsed_ms":1234}
// per-violation (only on failure)
{"suite":"metamorphic","relation":"MR-2","case_id":"rot90_doc07","kind":"bbox_map",
 "expected":[12,40,88,120],"actual":[11,41,88,121],"tol_px":1,"detail":"x2 off by 2px"}
```

---

## MR-1 — Identity-resize invariance (recognized text)

| | |
|---|---|
| **Transform `T`** | Resize the input image to **its own current dimensions** (a no-op resize): `img.resize((w, h))` where `(w, h) == img.size`. Also covers re-encoding the same pixels to the same on-disk format. |
| **Relation `R`** | `text(focr(x)) == text(focr(T(x)))` — the recognized text is **byte-identical**. |
| **Strength** | `STRICT`. |
| **Source backing** | The preprocess front end is a pure function of the decoded RGB pixel buffer: EXIF-transpose on load (`[SPEC-020]`), aspect-preserving pad to square with fill `(127,127,127)` (`[SPEC-022]`), `ToTensor` then `Normalize(0.5,0.5)` → `[-1,1]` (`[SPEC-021]`). A no-op resize does not touch the pixel buffer that reaches `BasicImageTransform`, so the entire downstream forward is identical. |

**Why this is a real test, not a tautology.** It is the cheapest possible probe
that the **decode → preprocess** boundary is *idempotent and side-effect-free*: no
hidden global state, no run-counter in the buffer, no accidental dependence on the
file path or mtime, no off-by-one in the resize/pad geometry that a same-size
resize would still trigger. A surprising number of preprocessing bugs (a stray
`+1` in a tile-count, a buffer reused across calls, an RNG seeded from the clock)
surface here first. It is also the **degenerate case of MR-3** (zero-width
whitespace pad), so a failure here localizes the bug to resize/identity before the
padding logic is even exercised.

**Generator.** Take each fixture in the metamorphic corpus (§7), read its
`(w, h)`, produce the pair `(x, x.resize((w, h)))`. Include at least one image
whose dimensions are **not** a multiple of the tile size (640) and one that is, so
the pad/tile geometry is exercised both ways.

**Caveat.** This is invariance of the **decoded-pixel → text** path, so the two
inputs must decode to the **same RGB buffer**. Re-encoding through a *lossy* codec
(JPEG round-trip) is **not** an identity transform and must not be asserted under
MR-1 (it changes pixels); use the original decoded buffer or a lossless re-encode
(PNG). The test generates the resized image **in-memory from the decoded buffer**
to avoid an accidental lossy round-trip.

---

## MR-2 — 90°-rotation / transpose bbox relationships

| | |
|---|---|
| **Transform `T`** | Rotate the page by a multiple of 90° (`rot90`, `rot180`, `rot270`) or transpose it. These are the **exact-pixel-permutation** rotations (no resampling, no interpolation, lossless). |
| **Relation `R`** | The recognized **content** is preserved (same labels/text spans, possibly reordered), and every emitted bounding box transforms by the **known coordinate map** for that rotation, within a ±1-pixel rounding tolerance. |
| **Strength** | `STRUCTURAL` (equal up to the documented coordinate map; **not** `STRICT`, because reading order / span ordering may legitimately change with orientation). |
| **Source backing** | Boxes are emitted by the model as **normalized integer coordinates in `[0, 999]`** (`[SPEC-113]`), then rescaled to pixels at draw time via `x = int(coord / 999 * image_width)` (`modeling_unlimitedocr.py:104-111`). The rotation map operates in the **normalized `[0,999]` space**, before pixel rescale. EXIF-transpose-on-load (`[SPEC-020]`) means the *engine* sees the rotated raster directly. |

**The coordinate map (the load-bearing math).** A box is the 4-tuple
`(x1, y1, x2, y2)` of normalized coordinates with `0 ≤ · ≤ 999`, `x1 ≤ x2`,
`y1 ≤ y2`. Working in the **normalized square** (treat the page as `[0,999]²`; the
±1 tolerance below absorbs the difference between the true aspect-correct map and
the normalized-square map — see the caveat), the rotation of a point `(x, y)` is:

| Rotation `T` | Point map `(x, y) →` | Box map `(x1,y1,x2,y2) →` (re-sorted to min/max) |
|---|---|---|
| `rot90` (CCW) | `(y, 999 − x)` | `(y1, 999 − x2, y2, 999 − x1)` |
| `rot90` (CW) / `rot270` | `(999 − y, x)` | `(999 − y2, x1, 999 − y1, x2)` |
| `rot180` | `(999 − x, 999 − y)` | `(999 − x2, 999 − y2, 999 − x1, 999 − y1)` |
| `transpose` (main diag) | `(y, x)` | `(y1, x1, y2, x2)` |

After the map, the test **re-sorts** each box to `(min x, min y, max x, max y)` so
`x1 ≤ x2`, `y1 ≤ y2` (the rotation can swap which corner is the minimum). The
relation `R` is then: **the multiset of mapped boxes from `focr(x)` equals the
multiset of boxes from `focr(T(x))`**, matched by their associated label/text,
each coordinate within `±1` (the `int(... )` floor in the `/999` rescale loses up
to one unit each way; the corner re-sort and the normalized-square approximation
each contribute at most one more — budget `tol_px = 1` in normalized units, `≤ 2`
after pixel rescale on a 1000px page; widen proportionally for larger pages and
record the chosen tolerance in the structured log).

**Why exact-90° only.** Arbitrary-angle rotation **resamples** pixels
(interpolation), which changes the decoded RGB buffer and therefore can legitimately
change recognition — it is **not** a metamorphic-safe transform and must not be
asserted. Only the four lossless orientations (`rot90`/`rot180`/`rot270`/transpose),
which are pure pixel permutations, preserve content. The generator is restricted to
these.

**Content sub-relation (`STRICT` half).** Independent of geometry, the **set of
recognized text spans** (after stripping orientation-dependent reading order — sort
spans canonically, e.g. by mapped top-left) must be **equal** between `x` and
`T(x)`. A rotation that *drops* or *invents* a text span is a bug even if every
surviving box maps correctly. The test asserts both halves and logs which half
failed.

**Caveat — aspect ratio and the normalized space.** The normalized `[0,999]²`
coordinate map above is exact only when the box coordinates are interpreted in the
**square normalized frame** the model emits them in. The model normalizes against
the *padded square* view, not the original aspect ratio, so the `[0,999]²` rotation
is the correct frame and the `±1` tolerance covers the integer floor. If a future
mode emits boxes against the *unpadded* image, the map must compose with the
pad-offset transform; that is recorded as a follow-up if such a mode is added. For
v1 (base/Gundam, padded-square views), the `[0,999]²` map is correct.

**Generator.** For each fixture, emit pairs `(x, rot90(x))`, `(x, rot180(x))`,
`(x, rot270(x))`, `(x, transpose(x))`, using exact-pixel-permutation rotation
(`image::imageops::rotate90` etc. — no resampling). Prefer fixtures with **at
least one grounding/`<|det|>` box** so the geometry relation is actually exercised
(a pure-prose page with no boxes only exercises the content sub-relation).

---

## MR-3 — Whitespace-pad invariance (recognized text)

| | |
|---|---|
| **Transform `T`** | Surround the page with a border of **uniform background-colored** pixels (extend the canvas, place the original content unchanged inside, fill the new border). |
| **Relation `R`** | `text(focr(x)) == text(focr(T(x)))` — recognized text is **byte-identical**; bboxes shift by the known pad offset (the `STRUCTURAL` half, analogous to MR-2's map but a pure translation + rescale). |
| **Strength** | `STRICT` on text; `STRUCTURAL` on box geometry. |
| **Source backing** | The preprocess front end *already* pads every input to a square with fill `(127,127,127)` before the forward (`[SPEC-022]`, `ImageOps.pad(..., color=(127,127,127))`). Adding more uniform border before that step should be absorbed by the same aspect-preserving pad/resize and leave the *content* region's features unchanged up to the resize resampling. |

**⚠️ The fill-color subtlety (the gotcha the bead flags).** The relation is
"**whitespace**-pad invariance," but the model's own pad color is **gray
`(127,127,127)`** — the dataset mean, mapped to `0` under the `[-1,1]` normalize
(`[SPEC-022]`, `[SPEC-021]`), **not** white `(255,255,255)`. There are therefore
**two distinct transforms**, and the relation must use the right one:

1. **MR-3a — mean-gray pad (the safe, primary relation).** Pad with the model's
   own fill `(127,127,127)`. This maps to exactly `0` after normalization — the
   same value the internal pad produces — so the bordered region is
   representationally identical to the model's own padding. This is the
   **defensible STRICT relation** and the one the suite asserts as a hard gate.
2. **MR-3b — true-white pad (a SHOULD, not a MUST).** Pad with white
   `(255,255,255)`. White is **not** the model's pad color; it maps to `+1` after
   normalization and is *visible content* to the encoder. Padding with white can
   legitimately change recognition (the model may treat a white margin as page
   structure). MR-3b is asserted only as an **EXISTENTIAL/SHOULD** observation
   (record whether text changed; do **not** fail the build on a change) and is
   ledgered as a known sensitivity, not a bug.

**Verify the pad color against the actual implementation, not the prose.** The
test reads the fill color from the same constant the preprocess uses (it must not
hard-code `127` independently); if the preprocess pad color ever changes,
MR-3a's transform changes with it. This is the bead's explicit risk note
("Whitespace-padding invariance assumes the pad gray (127) is outside the
recognized content — verify against the actual pad color").

**Box geometry sub-relation.** When MR-3a adds a border of `p` pixels on a side,
every emitted box translates by the pad offset and rescales by the new dimensions:
in normalized space, a content box at normalized `c` on the unpadded `W` maps to
`int((c/999·W + p) / W_padded · 999)` — a pure affine map the test applies and
checks within `±1` normalized unit, the same tolerance discipline as MR-2.

**Generator.** For each fixture, emit `(x, pad(x, p, fill=mean_gray))` for a few
border widths `p ∈ {8, 32, 128}` px on each side and asymmetric (left/top only),
to exercise both the geometry offset and the resize-to-square interaction. Include
the **zero-width** pad (`p = 0`) as the bridge to MR-1 (identity).

---

## MR-4 — Base-mode determinism across runs

| | |
|---|---|
| **Transform `T`** | **Identity** — run the *same* image with the *same* args **twice** (and at two thread counts: `FOCR_THREADS=1` and `FOCR_THREADS=N`). |
| **Relation `R`** | `focr(x)` is **byte-identical** across runs and across thread counts: same decoded text, same token sequence, same bbox tuples, same `--json` payload (after scrubbing only timing/run-id, never content). |
| **Strength** | `STRICT`. |
| **Source backing** | Greedy decode (`temperature=0` → `do_sample=False`, plan §7.3); the int8 GEMM is **bit-identical across SIMD paths** (integer add is exact/associative — plan §5.4); the engine is a pure function of (image, args). The f32 forward must reduce in a fixed order so thread count does not perturb the result (this is the engine's design obligation, plan §1.1 G5 / §7.3). |

**Why this is the keystone relation.** It is both a test *and* a precondition.
Every `focr(x) == focr(T(x))` relation (MR-1, MR-3a, the strict half of MR-2)
silently assumes `focr(x)` is reproducible; MR-4 is what makes that assumption
checkable. A determinism leak — a HashMap iteration order in the router, a
thread-count-dependent reduction order in a GEMM accumulator, a clock-seeded RNG,
an uninitialized buffer — is the single most corrosive bug class here because it
makes *every other relation flap intermittently*. MR-4 catches it directly, with
no oracle and no transform.

**Thread-count axis is mandatory.** Running twice at the *same* thread count
catches RNG/iteration-order leaks; running at **two different thread counts**
additionally catches **reduction-order non-associativity** (the classic
"`@8` ≠ `@32`" float-sum drift). The f32 reference forward must therefore use a
deterministic reduction (fixed tiling / tree-reduce independent of thread count),
not a `rayon` `sum()` whose order depends on work-stealing. MR-4 is the gate that
holds the engine to that.

**Generator.** For each fixture, run `{(seed_run=1, threads=1), (seed_run=2,
threads=1), (seed_run=3, threads=N)}` and assert all three byte-identical
(content fields). The structured log records the first divergent field and offset
on failure, so a determinism regression is immediately localized.

**Relationship to the golden suite.** MR-4 proves the *engine* is deterministic;
[`GOLDEN.md`](GOLDEN.md)'s scrubbed-NDJSON and exact-JSON snapshots then freeze
that deterministic output forever. MR-4 is the upstream guarantee that makes the
golden snapshots non-flaky; the golden suite is the downstream regression catch.

---

## MR-5 — Multi-page CROSS-PAGE dependence (the corrected relation, OQ-13)

> **⚠️ THE TRAP.** The naïve metamorphic relation here is
> *"multi-page concat = sum of single-page parses"* — i.e. `parse([p1, p2, …,
> pN])` equals `concat(parse(p1), parse(p2), …, parse(pN))`. **This relation is
> FALSE and must NEVER be asserted.** Encoding it would bake a falsehood into the
> conformance suite — the most dangerous metamorphic failure mode (a green test
> that certifies wrong behavior). OQ-13 is **RESOLVED** and it resolves
> *against* independence.

### What OQ-13 actually established

In multi-page mode (`infer_multi`), **all pages' image tokens are concatenated
into a single contiguous prefill in one `generate()` call**
(`modeling_unlimitedocr.py:1198-1212, 1233-1237, 1240-1256`;
[`../truth-pack/oq/rswa-attention.md`](../truth-pack/oq/rswa-attention.md) OQ-13).
During that single prefill, page *N*'s tokens attend **causally** to every token
of pages `1..N-1`. Under R-SWA the entire multi-page prefix is the **permanent,
never-evicted reference block** (`config.sliding_window` is nulled precisely so
`DynamicCache` does not truncate it; the 128-slot ring bounds only the *generated*
tail — OQ-1/OQ-3/OQ-13). **Therefore the multi-page decode is cross-page
DEPENDENT**: page *N*'s output is a function of pages `1..N`, not of page *N*
alone. The output is **not** a concatenation of independent single-page parses.

### The correct relation (the OPPOSITE of independence)

| | |
|---|---|
| **Transform `T`** | Reorder the page sequence (e.g. swap pages `i` and `j`, or reverse) **or** mutate an earlier page's content, in a multi-page parse. |
| **Relation `R` (MR-5a, EXISTENTIAL — dependence exists)** | There **exists** a multi-page input for which changing page order or an earlier page's content **changes** a later page's output. The relation asserts the *possibility* of cross-page influence — it is a **dependence-existence** property, never an independence property. |
| **Relation `R` (MR-5b, STRICT — single-page is self-consistent)** | The **first** page of a multi-page parse, when that page has no preceding context, is governed by the single-page relations (MR-1/MR-3a/MR-4); i.e. `parse([p1])` is deterministic and equals the single-image parse of `p1`. This anchors one end without claiming independence for the tail. |
| **Relation `R` (MR-5c, STRUCTURAL — prefix-stability under append)** | **Appending** a page to the *end* of the sequence does not change the prefill KV of the earlier pages (the reference prefix is causal and append-only at prefill), so the **earlier pages' contribution to the reference block is stable**; this is a structural property of the prefill assembly, not an output-equality claim on the generated text (the generated text still depends on the full prefix at decode time). |
| **Strength** | `EXISTENTIAL` (MR-5a) · `STRICT` (MR-5b) · `STRUCTURAL` (MR-5c). |
| **Source backing** | OQ-13 (`modeling_unlimitedocr.py:1198-1212, 1233-1237, 1240-1256`); OQ-1 (permanent prefill prefix, `modeling_deepseekv2.py:1322, 1363-1364`); OQ-3 (ring bounds only the decoded tail). |

**Why phrase MR-5a as existential, not universal.** We assert that cross-page
dependence *can* occur (and we construct at least one input pair that
demonstrates it), **not** that *every* reorder changes output — many reorders of
independent pages may coincidentally produce identical output, and asserting
universal change would be as false as asserting independence. The defensible,
non-falsifiable-into-a-lie statement is the existence one: *the engine's
multi-page output is a function of the whole page set, demonstrably so on at
least one constructed case.* This is the bead's explicit requirement: "asserts
cross-page dependence, NOT independence."

**Gating.** MR-5 is **gated on a working multi-page e2e path** (the model and the
`infer_multi`-equivalent prefill assembly) **and** is anchored by the
already-RESOLVED OQ-13. Until the multi-page path exists, only MR-1..MR-4
(single-page) are asserted; MR-5 is enumerated here (so it is not silent coverage
debt) and lands with the multi-page decode bead. **The suite must never, at any
phase, assert the independence/sum-of-parses relation** — that prohibition is
unconditional and predates the multi-page path existing.

**Construction recipe for the MR-5a witness.** Build a 2-page input where page 2's
correct parse is *ambiguous in isolation* but disambiguated by page 1 (e.g. a
table whose header is on page 1, continued rows on page 2; or a document where an
earlier page establishes a definition a later page references). Show that
`parse([p1, p2])` differs from `parse([blank, p2])` on page 2's region. One such
witness in the corpus discharges MR-5a.

---

## MR-6 — Unlimited-OCR-2 re-prefill ≡ incremental decode (U2-C2)

| | |
|---|---|
| **Transform `T`** | Fold the first `t` generated ids into the prompt: prefill `prompt ++ ids[..t]` afresh instead of decoding them one by one. The re-prefill rebuilds the page coordinates from the whole id sequence (`PageCoords::build`: a generated `<PAGE>` takes `bucket = cum − N`, clamped at 0, and the same query-page rule). |
| **Relation `R`** | At step `t`: the same greedy token; logits within the path's accumulation-order bound; each layer's selected remote blocks for the row equal (the biases' gap is logged), except where the window precondition below excuses it. Separately, the incremental decode with block retirement on and off gives bit-equal hidden states at every step. |
| **Strength** | `STRICT` tokens and selections; `STRUCTURAL` logits (a documented bound); `STRICT` (bitwise) retirement. |
| **Source backing** | Plan §5.8 (decode equals a re-prefill under the coordinate rule above); U2-B2 (the router window M is set per forward); U2-B4 (retirement frees only blocks no later row reads). |

**Why the logits are bounded, not bitwise.** A prefill runs multi-row GEMMs and a
decode step m = 1 GEMVs, which accumulate in different orders. On the shipping
int8-FFN cache, a last-ulp difference can also flip an int8 activation rounding in
each layer's FFN. U2-B5 set that path's bound at 10% of the largest logit, a
wiring check (it measured 2.4% on v2 and 5.0% on v1 over 12 steps); over these
cases U2-C2 measured up to 4.4% (S1) and 8.7% (S2). The all-f32 cache has no
such amplifier: it measured at most 9.0e-7 of the largest logit over both cases'
54 sampled steps, and is held to 1e-5. On that path the relation is a precision
check.

**The window precondition.** A prefill sets every row's router window
`M = min(16, ceil(L/32))` from its full length `L`; an incremental step sets it
from its own position. The compared row therefore has the same M on both sides,
but below the cap the re-prefill widens the window of every earlier row, prompt
rows included. A wider window adds non-candidate pads at the entmax input −1e4.
These leave a row's selection and biases unchanged when its best candidate input
is at least −1e4 + 2 and no candidate sits at or below −1e4
(`srswa::window_pad_sensitive`). The traced re-prefill records the rows where that
fails. A selection mismatch at layer `l` is excused (logged, not counted) only
when the window grows and some row is flagged at a layer below `l`: a flagged
row's output at layer `j` feeds the keys every layer above `j` routes on, and
nothing else of the window reaches the compared row. A prompt of more than 480
rows has M = 16 on every row, and its cases have no excuse.

**Generator.** A recorded generation (`tests/fixtures/uocr2_oracle/<fixture>.ids.txt`)
over its corpus pages: S1 `page_0014` (one page; its 279-row prompt has a 9-block
window that the re-prefills widen) and S2 (three pages). The relation compares the
model with itself, so the ids only need to be a valid sequence. Sampled steps:
every 64th, the last, and ±2 around every generated `<PAGE>`. Both cached paths
run: the shipping int8-FFN cache and the all-f32 one.

**Structured log.** One line per case (`case_id` = `<path>/<fixture>`) with the
sampled steps, the relative logit gap per step and its maximum, the largest bias
gap, the excused selections and the retired block count; one line per violation
(a token mismatch carries the incremental margin between the two ids); one
summary line. MR-6 reports its own summary and is not part of the MR-1..MR-5
rollup.

---

## 6. The single most important rule of this catalog

**Never assert a relation that is not entailed by the model's documented
semantics.** A wrong metamorphic relation is strictly worse than a missing one: a
missing relation leaves a gap (visible coverage debt), but a *wrong* relation is a
**green test that certifies incorrect behavior** and actively prevents the bug it
should catch from ever surfacing. The two ways to get this wrong here, both
explicitly guarded above:

1. **Asserting multi-page independence** (MR-5) — refuted by OQ-13. The suite
   asserts the *opposite* (cross-page dependence exists).
2. **Asserting white-pad invariance as a hard gate** (MR-3) — white is not the
   model's pad color; only **mean-gray** pad is representationally invariant.
   White-pad is a SHOULD observation, not a MUST.

Each relation above is line-backed to a `[SPEC-NNN]` clause or an OQ answer; a
relation with no source backing does not ship.

---

## 7. The metamorphic corpus & generator

The suite needs **inputs to transform**, not reference outputs (that is the whole
point of being oracle-free). The corpus is a small set of committable document
images spanning the regimes the relations exercise:

| Corpus slice | Why | Exercises |
|---|---|---|
| Dense prose page | the common case | MR-1, MR-3a, MR-4 |
| Page with grounding boxes (`<\|det\|>`) | geometry relations need boxes | MR-2 (both halves), MR-3a (box offset) |
| Non-square page (aspect ≠ 1) | the pad-to-square + rotation interaction | MR-2, MR-3a |
| Dimensions not a multiple of 640 | tile-geometry edge | MR-1, MR-3a |
| 2-page constructed dependence pair | the OQ-13 witness | MR-5a |
| Recorded Unlimited-OCR-2 generations (`tests/fixtures/uocr2_oracle`, S1 `page_0014`, S2) | valid id sequences to teacher-force | MR-6 |

The **generator** is a pure-Rust transform library (`image` crate: lossless
`rotate90/180/270`, `flip`, in-memory same-size resize, mean-gray border pad) that
takes a corpus image and emits the `(x, T(x))` pair for each relation. It uses
**no oracle and no model weights** for the pixel-domain transforms; the e2e
relations consume the same model-gated path as the L5 gate (skip-with-SUCCESS when
weights are absent, per plan §8.3 model-gated e2e). Every generated case carries a
stable `case_id` (e.g. `rot90_doc07`) for the structured log.

## 8. Test inventory (what `tests/metamorphic.rs` must contain)

- **One `#[test]` per relation** (MR-1..MR-6), each emitting the summary NDJSON
  line and per-violation lines (§0).
- **The generator** producing transformed input pairs (§7), unit-tested for the
  coordinate maps themselves (MR-2's `(x,y)→` maps and MR-3a's affine offset are
  pure functions — test them directly with hand-worked examples, independent of
  the model).
- **MR-4 thread-axis harness**: runs the engine at `FOCR_THREADS=1` and
  `FOCR_THREADS=N` and asserts byte-identity.
- **The negative guard**: an explicit comment / doc-test asserting the suite
  **does not** and **must not** contain a multi-page sum-of-parts assertion —
  pointing at this document and OQ-13. (A test that *fails to compile* if such an
  assertion is reintroduced is ideal, but at minimum a load-bearing comment.)
- **Coverage line**: the suite emits a final
  `{"suite":"metamorphic","relations_total":5,"relations_run":R,"relations_gated":G}`
  rollup so the gauntlet (plan §8.5) can account this leg. It covers MR-1..MR-5;
  MR-6 (model-gated on the raw Unlimited-OCR-2 checkpoint) emits its own summary
  line.

## 9. Relationship to the conformance pillar (plan §8.3 / §8.5)

| Suite | Question it answers | This doc's role |
|---|---|---|
| **Differential** (`bd-re8.9`) | "same as the bf16 reference?" | complementary — runs on the golden corpus, inherits oracle bugs |
| **Metamorphic** (this, `bd-re8.10`) | "self-consistent under transforms, no oracle?" | **catalogued here** |
| **Golden** ([`GOLDEN.md`](GOLDEN.md), `bd-re8.11`) | "did the frozen surface/numeric output change?" | downstream of MR-4's determinism guarantee |
| **Fault** (`VERIFY-r1-fault-suite`) | "graceful on malformed input?" | out of scope here (§Scope boundary) |

The metamorphic leg feeds the gauntlet's **conformance pillar**: each relation is
a `ConformanceTest` (requirement level `MUST` for MR-1/MR-2/MR-3a/MR-4, `MUST`
for the MR-5 *prohibition* and `SHOULD`/gated for MR-5a..c, `SHOULD` for MR-3b,
`MUST`/model-gated for MR-6 on Unlimited-OCR-2),
contributing to the ≥0.95 MUST-coverage accounting (plan §8.6). A relation
violation is a hard parity-cell failure — the release scorecard cannot ship with a
red metamorphic cell (plan §8.4).
