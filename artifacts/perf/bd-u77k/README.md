# Warm resident Unlimited-OCR-2 runs against inline runs

Evidence for bead `bd-u77k` and the `RESIDENT-warm-v2-0009-20261031` row of
`docs/PERF_LEDGER.md`. The question: once the resident daemon serves
Unlimited-OCR-2, what does a run that finds the model loaded save?

`warm_bench.py` is the bd-fuh2 harness's method (`artifacts/perf/bd-fuh2/`) on
v2: both arms on the same page in one invocation, in ABBA order, so a drift in
host load falls on both. `warm_bench.json` is its raw output. The first served
run spawns the daemon and loads the model; it is recorded as `cold` and left out
of the pairs. The script stops if the two arms' stdout differ, if a warm client
holds the model, or if an inline run does not. Pages are capped at
`--max-page-tokens 256`, so a run takes seconds; the saving is one model load
whatever the decode length. Three inline `--robot` runs then time the `load`
stage from its stage event to the `forward` one.

## Result

| run | pairs | inline median | warm median | median ratio | ratio range | client peak, inline → warm |
|---|---:|---:|---:|---:|---:|---:|
| `focr ocr --json --max-page-tokens 256`, one page | 9 | 13.05 s | 11.30 s | 1.16 | 0.91 to 2.00 | ≥ 7.76 GB → 32 MB |

Stdout (`--json`) matched on every pair, sha256 `8c113e8a…41a`. The cold run took
12.87 s. The `load` stage took 515, 564 and 549 ms.

A warm run saved a median 1.9 s, and only about 0.55 s of that is the `load`
stage. The rest falls inside the `forward` stage. This run does not say what it
is, so no claim is made about it.

The ratio range is wide because position in the pair mattered, and not equally
for both arms. Either arm ran slower second than first. Inline averaged 12.9 s
first and 17.1 s second, and warm 10.7 s first and 12.4 s second. So the five
inline-first pairs gave ratios of 0.91 to 1.25, and the four warm-first pairs
1.16 to 2.00. ABBA balances an order effect only when it is the same for both
arms, and this one was not. Comparing within a position gives about the same
saving: medians of 12.77 s against 10.66 s for runs in first position, about
2.1 s. The 2.00 is not a typical saving. Why an inline run after a warm one
lost about 4 s, while the idle daemon held its model, was not measured.

The memory figures are the client's. A warm client holds no model because the
daemon does, for the idle period after the run.

## Conditions

- Apple M5 Ultra, 36 physical cores, default thread budget, system allocator.
- Load average 60 to 76 throughout, from the U2-C5a evaluation running beside
  it, and the harness ran at background QoS (`taskpolicy -b`). Only the paired
  comparison is meaningful; the absolute times are not a throughput claim.
- `release-perf` (thin LTO, one codegen unit) built in a separate target
  directory, because the evaluation owns the default `release-perf` binary.
- `unlimited-ocr-2.v0.1.0.int8.focrq`, sha256
  `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`, beside its
  pinned `tokenizer.json`.
- Corpus page `page_0009.png`, sha256
  `62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da`.

The byte-identity contract itself, including `warnings[]` and exit 5 at the
position limit, is `tests/resident_e2e.rs`'s `a_resident_v2_run_matches_an_inline_run`.
