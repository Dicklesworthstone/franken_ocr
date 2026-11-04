# Warm resident runs against inline runs

Evidence for bead `bd-fuh2`, the `RESIDENT-warm-0009-20261031` row of
`docs/PERF_LEDGER.md`, and the `CLAIM-resident-ocr-batch` entry of
`docs/NEGATIVE_EVIDENCE.md`. The question: what does the resident daemon save a
run that finds the model loaded?

`warm_bench.py` runs both arms on the same input in one invocation, in ABBA
order, so a drift in host load falls on both. `warm_bench.json` is its raw
output. The first served run spawns the daemon and loads the model. It is
recorded as `cold` and left out of the pairs. The script stops if the two arms'
stdout differ, if a warm client holds the model, or if an inline run does not.

## Result

| run | pairs | inline median | warm median | median ratio | ratio range | client peak, inline → warm |
|---|---:|---:|---:|---:|---:|---:|
| `focr ocr`, one page | 9 | 14.47 s | 12.51 s | 1.16 | 1.08 to 1.32 | 8.38 GB → 31 MB |
| `focr ocr-batch`, three pages | 3 | 205.63 s | 207.16 s | 0.996 | 0.993 to 1.007 | 10.09 GB → 19 MB |

Stdout matched on every pair. The cold run took 16.09 s, 1.6 s more than the
inline median, so a run that spawns the daemon costs a little more than one that
does not.

A warm run saves about one model load, here about 2 s with the artifact in the
OS page cache. That is 14% of a 14 s page. A batch saves the same one load once,
and three pages of decode hide it: the two arms are within noise.

The memory figures are the client's. A warm client holds no model because the
daemon does, for the idle period after the run. Memory moves; it does not go away.

## Decision

`focr ocr` keeps the daemon. `focr ocr-batch` does not get it: no wall-clock gain,
and a batch that used to release its memory at exit would have held 8 GB to 10 GB
for ten idle minutes. The change was reverted.

## Conditions

- Apple M5 Ultra, 36 physical cores, default thread budget, system allocator.
- Load average 65 to 78 throughout, from the U2-C5a evaluation running beside it.
  Only the paired comparison is meaningful; the absolute times are not a
  throughput claim.
- `release-perf` with `CARGO_PROFILE_RELEASE_PERF_LTO=false` and
  `CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS=16`, in a separate target directory,
  because the evaluation owns the default `release-perf` binary.
- `unlimited-ocr.int8.focrq`, sha256
  `573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592`.
- Corpus pages `page_0009.png` (sha256
  `62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da`) for the single
  page, with `page_0014.png` and `page_0019.png` added for the batch.

The batch arm in this run was the code as it stood before the revert, so that
code path exists only in this measurement. A slower disk or an uncached artifact
would raise the saving. Neither was measured.
