# bd-av64.10 — SIMD/polynomial-exp softmax lever: measured dead (2026-07-07)

Evidence pointers for the NEGATIVE_EVIDENCE.md entry
(`evidence_id: artifacts/perf/bd-av64.10-simd-exp/`,
CLAIM-bd-av64.10-simd-exp):

* Measured before→after tables live INLINE in the ledger entry (paired A/B,
  2 runs per arm, adjacent same-regime; `FOCR_TIMING=1 [FOCR_SAM_FAST_EXP=1]
  focr ocr --model unlimited-ocr.int8.focrq page_0009.png`).
* The paired logs + diffs were produced in that session's scratchpad
  (`fastexp/`, ephemeral); the durable record is the ledger entry itself plus
  the commit that reverted the lever (ab6e083).

The lever was REVERTED on the measurements recorded in the ledger; this
directory satisfies the artifact-graph contract (`check_ledgers.py`:
NEGATIVE evidence_id must be an `artifacts/perf/` path).
