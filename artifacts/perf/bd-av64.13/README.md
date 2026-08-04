# bd-av64.13 — TrOMR accuracy levers: measured negatives (2026-07-07)

Evidence pointers for the two NEGATIVE_EVIDENCE.md entries (the ledger's
`evidence_id: artifacts/perf/bd-av64.13/`):

* **1-crop-page routing** (CLAIM-bd-av64.13-onecrop-route): measured
  before→after tables live INLINE in the ledger entry itself; the run recipe
  is `FOCR_BIN=<release focr> bash scripts/realscan_music_gate.sh` against
  the committed realscan_music corpus v1. Full narrative: the bd-av64.13
  close note (`br show bd-av64.13`).
* **micro-rotation TTA vote** (CLAIM-bd-av64.13-tta-vote): per-candidate
  logs inline in the ledger entry; same gate script with
  `FOCR_TROMR_TTA=3`. Full narrative: the bd-av64.13 close note.

Both levers were REVERTED same-day on the measurements recorded in the
ledger; this directory exists to satisfy the artifact-graph contract
(`check_ledgers.py`: NEGATIVE evidence_id must be an `artifacts/perf/` path).
