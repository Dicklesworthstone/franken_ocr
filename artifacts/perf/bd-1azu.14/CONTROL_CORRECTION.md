bd-1azu.14 evidence CORRECTION (fresh-eyes review, 2026-07-02):
The original sweep's 'FOCR_BATCH_SPINE=0 control' never tested the kill-switch —
spine arming was presence-parsed, so =0 ARMED it; the recorded control fell back
to sequential only because the f32 artifact left the int8 conjunct false.

Fixed (value-parse) and re-proven at the CLI level on the real model:
  FOCR_BATCH_SPINE=0 ocr-batch: rc=0, 0 batched-vision timing lines (sequential)
  FOCR_BATCH_SPINE=1 ocr-batch: rc=0, 1 batched-vision line (spine)
  outputs BYTE-IDENTICAL across both states (2x page_0009)
Also: GOT artifact + FOCR_BATCH_SPINE=1 now takes the sequential arch-dispatched
path (new arch gate) instead of crashing into Baidu tensor names.
