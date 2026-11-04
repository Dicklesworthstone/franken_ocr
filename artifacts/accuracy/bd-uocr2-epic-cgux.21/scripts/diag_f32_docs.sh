#!/bin/zsh
# Post hoc, gates nothing: v2 all-f32 document reads of every U2-C5a document.
# Why: doc 0789 read at f32 scored ED_all 0.047 against int8's 0.855 (page 3 closed as a
# table instead of counting "814 815 ..." to the page cap). C5a's precision check covered
# 40 single pages only, so no document had an f32 read.
# Question, fixed before any result: per-document ED_all of v2 f32 vs the scored v2 int8
# (paired bootstrap, the driver's _bootstrap), page_cap pages per precision, and the C5a
# document comparison with f32 v2 in place of int8 (post hoc sensitivity, not a verdict).
# Same command shape as the driver's v2_doc cell and diag_f32_doc.sh: raw checkpoint,
# FOCR_DECODE_F32=1, no stage budget, default width. 0789 is already done (diag_f32/).
# Waits for diag_0789_p3_r2.sh (diag_f32/p3_r2_done) so it never shares the host.
# Output: diag_f32/docs/<id>.{json,stdout,err,time}, diag_f32/docs/runs.tsv, log lines.
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
O=$S/diag_f32/docs
L=$S/c5a_concurrent.log
while [ ! -f $S/diag_f32/p3_r2_done ]; do sleep 60; done
mkdir -p $O
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): f32 document sweep start" >> $L
for d in $S/c5a_data/docs/*(/); do
  id=${d:t}
  [[ $id == *0789* ]] && continue
  [ -f $O/$id.json ] && continue
  env FOCR_DECODE_F32=1 FOCR_NO_RESIDENT=1 FOCR_STAGE_BUDGET_FORWARD_MS=0 \
    /usr/bin/time -l -o $O/$id.time $B ocr $d --model $S/uocr2_checkpoint --multi-page --json \
    -o $O/$id.json > $O/$id.stdout 2> $O/$id.err
  rc=$?
  wall=$(awk '/ real /{print $1; exit}' $O/$id.time)
  printf '%s\t%s\t%s\n' $id $rc $wall >> $O/runs.tsv
  echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): f32 doc $id exit $rc ${wall}s" >> $L
done
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): f32 document sweep done" >> $L
echo done > $O/done
