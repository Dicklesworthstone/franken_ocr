#!/bin/zsh
# Post-hoc diagnostic, gates nothing: U2-C5a's document verdict against v1 per page
# rests on one document, marathon-mp3lan_0789-p223-227, where int8 v2 (document
# mode) ran page 3 to the 20,000-token page cap and opened page 4 with 20 malformed
# `<BLOCK>title<LEVEL1>` prefixes. Does the all-f32 path do the same? Same command
# as the eval's v2_doc cell, but the raw checkpoint, FOCR_DECODE_F32=1 and no stage
# budget. Waits for diag_f32.sh (diag_f32/done) so it never shares the host.
# Output: diag_f32/0789_v2_f32_doc.{json,stdout,err,time}; a line in c5a_concurrent.log.
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
O=$S/diag_f32
L=$S/c5a_concurrent.log
while [ ! -f $O/done ]; do sleep 60; done
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): 0789 document f32 start" >> $L
env FOCR_DECODE_F32=1 FOCR_NO_RESIDENT=1 FOCR_STAGE_BUDGET_FORWARD_MS=0 \
  /usr/bin/time -l -o $O/0789_v2_f32_doc.time $B ocr $S/c5a_data/docs/marathon-mp3lan_0789-p223-227 \
  --model $S/uocr2_checkpoint --multi-page --json -o $O/0789_v2_f32_doc.json \
  > $O/0789_v2_f32_doc.stdout 2> $O/0789_v2_f32_doc.err
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): 0789 document f32 exit $?" >> $L
echo done > $O/doc_done
