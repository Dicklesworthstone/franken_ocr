#!/bin/zsh
# Rerun of diag_0789_p3.sh, whose 17:35 run never read the page: it passed the model
# DIRECTORY to --model (focr: model not found, exit 3), and its log line said "exit 0"
# because `$(date)` in the echo resets $? before it expands.
# Post hoc, gates nothing: int8 v2 reads U2-C5a doc 0789 page 3 ALONE, with the model
# resolved as the C5a cells did (--model unlimited-ocr-2 + FOCR_MODEL_DIR), no stage
# budget, default width. Waits for guard_live.sh (guard_live/done) so it never shares
# the host with that timed run.
# Output: diag_f32/0789_p3_v2_single_r2.{json,stdout,err,time}; lines in c5a_concurrent.log.
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
O=$S/diag_f32
L=$S/c5a_concurrent.log
while [ ! -f $S/guard_live/done ]; do sleep 60; done
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): 0789 page 3 alone r2, int8 v2, start" >> $L
env FOCR_NO_RESIDENT=1 FOCR_STAGE_BUDGET_FORWARD_MS=0 \
  FOCR_MODEL_DIR=$S/v2_model_dir:/Users/jemanuel/.cache/franken_ocr_local_models \
  /usr/bin/time -l -o $O/0789_p3_v2_single_r2.time $B ocr $S/c5a_data/docs/marathon-mp3lan_0789-p223-227/p3.png \
  --model unlimited-ocr-2 --json -o $O/0789_p3_v2_single_r2.json \
  > $O/0789_p3_v2_single_r2.stdout 2> $O/0789_p3_v2_single_r2.err
rc=$?
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): 0789 page 3 alone r2 exit $rc" >> $L
echo $rc > $O/p3_r2_done
