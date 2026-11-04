#!/bin/zsh
# bd-gftg, post hoc, gates nothing: the opt-in v1 RunawayGuard (FOCR_RUNAWAY_GUARD=1)
# on real decodes. The offline replay (guard_calib.py) re-encoded post-processed text;
# this checks the abort point with the model's own token ids.
# Pages: the 7 v1 single reruns the replay says abort (at 8,704-14,592 tokens) plus
# newspaper b615 (a runaway the replay says the raw guard misses). No legit v1 single
# page reaches the 8,192-token floor, so the guard cannot change any other page.
# Same binary, model dir and env as the registered reruns, plus the guard.
# Waits for diag_0789_p3.sh (diag_f32/p3_done) so it never shares the host.
# Output: guard_live/<id>.{json,err,time}, guard_live/results.tsv, lines in c5a_concurrent.log.
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
O=$S/guard_live
L=$S/c5a_concurrent.log
mkdir -p $O
while [ ! -f $S/diag_f32/p3_done ]; do sleep 60; done
echo "$(date '+%Y-%m-%dT%H:%M:%S') guard_live (post_c5a follow-up): start, 8 pages" >> $L
while read id p; do
  env FOCR_NO_RESIDENT=1 FOCR_STAGE_BUDGET_FORWARD_MS=0 FOCR_RUNAWAY_GUARD=1 \
    FOCR_MODEL_DIR=$S/v2_model_dir:/Users/jemanuel/.cache/franken_ocr_local_models \
    /usr/bin/time -l -o $O/$id.time $B ocr-batch --model unlimited-ocr --json $S/c5a_data/omni/$p \
    > $O/$id.json 2> $O/$id.err
  rc=$?
  wall=$(awk '/real/ {print $1; exit}' $O/$id.time)
  msg=$(grep -o 'runaway token guard triggered at [0-9]* emitted tokens[^;]*' $O/$id.err | head -1)
  printf '%s\t%s\t%s\t%s\n' $id $rc $wall "$msg" >> $O/results.tsv
  echo "$(date '+%Y-%m-%dT%H:%M:%S') guard_live (post_c5a follow-up): $id exit $rc ${wall}s ${msg:-no trigger}" >> $L
done < $S/guard_live_items.txt
echo "$(date '+%Y-%m-%dT%H:%M:%S') guard_live (post_c5a follow-up): done" >> $L
echo done > $O/done
