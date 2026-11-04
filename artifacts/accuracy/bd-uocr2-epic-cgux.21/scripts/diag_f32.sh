#!/bin/zsh
# Post-hoc diagnostic, gates nothing: do the four pages behind v2's as-run
# `note` / `en_ch_mixed` category regressions behave the same on the all-f32
# path? v2 from the raw checkpoint, FOCR_DECODE_F32=1, no stage budget, same
# binary. Pages: notes_..._50 (int8 counted ordinals to the 600 s budget),
# notes_..._69 (row-major reading of a two-column list), needrop_en_1118 and
# needrop_en_443 (markup / furniture-line differences).
# Waits for the U2-F1 bench (PID 95698) so it never shares the host with a timed run.
# Output: diag_f32/v2_f32.{json,err,time}; a progress line in c5a_concurrent.log.
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
O=$S/diag_f32
I=$S/c5a_data/omni
L=$S/c5a_concurrent.log
while kill -0 95698 2>/dev/null; do sleep 60; done
mkdir -p $O
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): start" >> $L
env FOCR_DECODE_F32=1 FOCR_NO_RESIDENT=1 FOCR_STAGE_BUDGET_FORWARD_MS=0 \
  /usr/bin/time -l -o $O/v2_f32.time $B ocr-batch --model $S/uocr2_checkpoint --json \
  $I/notes_f7f010b78016aeebd76e56d9283eb67f_50.jpg \
  $I/notes_f7f010b78016aeebd76e56d9283eb67f_69.jpg \
  $I/jiaocaineedrop_jiaocai_needrop_en_1118.jpg \
  $I/jiaocaineedrop_jiaocai_needrop_en_443.jpg \
  > $O/v2_f32.json 2> $O/v2_f32.err
echo "$(date '+%Y-%m-%dT%H:%M:%S') diag_f32 (post_c5a follow-up): v2 f32 exit $?" >> $L
echo done > $O/done
