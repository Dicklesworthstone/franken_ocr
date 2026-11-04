#!/bin/zsh
# bd-zex4: thread-width sweep on a quiet host. Same release-perf focr the
# U2-C5a evaluation uses; v1 and v2; corpus page_0014, page_0023, page_0590;
# FOCR_THREADS in {8,12,18,24,36} plus the incumbent default (unset, i.e.
# num_cpus::get_physical()); 3 reps with the width order rotated per rep.
# Normal QoS on purpose: background QoS would push the work onto the E-cores.
# Every run records real/user/sys, max RSS, peak footprint and the output's
# sha256 (the token output must not move with width).
# Output: zex4/runs.tsv, zex4/host.txt, zex4/<model>.<page>.<arm>.r<rep>.{md,time}
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
P=/Users/jemanuel/.cache/franken_ocr/corpus/pages
O=$S/zex4
if pgrep -f c5a_run.sh > /dev/null; then
  echo "U2-C5a is still running; this sweep needs a quiet host"
  exit 1
fi
mkdir -p $O
{
  echo "binary $B"
  echo "sha256 $(shasum -a 256 $B | cut -d' ' -f1)"
  sysctl hw.ncpu hw.physicalcpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
  date '+%Y-%m-%dT%H:%M:%S%z'
  uptime
} > $O/host.txt
T=$O/runs.tsv
print -r -- $'model\tpage\tarm\trep\treal_s\tuser_s\tsys_s\tmaxrss_b\tfootprint_b\tout_sha256\texit' > $T
echo "$(date '+%Y-%m-%dT%H:%M:%S') start zex4_sweep" >> $S/c5a_concurrent.log
arms=(default 8 12 18 24 36)
# Rep 4 is the low-width pass for the AF-5 USL fit (scripts/af5_usl_fit.py
# needs an N = 1 anchor): one run each at 1, 2 and 4 threads, after the
# best-of-3 pass so it cannot delay the bead's own arms.
for rep in 1 2 3 4; do
  # Rotate the arm order each rep so slow drift does not favour one arm.
  order=(${arms[$rep,-1]} ${arms[1,$((rep-1))]})
  [ $rep = 4 ] && order=(4 2 1)
  for model in unlimited-ocr unlimited-ocr-2; do
    for page in page_0014 page_0023 page_0590; do
      for arm in $order; do
        stem=$O/$model.$page.$arm.r$rep
        if [ "$arm" = default ]; then
          env -u FOCR_THREADS FOCR_NO_RESIDENT=1 \
            FOCR_MODEL_DIR=$S/v2_model_dir:/Users/jemanuel/.cache/franken_ocr_local_models \
            /usr/bin/time -l $B ocr $P/$page.png --model $model -o $stem.md 2> $stem.time
        else
          env FOCR_THREADS=$arm FOCR_NO_RESIDENT=1 \
            FOCR_MODEL_DIR=$S/v2_model_dir:/Users/jemanuel/.cache/franken_ocr_local_models \
            /usr/bin/time -l $B ocr $P/$page.png --model $model -o $stem.md 2> $stem.time
        fi
        rc=$?
        times=($(awk '/ real .* user .* sys/{print $1, $3, $5}' $stem.time))
        rss=$(awk '/maximum resident set size/{print $1}' $stem.time)
        fp=$(awk '/peak memory footprint/{print $1}' $stem.time)
        sha=$( [ -f $stem.md ] && shasum -a 256 $stem.md | cut -d' ' -f1 || echo none)
        print -r -- "$model	$page	$arm	$rep	$times[1]	$times[2]	$times[3]	$rss	$fp	$sha	$rc" >> $T
      done
    done
  done
done
echo "$(date '+%Y-%m-%dT%H:%M:%S') end zex4_sweep" >> $S/c5a_concurrent.log
echo done >> $O/host.txt
