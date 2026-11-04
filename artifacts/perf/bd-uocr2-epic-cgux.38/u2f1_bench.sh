#!/bin/zsh
# U2-F1 (bd-uocr2-epic-cgux.38): v1 and v2 in the same invocation, same thread
# count, same inputs, the U2-C3/C4 conservative int8 artifacts (the ones U2-C5a
# ran). Cells: S1 single pages (page_0009, page_0014, page_0590; best of 3), and
# the S2 (3-page), p10 and p20 documents as v2 document mode, v1 multi-page and
# v1 per page (one run each). FOCR_TIMING=1 for the stage rows, /usr/bin/time -l
# for wall and peak RSS. Width: FOCR_THREADS=$1, or else the fastest width in the
# bd-zex4 sweep (best-of-3 page-set total summed over both models).
# Waits for post_c5a.done so it has the host to itself.
# Output: u2f1/runs.tsv and u2f1/<cell>.{md,time,timing}
S=/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/0e68861b-d16c-4ae1-8786-16748171bc4f/scratchpad
B=/Users/jemanuel/.local/state/dsr/siblings/franken_ocr-lock-pinned/franken_ocr/target/release-perf/focr
P=/Users/jemanuel/.cache/franken_ocr/corpus/pages
O=$S/u2f1
while [ ! -f $S/post_c5a.done ]; do sleep 60; done
W=${1:-$(python3 -I -c "
import csv, collections
rows = [r for r in csv.DictReader(open('$S/zex4/runs.tsv'), delimiter='\t') if r['exit'] == '0' and r['arm'] not in ('default', '1', '2', '4')]
best = collections.defaultdict(lambda: float('inf'))
for r in rows:
    k = (r['arm'], r['model'], r['page'])
    best[k] = min(best[k], float(r['real_s']))
tot = collections.defaultdict(float)
for (arm, _, _), v in best.items():
    tot[arm] += v
print(min(tot, key=tot.get))
")}
mkdir -p $O/docs
{
  echo "binary $B"
  echo "sha256 $(shasum -a 256 $B | cut -d' ' -f1)"
  echo "FOCR_THREADS $W"
  date '+%Y-%m-%dT%H:%M:%S%z'
  uptime
} > $O/host.txt
echo "$(date '+%Y-%m-%dT%H:%M:%S') start u2f1_bench FOCR_THREADS=$W" >> $S/c5a_concurrent.log
docs=(s2 "page_0009 page_0014 page_0019"
      p10 "page_0009 page_0014 page_0019 page_0023 page_0086 page_0091 page_0105 page_0107 page_0108 page_0126"
      p20 "page_0009 page_0014 page_0019 page_0023 page_0086 page_0091 page_0105 page_0107 page_0108 page_0126 page_0136 page_0179 page_0250 page_0358 page_0400 page_0437 page_0547 page_0557 page_0590 page_0645")
for name pages in $docs; do
  mkdir -p $O/docs/$name
  for p in ${=pages}; do [ -e $O/docs/$name/$p.png ] || ln -s $P/$p.png $O/docs/$name/$p.png; done
done
T=$O/runs.tsv
print -r -- $'cell\tinput\tmodel\tmode\trep\treal_s\tuser_s\tsys_s\tmaxrss_b\tfootprint_b\tout_chars\texit' > $T
one() {  # cell input model mode rep -- focr args...
  local cell=$1 input=$2 model=$3 mode=$4 rep=$5; shift 5
  local stem=$O/$cell.$input.$model.$mode.r$rep
  env FOCR_THREADS=$W FOCR_NO_RESIDENT=1 FOCR_TIMING=1 \
    FOCR_MODEL_DIR=$S/v2_model_dir:/Users/jemanuel/.cache/franken_ocr_local_models \
    /usr/bin/time -l $B "$@" --model $model -o $stem.md 2> $stem.time
  local rc=$?
  local times=($(awk '/ real .* user .* sys/{print $1, $3, $5}' $stem.time))
  local rss=$(awk '/maximum resident set size/{print $1}' $stem.time)
  local fp=$(awk '/peak memory footprint/{print $1}' $stem.time)
  local chars=$( [ -f $stem.md ] && wc -m < $stem.md | tr -d ' ' || echo 0)
  print -r -- "$cell	$input	$model	$mode	$rep	$times[1]	$times[2]	$times[3]	$rss	$fp	$chars	$rc" >> $T
}
for rep in 1 2 3; do
  for page in page_0009 page_0014 page_0590; do
    # Alternate which model goes first so neither always runs on a warmer cache.
    if [ $((rep % 2)) = 1 ]; then order=(unlimited-ocr unlimited-ocr-2); else order=(unlimited-ocr-2 unlimited-ocr); fi
    for model in $order; do one single $page $model base $rep ocr $P/$page.png; done
  done
done
for name pages in $docs; do
  one doc $name unlimited-ocr-2 multi 1 ocr $O/docs/$name --multi-page
  one doc $name unlimited-ocr multi 1 ocr $O/docs/$name --multi-page
  one doc $name unlimited-ocr per-page 1 ocr $O/docs/$name
done
echo "$(date '+%Y-%m-%dT%H:%M:%S') end u2f1_bench" >> $S/c5a_concurrent.log
echo done >> $O/host.txt
