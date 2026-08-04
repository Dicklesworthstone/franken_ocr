#!/bin/sh
# realscan_music_gate.sh — the real-scan music corpus gate (bd-av64.6).
#
# Runs the REAL `focr` binary over the committed real-scan fixtures
# (tests/fixtures/realscan_music/, 1843 Spohr violin school — public domain)
# and gates on the THREE-TIER truth design documented in the corpus README:
#   tier 1  truth/attributes.json — human-verified clefs/key/time/staff
#           counts/bar floors/spot notes, asserted EXACTLY;
#   tier 2  goldens/*.musicxml — frozen model output; byte-diff = the output
#           changed (fail loud; re-freeze only deliberately);
#   tier 3  robustness — full pages + vocab-external content must complete
#           without aborting (staff-count floors via robot `staff` events).
#
# Model-gated: SKIP-with-SUCCESS when the tromr artifact is absent.
# Env: FOCR_TROMR_DIR (default USB zoo), FOCR_BIN (skip build).
# Logging: stdout DATA-ONLY NDJSON "realscan_music/v1"; human on stderr.
# POSIX sh; passes `sh -n`. python3 required.
set -eu

log()  { printf 'RSMU %s\n' "$*" >&2; }
ok()   { printf 'RSMU   PASS  %s\n' "$*" >&2; }
skip() { printf 'RSMU   SKIP  %s\n' "$*" >&2; }
fail() { printf 'RSMU   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "realscan_music/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FIX="$REPO_ROOT/tests/fixtures/realscan_music"
ZOO="${FOCR_TROMR_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr}"
MODEL="$ZOO/tromr.focrq"

if [ ! -f "$MODEL" ] || [ ! -f "$ZOO/tokenizer_rhythm.json" ]; then
  skip "tromr artifact absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights
  ndj event=result result=skip
  exit 0
fi
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
else
  ( cd "$REPO_ROOT" && cargo build --release --bin focr ) >&2
  BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/focr"
fi
ndj event=gate result=pass bin="$BIN"

FAILURES=0

# ── Tier 1+2: single-staff and system crops, attributes from --json + XML ──
check_staff() {
  # $1 fixture rel path
  rel="$1"
  img="$FIX/$rel"
  out=$("$BIN" ocr --model "$MODEL" --task music "$img" --json 2>/dev/null) || {
    fail "$rel: run failed"
    ndj event=case fixture="$rel" result=fail reason=run_failed
    FAILURES=$((FAILURES + 1))
    return 0
  }
  outf=$(mktemp "${TMPDIR:-/tmp}/rsmu.XXXXXX")
  printf '%s' "$out" > "$outf"
  verdict=$(python3 - "$rel" "$FIX/truth/attributes.json" "$outf" <<'PY'
import json, re, sys
rel, truth_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(out_path))
truth = json.load(open(truth_path))[f"staves/{rel.split('/', 1)[1]}"]
xml = d["markdown"]
problems = []
# clefs, in part order
clefs = re.findall(r"<clef><sign>(\w)</sign><line>(\d)</line></clef>", xml)
got_clefs = [f"{s}{l}" for s, l in clefs]
want_clefs = truth.get("clefs") or [truth["clef"]]
if got_clefs[: len(want_clefs)] != want_clefs:
    problems.append(f"clefs {got_clefs} != {want_clefs}")
# key
m = re.search(r"<fifths>(-?\d+)</fifths>", xml)
if "fifths" in truth and (not m or int(m.group(1)) != truth["fifths"]):
    problems.append(f"fifths {m and m.group(1)} != {truth['fifths']}")
# time
if "time" in truth:
    beats, beat_type = truth["time"].split("/")
    if f"<beats>{beats}</beats><beat-type>{beat_type}</beat-type>" not in xml:
        problems.append(f"time != {truth['time']}")
# staff count via staves array
if "staff_count" in truth:
    got = len(d.get("staves", []))
    if got != truth["staff_count"]:
        problems.append(f"staff_count {got} != {truth['staff_count']}")
# bar floor (first part)
if "bars_min" in truth:
    first_part = xml.split("</part>")[0]
    bars = len(re.findall(r"<measure number=", first_part))
    if bars < truth["bars_min"]:
        problems.append(f"bars {bars} < min {truth['bars_min']}")
# spot notes: bar 1 of part 1
if "bar1_notes" in truth:
    m1 = re.search(r"<measure number=\"1\">(.*?)</measure>", xml, re.S)
    notes = re.findall(
        r"<step>(\w)</step>(?:<alter>-?\d</alter>)?<octave>(\d)</octave></pitch>"
        r"<duration>\d+</duration><type>(\w+)</type>",
        m1.group(1) if m1 else "",
    )
    got1 = [f"{s}{o}:{t}" for s, o, t in notes]
    if got1 != truth["bar1_notes"]:
        problems.append(f"bar1 {got1} != {truth['bar1_notes']}")
print(json.dumps({"problems": problems}))
PY
)
  rm -f "$outf"
  problems=$(printf '%s' "$verdict" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['problems']))")
  xfail=$(python3 -c "
import json, sys
t = json.load(open('$FIX/truth/attributes.json'))['staves/$(basename "$rel")']
x = t.get('xfail')
print('1' if x else '0')")
  if [ "$problems" -eq 0 ] && [ "$xfail" = "0" ]; then
    ok "$rel: attributes match truth"
    ndj event=case fixture="$rel" result=pass tier=attributes
  elif [ "$problems" -gt 0 ] && [ "$xfail" = "1" ]; then
    log "  XFAIL $rel: $verdict (documented divergence; see xfail_note)"
    ndj event=case fixture="$rel" result=xfail tier=attributes detail="$verdict"
  elif [ "$problems" -eq 0 ] && [ "$xfail" = "1" ]; then
    fail "$rel: marked xfail but PASSES — promote it (remove the xfail from truth)"
    ndj event=case fixture="$rel" result=fail tier=attributes reason=xpass
    FAILURES=$((FAILURES + 1))
  else
    fail "$rel: $verdict"
    ndj event=case fixture="$rel" result=fail tier=attributes detail="$verdict"
    FAILURES=$((FAILURES + 1))
  fi
  # tier 2 golden, when one exists
  base=$(basename "$rel" .png)
  golden="$FIX/goldens/$base.musicxml"
  if [ -f "$golden" ]; then
    live=$("$BIN" ocr --model "$MODEL" --task music "$img" 2>/dev/null)
    if [ "$live" = "$(cat "$golden")" ]; then
      ok "$rel: golden byte-stable"
      ndj event=case fixture="$rel" result=pass tier=golden
    else
      fail "$rel: OUTPUT CHANGED vs frozen golden (investigate; re-freeze only deliberately)"
      ndj event=case fixture="$rel" result=fail tier=golden
      FAILURES=$((FAILURES + 1))
    fi
  fi
}

# ── Tier 3: full pages — staff-count floors via robot staff events ─────────
check_page() {
  rel="$1"
  img="$FIX/$rel"
  min=$(python3 -c "
import json
print(json.load(open('$FIX/truth/attributes.json'))['$1']['min_recognized'])")
  count=$("$BIN" robot run --model "$MODEL" --task music "$img" 2>/dev/null \
    | python3 -c '
import json, sys
n = 0
for line in sys.stdin:
    e = json.loads(line)
    if e.get("event") == "staff" and e.get("status") == "ok":
        n += 1
print(n)') || {
    fail "$rel: page run failed"
    ndj event=case fixture="$rel" result=fail reason=run_failed
    FAILURES=$((FAILURES + 1))
    return 0
  }
  xfail=$(python3 -c "
import json
t = json.load(open('$FIX/truth/attributes.json'))['$1']
print('1' if t.get('xfail') else '0')")
  if [ "$count" -ge "$min" ] && [ "$xfail" = "0" ]; then
    ok "$rel: $count staves recognized (floor $min)"
    ndj event=case fixture="$rel" result=pass tier=page recognized="$count" floor="$min"
  elif [ "$count" -lt "$min" ] && [ "$xfail" = "1" ]; then
    log "  XFAIL $rel: $count/$min staves (documented; flips with bd-av64.14)"
    ndj event=case fixture="$rel" result=xfail tier=page recognized="$count" floor="$min"
  elif [ "$count" -ge "$min" ] && [ "$xfail" = "1" ]; then
    fail "$rel: marked xfail but MEETS the floor — promote it"
    ndj event=case fixture="$rel" result=fail tier=page reason=xpass
    FAILURES=$((FAILURES + 1))
  else
    fail "$rel: only $count staves recognized (floor $min)"
    ndj event=case fixture="$rel" result=fail tier=page recognized="$count" floor="$min"
    FAILURES=$((FAILURES + 1))
  fi
}

check_staff "staves/spohr_no17_top.png"
check_staff "staves/spohr_no17_sys.png"
check_staff "staves/spohr_no21_sys.png"
check_staff "staves/spohr_p116_sys29.png"
check_page "pages/spohr_p055.png"
check_page "pages/spohr_p100.png"

if [ "$FAILURES" -eq 0 ]; then
  log "ALL CASES PASSED"
  ndj event=result result=pass
  exit 0
fi
fail "$FAILURES case(s) failed"
ndj event=result result=fail failures="$FAILURES"
exit 1
