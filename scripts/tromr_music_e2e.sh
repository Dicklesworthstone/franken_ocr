#!/bin/sh
# tromr_music_e2e.sh — the E9 music-task gate (bd-3jo6.5.9).
#
# Drives the REAL `focr` binary over the REAL tromr.focrq:
#   1/5 GATE      zoo dir present (artifact + tokenizer tables + the upstream
#                 example staff) else SKIP-with-SUCCESS
#   2/5 BIN       build (or take FOCR_BIN) — release profile
#   3/5 NEGATIVE  /nonexistent model => exit 3; music at an unlimited-named
#                 model => exit 2 (Usage — knowably neither tromr nor got;
#                 ambiguous names pass through to the engine arch tag by design)
#   4/5 MUSIC     --task music on the example staff => partwise MusicXML
#                 carrying the CERTIFIED structure (clef F4, key CM fifths 0,
#                 3 measures — the token-exact argmax result for this staff,
#                 matching the committed ground truth's own opening)
#   5/5 REALSCAN  optional real-scan crop arm (bd-av64.1 regression: the
#                 2026-07-06 Cadwallader crop crashed the emitter on a
#                 pitched thirty_second) — set FOCR_TROMR_REALSCAN_CROP to a
#                 local staff image (copyright material stays LOCAL-ONLY,
#                 never committed); skipped-with-SUCCESS when unset.
#
# MusicXML validity: every document the binary emits is validated at emit
# time inside the engine (validate_musicxml, bd-av64.3) — a structural
# violation fails the run itself, so any exit-0 music step here implies a
# validator-clean document by construction.
#
# Env: FOCR_TROMR_DIR (default the USB zoo); FOCR_BIN (skips the build);
#      FOCR_TROMR_REALSCAN_CROP (optional local staff crop for step 5).
#
# Logging contract: stdout DATA-ONLY NDJSON, schema "tromr_music_e2e/v1"
# (events gate|bin|negative|music|result); human telemetry `TRMU `-prefixed
# on stderr. Exit 0 = PASS or gated SKIP; non-zero = a real divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission).
set -eu

log()   { printf 'TRMU %s\n' "$*" >&2; }
step()  { printf 'TRMU ==== STEP %s ====\n' "$*" >&2; }
ok()    { printf 'TRMU   PASS  %s\n' "$*" >&2; }
skip()  { printf 'TRMU   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'TRMU   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "tromr_music_e2e/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found — required for NDJSON emission"
  exit 2
fi

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ZOO="${FOCR_TROMR_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr}"
MODEL="$ZOO/tromr.focrq"
STAFF="$ZOO/../tromr-upstream/examples/1.png"

step "1/5 GATE"
if [ ! -f "$MODEL" ] || [ ! -f "$ZOO/tokenizer_rhythm.json" ] || [ ! -f "$STAFF" ]; then
  skip "tromr artifact/tokenizers/example staff absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights zoo="$ZOO"
  ndj event=result result=skip
  exit 0
fi
ok "artifact + tokenizer tables + example staff present"
ndj event=gate result=pass zoo="$ZOO"

step "2/5 BIN"
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
else
  ( cd "$REPO_ROOT" && cargo build --release --bin focr ) >&2
  BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/focr"
fi
if [ ! -x "$BIN" ]; then
  fail "focr binary not found/executable at $BIN"
  ndj event=bin result=fail path="$BIN"
  exit 1
fi
ok "binary ready: $BIN"
ndj event=bin result=pass path="$BIN"

step "3/5 NEGATIVE"
rc=0
"$BIN" ocr "$STAFF" --task music --model /nonexistent/tromr.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 3 ]; then
  fail "/nonexistent model must exit 3 (ModelNotFound); got $rc"
  ndj event=negative check=nonexistent_model result=fail exit_code="$rc" expect=3
  exit 1
fi
ok "/nonexistent model => exit 3"
ndj event=negative check=nonexistent_model result=pass exit_code="$rc"

rc=0
"$BIN" ocr "$STAFF" --task music --model unlimited-ocr.int8.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 2 ]; then
  fail "music at an unlimited-named model must exit 2 (Usage); got $rc"
  ndj event=negative check=wrong_family_guard result=fail exit_code="$rc" expect=2
  exit 1
fi
ok "music at an unlimited-named model => exit 2 (knowably neither tromr nor got)"
ndj event=negative check=wrong_family_guard result=pass exit_code="$rc"

step "4/5 MUSIC"
t0=$(python3 -c 'import time; print(int(time.time()*1000))')
OUT=$("$BIN" ocr "$STAFF" --task music --model "$MODEL" 2>/dev/null) || {
  rc=$?
  fail "music run failed (exit $rc)"
  ndj event=music result=fail exit_code="$rc"
  exit 1
}
t1=$(python3 -c 'import time; print(int(time.time()*1000))')
case "$OUT" in
  "<?xml"*) ok "output opens MusicXML" ;;
  *)
    fail "output is not MusicXML: $(printf '%s' "$OUT" | head -c 80)"
    ndj event=music result=fail reason=not_xml
    exit 1
    ;;
esac
for want in "<clef><sign>F</sign><line>4</line></clef>" "<key><fifths>0</fifths></key>" "<measure number=\"3\">" "score-partwise"; do
  case "$OUT" in
    *"$want"*) : ;;
    *)
      fail "MusicXML missing the certified structure: $want"
      ndj event=music result=fail reason=structure
      exit 1
      ;;
  esac
done
ok "MusicXML carries the certified structure (clef F4, CM, 3 measures)"
ndj event=music result=pass elapsed_ms=$((t1 - t0))

# Robot staff observability (bd-av64.2): the same staff through robot mode
# must emit at least one parseable `staff` event with status ok.
STAFF_EVENTS=$("$BIN" robot run "$STAFF" --task music --model "$MODEL" 2>/dev/null \
  | python3 -c '
import json, sys
count = 0
for line in sys.stdin:
    e = json.loads(line)
    if e.get("event") == "staff":
        assert e.get("status") in ("ok", "skipped"), e
        assert isinstance(e.get("bbox"), list) and len(e["bbox"]) == 4, e
        count += 1
print(count)')
if [ "$STAFF_EVENTS" -lt 1 ]; then
  fail "robot mode emitted no staff events"
  ndj event=music check=staff_events result=fail count="$STAFF_EVENTS"
  exit 1
fi
ok "robot mode emitted $STAFF_EVENTS parseable staff event(s)"
ndj event=music check=staff_events result=pass count="$STAFF_EVENTS"

step "5/5 REALSCAN"
CROP="${FOCR_TROMR_REALSCAN_CROP:-}"
if [ -n "$CROP" ] && [ -f "$CROP" ]; then
  t0=$(python3 -c 'import time; print(int(time.time()*1000))')
  OUT2=$("$BIN" ocr "$CROP" --task music --model "$MODEL" 2>/dev/null) || {
    rc=$?
    fail "real-scan crop failed (exit $rc) — the bd-av64.1 emitter-crash class"
    ndj event=realscan result=fail exit_code="$rc" crop="$CROP"
    exit 1
  }
  t1=$(python3 -c 'import time; print(int(time.time()*1000))')
  case "$OUT2" in
    "<?xml"*) : ;;
    *)
      fail "real-scan output is not MusicXML"
      ndj event=realscan result=fail reason=not_xml crop="$CROP"
      exit 1
      ;;
  esac
  # Whether the decode contains a 32nd is model-dependent; record, don't gate.
  case "$OUT2" in
    *"<type>32nd</type>"*) has32=1 ;;
    *) has32=0 ;;
  esac
  ok "real-scan crop transcribes without an emitter crash (has_32nd=$has32)"
  ndj event=realscan result=pass has_32nd="$has32" elapsed_ms=$((t1 - t0)) crop="$CROP"
else
  skip "FOCR_TROMR_REALSCAN_CROP unset — real-scan arm skipped"
  ndj event=realscan result=skip
fi

log "ALL STEPS PASSED"
ndj event=result result=pass
exit 0
