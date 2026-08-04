#!/usr/bin/env python3
"""Parse `FOCR_TIMING=1` stderr into per-stage gauntlet records (bd-re8.17).

`focr` emits `[focr-timing] <stage> <secs>s` lines to stderr when `FOCR_TIMING`
is set (`src/native_engine/mod.rs::timing_log`, plus the GOT decode line in
`src/native_engine/decoder_qwen2.rs`). This module turns N warm runs of that
stderr (captured by `scripts/gauntlet_focr.sh`) into the shared
`focr-gauntlet-stage/v1` JSON records that `scripts/gauntlet_row.py` merges
with the reference side (`scripts/gauntlet_reference.py`) and the roofline
(`scripts/gauntlet_roofline.py`) into a PERF_LEDGER row.

Honesty contract: every number here is read from a real run's stderr/meta
files. There is no default, estimate, or fill-in anywhere — a run whose stderr
carries no `[focr-timing]` lines is a hard error, never a zero.

Stage vocabulary (docs/PERF_LEDGER.md): `preprocess`, `vision_encode`,
`prefill`, `decode_per_token`, `end_to_end` (underscored here; the ledger row
uses hyphens). Extra informational stages (`weight_cache_build`,
`decode_total`, `vision_sam`, ...) are carried alongside but are not ledger
stages.

Usage:
  gauntlet_timing.py aggregate --run-dir DIR --out FILE [--threads N]
                               [--precision P] [--allocator A] [--synthetic]
  gauntlet_timing.py aggregate-ab --run-dir DIR --out FILE --ab-env FOCR_VAR
                                  --a-label L --a-value V --b-label L --b-value V
  gauntlet_timing.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
import time

SCHEMA_STAGE = "focr-gauntlet-stage/v1"
SCHEMA_DOC = "focr-gauntlet-stages/v1"
SCHEMA_RAW_TIMING = "focr-gauntlet-raw-timing/v1"
SCHEMA_AB = "focr-gauntlet-ab/v1"
MAX_MEASURED_RUNS = 256
MAX_TIMING_FILE_BYTES = 64 * 1024 * 1024
MAX_TIMING_TOTAL_BYTES = 512 * 1024 * 1024
MAX_RAW_DIRECTORY_ENTRIES = 2048
MIN_AB_RUNS_PER_ARM = 5

DECODE_PHASE_STAGES = (
    "decode_lm_head",
    "decode_attn",
    "decode_experts",
    "decode_route",
)

PRECISION_MIXED_FFN_INT8 = "focr-mixed-ffn-int8"
PRECISION_FULL_INT8 = "focr-full-int8"
UNLIMITED_PRECISIONS = frozenset((PRECISION_MIXED_FFN_INT8, PRECISION_FULL_INT8))
DECODE_MODE_BY_PRECISION = {
    PRECISION_MIXED_FFN_INT8: "mixed-ffn-int8",
    PRECISION_FULL_INT8: "full-int8",
}
UNLIMITED_QUANT_RECIPE = "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1"
PRECISION_GATE_VARS = (
    "FOCR_DECODE_INT8",
    "FOCR_INT8_ATTN",
    "FOCR_INT8_LMHEAD",
    "FOCR_ATTN_GEMM",
    "FOCR_INT8_KV",
    "FOCR_SPEC_DECODE",
    "FOCR_DECODE_STATELESS",
)

LEDGER_STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")

PREFIX = "[focr-timing]"

# One regex per timing_log format string in src (kept in emission order of the
# pipeline; the GOT decode line must be tried before the Unlimited decode line
# because both start with "decode").
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "precision",
        re.compile(r"^precision (?P<precision>focr-(?:mixed-ffn-int8|full-int8))$"),
    ),
    ("preprocess", re.compile(r"^preprocess (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_sam", re.compile(r"^vision\.sam (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_clip", re.compile(r"^vision\.clip (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_bridge", re.compile(r"^vision\.bridge (?P<s>\d+(?:\.\d+)?)s$")),
    # Batched-vision lines (bd-t6a / bd-1azu.10, spine multi-page path,
    # src/native_engine/mod.rs::vision_tower_batched_pages). The sam/clip/bridge
    # batch lines fold into the same stage names as their per-view twins (both
    # are the run's total time in that tower); hydrate is its own info stage.
    (
        "vision_hydrate",
        re.compile(r"^vision\.hydrate\(batch\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "unlimited_vision_hydrate",
        re.compile(
            r"^unlimited_vision\.hydrate\((?:cached|batch-local)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "vision_sam_batch",
        re.compile(
            r"^vision\.sam\(batch of (?P<views>\d+), side (?P<side>\d+)\)"
            r" (?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "vision_clip_batch",
        re.compile(r"^vision\.clip\(batch of (?P<views>\d+)\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "vision_bridge_batch",
        re.compile(r"^vision\.bridge\(batch of (?P<views>\d+)\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    ("vision_encode", re.compile(r"^vision_tower (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "weight_cache_build",
        re.compile(r"^weight_cache_build(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "prefill",
        re.compile(r"^prefill(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s \((?P<tok>\d+) tokens\)$"),
    ),
    (
        "got_decode",
        re.compile(
            r"^decode (?P<tok>\d+) tok in (?P<s>\d+(?:\.\d+)?)s \(\d+(?:\.\d+)? tok/s\)"
            r" \| seed\(prefill(?: (?P<seedtok>\d+) tok)?\) (?P<seed>\d+(?:\.\d+)?)s"
            r" \| layers (?P<layers>\d+(?:\.\d+)?)s"
            r" \(attn (?P<attn>\d+(?:\.\d+)?)s, gemv\+misc (?P<gemv>\d+(?:\.\d+)?)s\)"
            r" \| lm_head (?P<head>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "decode_total",
        re.compile(
            r"^decode(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s"
            r" \((?P<tok>\d+) tokens, \d+(?:\.\d+)?s/tok\)$"
        ),
    ),
    (
        "decode_phases",
        re.compile(
            r"^decode(?:_i8)? phases \(ms\): lm_head (?P<head>\d+)\s+attn (?P<attn>\d+)"
            r"\s+experts (?P<experts>\d+)\s+route (?P<route>\d+)$"
        ),
    ),
    (
        "multi_page_vision",
        re.compile(
            r"^multi_page\.vision \((?P<pages>\d+) pages\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "multi_page_decode",
        re.compile(
            r"^multi_page\.decode (?P<tok>\d+) tokens (?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "got_hydrate",
        re.compile(r"^got\.hydrate\(cached\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    ("got_vision_splice", re.compile(r"^got\.vision\+splice (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "got_vision_splice_batch",
        re.compile(
            r"^got\.vision\+splice\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    ("got_generate", re.compile(r"^got\.generate (?P<tok>\d+) tokens (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "got_generate_batch",
        re.compile(
            r"^got\.generate\(batch of (?P<streams>\d+)\) (?P<tok>\d+) tokens "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    ("got_forward", re.compile(r"^got forward (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "got_forward_batch",
        re.compile(
            r"^got-ocr2 forward\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    # The SmolVLM2 / OneChart lanes (A11, bd-3jo6.1.11) — same shape as GOT's:
    # vision+splice, generate, forward-total. Their decode/seed(prefill)
    # breakdown arrives via the SHARED engine line (`got_decode` above —
    # `generate_greedy_kvcache` emits it for every dense lane).
    (
        "smolvlm2_vision_splice",
        re.compile(
            r"^smolvlm2\.vision\+splice (?P<s>\d+(?:\.\d+)?)s"
            r" \((?P<frames>\d+) frames, (?P<prompt>\d+) prompt ids\)$"
        ),
    ),
    (
        "smolvlm2_hydrate",
        re.compile(r"^smolvlm2\.hydrate\(cached\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "smolvlm2_vision_splice_batch",
        re.compile(
            r"^smolvlm2\.vision\+splice\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "smolvlm2_generate",
        re.compile(r"^smolvlm2\.generate (?P<tok>\d+) tokens (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "smolvlm2_generate_batch",
        re.compile(
            r"^smolvlm2\.generate\(batch of (?P<streams>\d+)\) (?P<tok>\d+) tokens "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    ("smolvlm2_forward", re.compile(r"^smolvlm2 forward (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "smolvlm2_forward_batch",
        re.compile(
            r"^smolvlm2 forward\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "onechart_hydrate",
        re.compile(r"^onechart\.hydrate\(cached\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "onechart_vision_splice",
        re.compile(r"^onechart\.vision\+splice (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "onechart_vision_splice_batch",
        re.compile(
            r"^onechart\.vision\+splice\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "onechart_generate",
        re.compile(r"^onechart\.generate (?P<tok>\d+) tokens (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "onechart_generate_batch",
        re.compile(
            r"^onechart\.generate\(batch of (?P<streams>\d+)\) (?P<tok>\d+) tokens "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "onechart_forward_batch",
        re.compile(
            r"^onechart forward\(batch of (?P<views>\d+)\) "
            r"(?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    # Polyphonic-TrOMR (bd-2sez): encode folds straight into the canonical
    # vision_encode stage; generate is the decode TOTAL (per-token derives from
    # it); the forward line is the lane's info stage. Multi-staff pages emit
    # one encode+generate pair per staff — summed with occurrences, as for
    # multi-view vision.
    (
        "vision_encode",
        re.compile(r"^tromr\.encode (?P<s>\d+(?:\.\d+)?)s \(w \d+, \d+ ctx tokens\)$"),
    ),
    (
        "decode_total",
        re.compile(r"^tromr\.generate (?P<tok>\d+) steps (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "tromr_split",
        re.compile(
            r"^tromr\.split seg (?P<segment>\d+) "
            r"\[(?P<start>\d+)\.\.(?P<end>\d+)\] "
            r"(?P<s>\d+(?:\.\d+)?)s \((?P<chars>\d+) chars\)$"
        ),
    ),
    (
        "tromr_forward",
        re.compile(
            r"^tromr forward (?P<s>\d+(?:\.\d+)?)s "
            r"\(\d+/\d+ staves recognized, semantic \d+ chars total\)$"
        ),
    ),
    (
        "onechart_forward",
        re.compile(r"^onechart forward (?P<s>\d+(?:\.\d+)?)s \(reliable_distance .*\)$"),
    ),
    # bd-av64.10 vision sub-stage instrumentation (2026-07-06/07): per-tower
    # hydrate/blocks/forward lines plus per-block detail. Folded as their own
    # stages (the row consumes the top-level stages; these are drill-down
    # evidence, and the parser's strictness contract still rejects truly
    # unknown shapes).
    (
        "sam_hydrate",
        re.compile(r"^sam\.hydrate (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "sam_blocks",
        re.compile(r"^sam\.blocks (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "sam_forward",
        re.compile(r"^sam\.forward (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "sam_block_detail",
        re.compile(r"^sam\.block (?:attn\(win\)|attn\(GLOBAL\)|mlp) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "clip_hydrate",
        re.compile(r"^clip\.hydrate(?:\(cached\))? (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "clip_blocks",
        re.compile(r"^clip\.blocks (?P<s>\d+(?:\.\d+)?)s$"),
    ),
]

_KNOWN_EVENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^pdf page (?P<page>\d+): spread split at x=(?P<gutter>\d+) "
        r"\((?P<w>\d+)x(?P<h>\d+)\)$"
    ),
    re.compile(
        r"^pdf page (?P<page>\d+): no spread detected "
        r"\((?P<w>\d+)x(?P<h>\d+)\), unsplit$"
    ),
    re.compile(r"^tromr\.staff_detect (?P<staves>\d+) staves$"),
    re.compile(
        r"^tromr\.staff (?P<staff>\d+) split-recognized "
        r"\((?P<w>\d+)x(?P<h>\d+)\)$"
    ),
    re.compile(
        r"^tromr\.staff (?P<staff>\d+) ok "
        r"\((?P<w>\d+)x(?P<h>\d+), semantic (?P<chars>\d+) chars\)$"
    ),
    re.compile(
        r"^tromr\.staff (?P<staff>\d+) SKIP "
        r"\((?P<w>\d+)x(?P<h>\d+)\): (?P<reason>.+)$"
    ),
    re.compile(
        r"^tromr\.sanity "
        r"(?P<kind>overfull_bar|underfull_bar|impossible_duration|key_mismatch) "
        r"part (?P<part>\d+) measure (?P<measure>\d+): (?P<detail>.+)$"
    ),
)


class TimingParseError(ValueError):
    """A run's stderr could not be parsed into timing evidence."""


def parse_run(stderr_text: str) -> dict:
    """Parse one run's stderr into `{stage: {"s": secs, "tokens": n?}, ...}`.

    Repeated occurrences of a stage (multi-view vision, multi-page PDFs) are
    summed and counted in `occurrences`; token counts sum likewise. Raises
    [`TimingParseError`] when no `[focr-timing]` line is present (the honest
    refusal: no evidence, no record) or when a `[focr-timing]` line does not
    match any known format (the binary's format moved — fix the parser, do not
    guess).
    """
    stages: dict[str, dict] = {}
    saw_prefix = False
    for raw in stderr_text.splitlines():
        line = raw.strip()
        if not line.startswith(PREFIX):
            continue
        saw_prefix = True
        body = line[len(PREFIX) :].strip()
        matched = False
        for name, pattern in _PATTERNS:
            m = pattern.match(body)
            if m is None:
                continue
            matched = True
            _fold(stages, name, m)
            break
        if not matched and any(pattern.match(body) for pattern in _KNOWN_EVENT_PATTERNS):
            # These are structured lifecycle/warning events, not durations.
            # Recognize their exact vocabulary without fabricating a zero-time
            # stage that numeric aggregation could mistake for measurement.
            matched = True
        if not matched:
            raise TimingParseError(f"unrecognized [focr-timing] line: {body!r}")
    if not saw_prefix:
        raise TimingParseError(
            "no [focr-timing] lines in stderr — was FOCR_TIMING=1 exported and is "
            "this a real focr run?"
        )
    if not stages:
        raise TimingParseError("[focr-timing] prefix seen but no stage line parsed")
    phase_lines = stages.pop("_decode_phase_lines", None)
    if phase_lines is not None:
        decode = stages.get("decode_total")
        expected = decode.get("occurrences") if decode is not None else None
        observed = {
            stage: stages.get(stage, {}).get("occurrences")
            for stage in DECODE_PHASE_STAGES
        }
        if (
            not isinstance(expected, int)
            or phase_lines["occurrences"] != expected
            or any(value != expected for value in observed.values())
        ):
            raise TimingParseError(
                "decode phase lines are incomplete or duplicated relative to decode totals: "
                f"decode_total={expected!r}, phase_lines={phase_lines['occurrences']!r}, "
                f"phase_occurrences={observed!r}"
            )
    _validate_run_coherence(stages)
    return stages


def _fold(stages: dict[str, dict], name: str, m: re.Match[str]) -> None:
    groups = m.groupdict()
    if name == "precision":
        identity = groups["precision"]
        entry = stages.get(name)
        if entry is not None and entry.get("identity") != identity:
            raise TimingParseError(
                f"one run emitted conflicting precision identities: "
                f"{entry.get('identity')!r} vs {identity!r}"
            )
        if entry is None:
            stages[name] = {"identity": identity, "occurrences": 1}
        else:
            entry["occurrences"] += 1
        return
    if name == "decode_phases":
        marker = stages.setdefault("_decode_phase_lines", {"occurrences": 0})
        marker["occurrences"] += 1
        for key, stage in (
            ("head", "decode_lm_head"),
            ("attn", "decode_attn"),
            ("experts", "decode_experts"),
            ("route", "decode_route"),
        ):
            _add(stages, stage, float(groups[key]) / 1000.0, None)
        return
    if name == "got_decode":
        # The shared-engine decode breakdown (all dense lanes) carries BOTH the
        # decode total and the seeding prefill; fan it out to the shared stage
        # names. The seed token count (newer binaries) feeds the prefill floor.
        seedtok = int(groups["seedtok"]) if groups.get("seedtok") else None
        _add(stages, "decode_total", float(groups["s"]), int(groups["tok"]))
        _add(stages, "prefill", float(groups["seed"]), seedtok)
        _add(stages, "decode_layers", float(groups["layers"]), None)
        _add(stages, "decode_attn", float(groups["attn"]), None)
        _add(stages, "decode_lm_head", float(groups["head"]), None)
        return
    if name == "multi_page_vision":
        existing = stages.get("vision_encode")
        if existing is not None and existing.get("pages") is None:
            raise TimingParseError(
                "one run mixed single-page vision_tower totals with multi_page.vision"
            )
        _add(stages, "vision_encode", float(groups["s"]), None)
        entry = stages["vision_encode"]
        entry["pages"] = entry.get("pages", 0) + int(groups["pages"])
        return
    if name == "vision_encode" and stages.get(name, {}).get("pages") is not None:
        raise TimingParseError(
            "one run mixed multi_page.vision with single-page vision_tower totals"
        )
    if name.endswith("_generate_batch"):
        base = name[: -len("_batch")]
        _add(stages, base, float(groups["s"]), int(groups["tok"]))
        entry = stages[base]
        entry["batched"] = True
        entry["streams"] = entry.get("streams", 0) + int(groups["streams"])
        return
    if name.endswith("_batch"):
        # Batched vision (bd-t6a): same tower work as the per-view lines, one
        # line per side-group. Fold into the per-view stage name and keep the
        # batching evidence (`batched`, summed `views`) alongside.
        base = name[: -len("_batch")]
        _add(stages, base, float(groups["s"]), None)
        entry = stages[base]
        entry["batched"] = True
        entry["views"] = entry.get("views", 0) + int(groups["views"])
        return
    if name == "smolvlm2_vision_splice":
        # The frame count is the SigLIP "view" count (A11 roofline input).
        _add(stages, name, float(groups["s"]), None)
        entry = stages[name]
        entry["views"] = entry.get("views", 0) + int(groups["frames"])
        return
    tokens = int(groups["tok"]) if groups.get("tok") is not None else None
    _add(stages, name, float(groups["s"]), tokens)
    if groups.get("i8") is not None:
        stages[name]["int8"] = True


def _add(stages: dict[str, dict], name: str, secs: float, tokens: int | None) -> None:
    entry = stages.setdefault(name, {"s": 0.0, "occurrences": 0})
    entry["s"] += secs
    entry["occurrences"] += 1
    if tokens is not None:
        entry["tokens"] = entry.get("tokens", 0) + tokens


def _validate_run_coherence(stages: dict[str, dict]) -> None:
    multi_vision = stages.get("vision_encode")
    multi_pages = multi_vision.get("pages") if multi_vision is not None else None
    multi_decode = stages.get("multi_page_decode")
    if multi_pages is not None or multi_decode is not None:
        if multi_pages is None or multi_decode is None:
            raise TimingParseError(
                "multi-page timing requires one paired vision and outer decode total"
            )
        if multi_vision["occurrences"] != 1 or multi_decode["occurrences"] != 1:
            raise TimingParseError(
                "one timing capture must contain exactly one multi-page transaction: "
                f"vision_occurrences={multi_vision['occurrences']!r}, "
                f"decode_occurrences={multi_decode['occurrences']!r}"
            )
        pages = multi_vision["pages"]
        if not isinstance(pages, int) or pages <= 0:
            raise TimingParseError(f"multi-page vision reported invalid page count {pages!r}")
        detail_counts = {
            stage: stages.get(stage, {}).get("occurrences")
            for stage in ("vision_sam", "vision_clip", "vision_bridge")
        }
        if any(count != pages for count in detail_counts.values()):
            raise TimingParseError(
                "multi-page vision detail counts do not match its page count: "
                f"pages={pages!r}, detail_occurrences={detail_counts!r}"
            )
        decode = stages.get("decode_total")
        expected = {
            "occurrences": multi_decode.get("occurrences"),
            "tokens": multi_decode.get("tokens"),
        }
        observed = (
            {
                "occurrences": decode.get("occurrences"),
                "tokens": decode.get("tokens"),
            }
            if decode is not None
            else None
        )
        if observed != expected:
            raise TimingParseError(
                "multi-page outer decode disagrees with the decoder total: "
                f"outer={expected!r}, decode_total={observed!r}"
            )

    for lane in ("got", "smolvlm2", "onechart"):
        vision = stages.get(f"{lane}_vision_splice")
        generate = stages.get(f"{lane}_generate")
        forward = stages.get(f"{lane}_forward")
        entries = (vision, generate, forward)
        batch_markers = [bool(entry and entry.get("batched")) for entry in entries]
        if not any(batch_markers):
            continue
        if not all(batch_markers):
            raise TimingParseError(
                f"{lane} dense-batch timing requires vision, generate, and forward spans"
            )
        occurrences = [entry.get("occurrences") for entry in entries]
        if occurrences != [1, 1, 1]:
            raise TimingParseError(
                f"{lane} dense-batch timing is duplicated or mixed with sequential spans: "
                f"occurrences={occurrences!r}"
            )
        counts = [vision.get("views"), generate.get("streams"), forward.get("views")]
        if any(not isinstance(count, int) or count <= 0 for count in counts):
            raise TimingParseError(
                f"{lane} dense-batch timing has invalid cardinalities: {counts!r}"
            )
        if len(set(counts)) != 1:
            raise TimingParseError(
                f"{lane} dense-batch cardinalities disagree: "
                f"vision_views={counts[0]!r}, generate_streams={counts[1]!r}, "
                f"outer_inputs={counts[2]!r}"
            )


def infer_precision(runs: list[dict]) -> str | None:
    """Return the runtime-declared Unlimited-OCR precision identity.

    The marker is emitted by the branch that actually owns the decoder cache;
    filenames, operator flags, and `_i8` suffixes are only consistency proofs.
    Dense zoo lanes have no Unlimited marker and continue to require an explicit
    caller precision. Missing or contradictory Unlimited markers fail closed.
    """
    identities: list[str] = []
    saw_unlimited = False
    for run in runs:
        is_zoo = any(
            k in run
            for k in ("got_forward", "smolvlm2_forward", "onechart_forward", "tromr_forward")
        )
        relevant = [run.get(name) for name in ("weight_cache_build", "prefill", "decode_total")]
        saw_runtime_stage = any(entry is not None for entry in relevant)
        marker = run.get("precision")
        if is_zoo and marker is not None:
            raise TimingParseError("a dense zoo run emitted an Unlimited-OCR precision marker")
        if is_zoo:
            continue
        if not saw_runtime_stage and marker is None:
            continue
        saw_unlimited = True
        if marker is None:
            raise TimingParseError(
                "Unlimited-OCR timing lines lack the runtime precision marker"
            )
        identity = marker.get("identity")
        if identity not in UNLIMITED_PRECISIONS:
            raise TimingParseError(f"unsupported Unlimited-OCR precision {identity!r}")
        if any(entry is None for entry in relevant):
            raise TimingParseError(
                "Unlimited-OCR precision proof requires cache-build, prefill, and decode timing lines"
            )
        marked = [bool(entry.get("int8")) for entry in relevant]
        if identity == PRECISION_FULL_INT8 and not all(marked):
            raise TimingParseError(
                "runtime declared focr-full-int8 but an all-int8 timing suffix is missing"
            )
        if identity == PRECISION_MIXED_FFN_INT8 and any(marked):
            raise TimingParseError(
                "runtime declared focr-mixed-ffn-int8 but an all-int8 timing suffix is present"
            )
        identities.append(identity)
    if not saw_unlimited:
        return None
    if len(identities) != len(runs) or len(set(identities)) != 1:
        raise TimingParseError(
            f"measured runs disagree on runtime precision identity: {identities!r}"
        )
    return identities[0]


def _truthy(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "on", "yes"}


def _batch_spine_enabled(value: object) -> bool:
    """Mirror `decoder::batch_spine_enabled` including nonstandard armed values."""
    return (
        isinstance(value, str)
        and value.strip().lower() not in {"", "0", "off", "false", "no"}
    )


def _validate_unlimited_contract(precision: str, meta: dict) -> None:
    """Bind a canonical runtime marker to artifact and gate evidence."""
    if precision not in UNLIMITED_PRECISIONS:
        return
    model = meta.get("model")
    if (
        not isinstance(model, str)
        or not model.endswith(".focrq")
        or meta.get("model_kind") != "file"
        or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("model_sha256", "")))
        or not isinstance(meta.get("model_size"), int)
        or isinstance(meta.get("model_size"), bool)
        or meta["model_size"] <= 0
    ):
        raise TimingParseError(
            "canonical Unlimited-OCR evidence requires a hashed regular .focrq subject artifact"
        )
    identity = meta.get("model_identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
        or any(not isinstance(value, int) or isinstance(value, bool) for value in identity.values())
        or identity["size"] != meta["model_size"]
    ):
        raise TimingParseError(
            "canonical Unlimited-OCR evidence requires a per-run stable model inode receipt"
        )
    if meta.get("quant_recipe") != UNLIMITED_QUANT_RECIPE:
        raise TimingParseError(
            "canonical Unlimited-OCR evidence requires exact artifact recipe "
            f"{UNLIMITED_QUANT_RECIPE!r}, got {meta.get('quant_recipe')!r}"
        )
    gates = meta.get("precision_gate_states")
    if not isinstance(gates, dict) or set(gates) != set(PRECISION_GATE_VARS):
        raise TimingParseError("precision gate evidence is missing or incomplete")
    risky_presence = ("FOCR_ATTN_GEMM", "FOCR_INT8_KV", "FOCR_SPEC_DECODE", "FOCR_DECODE_STATELESS")
    armed_risky = [name for name in risky_presence if gates.get(name) != "<unset>"]
    if armed_risky:
        raise TimingParseError(
            "canonical precision evidence forbids presence-gated alternate paths: "
            + ", ".join(armed_risky)
        )
    master = _truthy(gates.get("FOCR_DECODE_INT8"))
    attn = _truthy(gates.get("FOCR_INT8_ATTN"))
    lmhead = _truthy(gates.get("FOCR_INT8_LMHEAD"))
    expected = (False, False, False) if precision == PRECISION_MIXED_FFN_INT8 else (True, True, True)
    if (master, attn, lmhead) != expected:
        raise TimingParseError(
            f"{precision} contradicts recipe gates: "
            f"FOCR_DECODE_INT8={gates.get('FOCR_DECODE_INT8')!r}, "
            f"FOCR_INT8_ATTN={gates.get('FOCR_INT8_ATTN')!r}, "
            f"FOCR_INT8_LMHEAD={gates.get('FOCR_INT8_LMHEAD')!r}"
        )


def _validate_profiled_decode_capture(capture: dict, precision: str) -> None:
    """Require the focused-stage evidence declared by the Unlimited campaign."""
    if precision not in UNLIMITED_PRECISIONS:
        return
    pins = capture["meta0"].get("env_pins")
    if not isinstance(pins, dict) or pins.get("FOCR_PROFILE_DECODE") != "1":
        raise TimingParseError(
            "canonical Unlimited-OCR evidence requires FOCR_PROFILE_DECODE=1 "
            "in the recorded environment pins"
        )
    for index, run in enumerate(capture["runs"], start=1):
        missing = [stage for stage in DECODE_PHASE_STAGES if stage not in run]
        if missing:
            raise TimingParseError(
                f"run {index} lacks required decode phase evidence: {', '.join(missing)}"
            )


def _validate_sequential_batch_result(
    meta: dict,
    parsed: dict,
    stderr_text: str,
    stdout_bytes: bytes,
    *,
    source: str,
) -> None:
    """Bind one successful result and decode occurrence to every batch page."""
    workload = meta.get("workload")
    command = meta.get("command")
    argv_is_batch = (
        isinstance(command, list)
        and len(command) >= 2
        and command[1] == "ocr-batch"
    )
    workload_is_batch = (
        isinstance(workload, dict) and workload.get("command") == "ocr-batch"
    )
    if not argv_is_batch and not workload_is_batch:
        return
    if not argv_is_batch or not workload_is_batch:
        raise TimingParseError(
            f"{source}: batch argv and workload metadata disagree"
        )
    switch_states = meta.get("performance_switch_states")
    focr_env = meta.get("focr_env")
    if (
        not isinstance(switch_states, dict)
        or "FOCR_BATCH_SPINE" not in switch_states
        or not isinstance(focr_env, dict)
    ):
        raise TimingParseError(
            f"{source}: batch capture lacks a complete FOCR_BATCH_SPINE receipt"
        )
    switch_value = switch_states["FOCR_BATCH_SPINE"]
    env_has_spine = "FOCR_BATCH_SPINE" in focr_env
    if not isinstance(switch_value, str):
        raise TimingParseError(
            f"{source}: FOCR_BATCH_SPINE receipt value is not a string"
        )
    if switch_value == "<unset>":
        if env_has_spine:
            raise TimingParseError(
                f"{source}: FOCR_BATCH_SPINE absence sentinel contradicts focr_env"
            )
        switch_armed = False
    else:
        if not env_has_spine or focr_env["FOCR_BATCH_SPINE"] != switch_value:
            raise TimingParseError(
                f"{source}: FOCR_BATCH_SPINE receipts disagree"
            )
        switch_armed = _batch_spine_enabled(switch_value)
    if switch_armed:
        raise TimingParseError(
            f"{source}: FOCR_BATCH_SPINE has no canonical per-page timing contract"
        )
    pages = meta.get("pages")
    page_count = workload.get("page_count")
    if (
        not isinstance(pages, list)
        or not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count <= 0
        or page_count != len(pages)
    ):
        raise TimingParseError(
            f"{source}: sequential batch metadata has an invalid page count"
        )
    if (
        not isinstance(command, list)
        or len(command) < 2 + page_count
        or command[1] != "ocr-batch"
        or any(not isinstance(value, str) for value in command)
    ):
        raise TimingParseError(f"{source}: sequential batch command is invalid")
    command_pages = command[2 : 2 + page_count]
    command_options = command[2 + page_count :]
    metadata_paths = [
        page.get("path") if isinstance(page, dict) else None for page in pages
    ]
    if (
        any(not isinstance(path, str) for path in metadata_paths)
        or [os.path.abspath(path) for path in command_pages] != metadata_paths
    ):
        raise TimingParseError(
            f"{source}: sequential batch command pages do not bind metadata pages"
        )
    if "--multi-page" in command_options:
        raise TimingParseError(
            f"{source}: --multi-page has no sequential batch evidence contract"
        )
    occurrences = parsed.get("decode_total", {}).get("occurrences")
    if occurrences != page_count:
        raise TimingParseError(
            f"{source}: sequential batch decoded {occurrences!r} of "
            f"{page_count} declared pages"
        )
    failed = [
        line
        for line in stderr_text.splitlines()
        if line.startswith("[focr] ") and " FAILED (" in line
    ]
    if failed:
        raise TimingParseError(
            f"{source}: sequential batch reported {len(failed)} failed page(s)"
        )
    if "--json" in command:
        payload = _bounded_json_bytes(stdout_bytes, f"{source} stdout")
        results = payload.get("results")
        if (
            payload.get("command") != "ocr-batch"
            or payload.get("count") != page_count
            or not isinstance(results, list)
            or len(results) != page_count
            or any(
                not isinstance(result, dict)
                or result.get("ok") is not True
                or result.get("image") != command_pages[index]
                or not isinstance(result.get("markdown"), str)
                for index, result in enumerate(results)
            )
        ):
            raise TimingParseError(
                f"{source}: sequential JSON batch output lacks one success per page"
            )
        return
    success_pattern = re.compile(
        r"^\[focr\] (?P<path>.*) \(\d+(?:\.\d+)?s\)$"
    )
    success_paths = [
        match.group("path")
        for line in stderr_text.splitlines()
        if (match := success_pattern.fullmatch(line)) is not None
    ]
    if success_paths != command_pages:
        raise TimingParseError(
            f"{source}: sequential batch lacks one ordered success record per page"
        )
    try:
        stdout_text = stdout_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TimingParseError(
            f"{source}: sequential batch stdout is not UTF-8: {error}"
        ) from error
    expected_headers = [f"===== {path} =====" for path in command_pages]
    expected_header_set = set(expected_headers)
    observed_headers = [
        line for line in stdout_text.splitlines() if line in expected_header_set
    ]
    if observed_headers != expected_headers:
        raise TimingParseError(
            f"{source}: sequential batch stdout lacks one ordered page header per result"
        )


def _stats(samples_ms: list[float]) -> dict:
    best = min(samples_ms)
    mean = statistics.fmean(samples_ms)
    ordered = sorted(samples_ms)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    p99 = ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]
    cv_pct = None
    if len(samples_ms) > 1 and mean > 0:
        cv_pct = statistics.stdev(samples_ms) / mean * 100.0
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "best_ms": round(best, 6),
        "p50_ms": round(statistics.median(samples_ms), 6),
        "p95_ms": round(p95, 6),
        "p99_ms": round(p99, 6),
        "mean_ms": round(mean, 6),
        "cv_pct": None if cv_pct is None else round(cv_pct, 3),
        "n": len(samples_ms),
    }


def parse_resource_usage(stderr_text: str) -> dict[str, int]:
    """Parse `/usr/bin/time` RSS output without confusing it with focr timings."""
    mac_rss = re.search(
        r"^\s*(?P<value>\d+)\s+maximum resident set size\s*$",
        stderr_text,
        re.MULTILINE,
    )
    linux_rss = re.search(
        r"^\s*Maximum resident set size \(kbytes\):\s*(?P<value>\d+)\s*$",
        stderr_text,
        re.MULTILINE,
    )
    peak_footprint = re.search(
        r"^\s*(?P<value>\d+)\s+peak memory footprint\s*$",
        stderr_text,
        re.MULTILINE,
    )
    result: dict[str, int] = {}
    if mac_rss:
        result["maximum_resident_set_size_bytes"] = int(mac_rss.group("value"))
    elif linux_rss:
        result["maximum_resident_set_size_bytes"] = int(linux_rss.group("value")) * 1024
    if peak_footprint:
        result["peak_memory_footprint_bytes"] = int(peak_footprint.group("value"))
    return result


def _byte_stats(samples: list[int]) -> dict:
    if not samples:
        raise TimingParseError("no RSS samples to aggregate")
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    cv_pct = None
    if len(samples) > 1 and mean > 0:
        cv_pct = statistics.stdev(samples) / mean * 100.0
    return {
        "samples_bytes": samples,
        "best_bytes": min(samples),
        "p50_bytes": round(statistics.median(samples), 3),
        "p95_bytes": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "p99_bytes": ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)],
        "mean_bytes": round(mean, 3),
        "cv_pct": None if cv_pct is None else round(cv_pct, 3),
        "n": len(samples),
    }


def raw_observation(
    run: dict,
    wall_ms: float,
    *,
    run_id: str,
    resources: dict[str, int] | None = None,
) -> dict:
    """Preserve one measured run before any best/median/CV aggregation."""
    stages: dict[str, dict] = {}
    for name, entry in run.items():
        if name == "precision" or "s" not in entry:
            continue
        sample = {"ms": round(float(entry["s"]) * 1000.0, 6)}
        if entry.get("tokens") is not None:
            sample["tokens"] = int(entry["tokens"])
        stages[name] = sample
    decode = run.get("decode_total")
    if decode is not None and decode.get("tokens"):
        stages["decode_per_token"] = {
            "ms": round(float(decode["s"]) * 1000.0 / int(decode["tokens"]), 6),
            "tokens": int(decode["tokens"]),
        }
    stages["end_to_end"] = {"ms": round(float(wall_ms), 6)}
    observation = {"run_id": run_id, "stages": stages}
    if resources:
        observation["resources"] = resources
    return observation


def _read_bounded(path: str, max_bytes: int = MAX_TIMING_FILE_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TimingParseError(f"timing input is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise TimingParseError(
                f"timing input exceeds {max_bytes} bytes: {path}"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise TimingParseError(
                    f"timing input exceeds {max_bytes} bytes while reading: {path}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise TimingParseError(f"timing input changed while being read: {path}")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise TimingParseError(f"timing input changed length while reading: {path}")
        return content
    finally:
        os.close(descriptor)


def _bounded_json_bytes(content: bytes, path: str) -> dict:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingParseError(f"malformed timing metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TimingParseError(f"timing metadata is not an object: {path}")
    return payload


def _bounded_meta_names(names, *, prefix: str = "") -> list[str]:
    measured: list[str] = []
    pattern = re.compile(re.escape(prefix) + r"run_\d{3}\.meta\.json")
    for observed, name in enumerate(names, start=1):
        if observed > MAX_RAW_DIRECTORY_ENTRIES:
            raise TimingParseError(
                f"raw directory exceeds {MAX_RAW_DIRECTORY_ENTRIES} entries"
            )
        if pattern.fullmatch(name):
            measured.append(name)
            if len(measured) > MAX_MEASURED_RUNS:
                raise TimingParseError(
                    f"measured run count exceeds {MAX_MEASURED_RUNS}"
                )
    return sorted(measured)


def _measured_meta_paths(run_dir: str, *, prefix: str = "") -> list[str]:
    try:
        with os.scandir(run_dir) as entries:
            names = _bounded_meta_names(
                (entry.name for entry in entries), prefix=prefix
            )
    except OSError as error:
        raise TimingParseError(f"cannot enumerate raw timing directory: {error}") from error
    return [os.path.join(run_dir, name) for name in names]


def _checked_evidence_total(current: int, added: int) -> int:
    total = current + added
    if added < 0 or total > MAX_TIMING_TOTAL_BYTES:
        raise TimingParseError(
            f"raw timing evidence exceeds {MAX_TIMING_TOTAL_BYTES} bytes"
        )
    return total


def _resolve_threads(requested: int | None, meta_threads: object) -> int:
    if (
        not isinstance(meta_threads, int)
        or isinstance(meta_threads, bool)
        or meta_threads <= 0
    ):
        raise TimingParseError("raw metadata lacks a positive thread count")
    if requested is not None and requested != meta_threads:
        raise TimingParseError(
            f"--threads {requested} contradicts raw metadata threads {meta_threads}"
        )
    return meta_threads


def _raw_member(run_dir: str, value: object, suffix: str, *, tag: str | None = None) -> str:
    expected = re.escape(tag) if tag is not None else r"run_\d{3}"
    if (
        not isinstance(value, str)
        or os.path.basename(value) != value
        or not re.fullmatch(expected + re.escape(suffix), value)
    ):
        raise TimingParseError(f"noncanonical raw timing member: {value!r}")
    return os.path.join(run_dir, value)


def aggregate_runs(
    runs: list[dict],
    wall_ms: list[float],
    *,
    threads: int,
    precision: str,
    allocator: str,
    backend: str = "focr",
    warmup_discarded: int = 0,
    synthetic: bool = False,
) -> list[dict]:
    """Fold per-run parses into best-of-N stage records.

    `runs` and `wall_ms` are index-aligned (warmups already discarded by the
    caller). Token counts must agree across runs for `prefill`/`decode_total`
    — greedy decode is deterministic, so a variance is flagged
    (`tokens_consistent: false`) rather than averaged away; `gauntlet_row.py`
    refuses such records.
    """
    if len(runs) != len(wall_ms):
        raise TimingParseError(f"runs ({len(runs)}) and wall_ms ({len(wall_ms)}) misaligned")
    if not runs:
        raise TimingParseError("no measured runs to aggregate")

    names: list[str] = []
    for run in runs:
        for name in run:
            if name not in names:
                names.append(name)

    records: list[dict] = []
    for name in names:
        if name == "precision":
            continue
        present = [run[name] for run in runs if name in run]
        if len(present) != len(runs):
            raise TimingParseError(
                f"stage {name!r} present in only {len(present)}/{len(runs)} runs — "
                "runs are not comparable"
            )
        samples_ms = [entry["s"] * 1000.0 for entry in present]
        tokens = [entry.get("tokens") for entry in present]
        occurrences = [entry.get("occurrences", 1) for entry in present]
        if len(set(occurrences)) != 1:
            raise TimingParseError(
                f"stage {name!r} occurrence counts drift across runs: {occurrences!r}"
            )
        record = {
            "schema": SCHEMA_STAGE,
            "source": "focr",
            "stage": name,
            "ledger_stage": name in LEDGER_STAGES,
            "unit": "ms",
            **_stats(samples_ms),
            "warmup_discarded": warmup_discarded,
            "threads": threads,
            "precision": precision,
            "backend": backend,
            "allocator": allocator,
            "occurrences": occurrences[0],
            "synthetic": synthetic,
        }
        for dimension in ("views", "pages", "streams"):
            values = [entry.get(dimension) for entry in present]
            if any(value is not None for value in values):
                if any(value is None for value in values) or len(set(values)) != 1:
                    raise TimingParseError(
                        f"stage {name!r} {dimension} counts drift across runs: {values!r}"
                    )
                record[dimension] = values[0]
                record[f"{dimension}_consistent"] = True
        if any(t is not None for t in tokens):
            record["tokens"] = tokens[0]
            record["tokens_consistent"] = len(set(tokens)) == 1
        records.append(record)

    # decode_per_token derives per run from the decode TOTAL and its token
    # count (never from the printed rounded s/tok).
    decode = [run.get("decode_total") for run in runs]
    if all(d is not None and d.get("tokens") for d in decode):
        per_tok_ms = [d["s"] * 1000.0 / d["tokens"] for d in decode]
        tokens = [d["tokens"] for d in decode]
        records.append(
            {
                "schema": SCHEMA_STAGE,
                "source": "focr",
                "stage": "decode_per_token",
                "ledger_stage": True,
                "unit": "ms",
                **_stats(per_tok_ms),
                "warmup_discarded": warmup_discarded,
                "threads": threads,
                "precision": precision,
                "backend": backend,
                "allocator": allocator,
                "occurrences": 1,
                "tokens": tokens[0],
                "tokens_consistent": len(set(tokens)) == 1,
                "synthetic": synthetic,
            }
        )

    records.append(
        {
            "schema": SCHEMA_STAGE,
            "source": "focr",
            "stage": "end_to_end",
            "ledger_stage": True,
            "unit": "ms",
            **_stats(list(wall_ms)),
            "warmup_discarded": warmup_discarded,
            "threads": threads,
            "precision": precision,
            "backend": backend,
            "allocator": allocator,
            "occurrences": 1,
            "note": "process wall clock: includes binary startup + model load + weight-cache build",
            "synthetic": synthetic,
        }
    )
    return records


def sha256_file(path: str) -> str:
    return hashlib.sha256(_read_bounded(path)).hexdigest()


_CAPTURE_INVARIANT_KEYS = (
    "command",
    "env_pins",
    "focr_env",
    "precision_gate_states",
    "performance_switch_states",
    "binary",
    "binary_sha256",
    "binary_size",
    "binary_origin",
    "build_receipt",
    "build_receipt_sha256",
    "page",
    "page_sha256",
    "pages",
    "workload",
    "model",
    "model_kind",
    "model_sha256",
    "model_size",
    "model_identity",
    "quant_recipe",
    "threads",
    "warmup",
    "ab",
)


def _invariant_meta_value(meta: dict, key: str) -> object:
    value = meta.get(key)
    if key == "ab" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in ("schedule_index", "arm_run")}
    return value


def _load_capture(run_dir: str, *, prefix: str = "") -> dict:
    metas = _measured_meta_paths(run_dir, prefix=prefix)
    if not metas:
        raise TimingParseError(f"no {prefix}run_*.meta.json under {run_dir}")

    runs: list[dict] = []
    wall_ms: list[float] = []
    stdout_hashes: set[str] = set()
    raw_records: list[dict] = []
    metadata: list[dict] = []
    rss_bytes: list[int] = []
    meta0: dict = {}
    evidence_bytes = 0
    for index, meta_path in enumerate(metas, start=1):
        expected_tag = f"{prefix}run_{index:03d}"
        if os.path.basename(meta_path) != f"{expected_tag}.meta.json":
            raise TimingParseError(
                "measured run metadata must be contiguous: "
                f"expected {expected_tag}.meta.json"
            )
        meta_bytes = _read_bounded(meta_path)
        meta = _bounded_json_bytes(meta_bytes, meta_path)
        if meta.get("tag") != expected_tag:
            raise TimingParseError(
                f"metadata tag {meta.get('tag')!r} does not match {expected_tag!r}"
            )
        stderr_path = _raw_member(
            run_dir, meta.get("stderr"), ".stderr", tag=expected_tag
        )
        stdout_path = _raw_member(
            run_dir, meta.get("stdout"), ".stdout", tag=expected_tag
        )
        stderr_bytes = _read_bounded(stderr_path)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        stdout_bytes = _read_bounded(stdout_path)
        evidence_bytes = _checked_evidence_total(
            evidence_bytes,
            len(meta_bytes) + len(stderr_bytes) + len(stdout_bytes),
        )
        if not meta0:
            meta0 = meta
        else:
            drift = [
                key
                for key in _CAPTURE_INVARIANT_KEYS
                if _invariant_meta_value(meta, key)
                != _invariant_meta_value(meta0, key)
            ]
            if drift:
                raise TimingParseError(
                    f"{meta_path}: measured-run identity drifted: {', '.join(drift)}"
                )
        if meta.get("exit_code", 0) != 0:
            raise TimingParseError(
                f"{meta_path}: run exited {meta['exit_code']} -- a failed run "
                "is not perf evidence"
            )
        parsed = parse_run(stderr_text)
        _validate_sequential_batch_result(
            meta,
            parsed,
            stderr_text,
            stdout_bytes,
            source=meta_path,
        )
        try:
            wall = float(meta["wall_ms"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise TimingParseError(f"{meta_path}: invalid wall_ms: {error}") from error
        if not math.isfinite(wall) or wall <= 0.0:
            raise TimingParseError(f"{meta_path}: wall_ms must be positive and finite")
        resources = parse_resource_usage(stderr_text)
        if "maximum_resident_set_size_bytes" in resources:
            rss_bytes.append(resources["maximum_resident_set_size_bytes"])
        runs.append(parsed)
        wall_ms.append(wall)
        metadata.append(meta)
        stdout_hashes.add(hashlib.sha256(stdout_bytes).hexdigest())
        observation = raw_observation(
            parsed,
            wall,
            run_id=expected_tag,
            resources=resources,
        )
        observation["raw_files"] = {
            "meta": {
                "path": os.path.basename(meta_path),
                "sha256": hashlib.sha256(meta_bytes).hexdigest(),
            },
            "stderr": {
                "path": os.path.basename(stderr_path),
                "sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            },
            "stdout": {
                "path": os.path.basename(stdout_path),
                "sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            },
        }
        raw_records.append(observation)
    return {
        "runs": runs,
        "wall_ms": wall_ms,
        "stdout_hashes": stdout_hashes,
        "raw_records": raw_records,
        "metadata": metadata,
        "meta0": meta0,
        "rss_bytes": rss_bytes,
        "evidence_bytes": evidence_bytes,
    }


def _resolve_capture_precision(capture: dict, requested: str | None) -> str:
    precision = infer_precision(capture["runs"])
    if requested:
        if precision is not None and precision != requested:
            raise TimingParseError(
                f"--precision {requested} contradicts the timing lines ({precision})"
            )
        precision = requested
    if precision is None:
        raise TimingParseError(
            "precision not inferable from the timing lines (GOT path) -- "
            "pass an explicit zoo-lane --precision"
        )
    _validate_unlimited_contract(precision, capture["meta0"])
    _validate_profiled_decode_capture(capture, precision)
    return precision


def _cmd_aggregate(args: argparse.Namespace) -> int:
    try:
        capture = _load_capture(args.run_dir)
        precision = _resolve_capture_precision(capture, args.precision)
    except (OSError, TimingParseError) as err:
        print(f"ERROR: precision contract: {err}", file=sys.stderr)
        return 1

    runs = capture["runs"]
    wall_ms = capture["wall_ms"]
    stdout_hashes = capture["stdout_hashes"]
    raw_records = capture["raw_records"]
    meta0 = capture["meta0"]
    try:
        threads = _resolve_threads(args.threads, meta0.get("threads"))
    except TimingParseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    records = aggregate_runs(
        runs,
        wall_ms,
        threads=threads,
        precision=precision,
        allocator=args.allocator,
        warmup_discarded=int(meta0.get("warmup", 0)),
        synthetic=args.synthetic,
    )
    doc = {
        "schema": SCHEMA_DOC,
        "source": "focr",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": os.path.abspath(args.run_dir),
        "command": meta0.get("command"),
        "env_pins": meta0.get("env_pins", {}),
        "focr_env": meta0.get("focr_env", {}),
        "precision_gate_states": meta0.get("precision_gate_states", {}),
        "performance_switch_states": meta0.get("performance_switch_states", {}),
        "binary": meta0.get("binary"),
        "binary_sha256": meta0.get("binary_sha256"),
        "binary_size": meta0.get("binary_size"),
        "binary_origin": meta0.get("binary_origin"),
        "build_receipt": meta0.get("build_receipt"),
        "build_receipt_sha256": meta0.get("build_receipt_sha256"),
        "page": meta0.get("page"),
        "page_sha256": meta0.get("page_sha256"),
        "pages": meta0.get("pages", []),
        "workload": meta0.get("workload"),
        "model": meta0.get("model"),
        "model_kind": meta0.get("model_kind"),
        "model_sha256": meta0.get("model_sha256"),
        "model_size": meta0.get("model_size"),
        "model_identity": meta0.get("model_identity"),
        "quant_recipe": meta0.get("quant_recipe"),
        "threads": threads,
        "precision": precision,
        "decode_mode": DECODE_MODE_BY_PRECISION.get(precision),
        "allocator": args.allocator,
        "runs": len(runs),
        "warmup": int(meta0.get("warmup", 0)),
        "stdout_identical_across_runs": len(stdout_hashes) == 1,
        "resources": {
            "maximum_resident_set_size": _byte_stats(capture["rss_bytes"])
        }
        if len(capture["rss_bytes"]) == len(runs)
        else None,
        "raw_timing": {
            "schema": SCHEMA_RAW_TIMING,
            "source": "focr",
            "unit": "ms",
            "measured_runs": len(raw_records),
            "records": raw_records,
        },
        "stages": records,
        "synthetic": args.synthetic,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "focr_stages_written",
                "out": args.out,
                "runs": len(runs),
                "stages": [r["stage"] for r in records],
                "stdout_identical_across_runs": doc["stdout_identical_across_runs"],
                "synthetic": args.synthetic,
            }
        )
    )
    return 0


def _balanced_ab_schedule(runs_per_arm: int) -> list[str]:
    counts = {"a": 0, "b": 0}
    schedule: list[str] = []
    block = 0
    while counts["a"] < runs_per_arm or counts["b"] < runs_per_arm:
        pattern = ("a", "b", "b", "a") if block % 2 == 0 else ("b", "a", "a", "b")
        for arm in pattern:
            if counts[arm] >= runs_per_arm:
                continue
            counts[arm] += 1
            schedule.append(arm)
        block += 1
    return schedule


def _validate_ab_capture(
    arm_a: dict,
    arm_b: dict,
    *,
    ab_env: str,
    a_label: str,
    a_value: str,
    b_label: str,
    b_value: str,
) -> list[dict]:
    if len(arm_a["runs"]) != len(arm_b["runs"]):
        raise TimingParseError("A/B arms have unequal measured-run counts")
    runs_per_arm = len(arm_a["runs"])
    if runs_per_arm < MIN_AB_RUNS_PER_ARM:
        raise TimingParseError(
            f"A/B capture requires at least {MIN_AB_RUNS_PER_ARM} measured runs per arm, "
            f"got {runs_per_arm}"
        )

    expected_identity = {
        "a": {"label": a_label, "value": a_value},
        "b": {"label": b_label, "value": b_value},
    }
    schedule_rows: list[dict] = []
    for arm, capture in (("a", arm_a), ("b", arm_b)):
        for arm_run, meta in enumerate(capture["metadata"], start=1):
            ab = meta.get("ab")
            if not isinstance(ab, dict):
                raise TimingParseError(f"{arm} run {arm_run} lacks an A/B receipt")
            expected = expected_identity[arm]
            if (
                ab.get("env") != ab_env
                or ab.get("arm") != arm
                or ab.get("label") != expected["label"]
                or ab.get("value") != expected["value"]
                or ab.get("arm_run") != arm_run
            ):
                raise TimingParseError(f"{arm} run {arm_run} has contradictory A/B identity")
            schedule_index = ab.get("schedule_index")
            if not isinstance(schedule_index, int) or isinstance(schedule_index, bool):
                raise TimingParseError(f"{arm} run {arm_run} has invalid schedule index")
            schedule_rows.append(
                {
                    "schedule_index": schedule_index,
                    "arm": arm,
                    "arm_run": arm_run,
                    "label": expected["label"],
                }
            )

    schedule_rows.sort(key=lambda row: row["schedule_index"])
    expected_schedule = _balanced_ab_schedule(runs_per_arm)
    if [row["schedule_index"] for row in schedule_rows] != list(
        range(1, 2 * runs_per_arm + 1)
    ):
        raise TimingParseError("A/B schedule indices are not contiguous")
    if [row["arm"] for row in schedule_rows] != expected_schedule:
        raise TimingParseError("A/B run order does not match the balanced interleave")

    varying_keys = {"focr_env", "performance_switch_states", "ab"}
    common_keys = [key for key in _CAPTURE_INVARIANT_KEYS if key not in varying_keys]
    drift = [
        key
        for key in common_keys
        if _invariant_meta_value(arm_a["meta0"], key)
        != _invariant_meta_value(arm_b["meta0"], key)
    ]
    if drift:
        raise TimingParseError(f"cross-arm subject identity drifted: {', '.join(drift)}")

    for map_key in ("focr_env", "performance_switch_states"):
        maps = []
        for arm, capture, expected_value in (
            ("a", arm_a, a_value),
            ("b", arm_b, b_value),
        ):
            value = capture["meta0"].get(map_key)
            if not isinstance(value, dict):
                raise TimingParseError(f"{arm} arm lacks {map_key} receipt")
            observed = value.get(ab_env, "<unset>")
            if observed != expected_value:
                raise TimingParseError(
                    f"{arm} arm {map_key} records {ab_env}={observed!r}, "
                    f"expected {expected_value!r}"
                )
            maps.append({k: v for k, v in value.items() if k != ab_env})
        if maps[0] != maps[1]:
            raise TimingParseError(
                f"cross-arm {map_key} differs outside selected switch {ab_env}"
            )

    if len(arm_a["rss_bytes"]) != runs_per_arm or len(arm_b["rss_bytes"]) != runs_per_arm:
        raise TimingParseError("every A/B observation must carry maximum RSS evidence")
    if len(arm_a["stdout_hashes"]) != 1 or len(arm_b["stdout_hashes"]) != 1:
        raise TimingParseError("stdout is nondeterministic within an A/B arm")
    if arm_a["stdout_hashes"] != arm_b["stdout_hashes"]:
        raise TimingParseError("A/B stdout differs byte-for-byte")
    return schedule_rows


def _ab_arm_document(
    capture: dict,
    *,
    arm: str,
    label: str,
    value: str,
    threads: int,
    precision: str,
    allocator: str,
    synthetic: bool,
) -> dict:
    meta = capture["meta0"]
    stages = aggregate_runs(
        capture["runs"],
        capture["wall_ms"],
        threads=threads,
        precision=precision,
        allocator=allocator,
        warmup_discarded=int(meta.get("warmup", 0)),
        synthetic=synthetic,
    )
    return {
        "arm": arm,
        "label": label,
        "value": value,
        "runs": len(capture["runs"]),
        "focr_env": meta.get("focr_env", {}),
        "performance_switch_states": meta.get("performance_switch_states", {}),
        "stdout_sha256": next(iter(capture["stdout_hashes"])),
        "resources": {
            "maximum_resident_set_size": _byte_stats(capture["rss_bytes"]),
        },
        "raw_timing": {
            "schema": SCHEMA_RAW_TIMING,
            "source": "focr",
            "unit": "ms",
            "measured_runs": len(capture["raw_records"]),
            "records": capture["raw_records"],
        },
        "stages": stages,
    }


def _ab_stage_comparisons(arm_a: dict, arm_b: dict) -> list[dict]:
    a_by_stage = {record["stage"]: record for record in arm_a["stages"]}
    b_by_stage = {record["stage"]: record for record in arm_b["stages"]}
    if set(a_by_stage) != set(b_by_stage):
        raise TimingParseError("A/B stage sets differ")
    comparisons = []
    for stage in a_by_stage:
        a_p50 = float(a_by_stage[stage]["p50_ms"])
        b_p50 = float(b_by_stage[stage]["p50_ms"])
        comparisons.append(
            {
                "stage": stage,
                "a_p50_ms": a_p50,
                "a_p95_ms": float(a_by_stage[stage]["p95_ms"]),
                "a_p99_ms": float(a_by_stage[stage]["p99_ms"]),
                "b_p50_ms": b_p50,
                "b_p95_ms": float(b_by_stage[stage]["p95_ms"]),
                "b_p99_ms": float(b_by_stage[stage]["p99_ms"]),
                "b_minus_a_p50_ms": round(b_p50 - a_p50, 6),
                "b_vs_a_p50_pct": round((b_p50 - a_p50) / a_p50 * 100.0, 6)
                if a_p50 > 0.0
                else None,
                "a_over_b_speedup": round(a_p50 / b_p50, 6)
                if b_p50 > 0.0
                else None,
            }
        )
    return comparisons


def _cmd_aggregate_ab(args: argparse.Namespace) -> int:
    try:
        if not re.fullmatch(r"FOCR_[A-Z0-9_]+", args.ab_env):
            raise TimingParseError("--ab-env must be a FOCR_* variable")
        arm_a = _load_capture(args.run_dir, prefix="a_")
        arm_b = _load_capture(args.run_dir, prefix="b_")
        if _checked_evidence_total(
            arm_a["evidence_bytes"], arm_b["evidence_bytes"]
        ) < 1:
            raise TimingParseError("empty A/B evidence")
        precision_a = _resolve_capture_precision(arm_a, args.precision)
        precision_b = _resolve_capture_precision(arm_b, args.precision)
        if precision_a != precision_b:
            raise TimingParseError("A/B precision labels differ")
        threads_a = _resolve_threads(args.threads, arm_a["meta0"].get("threads"))
        threads_b = _resolve_threads(args.threads, arm_b["meta0"].get("threads"))
        if threads_a != threads_b:
            raise TimingParseError("A/B thread budgets differ")
        schedule = _validate_ab_capture(
            arm_a,
            arm_b,
            ab_env=args.ab_env,
            a_label=args.a_label,
            a_value=args.a_value,
            b_label=args.b_label,
            b_value=args.b_value,
        )
        a_doc = _ab_arm_document(
            arm_a,
            arm="a",
            label=args.a_label,
            value=args.a_value,
            threads=threads_a,
            precision=precision_a,
            allocator=args.allocator,
            synthetic=args.synthetic,
        )
        b_doc = _ab_arm_document(
            arm_b,
            arm="b",
            label=args.b_label,
            value=args.b_value,
            threads=threads_b,
            precision=precision_b,
            allocator=args.allocator,
            synthetic=args.synthetic,
        )
        comparisons = _ab_stage_comparisons(a_doc, b_doc)
    except (OSError, TimingParseError, TypeError, ValueError) as error:
        print(f"ERROR: A/B evidence refused: {error}", file=sys.stderr)
        return 1

    meta = arm_a["meta0"]
    doc = {
        "schema": SCHEMA_AB,
        "source": "focr",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": os.path.abspath(args.run_dir),
        "workload": meta.get("workload"),
        "command": meta.get("command"),
        "pages": meta.get("pages", []),
        "binary": meta.get("binary"),
        "binary_sha256": meta.get("binary_sha256"),
        "binary_size": meta.get("binary_size"),
        "binary_origin": meta.get("binary_origin"),
        "build_receipt": meta.get("build_receipt"),
        "build_receipt_sha256": meta.get("build_receipt_sha256"),
        "model": meta.get("model"),
        "model_kind": meta.get("model_kind"),
        "model_sha256": meta.get("model_sha256"),
        "model_size": meta.get("model_size"),
        "model_identity": meta.get("model_identity"),
        "quant_recipe": meta.get("quant_recipe"),
        "threads": threads_a,
        "precision": precision_a,
        "decode_mode": DECODE_MODE_BY_PRECISION.get(precision_a),
        "allocator": args.allocator,
        "warmup": int(meta.get("warmup", 0)),
        "ab_env": args.ab_env,
        "schedule": schedule,
        "cross_arm_output": {
            "byte_identical": True,
            "sha256": a_doc["stdout_sha256"],
        },
        "arms": {"a": a_doc, "b": b_doc},
        "comparisons": comparisons,
        "synthetic": args.synthetic,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "event": "focr_ab_written",
                "out": args.out,
                "workload": meta.get("workload"),
                "runs_per_arm": len(arm_a["runs"]),
                "output_byte_identical": True,
                "synthetic": args.synthetic,
            }
        )
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

_SYNTHETIC_UNLIMITED = """\
[focr-timing] precision focr-full-int8
[focr-timing] preprocess 0.12s
[focr-timing]   vision.sam 1.20s
[focr-timing]   vision.clip 0.40s
[focr-timing]   vision.bridge 0.05s
[focr-timing] vision_tower 1.65s
[focr-timing] weight_cache_build_i8 1.10s
[focr-timing] prefill_i8 2.40s (289 tokens)
[focr-timing] decode_i8 9.00s (750 tokens, 0.012s/tok)
[focr-timing] decode_i8 phases (ms): lm_head 1200  attn 2400  experts 4300  route 100
noise line the parser must ignore
"""

_SYNTHETIC_UNLIMITED_MIXED = """\
[focr-timing] precision focr-mixed-ffn-int8
[focr-timing] preprocess 0.12s
[focr-timing]     unlimited_vision.hydrate(cached) 0.11s
[focr-timing]   vision.sam 1.20s
[focr-timing]   vision.clip 0.40s
[focr-timing]   vision.bridge 0.05s
[focr-timing] vision_tower 1.65s
[focr-timing] weight_cache_build 6.10s
[focr-timing] prefill 2.80s (289 tokens)
[focr-timing] decode 12.00s (750 tokens, 0.016s/tok)
[focr-timing] decode phases (ms): lm_head 1600  attn 3200  experts 6000  route 200
"""

_SYNTHETIC_GOT = """\
[focr-timing]   got.vision+splice 1.30s
[focr-timing]   decode 512 tok in 12.00s (42.7 tok/s) | seed(prefill) 0.55s | \
layers 10.00s (attn 3.00s, gemv+misc 7.00s) | lm_head 1.50s
[focr-timing]   got.generate 512 tokens 12.60s
[focr-timing] got forward 14.10s
"""

# The bd-t6a spine batched-vision line set (multi-page: two side groups for
# sam, single lines for hydrate/clip/bridge; the second sam group proves the
# same-stage fold sums seconds and view counts).
_SYNTHETIC_BATCH = """\
[focr-timing]     unlimited_vision.hydrate(batch-local) 0.21s
[focr-timing]   vision.hydrate(batch) 0.62s
[focr-timing]   vision.sam(batch of 3, side 1024) 2.40s
[focr-timing]   vision.sam(batch of 2, side 640) 0.80s
[focr-timing]   vision.clip(batch of 5) 0.90s
[focr-timing]   vision.bridge(batch of 5) 0.12s
"""

_SYNTHETIC_ZOO_BATCH = """\
[focr-timing]   got.hydrate(cached) 0.10s
[focr-timing]   got.vision+splice(batch of 3) 1.20s
[focr-timing]   got.generate(batch of 3) 12 tokens 2.30s
[focr-timing] got-ocr2 forward(batch of 3) 3.60s
[focr-timing]   smolvlm2.hydrate(cached) 0.20s
[focr-timing]   smolvlm2.vision+splice(batch of 3) 1.30s
[focr-timing]   smolvlm2.generate(batch of 3) 15 tokens 2.40s
[focr-timing] smolvlm2 forward(batch of 3) 3.90s
[focr-timing]   onechart.hydrate(cached) 0.30s
[focr-timing]   onechart.vision+splice(batch of 3) 1.40s
[focr-timing]   onechart.generate(batch of 3) 18 tokens 2.50s
[focr-timing] onechart forward(batch of 3) 4.20s
"""

_SYNTHETIC_MULTI_PAGE = """\
[focr-timing]   vision.sam 1.00s
[focr-timing]   vision.clip 0.20s
[focr-timing]   vision.bridge 0.01s
[focr-timing]   vision.sam 1.10s
[focr-timing]   vision.clip 0.21s
[focr-timing]   vision.bridge 0.01s
[focr-timing]   vision.sam 1.20s
[focr-timing]   vision.clip 0.22s
[focr-timing]   vision.bridge 0.01s
[focr-timing] multi_page.vision (3 pages) 4.50s
[focr-timing] decode 1.00s (5 tokens, 0.200s/tok)
[focr-timing] multi_page.decode 5 tokens 2.00s
"""

_SYNTHETIC_PDF_EVENTS = """\
[focr-timing] pdf page 1: spread split at x=100 (200x300)
[focr-timing] pdf page 2: no spread detected (200x300), unsplit
[focr-timing] preprocess 0.01s
"""

_SYNTHETIC_TROMR_EVENTS = """\
[focr-timing]   tromr.staff_detect 2 staves
[focr-timing]   tromr.staff 0 split-recognized (100x20)
[focr-timing]   tromr.staff 0 ok (100x20, semantic 42 chars)
[focr-timing]   tromr.staff 1 SKIP (90x20): blank staff
[focr-timing]   tromr.sanity underfull_bar part 1 measure 2: short by one beat
[focr-timing] tromr.encode 0.30s (w 100, 20 ctx tokens)
[focr-timing]     tromr.split seg 0 [0..100] 0.20s (42 chars)
[focr-timing] tromr.generate 10 steps 0.50s
[focr-timing] tromr forward 1.20s (1/2 staves recognized, semantic 42 chars total)
"""


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    # Unlimited-OCR full-int8 line set parses stage-for-stage and the runtime
    # marker agrees with every structural `_i8` suffix.
    run = parse_run(_SYNTHETIC_UNLIMITED)
    check("unlimited-preprocess", math.isclose(run["preprocess"]["s"], 0.12))
    check("unlimited-vision", math.isclose(run["vision_encode"]["s"], 1.65))
    check("unlimited-prefill-tokens", run["prefill"]["tokens"] == 289)
    check("unlimited-decode-tokens", run["decode_total"]["tokens"] == 750)
    check("unlimited-int8-marker", run["decode_total"].get("int8") is True)
    check("unlimited-phases", math.isclose(run["decode_experts"]["s"], 4.3))
    check("unlimited-full-int8-precision", infer_precision([run]) == PRECISION_FULL_INT8)

    mixed_run = parse_run(_SYNTHETIC_UNLIMITED_MIXED)
    check(
        "unlimited-mixed-precision",
        infer_precision([mixed_run]) == PRECISION_MIXED_FFN_INT8,
    )
    check(
        "unlimited-mixed-vision-hydrate",
        math.isclose(mixed_run["unlimited_vision_hydrate"]["s"], 0.11),
    )
    check(
        "unlimited-mixed-phases",
        math.isclose(mixed_run["decode_lm_head"]["s"], 1.6)
        and math.isclose(mixed_run["decode_attn"]["s"], 3.2)
        and math.isclose(mixed_run["decode_experts"]["s"], 6.0)
        and math.isclose(mixed_run["decode_route"]["s"], 0.2),
    )
    two_page_mixed = parse_run(_SYNTHETIC_UNLIMITED_MIXED * 2)
    check(
        "unlimited-multipage-phases-sum-and-count",
        math.isclose(two_page_mixed["decode_experts"]["s"], 12.0)
        and two_page_mixed["decode_experts"]["occurrences"] == 2
        and two_page_mixed["decode_total"]["occurrences"] == 2,
    )
    phase_line = (
        "[focr-timing] decode phases (ms): lm_head 1600  attn 3200  "
        "experts 6000  route 200\n"
    )
    for name, text in (
        (
            "refuses-missing-multipage-phase-line",
            (_SYNTHETIC_UNLIMITED_MIXED * 2).replace(phase_line, "", 1),
        ),
        (
            "refuses-duplicated-multipage-phase-line",
            _SYNTHETIC_UNLIMITED_MIXED * 2 + phase_line,
        ),
    ):
        try:
            parse_run(text)
            check(name, False)
        except TimingParseError:
            check(name, True)
    for name, text in (
        (
            "refuses-missing-runtime-precision",
            _SYNTHETIC_UNLIMITED.replace(
                "[focr-timing] precision focr-full-int8\n", ""
            ),
        ),
        (
            "refuses-full-label-on-mixed-lines",
            _SYNTHETIC_UNLIMITED_MIXED.replace(
                "focr-mixed-ffn-int8", "focr-full-int8"
            ),
        ),
        (
            "refuses-mixed-label-on-full-lines",
            _SYNTHETIC_UNLIMITED.replace("focr-full-int8", "focr-mixed-ffn-int8"),
        ),
    ):
        try:
            infer_precision([parse_run(text)])
            check(name, False)
        except TimingParseError:
            check(name, True)
    try:
        infer_precision([run, mixed_run])
        check("refuses-cross-run-precision-drift", False)
    except TimingParseError:
        check("refuses-cross-run-precision-drift", True)

    # GOT line set: the one-line decode breakdown fans out to shared stages.
    got = parse_run(_SYNTHETIC_GOT)
    check("got-decode-total", math.isclose(got["decode_total"]["s"], 12.0))
    check("got-decode-tokens", got["decode_total"]["tokens"] == 512)
    check("got-prefill-seed", math.isclose(got["prefill"]["s"], 0.55))
    check("got-vision", math.isclose(got["got_vision_splice"]["s"], 1.30))
    check("got-precision-unknown", infer_precision([got]) is None)

    # bd-t6a batched-vision lines: hydrate is its own stage; sam/clip/bridge
    # batch lines fold into the per-view stage names, summing seconds + views.
    batch = parse_run(_SYNTHETIC_BATCH)
    check(
        "batch-unlimited-vision-hydrate",
        math.isclose(batch["unlimited_vision_hydrate"]["s"], 0.21),
    )
    check("batch-hydrate", math.isclose(batch["vision_hydrate"]["s"], 0.62))
    check("batch-sam-sum", math.isclose(batch["vision_sam"]["s"], 3.20))
    check("batch-sam-views", batch["vision_sam"]["views"] == 5)
    check("batch-sam-occurrences", batch["vision_sam"]["occurrences"] == 2)
    check("batch-sam-marker", batch["vision_sam"].get("batched") is True)
    check("batch-clip", math.isclose(batch["vision_clip"]["s"], 0.90))
    check("batch-bridge", math.isclose(batch["vision_bridge"]["s"], 0.12))
    check("batch-precision-unknown", infer_precision([batch]) is None)

    # Per-view and batched lines for the same tower coexist in one run (a
    # mixed spine run): the fold sums into a single stage entry.
    mixed = parse_run(_SYNTHETIC_BATCH + "[focr-timing]   vision.sam 1.00s\n")
    check("batch-mixed-sam-sum", math.isclose(mixed["vision_sam"]["s"], 4.20))
    check("batch-mixed-occurrences", mixed["vision_sam"]["occurrences"] == 3)

    # Current zoo lanes emit first-use hydration plus three dense-batch spans.
    # Batch folds retain the sequential stage names while recording cardinality.
    zoo_batch = parse_run(_SYNTHETIC_ZOO_BATCH)
    hydrate_seconds = {"got": 0.1, "smolvlm2": 0.2, "onechart": 0.3}
    for lane, tokens in (("got", 12), ("smolvlm2", 15), ("onechart", 18)):
        check(
            f"{lane}-hydrate",
            math.isclose(zoo_batch[f"{lane}_hydrate"]["s"], hydrate_seconds[lane]),
        )
        check(
            f"{lane}-batch-vision",
            zoo_batch[f"{lane}_vision_splice"]["batched"] is True
            and zoo_batch[f"{lane}_vision_splice"]["views"] == 3,
        )
        check(
            f"{lane}-batch-generate",
            zoo_batch[f"{lane}_generate"]["batched"] is True
            and zoo_batch[f"{lane}_generate"]["streams"] == 3
            and zoo_batch[f"{lane}_generate"]["tokens"] == tokens,
        )
        check(
            f"{lane}-batch-forward",
            zoo_batch[f"{lane}_forward"]["batched"] is True
            and zoo_batch[f"{lane}_forward"]["views"] == 3,
        )

    zoo_records = aggregate_runs(
        [zoo_batch, zoo_batch],
        [5000.0, 5100.0],
        threads=8,
        precision="zoo-mixed",
        allocator="system",
        synthetic=True,
    )
    zoo_by_stage = {record["stage"]: record for record in zoo_records}
    check(
        "zoo-batch-cardinality-aggregation",
        zoo_by_stage["got_vision_splice"]["views"] == 3
        and zoo_by_stage["got_vision_splice"]["views_consistent"] is True
        and zoo_by_stage["got_generate"]["streams"] == 3
        and zoo_by_stage["got_generate"]["streams_consistent"] is True,
    )
    multi_page = parse_run(_SYNTHETIC_MULTI_PAGE)
    check(
        "multi-page-vision-canonical-stage",
        math.isclose(multi_page["vision_encode"]["s"], 4.5)
        and multi_page["vision_encode"]["pages"] == 3
        and multi_page["vision_encode"]["occurrences"] == 1,
    )
    check(
        "multi-page-vision-detail-counts",
        multi_page["vision_sam"]["occurrences"] == 3
        and multi_page["vision_clip"]["occurrences"] == 3
        and multi_page["vision_bridge"]["occurrences"] == 3
        and math.isclose(multi_page["vision_sam"]["s"], 3.3),
    )
    check(
        "multi-page-outer-decode-is-informational",
        math.isclose(multi_page["decode_total"]["s"], 1.0)
        and math.isclose(multi_page["multi_page_decode"]["s"], 2.0)
        and multi_page["multi_page_decode"]["tokens"] == 5,
    )

    pdf_events = parse_run(_SYNTHETIC_PDF_EVENTS)
    check(
        "pdf-spread-events-do-not-fabricate-numeric-stages",
        set(pdf_events) == {"preprocess"},
    )

    for name, text in (
        (
            "refuses-dense-batch-live-count-drift",
            _SYNTHETIC_ZOO_BATCH.replace(
                "got.generate(batch of 3)", "got.generate(batch of 2)"
            ),
        ),
        (
            "refuses-dense-batch-outer-smaller-than-live",
            _SYNTHETIC_ZOO_BATCH.replace(
                "got-ocr2 forward(batch of 3)", "got-ocr2 forward(batch of 2)"
            ),
        ),
        (
            "refuses-dense-batch-reduced-live-workload",
            _SYNTHETIC_ZOO_BATCH.replace(
                "got-ocr2 forward(batch of 3)", "got-ocr2 forward(batch of 4)"
            ),
        ),
        (
            "refuses-incomplete-dense-batch",
            _SYNTHETIC_ZOO_BATCH.replace(
                "[focr-timing]   got.generate(batch of 3) 12 tokens 2.30s\n", ""
            ),
        ),
        (
            "refuses-zero-cardinality-dense-batch",
            _SYNTHETIC_ZOO_BATCH.replace("batch of 3", "batch of 0"),
        ),
        (
            "refuses-mixed-sequential-and-dense-batch",
            _SYNTHETIC_ZOO_BATCH + "[focr-timing]   got.vision+splice 0.10s\n",
        ),
        (
            "refuses-multi-page-token-drift",
            _SYNTHETIC_MULTI_PAGE.replace(
                "multi_page.decode 5 tokens", "multi_page.decode 6 tokens"
            ),
        ),
        (
            "refuses-multi-page-page-count-drift",
            _SYNTHETIC_MULTI_PAGE.replace("(3 pages)", "(4 pages)"),
        ),
        (
            "refuses-multi-page-without-outer-decode",
            _SYNTHETIC_MULTI_PAGE.replace(
                "[focr-timing] multi_page.decode 5 tokens 2.00s\n", ""
            ),
        ),
        (
            "refuses-multi-page-decode-without-vision-total",
            _SYNTHETIC_MULTI_PAGE.replace(
                "[focr-timing] multi_page.vision (3 pages) 4.50s\n", ""
            ),
        ),
        (
            "refuses-duplicate-multi-page-transaction",
            _SYNTHETIC_MULTI_PAGE * 2,
        ),
        (
            "refuses-ambiguous-vision-total",
            _SYNTHETIC_MULTI_PAGE.replace(
                "[focr-timing] multi_page.vision",
                "[focr-timing] vision_tower 4.40s\n[focr-timing] multi_page.vision",
            ),
        ),
    ):
        try:
            parse_run(text)
            check(name, False)
        except TimingParseError:
            check(name, True)

    tromr_events = parse_run(_SYNTHETIC_TROMR_EVENTS)
    check(
        "tromr-events-do-not-fabricate-numeric-stages",
        set(tromr_events) == {"vision_encode", "tromr_split", "decode_total", "tromr_forward"},
    )
    check(
        "tromr-split-timing",
        math.isclose(tromr_events["tromr_split"]["s"], 0.2),
    )

    for name, malformed in (
        ("refuses-near-miss-got-batch-name", "got-ocr2x forward(batch of 3) 1.00s"),
        ("refuses-malformed-tromr-split", "tromr.split seg 0 [a..b] 0.20s (4 chars)"),
        ("refuses-empty-tromr-skip-reason", "tromr.staff 1 SKIP (90x20):"),
        ("refuses-unknown-tromr-sanity-kind", "tromr.sanity oddity part 1 measure 2: detail"),
        ("refuses-malformed-pdf-spread", "pdf page 1: spread split x=100 (200x300)"),
    ):
        try:
            parse_run(f"[focr-timing] preprocess 0.01s\n[focr-timing] {malformed}\n")
            check(name, False)
        except TimingParseError:
            check(name, True)

    # No timing lines / unknown timing line both refuse (never fabricate).
    for name, text in (
        ("refuses-empty", "plain stderr, no timing\n"),
        ("refuses-unknown-line", "[focr-timing] warpdrive 9.99s\n"),
    ):
        try:
            parse_run(text)
            check(name, False)
        except TimingParseError:
            check(name, True)

    # Aggregation: best-of-N is the min; cv% is sample stdev over mean;
    # per-token decode derives from totals; token drift is flagged.
    run_a = parse_run(_SYNTHETIC_UNLIMITED)
    run_b = parse_run(_SYNTHETIC_UNLIMITED.replace("9.00s (750", "9.60s (750"))
    records = aggregate_runs(
        [run_a, run_b],
        [15000.0, 15400.0],
        threads=8,
        precision=PRECISION_FULL_INT8,
        allocator="system",
        warmup_discarded=1,
        synthetic=True,
    )
    by_stage = {r["stage"]: r for r in records}
    check("agg-decode-best", math.isclose(by_stage["decode_total"]["best_ms"], 9000.0))
    per_tok = by_stage["decode_per_token"]
    check("agg-per-token-best", math.isclose(per_tok["best_ms"], 9000.0 / 750))
    check("agg-per-token-consistent", per_tok["tokens_consistent"] is True)
    expected_cv = statistics.stdev([12.0, 12.8]) / statistics.fmean([12.0, 12.8]) * 100
    check("agg-per-token-cv", math.isclose(per_tok["cv_pct"], round(expected_cv, 3)))
    check("agg-e2e-best", math.isclose(by_stage["end_to_end"]["best_ms"], 15000.0))
    check(
        "agg-decode-phase-stats",
        by_stage["decode_experts"]["samples_ms"] == [4300.0, 4300.0]
        and by_stage["decode_experts"]["p50_ms"] == 4300.0
        and by_stage["decode_experts"]["cv_pct"] == 0.0,
    )
    check("agg-p95-nearest-rank", _stats([float(v) for v in range(1, 21)])["p95_ms"] == 19.0)
    check("agg-p99-nearest-rank", _stats([float(v) for v in range(1, 101)])["p99_ms"] == 99.0)
    check("rss-p99-nearest-rank", _byte_stats(list(range(1, 101)))["p99_bytes"] == 99)
    try:
        _validate_ab_capture(
            {"runs": [run_a] * (MIN_AB_RUNS_PER_ARM - 1)},
            {"runs": [run_a] * (MIN_AB_RUNS_PER_ARM - 1)},
            ab_env="FOCR_TEST",
            a_label="a",
            a_value="0",
            b_label="b",
            b_value="1",
        )
        check("refuses-under-sampled-ab", False)
    except TimingParseError:
        check("refuses-under-sampled-ab", True)
    check("agg-synthetic-stamp", all(r["synthetic"] for r in records))
    raw = raw_observation(run_a, 15000.0, run_id="run_001")
    check(
        "raw-observation-recomputes-decode-sample",
        raw["stages"]["decode_per_token"]
        == {"ms": 12.0, "tokens": 750},
    )
    check(
        "raw-observation-preserves-wall-sample",
        raw["stages"]["end_to_end"] == {"ms": 15000.0},
    )
    check(
        "raw-observation-preserves-decode-phases",
        raw["stages"]["decode_lm_head"] == {"ms": 1200.0}
        and raw["stages"]["decode_attn"] == {"ms": 2400.0}
        and raw["stages"]["decode_experts"] == {"ms": 4300.0}
        and raw["stages"]["decode_route"] == {"ms": 100.0},
    )
    mac_resources = parse_resource_usage(
        "  11677941760  maximum resident set size\n"
        "   7519610024  peak memory footprint\n"
    )
    check(
        "resource-parser-macos-rss-bytes",
        mac_resources.get("maximum_resident_set_size_bytes") == 11677941760,
    )
    linux_resources = parse_resource_usage(
        "Maximum resident set size (kbytes): 2048\n"
    )
    check(
        "resource-parser-linux-rss-kib",
        linux_resources.get("maximum_resident_set_size_bytes") == 2 * 1024 * 1024,
    )
    check(
        "balanced-ab-schedule",
        _balanced_ab_schedule(5) == ["a", "b", "b", "a", "b", "a", "a", "b", "a", "b"],
    )

    run_drift = parse_run(_SYNTHETIC_UNLIMITED.replace("(750 tokens", "(751 tokens"))
    drift = aggregate_runs(
        [run_a, run_drift],
        [1.0, 1.0],
        threads=8,
        precision=PRECISION_FULL_INT8,
        allocator="system",
        synthetic=True,
    )
    drift_tok = next(r for r in drift if r["stage"] == "decode_per_token")
    check("agg-token-drift-flagged", drift_tok["tokens_consistent"] is False)

    zoo_batch_four = parse_run(_SYNTHETIC_ZOO_BATCH.replace("batch of 3", "batch of 4"))
    try:
        aggregate_runs(
            [zoo_batch, zoo_batch_four],
            [1.0, 1.0],
            threads=8,
            precision="zoo-mixed",
            allocator="system",
            synthetic=True,
        )
        check("refuses-cross-run-batch-cardinality-drift", False)
    except TimingParseError:
        check("refuses-cross-run-batch-cardinality-drift", True)

    # A stage missing from one run is a comparability error, not a silent gap.
    try:
        aggregate_runs(
            [run_a, {"preprocess": {"s": 0.1, "occurrences": 1}}],
            [1.0, 1.0],
            threads=8,
            precision=PRECISION_FULL_INT8,
            allocator="system",
            synthetic=True,
        )
        check("agg-missing-stage-refused", False)
    except TimingParseError:
        check("agg-missing-stage-refused", True)
    try:
        aggregate_runs(
            [mixed_run, two_page_mixed],
            [1.0, 2.0],
            threads=8,
            precision=PRECISION_MIXED_FFN_INT8,
            allocator="system",
            synthetic=True,
        )
        check("agg-occurrence-drift-refused", False)
    except TimingParseError:
        check("agg-occurrence-drift-refused", True)

    sequential_paths = [os.path.abspath("a.png"), os.path.abspath("b.png")]
    sequential_meta = {
        "command": ["/evidence/focr", "ocr-batch", *sequential_paths],
        "pages": [{"path": path} for path in sequential_paths],
        "workload": {"label": "2-page", "command": "ocr-batch", "page_count": 2},
        "focr_env": {},
        "performance_switch_states": {"FOCR_BATCH_SPINE": "<unset>"},
    }
    sequential_stdout = (
        f"===== {sequential_paths[0]} =====\nalpha\n"
        f"===== {sequential_paths[1]} =====\nbeta\n"
    ).encode()
    sequential_stderr = "".join(
        f"[focr] {path} (1.00s)\n" for path in sequential_paths
    )
    try:
        _validate_sequential_batch_result(
            sequential_meta,
            two_page_mixed,
            sequential_stderr,
            sequential_stdout,
            source="synthetic.meta.json",
        )
        check("batch-results-match-declared-pages", True)
    except TimingParseError as err:
        check("batch-results-match-declared-pages", False, error=str(err))
    try:
        _validate_sequential_batch_result(
            sequential_meta,
            mixed_run,
            sequential_stderr,
            sequential_stdout,
            source="synthetic.meta.json",
        )
        check("refuses-incomplete-sequential-batch", False)
    except TimingParseError:
        check("refuses-incomplete-sequential-batch", True)
    for name, failed_stderr, stdout in (
        (
            "refuses-post-decode-page-failure",
            f"[focr] {sequential_paths[0]} (1.00s)\n"
            f"[focr] {sequential_paths[1]} FAILED (1.00s): tokenizer failure\n",
            f"===== {sequential_paths[0]} =====\nalpha\n".encode(),
        ),
        (
            "refuses-missing-page-success-record",
            f"[focr] {sequential_paths[0]} (1.00s)\n",
            sequential_stdout,
        ),
    ):
        try:
            _validate_sequential_batch_result(
                sequential_meta,
                two_page_mixed,
                failed_stderr,
                stdout,
                source="synthetic.meta.json",
            )
            check(name, False)
        except TimingParseError:
            check(name, True)
    multi_page_meta = json.loads(json.dumps(sequential_meta))
    multi_page_meta["command"].append("--multi-page")
    try:
        _validate_sequential_batch_result(
            multi_page_meta,
            two_page_mixed,
            sequential_stderr,
            sequential_stdout,
            source="synthetic.meta.json",
        )
        check("refuses-multi-page-as-sequential-batch", False)
    except TimingParseError:
        check("refuses-multi-page-as-sequential-batch", True)
    delimiter_stdout = sequential_stdout + b"===== chapter =====\nbody\n"
    try:
        _validate_sequential_batch_result(
            sequential_meta,
            two_page_mixed,
            sequential_stderr,
            delimiter_stdout,
            source="synthetic.meta.json",
        )
        check("accepts-markdown-delimiter-shaped-content", True)
    except TimingParseError as err:
        check("accepts-markdown-delimiter-shaped-content", False, error=str(err))
    mismatched_command = json.loads(json.dumps(sequential_meta))
    mismatched_command["command"][2] = os.path.abspath("other.png")
    try:
        _validate_sequential_batch_result(
            mismatched_command,
            two_page_mixed,
            sequential_stderr,
            sequential_stdout,
            source="synthetic.meta.json",
        )
        check("refuses-command-page-mismatch", False)
    except TimingParseError:
        check("refuses-command-page-mismatch", True)
    for name, mutate in (
        ("refuses-missing-batch-workload", lambda meta: meta.pop("workload")),
        (
            "refuses-mismatched-batch-workload",
            lambda meta: meta["workload"].update(command="ocr"),
        ),
        (
            "refuses-missing-spine-receipt",
            lambda meta: meta.pop("performance_switch_states"),
        ),
        (
            "refuses-non-string-spine-receipt",
            lambda meta: (
                meta["performance_switch_states"].update(FOCR_BATCH_SPINE=1),
                meta["focr_env"].update(FOCR_BATCH_SPINE=1),
            ),
        ),
    ):
        bad_meta = json.loads(json.dumps(sequential_meta))
        mutate(bad_meta)
        try:
            _validate_sequential_batch_result(
                bad_meta,
                two_page_mixed,
                sequential_stderr,
                sequential_stdout,
                source="synthetic.meta.json",
            )
            check(name, False)
        except TimingParseError:
            check(name, True)
    try:
        _validate_sequential_batch_result(
            sequential_meta,
            two_page_mixed,
            sequential_stderr,
            f"===== {sequential_paths[0]} =====\nalpha\n".encode(),
            source="synthetic.meta.json",
        )
        check("refuses-truncated-human-batch-output", False)
    except TimingParseError:
        check("refuses-truncated-human-batch-output", True)
    json_meta = json.loads(json.dumps(sequential_meta))
    json_meta["command"].append("--json")
    json_payload = {
        "command": "ocr-batch",
        "count": 2,
        "results": [
            {"image": path, "ok": True, "markdown": text}
            for path, text in zip(sequential_paths, ("alpha", "beta"))
        ],
    }
    try:
        _validate_sequential_batch_result(
            json_meta,
            two_page_mixed,
            "",
            json.dumps(json_payload).encode(),
            source="synthetic.meta.json",
        )
        check("accepts-complete-json-batch", True)
    except TimingParseError as err:
        check("accepts-complete-json-batch", False, error=str(err))
    for name, mutate in (
        (
            "refuses-json-page-failure",
            lambda payload: payload["results"][1].update(ok=False),
        ),
        (
            "refuses-json-page-reordering",
            lambda payload: payload["results"].reverse(),
        ),
        (
            "refuses-json-missing-markdown",
            lambda payload: payload["results"][1].pop("markdown"),
        ),
    ):
        bad_payload = json.loads(json.dumps(json_payload))
        mutate(bad_payload)
        try:
            _validate_sequential_batch_result(
                json_meta,
                two_page_mixed,
                "",
                json.dumps(bad_payload).encode(),
                source="synthetic.meta.json",
            )
            check(name, False)
        except TimingParseError:
            check(name, True)
    spine_meta = json.loads(json.dumps(sequential_meta))
    for value in ("1", "enabled", "2"):
        spine_meta["performance_switch_states"] = {"FOCR_BATCH_SPINE": value}
        spine_meta["focr_env"] = {"FOCR_BATCH_SPINE": value}
        try:
            _validate_sequential_batch_result(
                spine_meta,
                two_page_mixed,
                sequential_stderr,
                sequential_stdout,
                source="synthetic.meta.json",
            )
            check(f"refuses-uninstrumented-batch-spine-{value}", False)
        except TimingParseError:
            check(f"refuses-uninstrumented-batch-spine-{value}", True)
    check(
        "batch-spine-disabled-values-match-runtime",
        all(
            not _batch_spine_enabled(value)
            for value in (None, "", "0", "off", "FALSE", " no ")
        )
        and _batch_spine_enabled("<unset>"),
    )
    literal_sentinel_meta = json.loads(json.dumps(sequential_meta))
    literal_sentinel_meta["performance_switch_states"] = {
        "FOCR_BATCH_SPINE": "<unset>"
    }
    literal_sentinel_meta["focr_env"] = {"FOCR_BATCH_SPINE": "<unset>"}
    try:
        _validate_sequential_batch_result(
            literal_sentinel_meta,
            two_page_mixed,
            sequential_stderr,
            sequential_stdout,
            source="synthetic.meta.json",
        )
        check("refuses-literal-unset-batch-spine-value", False)
    except TimingParseError:
        check("refuses-literal-unset-batch-spine-value", True)

    profiled_capture = {
        "meta0": {"env_pins": {"FOCR_PROFILE_DECODE": "1"}},
        "runs": [mixed_run],
    }
    try:
        _validate_profiled_decode_capture(
            profiled_capture, PRECISION_MIXED_FFN_INT8
        )
        check("profiled-capture-has-focused-stages", True)
    except TimingParseError as err:
        check("profiled-capture-has-focused-stages", False, error=str(err))
    unprofiled_run = parse_run(
        _SYNTHETIC_UNLIMITED_MIXED.replace(phase_line, "")
    )
    try:
        _validate_profiled_decode_capture(
            {"meta0": profiled_capture["meta0"], "runs": [unprofiled_run]},
            PRECISION_MIXED_FFN_INT8,
        )
        check("refuses-missing-focused-stage-evidence", False)
    except TimingParseError:
        check("refuses-missing-focused-stage-evidence", True)

    base_meta = {
        "model": "/models/unlimited-ocr.conservative.focrq",
        "model_kind": "file",
        "model_sha256": "a" * 64,
        "model_size": 4_157_448_783,
        "model_identity": {
            "dev": 1,
            "ino": 2,
            "size": 4_157_448_783,
            "mtime_ns": 3,
            "ctime_ns": 4,
        },
        "quant_recipe": UNLIMITED_QUANT_RECIPE,
        "precision_gate_states": {
            name: "<unset>" for name in PRECISION_GATE_VARS
        },
    }
    base_meta["precision_gate_states"].update(
        {
            "FOCR_DECODE_INT8": "0",
            "FOCR_INT8_ATTN": "false",
            "FOCR_INT8_LMHEAD": "off",
        }
    )
    try:
        _validate_unlimited_contract(PRECISION_MIXED_FFN_INT8, base_meta)
        check("mixed-contract-accepts-falsy-gates", True)
    except TimingParseError as err:
        check("mixed-contract-accepts-falsy-gates", False, error=str(err))

    full_meta = json.loads(json.dumps(base_meta))
    full_meta["precision_gate_states"].update(
        {
            "FOCR_DECODE_INT8": "yes",
            "FOCR_INT8_ATTN": "1",
            "FOCR_INT8_LMHEAD": "true",
        }
    )
    try:
        _validate_unlimited_contract(PRECISION_FULL_INT8, full_meta)
        check("full-contract-requires-three-truthy-gates", True)
    except TimingParseError as err:
        check("full-contract-requires-three-truthy-gates", False, error=str(err))

    for name, mutate in (
        (
            "refuses-mixed-truthy-attention-gate",
            lambda meta: meta["precision_gate_states"].update(FOCR_INT8_ATTN="1"),
        ),
        (
            "refuses-presence-gated-int8-kv",
            lambda meta: meta["precision_gate_states"].update(FOCR_INT8_KV="0"),
        ),
        ("refuses-unhashed-model", lambda meta: meta.update(model_sha256=None)),
        ("refuses-model-directory", lambda meta: meta.update(model_kind="directory")),
        ("refuses-missing-quant-recipe", lambda meta: meta.update(quant_recipe=None)),
        (
            "refuses-wrong-quant-recipe",
            lambda meta: meta.update(
                quant_recipe="unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1"
            ),
        ),
    ):
        bad = json.loads(json.dumps(base_meta))
        mutate(bad)
        try:
            _validate_unlimited_contract(PRECISION_MIXED_FFN_INT8, bad)
            check(name, False)
        except TimingParseError:
            check(name, True)

    check(
        "runtime-precision-maps-to-short-decode-mode",
        DECODE_MODE_BY_PRECISION[PRECISION_MIXED_FFN_INT8] == "mixed-ffn-int8"
        and DECODE_MODE_BY_PRECISION[PRECISION_FULL_INT8] == "full-int8",
    )
    check("raw-thread-label-from-meta", _resolve_threads(None, 8) == 8)
    try:
        _resolve_threads(16, 8)
        check("refuses-thread-override-contradicting-meta", False)
    except TimingParseError:
        check("refuses-thread-override-contradicting-meta", True)
    try:
        _checked_evidence_total(MAX_TIMING_TOTAL_BYTES, 1)
        check("refuses-total-evidence-byte-overflow", False)
    except TimingParseError:
        check("refuses-total-evidence-byte-overflow", True)
    try:
        _bounded_meta_names(
            f"noise_{index}" for index in range(MAX_RAW_DIRECTORY_ENTRIES + 1)
        )
        check("refuses-unbounded-raw-directory-enumeration", False)
    except TimingParseError:
        check("refuses-unbounded-raw-directory-enumeration", True)
    try:
        _bounded_meta_names(
            f"run_{index:03d}.meta.json"
            for index in range(1, MAX_MEASURED_RUNS + 2)
        )
        check("refuses-too-many-measured-meta-files", False)
    except TimingParseError:
        check("refuses-too-many-measured-meta-files", True)

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-timing-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run the parser self-test")
    sub = parser.add_subparsers(dest="cmd")
    agg = sub.add_parser("aggregate", help="aggregate a gauntlet_focr.sh run dir")
    agg.add_argument("--run-dir", required=True)
    agg.add_argument("--out", required=True)
    agg.add_argument("--threads", type=int, default=None)
    agg.add_argument(
        "--precision",
        default=None,
        help=(
            "caller identity for marker-less zoo lanes; Unlimited-OCR must emit "
            "focr-mixed-ffn-int8 or focr-full-int8 at runtime"
        ),
    )
    agg.add_argument("--allocator", default="system")
    agg.add_argument(
        "--synthetic",
        action="store_true",
        help="stamp records synthetic (self-test stubs; gauntlet_row.py refuses them)",
    )
    ab = sub.add_parser(
        "aggregate-ab", help="aggregate a strict interleaved A/B gauntlet run dir"
    )
    ab.add_argument("--run-dir", required=True)
    ab.add_argument("--out", required=True)
    ab.add_argument("--threads", type=int, default=None)
    ab.add_argument("--precision", default=None)
    ab.add_argument("--allocator", default="system")
    ab.add_argument("--ab-env", required=True)
    ab.add_argument("--a-label", required=True)
    ab.add_argument("--a-value", required=True)
    ab.add_argument("--b-label", required=True)
    ab.add_argument("--b-value", required=True)
    ab.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.cmd == "aggregate":
        return _cmd_aggregate(args)
    if args.cmd == "aggregate-ab":
        return _cmd_aggregate_ab(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
