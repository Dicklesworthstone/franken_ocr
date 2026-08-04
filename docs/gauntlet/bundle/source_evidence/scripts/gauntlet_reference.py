#!/usr/bin/env python3
"""Torch-side reference timer for the head-to-head gauntlet (bd-re8.17).

Times the pinned HF CPU reference per stage and emits the SAME
`focr-gauntlet-stage/v1` JSON records as `scripts/gauntlet_timing.py`, plus (as
the last stdout line, one per measured stage) the timing envelope
`benches/gauntlet_harness.rs` / docs/gauntlet/BENCH_HARNESS.md §5 expects from
`FOCR_REFERENCE_CMD`:

    {"stage":"decode_per_token","result":"pass","p50_ms":14.5,
     "precision":"bf16","threads":8,"reference_backend":"hf", ...}

FAIRNESS IS MANDATORY AND FAIL-CLOSED (docs/PERF_LEDGER.md §9.3; the hardened
frankentorch lesson — NEVER benchmark torch at @64):

  * a positive thread budget must be pinned (FOCR_THREADS or --threads),
    budget > 32 is refused outright;
  * the BLAS/OMP pool env vars must already equal the budget BEFORE torch is
    imported (OMP reads them at import) — a missing or mismatched pin REFUSES
    to emit any timing record (`result:"error"`, non-zero exit);
  * after `torch.set_num_threads(N)`, `torch.get_num_threads()` must equal the
    budget or the run refuses;
  * the truth-pack runtime pins (torch==2.10.0, transformers==4.57.1 —
    docs/truth-pack/PINNED_SOURCES.md) are verified; a drifted stack refuses.

The model-specific measurement is injected via `--entry module:function`
(loading/setup belongs in `--setup module:function`, run once outside the
clock). The REAL wired entry is `scripts/gauntlet_ref_unlimited.py` — the
truth-pack CPU-patched HF baseline (`scripts/baseline/run_baidu_reference.py`
flow) with per-stage instance-level forward timers. Entry protocols:

  * `None`                  — the harness's outer wall clock times the call
                              (single-stage legacy);
  * `{"tokens": n}`         — outer wall; `decode_per_token` divides by n;
  * `{"stages": {...}}`     — the entry measured each stage internally:
                              `{"ms": float, "tokens": int?}` per stage; ONE
                              entry call yields EVERY stage's sample, so
                              `--stage all` costs one inference per run.

Without an entry the run emits `result:"skip"` — it NEVER invents a number.

`--smoke` proves the plumbing end-to-end (single run, warmup 0) and stamps
every output SYNTHETIC/non-citable (`gauntlet_row.py` refuses it); envelopes
carry `result:"smoke"`. Timings from a smoke run are NOT evidence.

Usage:
  gauntlet_reference.py --stage all --page PAGE --model-dir DIR \
      --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage \
      --setup gauntlet_ref_unlimited:setup [--runs 5] [--warmup 1] [--out FILE]
  FOCR_GAUNTLET_STAGE=prefill FOCR_THREADS=8 gauntlet_reference.py ...  # envelope mode
  gauntlet_reference.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import statistics
import sys
import time

SCHEMA_STAGE = "focr-gauntlet-stage/v1"
SCHEMA_DOC = "focr-gauntlet-stages/v1"

STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")
ALL = "all"  # measure every stage the entry exposes, one entry call per run


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Pool pins that must equal the budget BEFORE `import torch` (OMP/MKL read the
# environment at import time; setting them afterwards silently does nothing).
# Mirrors the benches/gauntlet_harness.rs reference env list.
PRE_IMPORT_PINS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Truth-pack runtime pins (docs/truth-pack/PINNED_SOURCES.md). A ratio against
# an unpinned stack is not comparable and is not added (docs/PERF_LEDGER.md).
PINNED_TORCH = "2.10.0"
PINNED_TRANSFORMERS = "4.57.1"

MAX_THREAD_BUDGET = 32  # NEVER @64 — oversubscription inflates fake wins


class FairnessError(RuntimeError):
    """A mandatory fairness control is not satisfied; no record may be emitted."""


def resolve_budget(arg_threads: int | None, env: dict[str, str]) -> int:
    """The pinned thread budget, or a refusal. There is NO default here — an
    unpinned reference run must never silently measure at the machine width."""
    raw = str(arg_threads) if arg_threads is not None else env.get("FOCR_THREADS", "")
    if not raw.strip():
        raise FairnessError(
            "thread budget unpinned: set FOCR_THREADS or pass --threads"
        )
    try:
        budget = int(raw)
    except ValueError as err:
        raise FairnessError(f"thread budget {raw!r} is not an integer") from err
    if budget <= 0:
        raise FairnessError(f"thread budget must be positive, got {budget}")
    if budget > MAX_THREAD_BUDGET:
        raise FairnessError(
            f"thread budget {budget} > {MAX_THREAD_BUDGET} — oversubscribed torch "
            "runs are rejected (measure at @8/@32, NEVER @64)"
        )
    return budget


def verify_env_pins(budget: int, env: dict[str, str]) -> None:
    """Every pool pin must be present and equal to the budget pre-import."""
    for key in PRE_IMPORT_PINS:
        value = env.get(key, "")
        if value.strip() != str(budget):
            raise FairnessError(
                f"{key}={value!r} does not pin the budget {budget}; export "
                f"{key}={budget} before running (torch/BLAS read it at import)"
            )


def verify_torch_pinned(budget: int, torch_threads: int) -> None:
    """Post-`set_num_threads` proof that torch actually honors the budget."""
    if torch_threads != budget:
        raise FairnessError(
            f"torch.get_num_threads()={torch_threads} != pinned budget {budget}"
        )


def verify_stack_pins(
    torch_version: str,
    transformers_version: str,
    pin_torch: str = PINNED_TORCH,
    pin_transformers: str = PINNED_TRANSFORMERS,
) -> None:
    """Refuse a stack that drifts from the EXPLICIT pins.

    The defaults are the Unlimited-OCR truth-pack stack. A zoo lane whose
    oracle certs were generated against a different stack passes ITS certified
    pins via `--pin-torch/--pin-transformers` (A11: GOT/OneChart certs are
    torch 2.12.1 + transformers 4.45.2 — 4.57's cache API breaks the pinned
    GOT modeling code). The pins remain fail-closed either way and land in
    the output records; there is no unpinned mode.
    """

    def base(v: str) -> str:
        return v.split("+", 1)[0]

    if (
        base(torch_version) != pin_torch
        or base(transformers_version) != pin_transformers
    ):
        raise FairnessError(
            f"unpinned reference stack: torch=={torch_version}, "
            f"transformers=={transformers_version} (this run pins "
            f"torch=={pin_torch}, transformers=={pin_transformers}); "
            "a ratio against a drifted stack is not comparable"
        )


def stats(samples_ms: list[float]) -> dict:
    mean = statistics.fmean(samples_ms)
    cv = (
        statistics.stdev(samples_ms) / mean * 100.0
        if len(samples_ms) > 1 and mean > 0
        else None
    )
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "best_ms": round(min(samples_ms), 6),
        "p50_ms": round(statistics.median(samples_ms), 6),
        "mean_ms": round(mean, 6),
        "cv_pct": None if cv is None else round(cv, 3),
        "n": len(samples_ms),
    }


def build_record(
    stage: str,
    samples_ms: list[float],
    *,
    budget: int,
    torch_threads: int,
    precision: str,
    backend: str,
    allocator: str,
    warmup_discarded: int,
    tokens: int | None = None,
    synthetic: bool = False,
) -> dict:
    if not samples_ms:
        raise FairnessError("no measured samples — a record cannot be built")
    record = {
        "schema": SCHEMA_STAGE,
        "source": "reference",
        "stage": stage,
        "ledger_stage": stage in STAGES,
        "unit": "ms",
        **stats(samples_ms),
        "warmup_discarded": warmup_discarded,
        "threads": budget,
        "thread_proof": {"budget": budget, "torch_num_threads": torch_threads},
        "precision": precision,
        "backend": backend,
        "allocator": allocator,
        "synthetic": synthetic,
    }
    if tokens is not None:
        record["tokens"] = tokens
    return record


def stage_sample(
    stage: str, result: object, outer_ms: float
) -> tuple[float, int | None]:
    """One measured `(ms, tokens|None)` for `stage` from one entry invocation.

    Protocols (module docstring): `None` → the harness's outer wall (legacy
    single-stage); `{"tokens": n}` → outer wall + token count;
    `{"stages": {...}}` → the entry measured the stage internally (`ms`
    mandatory and positive, `tokens` optional and positive)."""
    if result is None:
        return outer_ms, None
    if not isinstance(result, dict):
        raise FairnessError(
            f"entry returned {type(result).__name__}; want dict or None"
        )
    if "stages" in result:
        stages = result["stages"]
        if not isinstance(stages, dict):
            raise FairnessError("entry result 'stages' is not a dict")
        rec = stages.get(stage)
        if rec is None:
            raise FairnessError(
                f"entry measured no stage {stage!r} (measured: {sorted(stages)})"
            )
        ms = rec.get("ms")
        if not isinstance(ms, (int, float)) or isinstance(ms, bool) or ms <= 0:
            raise FairnessError(f"entry stage {stage!r} has no positive 'ms': {ms!r}")
        tokens = rec.get("tokens")
        if tokens is not None:
            tokens = int(tokens)
            if tokens <= 0:
                raise FairnessError(f"entry stage {stage!r} has non-positive tokens")
        return float(ms), tokens
    tokens = result.get("tokens")
    if tokens is None:
        return outer_ms, None
    tokens = int(tokens)
    if tokens <= 0:
        raise FairnessError("entry returned non-positive tokens")
    return outer_ms, tokens


def requested_from_result(stage_req: str, result: object) -> list[str]:
    """The stage list one entry call satisfies. `--stage all` requires the
    multi-stage protocol and keeps only ledger-vocabulary stages, in order."""
    if stage_req != ALL:
        return [stage_req]
    if not (isinstance(result, dict) and isinstance(result.get("stages"), dict)):
        raise FairnessError(
            "--stage all requires an entry returning {'stages': {...}} — the "
            "legacy outer-wall protocol times exactly one stage"
        )
    requested = [s for s in STAGES if s in result["stages"]]
    if not requested:
        raise FairnessError(
            f"entry measured no ledger-vocabulary stage (measured: {sorted(result['stages'])})"
        )
    return requested


def envelope_from_record(record: dict, result: str = "pass") -> dict:
    """The benches/gauntlet_harness.rs timing envelope (BENCH_HARNESS.md §5):
    requires `result`, a duration key, a thread proof, and a precision."""
    return {
        "stage": record["stage"],
        "result": result,
        "p50_ms": record["p50_ms"],
        "min_ms": record["best_ms"],
        "cv_pct": record["cv_pct"],
        "precision": record["precision"],
        "reference_precision": record["precision"],
        "threads": record["threads"],
        "reference_threads": record["threads"],
        "torch_threads": record["thread_proof"]["torch_num_threads"],
        "reference_backend": record["backend"],
        "n": record["n"],
    }


def _emit(obj: dict) -> None:
    print(json.dumps(obj, sort_keys=False))


def _load_callable(spec: str):
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise FairnessError(f"--entry/--setup must be module:function, got {spec!r}")
    # Resolve entry modules from the scripts dir (where gauntlet_ref_unlimited
    # lives) regardless of the caller's cwd, then the cwd for ad-hoc entries.
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_stage(args: argparse.Namespace, stage_req: str, budget: int) -> int:
    """Measure the requested stage(s) with the injected entry, or refuse/skip
    honestly. One entry call per run; with the multi-stage protocol that one
    call samples EVERY stage (so `--stage all` costs one inference per run)."""
    if args.entry is None:
        _emit(
            {
                "stage": stage_req,
                "result": "skip",
                "reason": "stage_entry_not_wired",
                "detail": "pass --entry module:function (the real wiring is "
                "--entry gauntlet_ref_unlimited:run_stage --setup "
                "gauntlet_ref_unlimited:setup); the harness never invents a number",
            }
        )
        return 0

    # Import torch only AFTER the env pins are proven (they are read at import).
    import torch  # noqa: PLC0415 — deliberate post-gate import
    import transformers  # noqa: PLC0415

    verify_stack_pins(
        torch.__version__,
        transformers.__version__,
        args.pin_torch,
        args.pin_transformers,
    )
    torch.set_num_threads(budget)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized by a prior op; intra-op pin below still gates
    verify_torch_pinned(budget, torch.get_num_threads())

    entry = _load_callable(args.entry)
    setup_state = None
    if args.setup is not None:
        # stdout is reserved for the envelope lines; anything the model code
        # prints is redirected to stderr (visible in logs, never parsed).
        with contextlib.redirect_stdout(sys.stderr):
            setup_state = _load_callable(args.setup)(
                stage_req, args.page, args.model_dir
            )

    requested: list[str] | None = None
    samples_ms: dict[str, list[float]] = {}
    tokens_by_stage: dict[str, int] = {}
    text_shas: list[str] = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result = entry(stage_req, args.page, args.model_dir, setup_state)
        outer_ms = (time.perf_counter() - t0) * 1000.0

        if requested is None:
            requested = requested_from_result(stage_req, result)
        if isinstance(result, dict) and result.get("text_sha256"):
            text_shas.append(str(result["text_sha256"]))
        for stage in requested:
            elapsed_ms, tokens = stage_sample(stage, result, outer_ms)
            if tokens is not None:
                prev = tokens_by_stage.get(stage)
                if prev is not None and prev != tokens:
                    raise FairnessError(
                        f"{stage}: token count drifted across runs ({prev} -> {tokens}); "
                        "a nondeterministic reference cannot land a ratio"
                    )
                tokens_by_stage[stage] = tokens
                if stage == "decode_per_token":
                    elapsed_ms /= tokens
            if i >= args.warmup:
                samples_ms.setdefault(stage, []).append(elapsed_ms)

    assert requested is not None  # runs >= 1 is enforced in main()
    records = [
        build_record(
            stage,
            samples_ms[stage],
            budget=budget,
            torch_threads=torch.get_num_threads(),
            precision=args.precision,
            backend=args.backend,
            allocator=args.allocator,
            warmup_discarded=args.warmup,
            tokens=tokens_by_stage.get(stage),
            synthetic=args.smoke,
        )
        for stage in requested
    ]
    doc = {
        "schema": SCHEMA_DOC,
        "source": "reference",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "env_pins": {
            k: os.environ.get(k, "") for k in PRE_IMPORT_PINS + ("FOCR_THREADS",)
        },
        "page": args.page,
        "page_sha256": sha256_file(args.page),
        "model": args.model_dir,
        "threads": budget,
        "precision": args.precision,
        "backend": args.backend,
        "allocator": args.allocator,
        "runs": args.runs,
        "warmup": args.warmup,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "stages": records,
        "synthetic": args.smoke,
        "smoke": args.smoke,
    }
    if text_shas:
        doc["text_sha256"] = sorted(set(text_shas))
        doc["text_identical_across_runs"] = len(set(text_shas)) == 1
    if args.out:
        existing = None
        if os.path.exists(args.out):
            with open(args.out, encoding="utf-8") as f:
                existing = json.load(f)
        if existing is not None and existing.get("schema") == SCHEMA_DOC:
            existing["stages"] = [
                r for r in existing["stages"] if r.get("stage") not in requested
            ] + records
            for key in (
                "page_sha256",
                "text_sha256",
                "text_identical_across_runs",
                "smoke",
                "synthetic",
            ):
                if key in doc:
                    existing[key] = doc[key]
            doc = existing
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
    # The envelope(s) MUST be the last stdout lines (the bench harness parses
    # the final line; single-stage behavior is unchanged).
    for record in records:
        _emit(envelope_from_record(record, result="smoke" if args.smoke else "pass"))
    return 0


# ── self-test (no torch required; the gate logic is pure) ───────────────────


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail"}))
        if not ok:
            failures.append(name)

    def refuses(name: str, fn) -> None:
        try:
            fn()
            check(name, False)
        except FairnessError:
            check(name, True)

    pinned8 = {k: "8" for k in PRE_IMPORT_PINS} | {"FOCR_THREADS": "8"}

    # Budget resolution: pinned accepted; unset/zero/64 refused.
    check("budget-env", resolve_budget(None, pinned8) == 8)
    check("budget-arg-overrides", resolve_budget(4, pinned8) == 4)
    refuses("budget-unpinned-refused", lambda: resolve_budget(None, {}))
    refuses("budget-zero-refused", lambda: resolve_budget(0, pinned8))
    refuses("budget-64-refused", lambda: resolve_budget(64, pinned8))
    refuses(
        "budget-garbage-refused",
        lambda: resolve_budget(None, {"FOCR_THREADS": "eight"}),
    )

    # Env pins: all present+equal passes; missing or drifted refuses.
    check("pins-ok", verify_env_pins(8, pinned8) is None)
    refuses(
        "pins-missing-refused", lambda: verify_env_pins(8, {"OMP_NUM_THREADS": "8"})
    )
    refuses(
        "pins-drifted-refused",
        lambda: verify_env_pins(8, pinned8 | {"MKL_NUM_THREADS": "64"}),
    )

    # torch thread proof and stack pins.
    check("torch-proof-ok", verify_torch_pinned(8, 8) is None)
    refuses("torch-proof-64-refused", lambda: verify_torch_pinned(8, 64))
    check("stack-pins-ok", verify_stack_pins("2.10.0+cpu", "4.57.1") is None)
    refuses("stack-drift-refused", lambda: verify_stack_pins("2.11.0", "4.57.1"))

    # Records refuse to exist without samples; the envelope carries every
    # mandatory field of the BENCH_HARNESS.md contract.
    refuses(
        "empty-samples-refused",
        lambda: build_record(
            "prefill",
            [],
            budget=8,
            torch_threads=8,
            precision="bf16",
            backend="hf",
            allocator="system",
            warmup_discarded=1,
        ),
    )
    record = build_record(
        "decode_per_token",
        [14.5, 15.0, 14.7],
        budget=8,
        torch_threads=8,
        precision="bf16",
        backend="hf",
        allocator="system",
        warmup_discarded=1,
        tokens=600,
        synthetic=True,
    )
    check("record-best", record["best_ms"] == 14.5)
    check("record-thread-proof", record["thread_proof"]["torch_num_threads"] == 8)
    envelope = envelope_from_record(record)
    for field in (
        "stage",
        "result",
        "p50_ms",
        "precision",
        "threads",
        "reference_backend",
    ):
        check(
            f"envelope-has-{field}",
            field in envelope and envelope[field] not in ("", None),
        )
    check("envelope-result-pass", envelope["result"] == "pass")
    check(
        "envelope-result-smoke",
        envelope_from_record(record, result="smoke")["result"] == "smoke",
    )

    # Entry protocols (stage_sample): outer wall, legacy tokens, multi-stage.
    check("sample-none-outer", stage_sample("prefill", None, 123.0) == (123.0, None))
    check(
        "sample-legacy-tokens",
        stage_sample("decode_per_token", {"tokens": 600}, 90.0) == (90.0, 600),
    )
    multi = {
        "stages": {
            "preprocess": {"ms": 120.0},
            "prefill": {"ms": 600.0, "tokens": 290},
            "decode_per_token": {"ms": 750.0, "tokens": 3},
            "end_to_end": {"ms": 3000.0},
        },
        "text_sha256": "ab" * 32,
    }
    check("sample-multi-ms", stage_sample("prefill", multi, 9e9) == (600.0, 290))
    check(
        "sample-multi-decode",
        stage_sample("decode_per_token", multi, 9e9) == (750.0, 3),
    )
    refuses(
        "sample-missing-stage-refused",
        lambda: stage_sample("vision_encode", multi, 1.0),
    )
    refuses(
        "sample-nonpositive-ms-refused",
        lambda: stage_sample("prefill", {"stages": {"prefill": {"ms": 0}}}, 1.0),
    )
    refuses(
        "sample-nonpositive-tokens-refused",
        lambda: stage_sample(
            "prefill", {"stages": {"prefill": {"ms": 1.0, "tokens": 0}}}, 1.0
        ),
    )
    refuses("sample-bad-type-refused", lambda: stage_sample("prefill", "nope", 1.0))

    # `--stage all` resolution: ledger order, multi-stage protocol required.
    check(
        "all-resolves-ledger-order",
        requested_from_result(ALL, multi)
        == ["preprocess", "prefill", "decode_per_token", "end_to_end"],
    )
    check(
        "single-stage-passthrough",
        requested_from_result("prefill", None) == ["prefill"],
    )
    refuses("all-requires-stages-refused", lambda: requested_from_result(ALL, None))
    refuses(
        "all-empty-vocab-refused",
        lambda: requested_from_result(ALL, {"stages": {"warpdrive": {"ms": 1.0}}}),
    )

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-reference-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stage", choices=STAGES + (ALL,), default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--model-dir", default=os.environ.get("FOCR_MODEL_DIR"))
    parser.add_argument("--backend", default=os.environ.get("FOCR_REFERENCE_BACKEND"))
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--allocator", default="system")
    parser.add_argument("--entry", default=None, help="module:function timed per stage")
    parser.add_argument(
        "--setup", default=None, help="module:function run once, unclocked"
    )
    parser.add_argument(
        "--out", default=None, help="stage-record JSON (merged per stage)"
    )
    parser.add_argument(
        "--pin-torch",
        default=PINNED_TORCH,
        help="the torch version this lane's oracle certs pinned (default: the truth pack)",
    )
    parser.add_argument(
        "--pin-transformers",
        default=PINNED_TRANSFORMERS,
        help="the transformers version this lane's oracle certs pinned (default: the truth pack)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="single untimed plumbing run: forces runs=1 warmup=0 and stamps every "
        "output synthetic/non-citable (gauntlet_row.py refuses it)",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.smoke:
        args.runs, args.warmup = 1, 0

    stage = (
        args.stage
        or os.environ.get("FOCR_GAUNTLET_STAGE")
        or os.environ.get("FOCR_STAGE")
    )
    if stage not in STAGES + (ALL,):
        _emit({"result": "error", "reason": "no_stage", "detail": f"stage={stage!r}"})
        return 2
    if not args.backend or not str(args.backend).strip():
        _emit({"stage": stage, "result": "error", "reason": "no_reference_backend"})
        return 2
    if args.runs < 1 or args.warmup < 0:
        _emit({"stage": stage, "result": "error", "reason": "bad_run_counts"})
        return 2

    try:
        budget = resolve_budget(args.threads, dict(os.environ))
        verify_env_pins(budget, dict(os.environ))
        return run_stage(args, stage, budget)
    except FairnessError as err:
        # Fail-closed: an unfair run emits an error envelope and NO timing row.
        _emit(
            {
                "stage": stage,
                "result": "error",
                "reason": "fairness",
                "detail": str(err),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
