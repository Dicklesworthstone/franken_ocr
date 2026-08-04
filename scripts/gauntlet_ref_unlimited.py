#!/usr/bin/env python3
"""The REAL Unlimited-OCR reference entry for `scripts/gauntlet_reference.py`
(bd-re8.17): the PROVEN CPU-patched HF baseline from the truth pack
(`scripts/baseline/run_baidu_reference.py`), instrumented per ledger stage.

Wire-up (see `scripts/gauntlet_runbook.sh` for the exact quiet-host commands):

    gauntlet_reference.py --stage all --backend hf --precision bf16 \
        --entry gauntlet_ref_unlimited:run_stage \
        --setup gauntlet_ref_unlimited:setup ...

`setup()` runs ONCE outside the clock: it installs the baseline's CPU
monkeypatches (`install_cpu_patches` — `Tensor.cuda`→identity,
`autocast("cuda")`→nullcontext, no CUDA), loads the pinned tokenizer + bf16
model exactly as `run_baidu_reference.py` does, and installs *instance-level*
forward timers (the modeling code is NEVER edited):

  * the CausalLM `forward` — every call's (q_len, start, duration);
  * the vision submodules (`sam_model`, `vision_model`, `projector`) — their
    summed wall time (they run inside the prefill forward).

`run_stage()` performs ONE full `model.infer(...)` with the baseline's exact
arguments (base_size=image_size=1024, crop_mode=False, eval_mode=True, greedy
temperature=0, no_repeat_ngram 35/1024, prompt "<image>document parsing.") and
maps the call log onto the PERF_LEDGER stage vocabulary:

  * `preprocess`        — infer() start → first forward start (PIL load,
                          pad/normalize transforms, tokenize, tensor stack);
  * `vision_encode`     — summed SAM + CLIP + projector module time;
  * `prefill`           — first forward (q_len > 1) MINUS the instrumented
                          vision span (the HF graph runs vision inline in the
                          prefill forward; focr's vocabulary separates them);
  * `decode_per_token`  — summed q_len == 1 forward time, with `tokens` =
                          the number of decode forwards (the harness divides:
                          decode-per-token = decode_wall / generated_tokens);
  * `end_to_end`        — the full infer() wall. NOTE: the model was loaded
                          in setup(), so unlike focr's end_to_end this
                          EXCLUDES model load — a bias in the REFERENCE's
                          favor, recorded in the stage note, never hidden.

Honesty: every number is a wall-clock reading of the real pinned model; a call
log that cannot be decomposed (no prefill forward, vision exceeding its parent
forward, zero decoded tokens) REFUSES rather than guesses.

Inference-affecting and evidence-output settings are explicit harness argv:
`--max-length 8192` and `--text-dir DIR`. Ambient `FOCR_REF_*` variables are
rejected by the citable harness so inherited shell state cannot change work.

Self-test (no torch, pure decomposition logic):  gauntlet_ref_unlimited.py --self-test
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import tempfile
import time

BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline")

# The baseline runner's exact infer() arguments (scripts/baseline/run_baidu_reference.py
# defaults) — the truth-pack-proven Base-mode CPU flow franken_ocr is measured against.
INFER_ARGS = {
    "prompt": "<image>document parsing.",
    "base_size": 1024,
    "image_size": 1024,
    "crop_mode": False,
    "eval_mode": True,
    "no_repeat_ngram_size": 35,
    "ngram_window": 1024,
    "temperature": 0.0,
}

DEFAULT_MAX_LENGTH = 8192


class ReferenceEntryError(RuntimeError):
    """The instrumented reference run cannot honestly produce stage numbers."""


# ── pure decomposition (self-testable without torch) ────────────────────────


def stages_from_call_log(
    infer_start_ns: int,
    infer_ns: int,
    calls: list[tuple[int, int, int]],
    vision_ns: int,
    vision_hooked: bool,
) -> dict:
    """Map one infer() call log onto the ledger stage vocabulary.

    `calls` is `[(q_len, start_ns, dur_ns), ...]` for every CausalLM forward,
    in call order. Raises [`ReferenceEntryError`] whenever the decomposition
    assumptions do not hold — a mis-decomposed stage must never become a row.
    """
    if not calls:
        raise ReferenceEntryError("no forward calls recorded — generate() never ran")
    prefill_calls = [c for c in calls if c[0] > 1]
    decode_calls = [c for c in calls if c[0] == 1]
    if len(prefill_calls) != 1:
        raise ReferenceEntryError(
            f"expected exactly 1 prefill forward (q_len > 1), saw {len(prefill_calls)} — "
            "the greedy eval_mode decomposition does not hold for this run"
        )
    prefill_q, prefill_start_ns, prefill_dur_ns = prefill_calls[0]
    if calls[0][0] <= 1:
        raise ReferenceEntryError("first forward is not the prefill — call order broken")
    if not decode_calls:
        raise ReferenceEntryError("no decode forwards (q_len == 1) — nothing was generated")
    if vision_hooked and vision_ns <= 0:
        raise ReferenceEntryError(
            "vision hooks installed but recorded 0 ns — vision never ran (missing image?)"
        )
    if vision_ns > prefill_dur_ns:
        raise ReferenceEntryError(
            f"vision span {vision_ns} ns exceeds its parent prefill forward "
            f"{prefill_dur_ns} ns — the nesting assumption is broken"
        )

    ms = 1e-6
    preprocess_ns = prefill_start_ns - infer_start_ns
    if preprocess_ns <= 0:
        raise ReferenceEntryError("prefill started before infer() — clock misuse")
    decode_ns = sum(dur for _q, _s, dur in decode_calls)

    stages = {
        "preprocess": {
            "ms": preprocess_ns * ms,
            "note": "infer() start -> first forward start (PIL+transform+tokenize+stack)",
        },
        "prefill": {
            "ms": (prefill_dur_ns - vision_ns) * ms,
            "tokens": prefill_q,
            "note": "first HF forward minus the instrumented inline vision span",
        },
        "decode_per_token": {
            "ms": decode_ns * ms,
            "tokens": len(decode_calls),
            "note": "summed q_len==1 forwards; harness divides by tokens",
        },
        "end_to_end": {
            "ms": infer_ns * ms,
            "note": "full infer() wall; EXCLUDES model load (done unclocked in setup) "
            "— a bias in the reference's favor vs focr's process wall clock",
        },
    }
    if vision_hooked:
        stages["vision_encode"] = {
            "ms": vision_ns * ms,
            "note": "summed sam_model+vision_model+projector module time (inside prefill)",
        }
    return stages


# ── instrumentation (instance-level; the pinned modeling code is untouched) ─


def _wrap_module_forward(module, on_call) -> None:
    """Time a module's forward via an instance attribute (Module.__call__
    resolves `self.forward` to the instance first). `functools.wraps` keeps
    the original signature visible to transformers' kwarg validation."""
    orig = module.forward

    @functools.wraps(orig)
    def timed(*args, **kwargs):
        t0 = time.perf_counter_ns()
        out = orig(*args, **kwargs)
        on_call(t0, time.perf_counter_ns() - t0, args, kwargs)
        return out

    module.forward = timed


def _q_len(args, kwargs) -> int:
    tensor = kwargs.get("input_ids")
    if tensor is None and args:
        tensor = args[0]
    if tensor is None:
        tensor = kwargs.get("inputs_embeds")
    if tensor is None or getattr(tensor, "ndim", 0) < 2:
        raise ReferenceEntryError(
            "forward call carries neither input_ids nor inputs_embeds — cannot classify"
        )
    return int(tensor.shape[1])


def setup(
    stage: str,
    page: str,
    model_dir: str,
    *,
    max_length: int,
    text_dir: str | None,
):
    """Load the pinned CPU-patched baseline ONCE (unclocked) and hook timers."""
    del stage  # one shared setup serves every stage
    if not page or not os.path.isfile(page):
        raise ReferenceEntryError(f"page fixture missing: {page!r}")
    if not model_dir or not os.path.isdir(model_dir):
        raise ReferenceEntryError(f"model dir missing: {model_dir!r}")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length != DEFAULT_MAX_LENGTH
    ):
        raise ReferenceEntryError(
            f"Unlimited-OCR citable max_length must be {DEFAULT_MAX_LENGTH}"
        )
    if text_dir is not None and (
        not isinstance(text_dir, str) or not text_dir or not os.path.isabs(text_dir)
    ):
        raise ReferenceEntryError("reference text output directory must be absolute")

    sys.path.insert(0, BASELINE_DIR)
    import run_baidu_reference as baseline  # scripts/baseline — the truth-pack runner

    baseline.install_cpu_patches()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.train(False)

    timers: dict = {"calls": [], "vision_ns": 0}

    def on_lm_call(t0: int, dur: int, args, kwargs) -> None:
        timers["calls"].append((_q_len(args, kwargs), t0, dur))

    def on_vision_call(_t0: int, dur: int, _args, _kwargs) -> None:
        timers["vision_ns"] += dur

    _wrap_module_forward(model, on_lm_call)
    inner = model.get_model() if hasattr(model, "get_model") else None
    vision_hooked = False
    for name in ("sam_model", "vision_model", "projector"):
        sub = getattr(inner, name, None) if inner is not None else None
        if sub is not None:
            _wrap_module_forward(sub, on_vision_call)
            vision_hooked = True

    scratch = tempfile.mkdtemp(prefix="focr-gauntlet-ref-")
    return {
        "torch": torch,
        "model": model,
        "tokenizer": tok,
        "timers": timers,
        "vision_hooked": vision_hooked,
        "scratch": scratch,
        "max_length": max_length,
        "text_dir": text_dir,
    }


def run_stage(stage: str, page: str, model_dir: str, state) -> dict:
    """ONE instrumented infer() of the pinned baseline; returns every stage.

    The harness (`gauntlet_reference.py`) clocks nothing here — all numbers
    come from the hooks — so the same call serves `--stage all` and any single
    requested stage.
    """
    del stage, model_dir  # the loaded state fixes both
    if state is None:
        raise ReferenceEntryError("setup() state missing — pass --setup gauntlet_ref_unlimited:setup")
    torch = state["torch"]
    timers = state["timers"]
    timers["calls"].clear()
    timers["vision_ns"] = 0

    t0 = time.perf_counter_ns()
    with torch.no_grad():
        text = state["model"].infer(
            state["tokenizer"],
            image_file=page,
            output_path=os.path.join(state["scratch"], "_scratch"),
            max_length=state["max_length"],
            **INFER_ARGS,
        )
    infer_ns = time.perf_counter_ns() - t0

    text = text if isinstance(text, str) else str(text)
    stages = stages_from_call_log(
        t0, infer_ns, list(timers["calls"]), timers["vision_ns"], state["vision_hooked"]
    )
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if state["text_dir"]:
        os.makedirs(state["text_dir"], exist_ok=True)
        stem = os.path.splitext(os.path.basename(page))[0]
        with open(os.path.join(state["text_dir"], f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(text)
    return {"stages": stages, "text_sha256": sha, "chars": len(text)}


# ── self-test (pure logic; no torch, no model) ───────────────────────────────


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
        except ReferenceEntryError:
            check(name, True)

    M = 1_000_000  # ns per ms
    # infer at t=0; preprocess 120 ms; prefill fwd 2000 ms (vision 1400 ms
    # inside); 3 decode forwards of 250 ms each; total 3000 ms.
    calls = [
        (290, 120 * M, 2000 * M),
        (1, 2200 * M, 250 * M),
        (1, 2500 * M, 250 * M),
        (1, 2800 * M, 250 * M),
    ]
    stages = stages_from_call_log(0, 3000 * M, calls, 1400 * M, True)
    check("preprocess-span", abs(stages["preprocess"]["ms"] - 120.0) < 1e-9)
    check("vision-span", abs(stages["vision_encode"]["ms"] - 1400.0) < 1e-9)
    check("prefill-minus-vision", abs(stages["prefill"]["ms"] - 600.0) < 1e-9)
    check("prefill-tokens", stages["prefill"]["tokens"] == 290)
    check("decode-sum", abs(stages["decode_per_token"]["ms"] - 750.0) < 1e-9)
    check("decode-tokens", stages["decode_per_token"]["tokens"] == 3)
    check("e2e", abs(stages["end_to_end"]["ms"] - 3000.0) < 1e-9)
    check(
        "ledger-stage-names",
        set(stages)
        == {"preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end"},
    )

    # Without vision hooks the stage is honestly absent and prefill is the
    # whole first forward (nothing is invented).
    unhooked = stages_from_call_log(0, 3000 * M, calls, 0, False)
    check("unhooked-no-vision-stage", "vision_encode" not in unhooked)
    check("unhooked-prefill-full", abs(unhooked["prefill"]["ms"] - 2000.0) < 1e-9)

    # Refusals: empty log, no prefill, two prefills, decode-first order,
    # no decode calls, hooked-but-zero vision, vision exceeding its parent.
    refuses("refuses-empty-log", lambda: stages_from_call_log(0, M, [], 0, False))
    refuses(
        "refuses-no-prefill",
        lambda: stages_from_call_log(0, M, [(1, 0, M)], 0, False),
    )
    refuses(
        "refuses-two-prefills",
        lambda: stages_from_call_log(0, M, [(290, 0, M), (290, M, M)], 0, False),
    )
    refuses(
        "refuses-decode-first",
        lambda: stages_from_call_log(
            0, 4000 * M, [(1, 100 * M, M), (290, 200 * M, M)], 0, False
        ),
    )
    refuses(
        "refuses-no-decode",
        lambda: stages_from_call_log(0, 3000 * M, [calls[0]], 1400 * M, True),
    )
    refuses(
        "refuses-hooked-zero-vision",
        lambda: stages_from_call_log(0, 3000 * M, calls, 0, True),
    )
    refuses(
        "refuses-vision-over-parent",
        lambda: stages_from_call_log(0, 3000 * M, calls, 2100 * M, True),
    )

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-ref-unlimited-self-test", "result": "pass"}))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
    raise SystemExit(2)
