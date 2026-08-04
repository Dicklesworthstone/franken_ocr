#!/usr/bin/env python3
"""The zoo-lane reference entries for `scripts/gauntlet_reference.py` (A11,
bd-3jo6.1.11): GOT-OCR2 / SmolVLM2-500M / OneChart, each instrumented per
ledger stage exactly like `gauntlet_ref_unlimited.py` (whose decomposition
helpers this module REUSES — one call-log grammar, one refusal discipline).

Wire-up (one lane per gauntlet run; see the A11 runbook commands):

    gauntlet_reference.py --stage all --backend hf --precision f32 \
        --entry gauntlet_ref_zoo:run_got      --setup gauntlet_ref_zoo:setup_got ...
        --entry gauntlet_ref_zoo:run_smolvlm2 --setup gauntlet_ref_zoo:setup_smolvlm2 ...
        --entry gauntlet_ref_zoo:run_onechart --setup gauntlet_ref_zoo:setup_onechart ...

Per lane, `setup_*()` loads the pinned model ONCE (unclocked, f32 — the same
dtype the lane's oracle certs ran) and installs instance-level forward timers
(the modeling code is NEVER edited): the top-level CausalLM forward (every
call's q_len/start/duration) plus the lane's vision submodules (summed wall
time; they run inline in the prefill forward, same nesting as Unlimited-OCR):

  * GOT      — `model.chat(tokenizer, page, ocr_type='ocr')`;
               vision = `vision_tower_high` + `mm_projector_vary`.
  * SmolVLM2 — chat-template + `processor(...)` + greedy `generate` (the
               C-wave describe flow, default model-card caption prompt or
               FOCR_REF_QUESTION); vision = `vision_model` + `connector`.
  * OneChart — `model.chat(tokenizer, page, reliable_check=True)` (the
               number head fires INSIDE the same generate — one prefill);
               vision = `vision_tower` + `mm_projector`.

Decode is UNCAPPED on both sides of the gauntlet (natural greedy EOS
termination, exactly what focr's lane forwards do); `decode_per_token`
normalizes any token-count delta and the counts land in the records.

Env knobs (read here, recorded by the harness):
  FOCR_REF_TEXT_DIR   — when set, decoded text is written as `<page-stem>.md`
                        (feeds the correctness-proof compare);
  FOCR_REF_QUESTION   — SmolVLM2 only: override the describe question.

Self-test (no torch; wiring + prompt constants only):
    gauntlet_ref_zoo.py --self-test
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from gauntlet_ref_unlimited import (
    ReferenceEntryError,
    _q_len,
    _wrap_module_forward,
    stages_from_call_log,
)

# The C-wave describe default (src/native_engine/smolvlm2.rs DEFAULT_QUESTION —
# the SmolVLM2 model-card caption prompt focr uses when --question is absent).
SMOLVLM2_DEFAULT_QUESTION = "Can you describe this image?"


def _require(page: str, model_dir: str) -> None:
    if not page or not os.path.isfile(page):
        raise ReferenceEntryError(f"page fixture missing: {page!r}")
    if not model_dir or not os.path.isdir(model_dir):
        raise ReferenceEntryError(f"model dir missing: {model_dir!r}")


def _hook_lane(model, vision_names: list[str]):
    """Install the CausalLM call-log timer + summed vision timers.

    Returns `(timers, vision_hooked)`. Vision submodules are resolved on
    `model.get_model()` when present (the LLaVA-style inner model GOT and
    OneChart use), else on `model.model` (SmolVLM2's HF-native layout).
    """
    timers: dict = {"calls": [], "vision_ns": 0}

    def on_lm_call(t0: int, dur: int, args, kwargs) -> None:
        timers["calls"].append((_q_len(args, kwargs), t0, dur))

    def on_vision_call(_t0: int, dur: int, _args, _kwargs) -> None:
        timers["vision_ns"] += dur

    _wrap_module_forward(model, on_lm_call)
    inner = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", None)
    vision_hooked = False
    for name in vision_names:
        sub = getattr(inner, name, None) if inner is not None else None
        if sub is not None:
            _wrap_module_forward(sub, on_vision_call)
            vision_hooked = True
    if not vision_hooked:
        raise ReferenceEntryError(
            f"none of the vision submodules {vision_names} resolved — wrong model layout?"
        )
    return timers, vision_hooked


def _finish(state, t0_ns: int, total_ns: int, text: str, page: str) -> dict:
    stages = stages_from_call_log(
        t0_ns,
        total_ns,
        list(state["timers"]["calls"]),
        state["timers"]["vision_ns"],
        state["vision_hooked"],
    )
    text = text if isinstance(text, str) else str(text)
    text_dir = os.environ.get("FOCR_REF_TEXT_DIR") or None
    if text_dir:
        os.makedirs(text_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(page))[0]
        with open(os.path.join(text_dir, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(text)
    return {
        "stages": stages,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def _reset(state) -> None:
    state["timers"]["calls"].clear()
    state["timers"]["vision_ns"] = 0


# ── GOT-OCR2 ─────────────────────────────────────────────────────────────────


def setup_got(stage: str, page: str, model_dir: str):
    del stage
    _require(page, model_dir)
    # GOT's upstream chat() hardcodes `.cuda()`; the truth-pack baseline's CPU
    # monkeypatches (Tensor.cuda→identity, autocast("cuda")→nullcontext)
    # neutralize that without editing the pinned modeling code.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
    import run_baidu_reference as baseline

    baseline.install_cpu_patches()
    import torch

    # chat() also hardcodes `.half()` on the image tensor; on this f32 CPU
    # reference that would feed Half activations into float weights. Identity
    # it, same spirit as the cuda patch (the model stays f32 end to end).
    torch.Tensor.half = lambda self, *a, **k: self  # type: ignore[method-assign]
    from transformers import AutoModel, AutoTokenizer

    # config.json auto_map registers GOTQwenForCausalLM under AutoModel.
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        .eval()
    )
    timers, hooked = _hook_lane(model, ["vision_tower_high", "mm_projector_vary"])
    return {"torch": torch, "model": model, "tokenizer": tok, "timers": timers, "vision_hooked": hooked}


def run_got(stage: str, page: str, model_dir: str, state) -> dict:
    del stage, model_dir
    if state is None:
        raise ReferenceEntryError("setup state missing — pass --setup gauntlet_ref_zoo:setup_got")
    _reset(state)
    torch = state["torch"]
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        text = state["model"].chat(state["tokenizer"], page, ocr_type="ocr")
    return _finish(state, t0, time.perf_counter_ns() - t0, text, page)


# ── SmolVLM2-500M (describe) ─────────────────────────────────────────────────


def setup_smolvlm2(stage: str, page: str, model_dir: str):
    del stage
    _require(page, model_dir)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    timers, hooked = _hook_lane(model, ["vision_model", "connector"])
    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "timers": timers,
        "vision_hooked": hooked,
        "question": os.environ.get("FOCR_REF_QUESTION", SMOLVLM2_DEFAULT_QUESTION),
    }


def run_smolvlm2(stage: str, page: str, model_dir: str, state) -> dict:
    del stage, model_dir
    if state is None:
        raise ReferenceEntryError(
            "setup state missing — pass --setup gauntlet_ref_zoo:setup_smolvlm2"
        )
    _reset(state)
    torch = state["torch"]
    processor = state["processor"]
    from PIL import Image

    t0 = time.perf_counter_ns()
    pil = Image.open(page).convert("RGB")
    chat = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": state["question"]}],
        }
    ]
    rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=rendered, images=[pil], return_tensors="pt")
    with torch.inference_mode():
        gen = state["model"].generate(**inputs, do_sample=False, max_new_tokens=8192)
    total_ns = time.perf_counter_ns() - t0
    new_ids = gen[0][inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _finish(state, t0, total_ns, text, page)


# ── OneChart (chart→dict + number head) ──────────────────────────────────────


def setup_onechart(stage: str, page: str, model_dir: str):
    del stage
    _require(page, model_dir)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=True)
        .float()
        .eval()
    )
    timers, hooked = _hook_lane(model, ["vision_tower", "mm_projector"])
    return {"torch": torch, "model": model, "tokenizer": tok, "timers": timers, "vision_hooked": hooked}


def run_onechart(stage: str, page: str, model_dir: str, state) -> dict:
    del stage, model_dir
    if state is None:
        raise ReferenceEntryError(
            "setup state missing — pass --setup gauntlet_ref_zoo:setup_onechart"
        )
    _reset(state)
    torch = state["torch"]
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        text = state["model"].chat(
            state["tokenizer"], page, reliable_check=True, print_prompt=False
        )
    return _finish(state, t0, time.perf_counter_ns() - t0, text, page)


# ── Polyphonic-TrOMR (music) — bd-2sez ──────────────────────────────────────
#
# NOT an HF CausalLM: TrOMR is the upstream custom arch (hybrid-CNN/ViT
# encoder + 4-head transformer decoder with its own generate()), so this lane
# builds the ledger stages DIRECTLY instead of via stages_from_call_log
# (there is no text prefill — the decoder seeds from BOS against the encoder
# memory; the record omits the `prefill` stage and says so).


def setup_tromr(stage: str, page: str, model_dir: str):
    """`model_dir` = the tromr-upstream checkout root (the pinned repo)."""
    del stage
    _require(page, model_dir)
    import torch

    sys.path.insert(0, os.path.join(model_dir, "tromr"))
    from configs import getconfig  # noqa: PLC0415 — upstream module
    from model.tromr_arch import TrOMR  # noqa: PLC0415

    conf = getconfig(os.path.join(model_dir, "tromr", "workspace", "config.yaml"))
    model = TrOMR(conf)
    ckpt = os.path.join(model_dir, "tromr", "workspace", "checkpoints", "img2score_epoch47.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.train(False)

    timers = {"encoder_ns": 0}

    def on_encoder(_t0: int, dur: int, _args, _kwargs) -> None:
        timers["encoder_ns"] += dur

    _wrap_module_forward(model.encoder, on_encoder)
    # focr's music lane decodes ARGMAX by default (DISC-007 / E-wave: argmax ==
    # top-k/T0.2 sampling on real staves); force the same on the reference so
    # the timed decode path and the output stream are comparable.
    torch.multinomial = lambda probs, n, **kw: probs.argmax(-1, keepdim=True)
    return {"torch": torch, "model": model, "timers": timers}


def run_tromr(stage: str, page: str, model_dir: str, state) -> dict:
    del stage, model_dir
    if state is None:
        raise ReferenceEntryError("setup state missing — pass --setup gauntlet_ref_zoo:setup_tromr")
    import cv2
    import numpy as np
    from gen_reference_fixtures_tromr import readimg  # the byte-faithful L0 preprocess (DISC-007 rule)

    torch = state["torch"]
    state["timers"]["encoder_ns"] = 0
    t0 = time.perf_counter_ns()
    x = readimg(cv2, np, page)
    xt = torch.from_numpy(x).unsqueeze(0)  # (1, 1, 128, W)
    pre_ns = time.perf_counter_ns() - t0
    g0 = time.perf_counter_ns()
    with torch.inference_mode():
        rhythm, pitch, lift = state["model"].generate(xt, temperature=0.2)
    gen_ns = time.perf_counter_ns() - g0
    total_ns = time.perf_counter_ns() - t0
    enc_ns = state["timers"]["encoder_ns"]
    if enc_ns <= 0:
        raise ReferenceEntryError("encoder hook recorded 0 ns — generate() never ran the encoder")
    if enc_ns > gen_ns:
        raise ReferenceEntryError("encoder span exceeds generate() wall — nesting broken")
    tokens = int(rhythm.shape[-1])
    if tokens <= 1:
        raise ReferenceEntryError("degenerate generate: <= 1 token")
    ms = 1e-6
    stages = {
        "preprocess": {
            "ms": pre_ns * ms,
            "note": "upstream readimg (cv2 load + 128-height resize + ToGray + normalize; DISC-007 ink rule)",
        },
        "vision_encode": {
            "ms": enc_ns * ms,
            "note": "TrOMR hybrid-CNN/ViT encoder forward (hooked; runs inside generate())",
        },
        "decode_per_token": {
            "ms": (gen_ns - enc_ns) * ms,
            "tokens": tokens,
            "note": "generate() minus the encoder span; multinomial argmax-forced (matches focr's default per DISC-007); NO prefill stage — the decoder seeds from BOS",
        },
        "end_to_end": {
            "ms": total_ns * ms,
            "note": "readimg -> generate wall; EXCLUDES model load (done unclocked in setup)",
        },
    }
    # The comparable output = the three token streams (focr's music lane is
    # token-stream certified vs this oracle, so the stream IS the text).
    text = json.dumps({
        "rhythm": [int(v) for v in rhythm[0].tolist()],
        "pitch": [int(v) for v in pitch[0].tolist()],
        "lift": [int(v) for v in lift[0].tolist()],
    })
    return {
        "stages": stages,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": tokens,
    }


# ── self-test (wiring only; no torch, no model) ──────────────────────────────


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail"}))
        if not ok:
            failures.append(name)

    lanes = {
        "got": (setup_got, run_got),
        "smolvlm2": (setup_smolvlm2, run_smolvlm2),
        "onechart": (setup_onechart, run_onechart),
        "tromr": (setup_tromr, run_tromr),
    }
    for lane, (setup_fn, run_fn) in lanes.items():
        check(f"{lane}-callables", callable(setup_fn) and callable(run_fn))
        # A missing fixture must refuse (never invent a run).
        try:
            setup_fn("all", "/nonexistent/page.png", "/nonexistent/dir")
            check(f"{lane}-refuses-missing-fixture", False)
        except ReferenceEntryError:
            check(f"{lane}-refuses-missing-fixture", True)
        # A missing setup state must refuse.
        try:
            run_fn("all", "p", "d", None)
            check(f"{lane}-refuses-missing-state", False)
        except ReferenceEntryError:
            check(f"{lane}-refuses-missing-state", True)
    check("shared-decomposition-imported", callable(stages_from_call_log))
    check("smolvlm2-default-question", SMOLVLM2_DEFAULT_QUESTION.endswith("?"))

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-ref-zoo-self-test", "result": "pass"}))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
    raise SystemExit(2)
