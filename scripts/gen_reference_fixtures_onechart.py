#!/usr/bin/env python3
"""OneChart torch reference-oracle fixtures (D3/D4/D6 seams; beads bd-3jo6.4.3+).

OFFLINE TOOLING ONLY — mirrors gen_reference_fixtures_got.py (the B5 pattern):
establish the oracle's OWN nondeterminism floor FIRST, then dump the seams the
env-gated Rust certs compare against.

Seams (census docs/zoo/onechart-spec.md §14):
  * L0b  preprocess: bicubic 1024² squash, raw [0,1] CHW (NO CLIP normalize —
         the census §6 no-op Normalize((0,0,0),(1,1,1))).
  * L0c  the exact 309-id prompt (§5 conv_vicuna_v1_1 string, 256 <imgpad>).
  * L2   vision_tower out / mm_projector out / 12 decoder hiddens / final LN.
  * L3   last-pos prefill logits.
  * L4   chat() greedy: text + pred_locs (the number head's 100 floats) +
         the reliable_check verdict.

Env (OQ-D2: the GOT venv, transformers 4.45.2, loads the Vary-lineage code):
  FOCR_ONECHART_DIR=/Volumes/USBNVME16TB/temp_agent_space/zoo/onechart \
      /private/tmp/got_oracle_venv/bin/python \
      scripts/gen_reference_fixtures_onechart.py

Outputs: tests/fixtures/onechart/oracle_fixtures.json (committed, compact) +
$FOCR_ONECHART_DIR/onechart_{preproc,proj_out,final_logits}.bin +
onechart_oracle_tensors.npz (off-repo).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "tests" / "fixtures" / "onechart" / "oracle_fixtures.json"
CHART = REPO / "tests" / "fixtures" / "onechart" / "sample_chart.png"

PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions. USER: <img>" + "<imgpad>" * 256 + "</img>"
    "Convert the key information of the chart to a python dict:\n ASSISTANT:"
)


def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "l2": float(np.linalg.norm(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "sha256_f32": hashlib.sha256(
            np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main() -> int:
    import torch
    import transformers
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel, AutoTokenizer

    model_dir = os.environ.get(
        "FOCR_ONECHART_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/onechart"
    )
    if not Path(model_dir, "model.safetensors").exists():
        print(f"FATAL: no model.safetensors under {model_dir}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, low_cpu_mem_usage=True
        )
        .float()
        .eval()
    )

    # ── L0b preprocess (census §6: squash-bicubic 1024², raw [0,1]) ─────────
    tf = transforms.Compose(
        [
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )
    pil = Image.open(CHART).convert("RGB")
    image_tensor = tf(pil)  # [3,1024,1024] f32
    ids = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_ids = [int(t) for t in ids[0].tolist()]

    # ── seams via hooks on ONE prefill forward ──────────────────────────────
    captured: dict = {}

    def grab(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().float().numpy()

        return hook

    hooks = [
        model.model.vision_tower.register_forward_hook(grab("vision_out")),
        model.model.mm_projector.register_forward_hook(grab("proj_out")),
        model.model.decoder.final_layer_norm.register_forward_hook(grab("final_ln")),
    ]
    for i, layer in enumerate(model.model.decoder.layers):
        hooks.append(layer.register_forward_hook(grab(f"dec_{i}")))

    def prefill_logits():
        with torch.inference_mode():
            out = model(input_ids=ids, images=[image_tensor.unsqueeze(0)], use_cache=False)
        return out.logits[0, -1].float().numpy()

    torch.set_num_threads(1)
    l1 = prefill_logits()
    l2 = prefill_logits()
    torch.set_num_threads(2)
    l3 = prefill_logits()
    torch.set_num_threads(1)
    for h in hooks:
        h.remove()

    # ── L4: the public chat() (greedy; sets self.pred_locs) ─────────────────
    with torch.inference_mode():
        answer = model.chat(tokenizer, str(CHART), reliable_check=True, print_prompt=False)
    pred_locs = list(getattr(model, "pred_locs", []) or [])

    topk = np.argsort(l1)[::-1][:20]
    fixtures = {
        "_meta": {
            "purpose": "OneChart reference-oracle fixtures (D3 vision / D4 decoder seams + L4 chat)",
            "script": "scripts/gen_reference_fixtures_onechart.py",
            "model_dir": model_dir,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "sample_chart": "tests/fixtures/onechart/sample_chart.png",
            "sample_chart_sha256": hashlib.sha256(CHART.read_bytes()).hexdigest(),
            "trust_remote_code": True,
        },
        "nondeterminism_floor": {
            "logit_maxabs_same_thread": maxabs(l1, l2),
            "logit_maxabs_cross_thread": maxabs(l1, l3),
        },
        "l0b_preprocess": stats(image_tensor.numpy()),
        "l0c_prompt": {"ids": prompt_ids, "n": len(prompt_ids)},
        "l2_seams": {
            "vision_out": stats(captured["vision_out"]),
            "proj_out": stats(captured["proj_out"]),
            "decoder": [stats(captured[f"dec_{i}"]) for i in range(12)],
            "final_ln": stats(captured["final_ln"]),
        },
        "l3_logits": {
            "last_pos": stats(l1),
            "argmax": int(topk[0]),
            "topk20": [[int(i), float(l1[i])] for i in topk],
        },
        "l4_chat": {"answer": answer, "pred_locs": pred_locs},
    }
    OUT_JSON.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", "utf-8")

    out_dir = Path(model_dir)
    np.ascontiguousarray(image_tensor.numpy(), np.float32).tofile(out_dir / "onechart_preproc.bin")
    np.ascontiguousarray(captured["proj_out"], np.float32).tofile(out_dir / "onechart_proj_out.bin")
    np.ascontiguousarray(l1, np.float32).tofile(out_dir / "onechart_final_logits.bin")
    np.savez_compressed(
        out_dir / "onechart_oracle_tensors.npz",
        vision_out=captured["vision_out"],
        proj_out=captured["proj_out"],
        final_ln=captured["final_ln"],
        logits=l1,
        **{f"dec_{i}": captured[f"dec_{i}"] for i in range(12)},
    )
    print(f"floor same={maxabs(l1, l2):.2e} cross={maxabs(l1, l3):.2e}", file=sys.stderr)
    print(f"answer: {answer!r}", file=sys.stderr)
    print(f"pred_locs[:8]: {pred_locs[:8]}", file=sys.stderr)
    print(f"wrote {OUT_JSON} + 4 blobs under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
