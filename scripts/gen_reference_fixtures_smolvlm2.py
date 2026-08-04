#!/usr/bin/env python3
"""SmolVLM2-500M torch reference-oracle fixtures (C5, bd-3jo6.3.5 — incl the
nondeterminism floor). OFFLINE TOOLING ONLY — franken_ocr's Rust engine never
runs this; it produces the fixtures the env-gated parity certs compare against,
exactly mirroring scripts/gen_reference_fixtures_got.py (bead B5's pattern).

Doctrine (AGENTS.md §Testing): establish the oracle's OWN nondeterminism floor
FIRST (two runs @1 thread, one @2 threads); every downstream tolerance derives
from the measured floor — never import a tolerance.

SEAM DESIGN — certify C5 decoder-only, exactly like B5 shipped before B3: a
TEXT-ONLY chat prompt (no vision anywhere), so the seam needs ONLY the C2
`.focrq` artifact plus the decoder deltas. `hidden_states[0]` is the pure
embed_tokens lookup for the prompt ids (so the Rust cert transitively covers
our embed table), `hidden_states[1..33]` the 32 layer outputs, plus last-pos
logits and a manual greedy L4 (upstream has NO repetition guard; eos 49279
`<end_of_utterance>`). The image/describe seams for C3/C4/C8 land with C3.

OQ-5 GUARD (version skew): the checkpoint was saved with transformers 4.47.1
(the pre-merge fork); this oracle runs the IN-TREE port (>=4.50, NO
trust_remote_code). Before emitting any fixture the script reproduces a
greedy sanity decode and asserts it is non-degenerate; if the in-tree port
drifts, pin the exact reproducing version before trusting fixtures.

Env (mirrors the GOT recipe; isolated venv, NOT the GOT/unlimited venvs):
  uv venv --python 3.13 /private/tmp/smolvlm2_oracle_venv
  uv pip install --python /private/tmp/smolvlm2_oracle_venv/bin/python \
      torch 'transformers>=4.50,<5' pillow accelerate 'numpy<2.3' num2words
  FOCR_SMOLVLM2_DIR=/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2 \
      /private/tmp/smolvlm2_oracle_venv/bin/python \
      scripts/gen_reference_fixtures_smolvlm2.py

Outputs (committed json is small; raw blobs land OFF-repo beside the model):
  tests/fixtures/smolvlm2/oracle_fixtures.json      (floor + prompt ids + L2
                                                     stats + L3 topk + L4 ids)
  $FOCR_SMOLVLM2_DIR/smolvlm2_decoder_input.bin     (hidden_0 [N,960] f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_final_logits.bin      (last-pos [49280] f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_oracle_tensors.npz    (all hiddens, exhaustive)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "tests" / "fixtures" / "smolvlm2" / "oracle_fixtures.json"
EOS_ID = 49279  # <end_of_utterance>
MAX_GREEDY_STEPS = 24
PROMPT_TEXT = "What is the capital of France?"


def sha256_f32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "l2": float(np.linalg.norm(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "sha256_f32": sha256_f32(arr),
    }


def maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ver = transformers.__version__
    major, minor = (int(x) for x in ver.split(".")[:2])
    if (major, minor) < (4, 50):
        print(f"FATAL: transformers {ver} < 4.50 (smolvlm is in-tree from 4.50)", file=sys.stderr)
        return 1

    model_dir = os.environ.get(
        "FOCR_SMOLVLM2_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2"
    )
    if not Path(model_dir, "model.safetensors").exists():
        print(f"FATAL: no model.safetensors under {model_dir}", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    # ── the TEXT-ONLY C5 seam prompt ────────────────────────────────────────
    chat = [{"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}]}]
    rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
    enc = processor.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    prompt_ids = [int(t) for t in input_ids[0].tolist()]

    def run_forward():
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        return (
            [h[0].float().numpy() for h in out.hidden_states],
            out.logits[0].float().numpy(),
        )

    # ── nondeterminism floor FIRST (2 runs @1 thread, 1 run @2 threads) ─────
    torch.set_num_threads(1)
    h1, l1 = run_forward()
    h2, l2 = run_forward()
    torch.set_num_threads(2)
    h3, l3 = run_forward()
    torch.set_num_threads(1)
    logit_floor_same = maxabs(l1, l2)
    logit_floor_threads = maxabs(l1, l3)
    hid_floor_same = max(maxabs(a, b) for a, b in zip(h1, h2))
    hid_floor_threads = max(maxabs(a, b) for a, b in zip(h1, h3))

    # ── OQ-5 sanity: greedy decode must be non-degenerate before we trust it ─
    ids = input_ids.clone()
    greedy: list[int] = []
    with torch.inference_mode():
        for _ in range(MAX_GREEDY_STEPS):
            logits = model(input_ids=ids).logits[0, -1]
            nxt = int(torch.argmax(logits).item())
            greedy.append(nxt)
            if nxt == EOS_ID:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    decoded = processor.tokenizer.decode([t for t in greedy if t != EOS_ID])
    if len(set(greedy)) < 2 and len(greedy) > 3:
        print(f"FATAL (OQ-5): degenerate greedy {greedy} — version-skew suspected", file=sys.stderr)
        return 1
    print(f"greedy sanity: {greedy} -> {decoded!r}", file=sys.stderr)

    hiddens, logits_all = h1, l1
    last_logits = logits_all[-1]
    topk = np.argsort(last_logits)[::-1][:20]

    fixtures = {
        "_meta": {
            "purpose": "SmolVLM2-500M reference-oracle fixtures (C5 text-only decoder seam: "
            "L0c/L2/L3/L4 + nondeterminism floor)",
            "script": "scripts/gen_reference_fixtures_smolvlm2.py",
            "model_dir": model_dir,
            "transformers": ver,
            "torch": torch.__version__,
            "prompt_text": PROMPT_TEXT,
            "eos_id": EOS_ID,
            "trust_remote_code": False,
        },
        "nondeterminism_floor": {
            "logit_maxabs_same_thread": logit_floor_same,
            "logit_maxabs_cross_thread": logit_floor_threads,
            "hidden_maxabs_same_thread": hid_floor_same,
            "hidden_maxabs_cross_thread": hid_floor_threads,
        },
        "l0c_prompt": {"rendered": rendered, "ids": prompt_ids, "n": len(prompt_ids)},
        "l2_hidden_states": [stats(h) for h in hiddens],
        "l3_logits": {
            "last_pos": stats(last_logits),
            "argmax": int(topk[0]),
            "topk20": [[int(i), float(last_logits[i])] for i in topk],
        },
        "l4_greedy": {"ids": greedy, "text": decoded},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_dir = Path(model_dir)
    np.ascontiguousarray(hiddens[0], dtype=np.float32).tofile(out_dir / "smolvlm2_decoder_input.bin")
    np.ascontiguousarray(last_logits, dtype=np.float32).tofile(out_dir / "smolvlm2_final_logits.bin")
    np.savez_compressed(
        out_dir / "smolvlm2_oracle_tensors.npz",
        **{f"hidden_{i}": h for i, h in enumerate(hiddens)},
        logits=logits_all,
    )
    print(
        f"floor: logit same={logit_floor_same:.2e} cross={logit_floor_threads:.2e} "
        f"hidden same={hid_floor_same:.2e} cross={hid_floor_threads:.2e}",
        file=sys.stderr,
    )
    print(f"wrote {OUT_JSON} + 3 blobs under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
