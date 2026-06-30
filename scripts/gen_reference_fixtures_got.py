#!/usr/bin/env python3
"""GOT-OCR2.0 reference-oracle fixtures (parity ladder L0b/L0c/L2/L3/L4 + the
nondeterminism floor). OFFLINE TOOLING ONLY — franken_ocr's Rust engine never
runs this; it certifies the Rust forward (B2/B3/B5/B7) against the upstream
`GOTQwenForCausalLM` (transformers trust_remote_code).

Doctrine (AGENTS.md §Testing): establish the oracle's OWN nondeterminism floor
FIRST (run the same forward twice + at two thread counts) and derive tolerances
from the measured floor — never import a tolerance. The token-id gate (L0a) is
already green in Rust (`scripts/gen_got_token_id_fixtures.py`); this adds the
image/prompt/hidden/logit seams.

Environment (an isolated venv — NOT the system python, NOT the unlimited-ocr
oracle which pins a different transformers):
    uv venv --python 3.13 /private/tmp/got_oracle_venv
    uv pip install --python /private/tmp/got_oracle_venv/bin/python \
        torch 'transformers>=4.44,<4.46' tiktoken pillow accelerate 'numpy<2.3'
    FOCR_GOT_DIR=/path/to/got-ocr2 \
        /private/tmp/got_oracle_venv/bin/python scripts/gen_reference_fixtures_got.py

Writes:
  * tests/fixtures/got/sample_text.png        — the SHARED input image (committed;
        both this oracle and franken_ocr load these exact bytes, so the contract
        is the file, not the renderer).
  * tests/fixtures/got/oracle_fixtures.json   — compact, committed: prompt ids
        (L0c), preprocess stats (L0b), per-layer hidden summaries (L2), final
        logits top-k (L3), greedy decode (L4), and the nondeterminism envelope.
  * <FOCR_GOT_DIR>/oracle_tensors.npz         — bulky, OFF-repo: full preprocessed
        image, all 25 hidden_states, final logits — for exhaustive seam compare
        while implementing B5.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("FOCR_GOT_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/got-ocr2"))
FIX_DIR = REPO_ROOT / "tests/fixtures/got"
IMG_PATH = FIX_DIR / "sample_text.png"
OUT_JSON = FIX_DIR / "oracle_fixtures.json"
OUT_NPZ = MODEL_DIR / "oracle_tensors.npz"

IMG_PAD_ID = 151859  # <imgpad>
IMAGE_TOKEN_LEN = 256


def sha256_f32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "l2": float(np.sqrt((a * a).sum())),
        "min": float(a.min()),
        "max": float(a.max()),
        "sha256_f32": sha256_f32(arr),
    }


def make_sample_image() -> Image.Image:
    """Render a fixed text image, or load the committed PNG (the contract is the
    committed bytes — rendering is only the first-run bootstrap)."""
    if IMG_PATH.is_file():
        return Image.open(IMG_PATH).convert("RGB")
    img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(img)
    lines = [
        "Hello GOT-OCR 2.0",
        "The quick brown fox jumps",
        "over the lazy dog.",
        "1234567890  +-=/%",
        "Invoice #A-4217  Total: $1,234.56",
    ]
    y = 80
    for ln in lines:
        d.text((80, y), ln, fill=(0, 0, 0))
        y += 90
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    img.save(IMG_PATH)
    return img


def build_prompt(got_mod) -> str:
    """Replicate GOTQwenForCausalLM.chat()'s plain-OCR prompt EXACTLY, using the
    module's own Conversation class + DEFAULT tokens + the source system string
    (regex-extracted so whitespace cannot drift)."""
    src = (MODEL_DIR / "modeling_GOT.py").read_text()
    systems = re.findall(r'system="""(.*?)"""', src, re.DOTALL)
    assert systems and len(set(systems)) == 1, "system literal drift"
    system = systems[0]

    qs = (
        got_mod.DEFAULT_IM_START_TOKEN
        + got_mod.DEFAULT_IMAGE_PATCH_TOKEN * IMAGE_TOKEN_LEN
        + got_mod.DEFAULT_IM_END_TOKEN
        + "\n"
        + "OCR: "
    )
    conv = got_mod.Conversation(
        system=system,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        version="mpt",
        messages=[],
        offset=0,
        sep_style=got_mod.SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_forward(model, input_ids, images):
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            images=images,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    return out


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoModel, AutoTokenizer

    print("loading tokenizer + model (trust_remote_code, float32, CPU)...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    # config.json auto_map registers GOTQwenForCausalLM under AutoModel (not the
    # CausalLM auto-class), so load via AutoModel.
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    got_mod = sys.modules[type(model).__module__]

    # ── inputs (shared image + plain-OCR prompt) ────────────────────────────
    pil = make_sample_image()
    processor = got_mod.GOTImageEvalProcessor(image_size=1024)
    image_tensor = processor(pil).float()  # [3,1024,1024]
    images = [image_tensor.unsqueeze(0)]   # [1,3,1024,1024], P=1 (no multi-crop)

    prompt = build_prompt(got_mod)
    enc = tok([prompt])
    input_ids = torch.as_tensor(enc.input_ids)  # [1, seqlen]
    n_imgpad = int((input_ids == IMG_PAD_ID).sum())
    assert n_imgpad == IMAGE_TOKEN_LEN, f"expected {IMAGE_TOKEN_LEN} <imgpad>, got {n_imgpad}"

    # ── nondeterminism floor FIRST (2 runs @1 thread, 1 run @2 threads) ─────
    torch.set_num_threads(1)
    o1 = run_forward(model, input_ids, images)
    o2 = run_forward(model, input_ids, images)
    torch.set_num_threads(2)
    o3 = run_forward(model, input_ids, images)
    torch.set_num_threads(1)

    def maxabs(a, b):
        return float((a - b).abs().max())

    logit_floor_same = maxabs(o1.logits, o2.logits)
    logit_floor_threads = maxabs(o1.logits, o3.logits)
    hid_floor_same = max(maxabs(a, b) for a, b in zip(o1.hidden_states, o2.hidden_states))
    hid_floor_threads = max(maxabs(a, b) for a, b in zip(o1.hidden_states, o3.hidden_states))

    out = o1
    logits = out.logits[0].float().numpy()          # [seqlen, vocab]
    hidden = [h[0].float().numpy() for h in out.hidden_states]  # 25 x [seqlen, 1024]
    last_logits = logits[-1]                         # next-token distribution

    # ── greedy decode (L4) — manual argmax loop, deterministic ──────────────
    gen_ids = []
    cur = input_ids
    cur_imgs = images
    for _ in range(24):
        go = run_forward(model, cur, cur_imgs)
        nxt = int(go.logits[0, -1].argmax())
        gen_ids.append(nxt)
        if nxt == 151645:  # <|im_end|>
            break
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
        cur_imgs = images  # P==1 path re-splices; cheap enough for 24 steps

    topk = torch.topk(torch.from_numpy(last_logits), 20)
    fixtures = {
        "_meta": {
            "purpose": "GOT-OCR2 reference-oracle fixtures (L0b/L0c/L2/L3/L4 + nondeterminism floor)",
            "model_id": "got-ocr2",
            "oracle": "GOTQwenForCausalLM via transformers trust_remote_code, float32, CPU",
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "image": "tests/fixtures/got/sample_text.png (committed; the shared input contract)",
            "image_sha256": hashlib.sha256(IMG_PATH.read_bytes()).hexdigest(),
            "prompt_mode": "plain OCR (use_im_start_end, MPT conv)",
            "seq_len": int(input_ids.shape[1]),
            "num_imgpad": n_imgpad,
        },
        "nondeterminism_floor": {
            "logit_maxabs_same_thread": logit_floor_same,
            "logit_maxabs_cross_thread": logit_floor_threads,
            "hidden_maxabs_same_thread": hid_floor_same,
            "hidden_maxabs_cross_thread": hid_floor_threads,
            "note": "L3 logit / L2 hidden tolerances derive from cross_thread; same_thread proves determinism",
        },
        "l0c_prompt_ids": enc.input_ids[0],
        "l0b_preprocess": stats(image_tensor.numpy()),
        "l3_logits": {
            **stats(last_logits),
            "argmax": int(last_logits.argmax()),
            "topk20": [[int(i), float(v)] for v, i in zip(topk.values.tolist(), topk.indices.tolist())],
        },
        "l2_hidden_states": [
            {"layer": i, **stats(h), "row0_head8": h[0, :8].tolist(), "rowlast_head8": h[-1, :8].tolist()}
            for i, h in enumerate(hidden)
        ],
        "l4_greedy_decode_ids": gen_ids,
        "l4_greedy_decode_text": tok.decode(gen_ids),
    }

    FIX_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        image=image_tensor.numpy(),
        logits=logits,
        **{f"hidden_{i}": h for i, h in enumerate(hidden)},
    )
    print(f"floor: logit same={logit_floor_same:.2e} cross={logit_floor_threads:.2e} "
          f"hidden same={hid_floor_same:.2e} cross={hid_floor_threads:.2e}", file=sys.stderr)
    print(f"L4 greedy: {gen_ids}\n  -> {fixtures['l4_greedy_decode_text']!r}", file=sys.stderr)
    print(f"wrote {OUT_JSON} + {OUT_NPZ}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
