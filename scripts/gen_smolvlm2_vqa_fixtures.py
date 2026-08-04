#!/usr/bin/env python3
"""SmolVLM2 VQA quality fixtures (C8's INFORMATIONAL signal, bd-3jo6.3.8).

OFFLINE TOOLING ONLY. Runs the torch oracle's greedy answers for a small VQA
set over the committed sample photo. Per the C8 bead: the PARITY GATE is the
L0-L5 ladder (certified elsewhere); this set is a downstream INFORMATIONAL
regression signal with a guard — our answers are scored against the ORACLE's
own greedy output (parity, not benchmark; spec §13 L5), by normalized
exact-match with a keyword-containment fallback.

Env (the C5/C7 venv):
  FOCR_SMOLVLM2_DIR=/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2 \
      /private/tmp/smolvlm2_oracle_venv/bin/python \
      scripts/gen_smolvlm2_vqa_fixtures.py

Output: tests/fixtures/smolvlm2/vqa_fixtures.json (committed; answers are
short so the whole file stays small).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "tests" / "fixtures" / "smolvlm2" / "vqa_fixtures.json"
SAMPLE_PHOTO = REPO / "tests" / "fixtures" / "smolvlm2" / "sample_photo.png"
EOS_ID = 49279
MAX_NEW = 24

# A mix of yes/no, color, object, and scene questions about the committed
# synthetic cityscape (sun upper-right, blue gradient sky, 4 buildings with
# lit/dark windows, a tree, a road).
QUESTIONS = [
    "What color is the sky?",
    "Is there a sun in the image?",
    "Are there any buildings in the image?",
    "Is there a tree in the image?",
    "What time of day does it appear to be?",
    "Is this a photo of the ocean?",
    "What kind of scene is shown in this image?",
]


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_dir = os.environ.get(
        "FOCR_SMOLVLM2_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2"
    )
    if not Path(model_dir, "model.safetensors").exists():
        print(f"FATAL: no model.safetensors under {model_dir}", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    torch.set_num_threads(max(os.cpu_count() // 2, 1))

    pil = Image.open(SAMPLE_PHOTO).convert("RGB")
    cases = []
    for q in QUESTIONS:
        chat = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}
        ]
        rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(text=rendered, images=[pil], return_tensors="pt")
        with torch.inference_mode():
            gen = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW, use_cache=True
            )
        new_ids = [int(t) for t in gen[0][inputs["input_ids"].shape[1] :].tolist()]
        answer = processor.tokenizer.decode(
            [t for t in new_ids if t != EOS_ID], skip_special_tokens=True
        ).strip()
        print(f"  {q!r} -> {answer!r}", file=sys.stderr)
        cases.append({"question": q, "ids": new_ids, "answer": answer})

    OUT_JSON.write_text(
        json.dumps(
            {
                "_meta": {
                    "purpose": "SmolVLM2 VQA INFORMATIONAL quality fixtures (C8, spec §13 L5) — "
                    "oracle greedy answers over the committed sample photo",
                    "script": "scripts/gen_smolvlm2_vqa_fixtures.py",
                    "model_dir": model_dir,
                    "sample_photo": "tests/fixtures/smolvlm2/sample_photo.png",
                    "max_new_tokens": MAX_NEW,
                    "eos_id": EOS_ID,
                    "scoring": "normalized exact-match, else content-word containment >= 0.5 "
                    "(vs the ORACLE's own greedy output — parity, not benchmark)",
                },
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
