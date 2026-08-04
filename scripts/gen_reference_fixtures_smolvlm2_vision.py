#!/usr/bin/env python3
"""SmolVLM2-500M torch reference-oracle VISION fixtures (C3/C4 + the C7/C8
anchors; beads bd-3jo6.3.3 / bd-3jo6.3.4 / bd-3jo6.1.9). OFFLINE TOOLING ONLY —
franken_ocr's Rust engine never runs this; it produces the fixtures the
env-gated parity certs compare against, extending the C5 text-only seam script
(scripts/gen_reference_fixtures_smolvlm2.py) to the image path.

Doctrine (AGENTS.md §Testing): establish the oracle's OWN nondeterminism floor
FIRST (two vision forwards @1 thread, one @2 threads); every downstream
tolerance derives from the measured floor — never import a tolerance.

SEAM DESIGN — decouple the towers exactly like bd-3s7v did for Baidu:
  * L0b  preprocess: the SmolVLM resize→512-ceil→split→normalize pipeline on
         the committed sample_photo.png → pixel_values [F,3,512,512] dumped
         raw (f32-LE) so encoder parity never depends on resize parity
         (OQ-2/OQ-3 stay their own measured rung).
  * L2v  SigLIP seams: embeddings out (patch conv + learned pos), each of the
         12 encoder layer outputs, post_layernorm out — stats committed,
         frame-0 tensors in the off-repo npz, post_ln for ALL frames as .bin.
  * L1ps pixel-shuffle: a weights-free synthetic exact case (inline values,
         transcribed permute sequence CROSS-CHECKED against the model's own
         connector.pixel_shuffle) + the real-shape sha256 pin — this is the
         A9 parity artifact (spec §3 IS the A9 spec).
  * C4   connector: pixel_shuffle out + modality_projection out on the real
         weights (.bin, all frames).
  * L0c  the exact rendered describe prompt (placeholder + expanded ids) —
         pins C7's prompt builder, including the OQ-4 newline merges.
  * L4v  greedy describe decode (max_new_tokens=64, eos 49279) — the C8
         end-to-end anchor.

Env (the C5 venv):
  uv venv --python 3.13 /private/tmp/smolvlm2_oracle_venv
  uv pip install --python /private/tmp/smolvlm2_oracle_venv/bin/python \
      torch 'transformers>=4.50,<5' pillow accelerate 'numpy<2.3' num2words
  FOCR_SMOLVLM2_DIR=/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2 \
      /private/tmp/smolvlm2_oracle_venv/bin/python \
      scripts/gen_reference_fixtures_smolvlm2_vision.py

Outputs (committed json is small; raw blobs land OFF-repo beside the model):
  tests/fixtures/smolvlm2/vision_oracle_fixtures.json
  $FOCR_SMOLVLM2_DIR/smolvlm2_pixel_values.bin       ([F,3,512,512] f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_vision_post_ln.bin     ([F,1024,768]  f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_pixel_shuffle_out.bin  ([F,64,12288]  f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_connector_out.bin      ([F,64,960]    f32-LE)
  $FOCR_SMOLVLM2_DIR/smolvlm2_vision_tensors.npz     (frame-0 per-layer seams)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "tests" / "fixtures" / "smolvlm2" / "vision_oracle_fixtures.json"
SAMPLE_PHOTO = REPO / "tests" / "fixtures" / "smolvlm2" / "sample_photo.png"
EOS_ID = 49279  # <end_of_utterance>
MAX_NEW_TOKENS = 64
QUESTION = "Can you describe this image?"
SCALE = 4  # pixel_shuffle scale_factor (config.scale_factor)


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


def pixel_shuffle_reference(x: np.ndarray, s: int) -> np.ndarray:
    """The EXACT SmolVLMConnector.pixel_shuffle permute sequence (spec §3),
    transcribed from modeling_smolvlm.py for the weights-free synthetic case.
    x: [seq, d] for one frame, seq = h*w with h == w."""
    seq, d = x.shape
    h = w = int(round(seq**0.5))
    assert h * w == seq, f"non-square token grid: {seq}"
    y = x.reshape(h, w // s, d * s)
    y = np.transpose(y, (1, 0, 2))  # [w/s, h, d*s]
    y = y.reshape(w // s, h // s, d * s * s)
    y = np.transpose(y, (1, 0, 2))  # [h/s, w/s, d*s*s]
    return y.reshape(seq // (s * s), d * s * s)


def main() -> int:
    import torch
    import transformers
    from PIL import Image
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
    if not SAMPLE_PHOTO.is_file():
        print(f"FATAL: committed fixture image missing: {SAMPLE_PHOTO}", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    assert int(model.config.scale_factor) == SCALE, model.config.scale_factor
    connector = model.model.connector
    vision = model.model.vision_model

    # ── L0c: the exact rendered describe prompt + expanded ids ──────────────
    pil = Image.open(SAMPLE_PHOTO).convert("RGB")
    chat = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
        }
    ]
    rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=rendered, images=[pil], return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_ids = [int(t) for t in input_ids[0].tolist()]
    pv = inputs["pixel_values"]  # [1, F, 3, 512, 512]
    assert pv.ndim == 5 and pv.shape[0] == 1, pv.shape
    pv_frames = pv[0].contiguous()  # [F, 3, 512, 512]
    n_frames = int(pv_frames.shape[0])
    img_sha = hashlib.sha256(SAMPLE_PHOTO.read_bytes()).hexdigest()

    # ── nondeterminism floor FIRST (vision tower; 2 @1 thread, 1 @2) ────────
    def run_vision():
        with torch.inference_mode():
            out = vision(pixel_values=pv_frames, output_hidden_states=True)
        return (
            [h.float().numpy() for h in out.hidden_states],
            out.last_hidden_state.float().numpy(),
        )

    torch.set_num_threads(1)
    vh1, pl1 = run_vision()
    vh2, pl2 = run_vision()
    torch.set_num_threads(2)
    vh3, pl3 = run_vision()
    torch.set_num_threads(1)
    vis_floor_same = max(max(maxabs(a, b) for a, b in zip(vh1, vh2)), maxabs(pl1, pl2))
    vis_floor_threads = max(max(maxabs(a, b) for a, b in zip(vh1, vh3)), maxabs(pl1, pl3))

    hiddens, post_ln = vh1, pl1  # hiddens[0] = embeddings out, [1..12] = layers

    # ── C4: pixel-shuffle + connector on the real weights ───────────────────
    with torch.inference_mode():
        ps_out = connector.pixel_shuffle(torch.from_numpy(post_ln), SCALE).float().numpy()
        conn_out = connector(torch.from_numpy(post_ln)).float().numpy()

    # ── L1ps: weights-free synthetic pixel-shuffle exact cases ──────────────
    # Small case (inline values): one frame, 8x8 grid, d=3, s=4 → [4, 48].
    small_in = (np.arange(64 * 3, dtype=np.float32).reshape(64, 3) * 0.25) - 20.0
    small_out = pixel_shuffle_reference(small_in, SCALE)
    with torch.inference_mode():
        small_out_model = (
            connector.pixel_shuffle(torch.from_numpy(small_in[None, ...]), SCALE)[0]
            .float()
            .numpy()
        )
    drift = maxabs(small_out, small_out_model)
    if drift != 0.0:
        print(f"FATAL: transcribed pixel_shuffle drifts from the model's ({drift})", file=sys.stderr)
        return 1
    # Real-shape case (sha pin): [1024, 768] deterministic input → [64, 12288].
    big_in = ((np.arange(1024 * 768, dtype=np.int64) % 17).astype(np.float32) - 8.0).reshape(
        1024, 768
    ) * 0.125
    big_out = pixel_shuffle_reference(big_in, SCALE)
    with torch.inference_mode():
        big_out_model = (
            connector.pixel_shuffle(torch.from_numpy(big_in[None, ...]), SCALE)[0].float().numpy()
        )
    if maxabs(big_out, big_out_model) != 0.0:
        print("FATAL: real-shape transcription drift", file=sys.stderr)
        return 1

    # ── L4v: greedy describe decode (the C8 anchor) + logits floor ──────────
    def full_last_logits():
        with torch.inference_mode():
            return model(**inputs).logits[0, -1].float().numpy()

    fl1 = full_last_logits()
    fl2 = full_last_logits()
    torch.set_num_threads(2)
    fl3 = full_last_logits()
    torch.set_num_threads(1)

    with torch.inference_mode():
        gen = model.generate(
            **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True
        )
    new_ids = [int(t) for t in gen[0][input_ids.shape[1] :].tolist()]

    # Per-step top-2 logits along the ORACLE's own greedy path — the near-tie
    # ledger: a reimplementation with a different f32 summation order can flip
    # a step only where gap = logit#1 - logit#2 is comparable to the numeric
    # drift; the Rust cert asserts every divergence step IS such a near-tie.
    # KV-cached replay: vision + prefill run ONCE, then one cached step/token.
    step_top2 = []
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        logits = out.logits[0, -1]
        past = out.past_key_values
        for tok in new_ids:
            top = torch.topk(logits, 2)
            step_top2.append(
                {
                    "top1": [int(top.indices[0]), float(top.values[0])],
                    "top2": [int(top.indices[1]), float(top.values[1])],
                    "gap": float(top.values[0] - top.values[1]),
                }
            )
            assert int(top.indices[0]) == tok, "greedy replay drifted from generate()"
            out = model(
                input_ids=torch.tensor([[tok]]), past_key_values=past, use_cache=True
            )
            logits = out.logits[0, -1]
            past = out.past_key_values
    decoded = processor.tokenizer.decode(
        [t for t in new_ids if t != EOS_ID], skip_special_tokens=True
    )
    if len(set(new_ids)) < 2 and len(new_ids) > 3:
        print(f"FATAL (OQ-5): degenerate describe greedy {new_ids}", file=sys.stderr)
        return 1
    print(f"describe greedy ({len(new_ids)} ids): {decoded!r}", file=sys.stderr)

    fixtures = {
        "_meta": {
            "purpose": "SmolVLM2-500M reference-oracle VISION fixtures "
            "(C3 SigLIP seams / C4 pixel-shuffle+connector / C7 prompt / C8 greedy anchor)",
            "script": "scripts/gen_reference_fixtures_smolvlm2_vision.py",
            "model_dir": model_dir,
            "transformers": ver,
            "torch": torch.__version__,
            "sample_photo": "tests/fixtures/smolvlm2/sample_photo.png",
            "sample_photo_sha256": img_sha,
            "question": QUESTION,
            "eos_id": EOS_ID,
            "scale_factor": SCALE,
            "trust_remote_code": False,
        },
        "nondeterminism_floor": {
            "vision_maxabs_same_thread": vis_floor_same,
            "vision_maxabs_cross_thread": vis_floor_threads,
            "describe_logit_maxabs_same_thread": maxabs(fl1, fl2),
            "describe_logit_maxabs_cross_thread": maxabs(fl1, fl3),
        },
        "l0b_preprocess": {
            "n_frames": n_frames,
            "note": "R*C 512^2 tiles row-major then the global frame LAST "
            "(spec §6); pixel_values dumped raw so encoder parity is "
            "independent of resize parity (OQ-2/OQ-3)",
            "pixel_values": stats(pv_frames.numpy()),
            "per_frame_mean": [float(pv_frames[i].mean()) for i in range(n_frames)],
        },
        "l1_pixel_shuffle": {
            "note": "weights-free exact cases; transcription cross-checked "
            "bit-exact against SmolVLMConnector.pixel_shuffle",
            "small": {
                "grid": 8,
                "d": 3,
                "s": SCALE,
                "input": [[float(v) for v in row] for row in small_in],
                "output": [[float(v) for v in row] for row in small_out],
            },
            "real_shape": {
                "input_spec": "((arange(1024*768) % 17) - 8) * 0.125 as f32 row-major [1024,768]",
                "input_sha256_f32": sha256_f32(big_in),
                "output": stats(big_out),
            },
        },
        "l2_vision_seams": {
            "embeddings_out": stats(hiddens[0]),
            "layers": [stats(h) for h in hiddens[1:]],
            "post_layernorm_out": stats(post_ln),
            "pixel_shuffle_out": stats(ps_out),
            "connector_out": stats(conn_out),
        },
        "l0c_describe_prompt": {
            "rendered_with_placeholder": rendered,
            "ids": prompt_ids,
            "n": len(prompt_ids),
            "n_image_slots": prompt_ids.count(49190),
        },
        "l4_describe_greedy": {"ids": new_ids, "text": decoded, "step_top2": step_top2},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_dir = Path(model_dir)
    np.ascontiguousarray(pv_frames.numpy(), dtype=np.float32).tofile(
        out_dir / "smolvlm2_pixel_values.bin"
    )
    np.ascontiguousarray(post_ln, dtype=np.float32).tofile(out_dir / "smolvlm2_vision_post_ln.bin")
    np.ascontiguousarray(ps_out, dtype=np.float32).tofile(
        out_dir / "smolvlm2_pixel_shuffle_out.bin"
    )
    np.ascontiguousarray(conn_out, dtype=np.float32).tofile(out_dir / "smolvlm2_connector_out.bin")
    np.savez_compressed(
        out_dir / "smolvlm2_vision_tensors.npz",
        **{f"vision_hidden_{i}_frame0": h[0] for i, h in enumerate(hiddens)},
        post_ln_frame0=post_ln[0],
        pixel_shuffle_out_frame0=ps_out[0],
        connector_out_frame0=conn_out[0],
    )
    print(
        f"floor: vision same={vis_floor_same:.2e} cross={vis_floor_threads:.2e}",
        file=sys.stderr,
    )
    print(f"wrote {OUT_JSON} + 5 blobs under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
