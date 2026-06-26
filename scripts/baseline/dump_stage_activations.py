#!/usr/bin/env python3
"""Dump baidu per-stage vision + decoder activations as .npy fixtures for parity.

Hooks the reference model's vision submodules and runs one page through infer()
(base mode), capturing each stage boundary so the native engine can be compared
stage-by-stage (cosine >= 0.9999):
  sam_in.npy        [1,3,1024,1024]  normalized image tensor SAM receives
  sam_out.npy       SAM ViT-B output (global_features_1, spatial)
  clip_out.npy      [1,257,1024]     CLIP-L output (before dropping CLS)
  projector_out.npy [1,256,1280]     hybrid concat -> projector (bridge output)

Decoder boundaries (DeepSeek-V2 MoE, 12 layers), captured on the PREFILL forward
(first invocation), decoupling decoder parity from prompt-scatter + KV-cache:
  inputs_embeds.npy   [seq,1280]   activation entering decoder layer 0, AFTER the
                                   vision-token masked_scatter (also raw LE f32
                                   `inputs_embeds.f32` for examples/decoder_dump.rs)
  decoder_prenorm.npy [seq,1280]   hidden after all 12 layers, BEFORE model.norm
                                   (apples-to-apples with franken decoder::forward)
  decoder_hidden.npy  [seq,1280]   hidden AFTER model.norm (the (b) deliverable)
  lm_head_logits_last.npy [129280] lm_head logits for the LAST prefill position
  first_token_id.txt               argmax(last logits) = baidu's first decode token

Feeding franken_ocr's vision_sam::forward the SAME sam_in.npy / decoder::forward
the SAME inputs_embeds.f32 decouples each stage's math from its inputs.

Usage: dump_stage_activations.py --model DIR --page PNG --out DIR
"""
import argparse
import contextlib
from pathlib import Path

import numpy as np


def install_cpu_patches():
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[attr-defined]
    _orig = torch.autocast

    class _Shim(contextlib.AbstractContextManager):
        def __init__(self, device_type, *a, **k):
            self._cm = contextlib.nullcontext() if device_type == "cuda" else _orig(device_type, *a, **k)

        def __enter__(self):
            return self._cm.__enter__()

        def __exit__(self, *e):
            return self._cm.__exit__(*e)

    torch.autocast = _Shim  # type: ignore[assignment]
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]


def to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, (tuple, list)):
        return to_np(x[0])
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=64,
                    help="generation cap; the prefill hooks fire on the first "
                         "forward, so a small value (e.g. 4) is enough for the "
                         "activation dump and stops CPU decode early")
    args = ap.parse_args()

    install_cpu_patches()
    import torch
    torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()

    inner = model.model  # UnlimitedOCRModel
    acts = {}

    def save_in(name):
        def fn(module, inp, out_):
            acts[name + "_in"] = to_np(inp[0])
            acts[name + "_out"] = to_np(out_)
        return fn

    def save_out(name):
        def fn(module, inp, out_):
            acts[name + "_out"] = to_np(out_)
        return fn

    h = []
    h.append(inner.sam_model.register_forward_hook(save_in("sam")))
    h.append(inner.vision_model.register_forward_hook(save_out("clip")))
    h.append(inner.projector.register_forward_hook(save_out("projector")))

    # --- Decoder hooks: capture ONLY the prefill forward (first invocation). ---
    dec = {}

    def layer0_pre_hook(module, hk_args, hk_kwargs):
        if "inputs_embeds" in dec:
            return
        hs = hk_kwargs.get("hidden_states", None) if hk_kwargs else None
        if hs is None and hk_args:
            hs = hk_args[0]
        dec["inputs_embeds"] = to_np(hs)  # [1, seq, 1280]

    def norm_hook(module, inp, out_):
        if "decoder_hidden" in dec:
            return
        dec["decoder_prenorm"] = to_np(inp[0])  # last-layer output (pre-norm)
        dec["decoder_hidden"] = to_np(out_)      # after model.norm

    def lm_head_hook(module, inp, out_):
        if "lm_head_logits" in dec:
            return
        dec["lm_head_logits"] = to_np(out_)      # [1, seq, vocab] (or [1,1,vocab])

    h.append(inner.layers[0].register_forward_pre_hook(layer0_pre_hook, with_kwargs=True))
    h.append(inner.norm.register_forward_hook(norm_hook))
    h.append(model.lm_head.register_forward_hook(lm_head_hook))

    # --- Capture the EXACT prompt id-stream + image mask infer() builds, and cap
    #     the CPU decode to a single step (the prefill hooks above fire on the
    #     first forward, so we never need the full autoregressive loop). ---
    prompt_cap = {}
    _orig_generate = model.generate

    def _wrapped_generate(*g_args, **g_kwargs):
        if "input_ids" not in prompt_cap:
            ii = g_kwargs.get("input_ids")
            ism = g_kwargs.get("images_seq_mask")
            if ii is not None:
                prompt_cap["input_ids"] = ii.detach().to("cpu").numpy()
            if ism is not None:
                prompt_cap["images_seq_mask"] = ism.detach().to("cpu").numpy()
        # Force exactly one new token so the prefill forward (and all hooks) run
        # but we don't pay for hundreds of no-cache CPU decode steps.
        g_kwargs.pop("max_length", None)
        g_kwargs["max_new_tokens"] = 1
        return _orig_generate(*g_args, **g_kwargs)

    model.generate = _wrapped_generate

    print(f"[dump] running {Path(args.page).name} (base mode) ...", flush=True)
    with torch.no_grad():
        text = model.infer(
            tok, prompt="<image>document parsing.", image_file=args.page,
            output_path=str(out / "_scratch"), base_size=1024, image_size=1024,
            crop_mode=False, eval_mode=True, max_length=args.max_length,
            no_repeat_ngram_size=35, ngram_window=1024, temperature=0.0,
        )
    for hh in h:
        hh.remove()

    for k, v in acts.items():
        np.save(out / f"{k}.npy", v)
        print(f"[dump] {k}: shape={v.shape} dtype={v.dtype} -> {k}.npy", flush=True)

    # --- Save the EXACT prompt id-stream + image mask (the scatter inputs). ---
    import json as _json
    if "input_ids" in prompt_cap:
        ii = prompt_cap["input_ids"]
        ii = ii[0] if ii.ndim == 2 else ii  # squeeze batch
        ii = ii.astype(np.int64)
        np.save(out / "input_ids.npy", ii)
        (out / "input_ids.json").write_text(_json.dumps([int(x) for x in ii.tolist()]))
        print(f"[dump] input_ids: shape={ii.shape} (image_token_id=128815 "
              f"count={int((ii == 128815).sum())}) -> input_ids.npy/.json", flush=True)
    if "images_seq_mask" in prompt_cap:
        ism = prompt_cap["images_seq_mask"]
        ism = ism[0] if ism.ndim == 2 else ism  # squeeze batch
        ism = ism.astype(bool)
        np.save(out / "images_seq_mask.npy", ism)
        (out / "images_seq_mask.json").write_text(_json.dumps([bool(x) for x in ism.tolist()]))
        print(f"[dump] images_seq_mask: shape={ism.shape} (True count={int(ism.sum())}) "
              f"-> images_seq_mask.npy/.json", flush=True)

    # --- Save the decoder prefill boundaries. ---
    def _squeeze_batch(x):
        return x[0] if (x.ndim == 3 and x.shape[0] == 1) else x

    if "inputs_embeds" in dec:
        ie = _squeeze_batch(dec["inputs_embeds"]).astype(np.float32)  # [seq, 1280]
        np.save(out / "inputs_embeds.npy", ie)
        ie.tofile(out / "inputs_embeds.f32")  # raw LE f32 for examples/decoder_dump.rs
        print(f"[dump] inputs_embeds: shape={ie.shape} -> inputs_embeds.npy/.f32", flush=True)
    if "decoder_prenorm" in dec:
        pn = _squeeze_batch(dec["decoder_prenorm"]).astype(np.float32)
        np.save(out / "decoder_prenorm.npy", pn)
        print(f"[dump] decoder_prenorm: shape={pn.shape} -> decoder_prenorm.npy", flush=True)
    if "decoder_hidden" in dec:
        dh = _squeeze_batch(dec["decoder_hidden"]).astype(np.float32)
        np.save(out / "decoder_hidden.npy", dh)
        print(f"[dump] decoder_hidden: shape={dh.shape} -> decoder_hidden.npy", flush=True)
    if "lm_head_logits" in dec:
        lg = _squeeze_batch(dec["lm_head_logits"]).astype(np.float32)  # [seq, vocab] or [1, vocab]
        last = lg[-1]  # [vocab]
        np.save(out / "lm_head_logits_last.npy", last)
        tok_id = int(np.argmax(last))
        (out / "first_token_id.txt").write_text(str(tok_id))
        print(f"[dump] lm_head_logits_last: shape={last.shape} argmax={tok_id} "
              f"(logit {float(last[tok_id]):.4f}) -> lm_head_logits_last.npy", flush=True)
        print(f"[dump] BAIDU_FIRST_TOKEN_ID {tok_id}", flush=True)

    print(f"[dump] decoded prefix: {text[:80]!r}", flush=True)
    print(f"[dump] wrote {len(acts)} vision + decoder activation fixtures to {out}", flush=True)


if __name__ == "__main__":
    main()
