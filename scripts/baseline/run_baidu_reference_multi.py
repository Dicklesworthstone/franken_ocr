#!/usr/bin/env python3
"""Generate the baidu/Unlimited-OCR MULTI-PAGE reference (bd-1gv.26).

Runs the pinned reference `infer_multi(...)` on CPU — the SAME monkeypatch port
as run_baidu_reference.py (Tensor.cuda -> identity, autocast("cuda") ->
nullcontext, bf16) — over an ordered page list, producing the cross-page oracle
franken_ocr's `recognize_multi_page` must match: ONE fused prefill over every
page's Base-640 block, ONE greedy decode with ngram_window=1024, `<PAGE>`
separators emitted by the model.

The RAW returned text (pre save_results post-processing) is the frozen oracle:
the Rust side applies its own `finalize_multi` to both sides for the compare.

Probe mode (DISC-004 evidence, never a fixture): `--dtype f32` runs the same
call in float32, the native subject's precision (an exact upcast of the bf16
shard; the hard-coded bf16 image cast becomes float32). `--max-new-tokens N`
caps the decode, and `--margins` writes each step's chosen token, runner-up and
logit gap (after the no-repeat processor) to `multi_margins.txt`. Together they
show whether a fork from the frozen bf16 text is a near-tie in the reference
model itself.

Usage:
  run_baidu_reference_multi.py --model DIR --out DIR page_0009.png page_0014.png \
      [--pages-dir DIR] [--max-length 32768] [--threads N] \
      [--dtype bf16|f32] [--max-new-tokens N] [--margins]
"""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baidu_reference import install_cpu_patches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pages-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=35)
    ap.add_argument("--ngram-window", type=int, default=1024)
    ap.add_argument("--prompt", default="<image>Multi page parsing.")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--margins", action="store_true")
    ap.add_argument("pages", nargs="+", help="ordered page image files")
    args = ap.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        # modeling_unlimitedocr.py imports matplotlib only inside its
        # save_results drawing branch, which this script never takes;
        # transformers' static import scan still demands the name.
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")

    install_cpu_patches()
    import torch

    if args.dtype == "f32":
        orig_to = torch.Tensor.to

        def to_no_bf16(self, *a, **k):
            a = tuple(torch.float32 if x is torch.bfloat16 else x for x in a)
            if k.get("dtype") is torch.bfloat16:
                k = dict(k)
                k["dtype"] = torch.float32
            return orig_to(self, *a, **k)

        torch.Tensor.to = to_no_bf16

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    import transformers
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages_dir = Path(args.pages_dir) if args.pages_dir else None
    files = [str(pages_dir / p) if pages_dir else p for p in args.pages]
    for f in files:
        if not Path(f).is_file():
            raise SystemExit(f"page not found: {f}")

    print(f"[ref-multi] loading tokenizer from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[ref-multi] loading model ({args.dtype}, CPU) ...", flush=True)
    t0 = time.time()
    model = (
        AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.float32 if args.dtype == "f32" else torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
    )
    captured = {}
    if args.max_new_tokens is not None or args.margins:
        orig_generate = model.generate

        def probe_generate(**kw):
            if args.max_new_tokens is not None:
                kw.pop("max_length", None)
                kw["max_new_tokens"] = args.max_new_tokens
            if not args.margins:
                return orig_generate(**kw)
            kw["output_scores"] = True
            kw["return_dict_in_generate"] = True
            res = orig_generate(**kw)
            captured["scores"] = res.scores
            captured["generated"] = res.sequences[0, -len(res.scores):].tolist()
            return res.sequences

        model.generate = probe_generate
    print(
        f"[ref-multi] model loaded in {time.time() - t0:.1f}s; "
        f"threads={torch.get_num_threads()}",
        flush=True,
    )

    print(f"[ref-multi] infer_multi over {len(files)} page(s): {files}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs, output_tokens = model.infer_multi(
            tok,
            prompt=args.prompt,
            image_files=files,
            output_path=str(out / "_scratch"),
            image_size=args.image_size,
            save_results=False,
            max_length=args.max_length,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            ngram_window=args.ngram_window,
            temperature=0.0,
        )
    dt = time.time() - t0
    outputs = outputs if isinstance(outputs, str) else str(outputs)

    raw_path = out / "multi_raw.md"
    raw_path.write_text(outputs)
    sha = hashlib.sha256(outputs.encode("utf-8")).hexdigest()
    meta = {
        "pages": [Path(f).name for f in files],
        "prompt": args.prompt,
        "image_size": args.image_size,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "ngram_window": args.ngram_window,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "chars": len(outputs),
        "output_tokens": int(output_tokens),
        "page_markers": outputs.count("<PAGE>"),
        "sha256": sha,
        "seconds": round(dt, 2),
        "raw": raw_path.name,
    }
    (out / "multi_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.margins:
        rows = []
        for step, (logits, chosen) in enumerate(zip(captured["scores"], captured["generated"])):
            top = torch.topk(logits[0].float(), 3)
            ids, vals = top.indices.tolist(), top.values.tolist()
            rows.append(
                f"{step}\t{chosen}\t{tok.decode([chosen])!r}\t{ids[1]}\t{tok.decode([ids[1]])!r}\t"
                f"{vals[0] - vals[1]:.4f}\t{tok.decode([ids[2]])!r}\t{vals[0] - vals[2]:.4f}"
            )
        header = "step\tchosen\tchosen_text\tsecond\tsecond_text\tgap\tthird_text\tgap3\n"
        (out / "multi_margins.txt").write_text(header + "\n".join(rows) + "\n")
    print(
        f"[ref-multi] done: {len(outputs)} chars, {output_tokens} toks, "
        f"{meta['page_markers']} <PAGE> markers, {dt:.1f}s, sha {sha[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
