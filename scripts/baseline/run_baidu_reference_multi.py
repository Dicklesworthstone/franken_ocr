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

Usage:
  run_baidu_reference_multi.py --model DIR --out DIR page_0009.png page_0014.png \
      [--pages-dir DIR] [--max-length 32768] [--threads N]
"""

import argparse
import hashlib
import json
import sys
import time
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
    ap.add_argument("pages", nargs="+", help="ordered page image files")
    args = ap.parse_args()

    install_cpu_patches()
    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
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
    print("[ref-multi] loading model (bf16, CPU) ...", flush=True)
    t0 = time.time()
    model = (
        AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
    )
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
        "chars": len(outputs),
        "output_tokens": int(output_tokens),
        "page_markers": outputs.count("<PAGE>"),
        "sha256": sha,
        "seconds": round(dt, 2),
        "raw": raw_path.name,
    }
    (out / "multi_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[ref-multi] done: {len(outputs)} chars, {output_tokens} toks, "
        f"{meta['page_markers']} <PAGE> markers, {dt:.1f}s, sha {sha[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
