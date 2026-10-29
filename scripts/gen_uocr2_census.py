#!/usr/bin/env python3
"""Generate the Unlimited-OCR-2 delta tensor census (U2-A5, plan D22).

OFFLINE TOOLING ONLY. Reads the pinned Unlimited-OCR-2 checkpoint directory
(``model.safetensors.index.json``, both shards, ``tokenizer.json``) and the
committed v1 census ``src/native_engine/unlimited_ocr_manifest.json``, and
writes ``src/native_engine/unlimited_ocr_2_manifest.json``:

- the pins: HF commit, index SHA-256 and ``total_size``, each shard's name,
  size and SHA-256 (all re-hashed here), and the tokenizer SHA-256;
- the tensors v2 adds over v1's 2,710 (every v1 tensor must be present in v2
  with the same shape and dtype), and the one I64 tensor, whose content must be
  ``arange(257)``;
- ``census_sha256``: SHA-256 of the canonical 2,819-entry list, one line per
  tensor in byte order of names: ``{name}\\t{dtype}\\t{d0,d1,...}\\n``;
- ``vision_identity_sha256``: the full-tensor SHA-256 of every vision tower,
  projector, ``image_newline`` and ``view_seperator`` tensor. With
  ``--v1-focrq`` (the verified conservative v1 artifact, which stores these
  tensors as their source BF16 bytes) each one must equal v1's bytes.

Usage:
    python3 -I scripts/gen_uocr2_census.py <v2_checkpoint_dir> \\
        --v1-focrq ~/.cache/franken_ocr_local_models/unlimited-ocr.int8.focrq

The Rust census (``unlimited_ocr_census.rs``) pins this file's SHA-256 and
re-derives ``census_sha256`` from the v1 census plus this delta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_MANIFEST = REPO_ROOT / "src/native_engine/unlimited_ocr_manifest.json"
OUT = REPO_ROOT / "src/native_engine/unlimited_ocr_2_manifest.json"

HF_COMMIT = "5632329a32e103d5f1add270681e0f5c1233c6e4"
INDEX_SHA256 = "dc60a31ecb77eb03e166345d18cbe25bc71d2a19cc6fc089d8a9ac1c7f209671"
SHARDS = {
    "model-00001-of-00002.safetensors": (
        4_998_430_312,
        "4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18",
    ),
    "model-00002-of-00002.safetensors": (
        1_689_875_264,
        "9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c",
    ),
}
TOKENIZER_SHA256 = "c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589"
QUANT_RECIPE = "unlimited-ocr-2-ffn-int8-attn-bf16-lmhead-bf16-v1"
POSITION_IDS = "model.vision_model.embeddings.position_ids"
VISION = re.compile(
    r"^model\.(sam_model|vision_model|projector)\.|^model\.(image_newline|view_seperator)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> "tuple[dict, int]":
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def read_range(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if len(data) != end - start:
        raise SystemExit(f"short read of {path} [{start}, {end})")
    return data


def focrq_tensors(path: Path) -> "tuple[dict, int]":
    with path.open("rb") as f:
        prefix = f.read(51)
        if prefix[:6] != b"FOCRQ\0":
            raise SystemExit(f"{path} is not a .focrq")
        (header_len,) = struct.unpack("<Q", prefix[43:51])
        header = json.loads(f.read(header_len))
    return header["tensors"], 51 + header_len


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("v2_dir", type=Path)
    parser.add_argument("--v1-focrq", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    index_path = args.v2_dir / "model.safetensors.index.json"
    if sha256_file(index_path) != INDEX_SHA256:
        raise SystemExit(f"{index_path} is not the pinned index")
    index = json.loads(index_path.read_text())
    tokenizer_sha = sha256_file(args.v2_dir / "tokenizer.json")
    if tokenizer_sha != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 {tokenizer_sha}, want {TOKENIZER_SHA256}")

    tensors: "dict[str, dict]" = {}
    where: "dict[str, tuple[Path, int]]" = {}
    shards = []
    for name in sorted(SHARDS):
        path = args.v2_dir / name
        size, pinned = SHARDS[name]
        if path.stat().st_size != size:
            raise SystemExit(f"{name} is {path.stat().st_size} bytes, want {size}")
        sha = sha256_file(path)
        if sha != pinned:
            raise SystemExit(f"{name} sha256 {sha}, want {pinned}")
        shards.append({"name": name, "size": size, "sha256": sha})
        header, base = safetensors_header(path)
        for tensor, record in header.items():
            if tensor in tensors:
                raise SystemExit(f"{tensor} is in two shards")
            tensors[tensor] = record
            where[tensor] = (path, base)
        print(f"{name}: {len(header)} tensors, sha256 ok", file=sys.stderr)
    if set(tensors) != set(index["weight_map"]):
        raise SystemExit("the shard headers and the index disagree on the tensor set")
    payload = sum(r["data_offsets"][1] - r["data_offsets"][0] for r in tensors.values())
    if payload != index["metadata"]["total_size"]:
        raise SystemExit(f"payload {payload} != index total_size")

    v1 = json.loads(V1_MANIFEST.read_text())["tensors"]
    for name, tensor in v1.items():
        record = tensors.get(name)
        if record is None or record["dtype"] != tensor["source_dtype"] or record["shape"] != tensor["shape"]:
            raise SystemExit(f"v1 tensor {name} is missing or changed in v2: {record}")
    added = {}
    int64 = {}
    for name in sorted(set(tensors) - set(v1)):
        record = tensors[name]
        if record["dtype"] == "I64":
            int64[name] = record
        else:
            added[name] = {"shape": record["shape"], "source_dtype": record["dtype"]}
    if list(int64) != [POSITION_IDS] or int64[POSITION_IDS]["shape"] != [1, 257]:
        raise SystemExit(f"unexpected I64 tensors {int64}")
    path, base = where[POSITION_IDS]
    begin, end = int64[POSITION_IDS]["data_offsets"]
    values = struct.unpack("<257q", read_range(path, base + begin, base + end))
    if list(values) != list(range(257)):
        raise SystemExit("position_ids is not arange(257)")

    canonical = "".join(
        f"{name}\t{tensors[name]['dtype']}\t{','.join(str(d) for d in tensors[name]['shape'])}\n"
        for name in sorted(tensors, key=lambda n: n.encode())
    )
    census_sha256 = hashlib.sha256(canonical.encode()).hexdigest()

    v1_focrq = focrq_tensors(args.v1_focrq) if args.v1_focrq else None
    vision = {}
    for name in sorted(n for n in v1 if VISION.search(n)):
        path, base = where[name]
        begin, end = tensors[name]["data_offsets"]
        data = read_range(path, base + begin, base + end)
        vision[name] = hashlib.sha256(data).hexdigest()
        if v1_focrq is not None:
            records, payload_base = v1_focrq
            record = records[name]
            if record["dtype"] != "BF16" or record["shape"] != tensors[name]["shape"]:
                raise SystemExit(f"v1 .focrq {name} is {record['dtype']} {record['shape']}")
            start = payload_base + record["byte_offset"]
            v1_bytes = read_range(args.v1_focrq, start, start + record["byte_len"])
            if v1_bytes != data:
                raise SystemExit(f"vision tensor {name} differs from v1")
    print(
        f"{len(tensors)} tensors; {len(added)} added + {len(int64)} I64; "
        f"{len(vision)} vision tensors hashed"
        + (", all byte-identical to v1" if v1_focrq else ""),
        file=sys.stderr,
    )

    manifest = {
        "schema": "franken_ocr.unlimited_ocr_2_tensor_delta.v1",
        "model_id": "unlimited-ocr-2",
        "base_model_id": "unlimited-ocr",
        "hf_commit": HF_COMMIT,
        "source_index_sha256": INDEX_SHA256,
        "source_index_total_size": index["metadata"]["total_size"],
        "source_shards": shards,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tensor_count": len(tensors),
        "census_sha256": census_sha256,
        "quant_recipe": QUANT_RECIPE,
        "verified_conservative_artifact_sha256": None,
        "added_tensors": added,
        "int64_tensors": {
            POSITION_IDS: {"shape": int64[POSITION_IDS]["shape"], "content": "arange"}
        },
        "vision_identity_sha256": vision,
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
