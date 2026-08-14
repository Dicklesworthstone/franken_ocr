#!/usr/bin/env python3
"""E2 (bd-3jo6.5.2): the OFFLINE Polyphonic-TrOMR checkpoint export —
`img2score_epoch47.pth` (torch pickle) → `model.safetensors`, with the
convert-time Weight-Standardization fold (tromr-spec §10.3/§11).

What it does, in census order:

1. **Provenance gate**: refuses unless the input .pth matches the census pin
   (86,254,711 bytes, sha256 02925259ef…, spec §Sources) — the 261-tensor
   inventory in §12 was extracted from exactly this file.
2. **WS fold**: every ResNetV2 backbone conv (`encoder.patch_embed.backbone.
   *conv*.weight`) is stored PRE-STANDARDIZED by replaying the exact
   `timm==0.6.5` `StdConv2dSame` expression (`F.batch_norm(..., eps
   1e-6)`, population variance — census §16). The export manifest records the
   exact Python/PyTorch/safetensors/numpy environment because fused floating
   point results can vary by build or host. Runtime then runs plain
   `nn::conv2d` — no WS kernel exists in Rust (§15/E3 delta).
3. **WS proof (L1)**: per conv, (a) a determinism re-run must `torch.equal`
   the fold, and (b) the analytic `(w-mean)/sqrt(var+eps)` formulation must
   agree within 1e-5 (guards a wrong-axis/eps/shape invocation; the analytic
   form itself differs from the fused kernel by ~5e-7 rounding, measured
   2026-07-05). Any violation refuses the export.
4. **Drop `decoder.note_mask`** (train-only, census §12) — everything else
   (260 tensors) carries over byte-identical (the fold touches ONLY backbone
   conv weights; norms/biases/ViT/decoder/heads are untouched f32).
5. Writes `model.safetensors` + `TROMR_EXPORT_MANIFEST.json` (complete
   source/output name-shape-dtype-byte-SHA inventories, environment, source
   pin, counts) beside the output.

Usage:
    python scripts/gen_tromr_safetensors.py \
        --pth  <zoo>/tromr-upstream/tromr/workspace/checkpoints/img2score_epoch47.pth \
        --out  <zoo>/tromr/model.safetensors

Requires: torch, safetensors, timm==0.6.5 (the pinned reference for the proof).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import struct
import sys

PIN_BYTES = 86_254_711
PIN_SHA256 = "02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910"
WS_EPS = 1e-6  # census §16: population variance, eps 1e-6
EXPECTED_TENSORS = 261  # census §12 (incl. the dropped note_mask)
DROP = ("decoder.note_mask",)  # train-only (census §12)
BACKBONE_PREFIX = "encoder.patch_embed.backbone."


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_replay_outcome(
    *,
    exact_pin_supplied: bool,
    expected_match: bool,
    accepted_comparison_supplied: bool,
    tolerance_match: bool,
) -> str:
    """Classify authority without turning absence of a pin into acceptance."""
    if expected_match:
        return "exact_bytes"
    if tolerance_match:
        return "value_tolerance"
    if not exact_pin_supplied and not accepted_comparison_supplied:
        return "unverified"
    return "mismatch"


def is_ws_conv(name: str, shape) -> bool:
    """The WS-folded set: backbone conv WEIGHTS only (4-D), never norms.

    ResNetV2's StdConv2dSame standardizes stem.conv, every blocks.*.conv{1,2,3}
    and every downsample.conv — i.e. every 4-D `.weight` under the backbone.
    GN weights/biases are 1-D and stay untouched.
    """
    return name.startswith(BACKBONE_PREFIX) and name.endswith(".weight") and len(shape) == 4


def ws_fold(w):
    """The fold IS the pinned timm 0.6.5 StdConv2dSame arithmetic — the stored
    weight must be BIT-IDENTICAL to what upstream's runtime WS computes (so our
    plain conv reproduces their standardized conv bitwise):

        weight = F.batch_norm(self.weight.reshape(1, out, -1), None, None,
                              training=True, momentum=0., eps=self.eps)
                  .reshape_as(self.weight)

    (The analytic (w-mean)/sqrt(var+eps) form differs from this fused kernel
    by ~5e-7 float rounding — measured 2026-07-05; hence invoke, don't mimic.)
    """
    import torch.nn.functional as F

    return F.batch_norm(
        w.reshape(1, w.shape[0], -1), None, None, training=True, momentum=0.0, eps=WS_EPS
    ).reshape_as(w)


def analytic_cross_check(w, folded) -> float:
    """Guard against invoking the reference WRONGLY (axis/eps/shape bugs): the
    analytic population-variance formulation must agree to ~float-rounding.
    Returns the maxabs delta (caller enforces the 1e-5 sanity bound)."""
    import torch

    v = w.reshape(w.shape[0], -1)
    mean = v.mean(dim=1, keepdim=True)
    var = v.var(dim=1, unbiased=False, keepdim=True)
    analytic = ((v - mean) / torch.sqrt(var + WS_EPS)).reshape_as(w)
    return (analytic - folded).abs().max().item()


def tensor_inventory(state: dict) -> list[dict]:
    """Return the canonical value inventory used by the lineage receipt."""
    rows: list[dict] = []
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        data = tensor.numpy().tobytes(order="C")
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def inventory_sha256(rows: list[dict]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def focrq_f32_inventory(path: str, expected_sha256: str) -> tuple[dict, list[dict]]:
    """Read the provider's exact f32 FOCRQ as a comparison authority."""
    import numpy

    blob = open(path, "rb").read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"accepted FOCRQ SHA-256 is {digest}, expected {expected_sha256}"
        )
    if len(blob) < 51 or blob[:6] != b"FOCRQ\0":
        raise ValueError("accepted FOCRQ has no valid 51-byte v1 preamble")
    version = struct.unpack("<I", blob[6:10])[0]
    if version != 1:
        raise ValueError(f"accepted FOCRQ format version is {version}, expected 1")
    header_len = struct.unpack("<Q", blob[43:51])[0]
    header_end = 51 + header_len
    if header_end > len(blob):
        raise ValueError("accepted FOCRQ header overruns its bytes")
    header = json.loads(blob[51:header_end])
    preamble_source = blob[11:43].hex()
    if header.get("source_sha256") != preamble_source:
        raise ValueError("accepted FOCRQ header/preamble source SHA-256 disagree")
    payload = memoryview(blob)[header_end:]
    rows: list[dict] = []
    for name, entry in sorted(header["tensors"].items()):
        if entry["dtype"] != "F32":
            raise ValueError(f"accepted FOCRQ tensor {name} is not F32")
        start = entry["byte_offset"]
        end = start + entry["byte_len"]
        if end > len(payload):
            raise ValueError(f"accepted FOCRQ tensor {name} overruns its payload")
        data = bytes(payload[start:end])
        expected_bytes = int(numpy.prod(entry["shape"], dtype=numpy.int64)) * 4
        if len(data) != expected_bytes:
            raise ValueError(f"accepted FOCRQ tensor {name} shape/bytes disagree")
        rows.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": entry["shape"],
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    summary = {
        "bytes": len(blob),
        "sha256": digest,
        "format_version": version,
        "model_id": header.get("model_id"),
        "source_sha256": header.get("source_sha256"),
        "tensor_count": len(rows),
        "value_bytes": sum(row["bytes"] for row in rows),
        "tensor_value_inventory_sha256": inventory_sha256(rows),
    }
    return summary, rows


def compare_value_inventories(
    generated_path: str, generated_rows: list[dict], accepted_path: str, accepted_rows: list[dict]
) -> dict:
    """Compare generated safetensors values with accepted f32 FOCRQ values."""
    import numpy

    generated_blob = open(generated_path, "rb").read()
    generated_header_len = struct.unpack("<Q", generated_blob[:8])[0]
    generated_header = json.loads(generated_blob[8 : 8 + generated_header_len])
    generated_payload = 8 + generated_header_len

    accepted_blob = open(accepted_path, "rb").read()
    accepted_header_len = struct.unpack("<Q", accepted_blob[43:51])[0]
    accepted_header = json.loads(accepted_blob[51 : 51 + accepted_header_len])
    accepted_payload = 51 + accepted_header_len

    generated_by_name = {row["name"]: row for row in generated_rows}
    accepted_by_name = {row["name"]: row for row in accepted_rows}
    names_shapes_equal = [
        (row["name"], row["shape"]) for row in generated_rows
    ] == [(row["name"], row["shape"]) for row in accepted_rows]
    differences: list[dict] = []
    exact = 0
    if names_shapes_equal:
        for name in sorted(generated_by_name):
            generated = generated_header[name]
            g0, g1 = generated["data_offsets"]
            generated_data = generated_blob[
                generated_payload + g0 : generated_payload + g1
            ]
            accepted = accepted_header["tensors"][name]
            a0 = accepted_payload + accepted["byte_offset"]
            accepted_data = accepted_blob[a0 : a0 + accepted["byte_len"]]
            if generated_data == accepted_data:
                exact += 1
                continue
            generated_values = numpy.frombuffer(generated_data, dtype="<f4")
            accepted_values = numpy.frombuffer(accepted_data, dtype="<f4")
            delta = numpy.abs(generated_values - accepted_values)
            differences.append(
                {
                    "name": name,
                    "generated_sha256": generated_by_name[name]["sha256"],
                    "accepted_sha256": accepted_by_name[name]["sha256"],
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                    "different_elements": int(numpy.count_nonzero(delta)),
                }
            )
    return {
        "names_shapes_dtypes_equal": names_shapes_equal,
        "exact_value_tensor_count": exact,
        "tolerance_value_tensor_count": len(differences),
        "max_abs": max((row["max_abs"] for row in differences), default=0.0),
        "mean_of_tensor_mean_abs": (
            sum(row["mean_abs"] for row in differences) / len(differences)
            if differences
            else 0.0
        ),
        "different_element_count": sum(
            row["different_elements"] for row in differences
        ),
        "differences": differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-torch")
    parser.add_argument("--expect-safetensors")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--accepted-focrq")
    parser.add_argument("--expected-focrq-sha256")
    parser.add_argument("--accepted-max-abs", type=float)
    args = parser.parse_args()

    size = os.path.getsize(args.pth)
    digest = sha256_file(args.pth)
    if size != PIN_BYTES or digest != PIN_SHA256:
        print(
            f"FATAL: {args.pth} does not match the census pin "
            f"(size {size} vs {PIN_BYTES}, sha256 {digest[:16]}… vs {PIN_SHA256[:16]}…)",
            file=sys.stderr,
        )
        return 1

    import numpy
    import safetensors
    import torch
    from safetensors.torch import save_file

    if sys.byteorder != "little":
        print("FATAL: TrOMR export currently supports little-endian hosts only", file=sys.stderr)
        return 1
    if args.expect_torch is not None and torch.__version__ != args.expect_torch:
        print(
            f"FATAL: torch=={torch.__version__}, expected {args.expect_torch}",
            file=sys.stderr,
        )
        return 1
    if args.expect_safetensors is not None and safetensors.__version__ != args.expect_safetensors:
        print(
            f"FATAL: safetensors=={safetensors.__version__}, expected {args.expect_safetensors}",
            file=sys.stderr,
        )
        return 1

    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    # torch 1.11 predates weights_only. The legacy pickle loader is used only
    # after the exact official checkpoint byte length and SHA-256 gate above.
    state = torch.load(args.pth, **load_kwargs)
    if len(state) != EXPECTED_TENSORS:
        print(f"FATAL: {len(state)} tensors, census expects {EXPECTED_TENSORS}", file=sys.stderr)
        return 1
    source_inventory = tensor_inventory(state)

    out_state: dict = {}
    folded_names: list[str] = []
    for name, tensor in state.items():
        if name in DROP:
            continue
        tensor = tensor.contiguous()
        if tensor.dtype != torch.float32:
            print(f"FATAL: {name} is {tensor.dtype}, census says all-fp32", file=sys.stderr)
            return 1
        if is_ws_conv(name, tensor.shape):
            folded = ws_fold(tensor)
            # Determinism proof: the reference arithmetic must reproduce itself
            # bit-exactly (a nondeterministic kernel could not be blessed).
            if not torch.equal(folded, ws_fold(tensor)):
                print(f"FATAL: WS fold nondeterministic for {name}", file=sys.stderr)
                return 1
            delta = analytic_cross_check(tensor, folded)
            if delta > 1e-5:
                print(
                    f"FATAL: WS fold sanity FAILED for {name} (analytic delta {delta:.3e} "
                    "> 1e-5 — wrong axis/eps/shape?)",
                    file=sys.stderr,
                )
                return 1
            out_state[name] = folded
            folded_names.append(name)
        else:
            out_state[name] = tensor

    # Census cross-checks: 3 stages × {2,3,7} blocks × conv{1,2,3} + 3
    # downsamples + the stem = 40 folded convs.
    if len(folded_names) != 40:
        print(f"FATAL: folded {len(folded_names)} convs, census layout expects 40", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out_state, args.out)
    output_inventory = tensor_inventory(out_state)
    output_sha256 = sha256_file(args.out)
    output_bytes = os.path.getsize(args.out)
    expected_match = (
        args.expected_output_sha256 is not None
        and output_sha256 == args.expected_output_sha256
    )
    if (args.accepted_focrq is None) != (args.expected_focrq_sha256 is None):
        print(
            "FATAL: --accepted-focrq and --expected-focrq-sha256 must be supplied together",
            file=sys.stderr,
        )
        return 1
    accepted_summary = None
    accepted_inventory = None
    accepted_comparison = None
    if args.accepted_focrq is not None:
        try:
            accepted_summary, accepted_inventory = focrq_f32_inventory(
                args.accepted_focrq, args.expected_focrq_sha256
            )
            accepted_comparison = compare_value_inventories(
                args.out, output_inventory, args.accepted_focrq, accepted_inventory
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"FATAL: accepted FOCRQ comparison failed: {error}", file=sys.stderr)
            return 1
    tolerance_match = (
        accepted_comparison is not None
        and args.accepted_max_abs is not None
        and accepted_comparison["names_shapes_dtypes_equal"]
        and {row["name"] for row in accepted_comparison["differences"]}
        == set(folded_names)
        and accepted_comparison["max_abs"] <= args.accepted_max_abs
    )
    accepted_replay_outcome = classify_replay_outcome(
        exact_pin_supplied=args.expected_output_sha256 is not None,
        expected_match=expected_match,
        accepted_comparison_supplied=accepted_comparison is not None,
        tolerance_match=tolerance_match,
    )

    manifest = {
        "purpose": "TrOMR E2 offline export (bd-3jo6.5.2) — WS-folded, note_mask dropped",
        "script": "scripts/gen_tromr_safetensors.py",
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "source_pth": {
            "bytes": PIN_BYTES,
            "sha256": PIN_SHA256,
            "tensor_count": len(source_inventory),
            "parameter_count": sum(tensor.numel() for tensor in state.values()),
            "value_bytes": sum(row["bytes"] for row in source_inventory),
            "tensor_value_inventory_sha256": inventory_sha256(source_inventory),
        },
        "ws_fold": {
            "eps": WS_EPS,
            "variance": "population (unbiased=False)",
            "reference": "timm==0.6.5 StdConv2dSame F.batch_norm expression",
            "proof": "determinism re-run torch.equal + analytic population-variance cross-check <= 1e-5 per conv; exact environment recorded because fused floating-point bytes are environment-sensitive",
            "folded_convs": folded_names,
        },
        "dropped": list(DROP),
        "tensors_out": len(out_state),
        "model_safetensors_bytes": output_bytes,
        "model_safetensors_sha256": output_sha256,
        "expected_model_safetensors_sha256": args.expected_output_sha256,
        "expected_model_safetensors_match": expected_match,
        "accepted_replay_outcome": accepted_replay_outcome,
        "accepted_max_abs_contract": args.accepted_max_abs,
        "tensor_value_inventory_sha256": inventory_sha256(output_inventory),
        "accepted_focrq": accepted_summary,
        "accepted_value_comparison": accepted_comparison,
        "source_tensor_inventory": source_inventory,
        "output_tensor_inventory": output_inventory,
        "accepted_tensor_inventory": accepted_inventory,
        "license": "Apache-2.0 (NetEase Polyphonic-TrOMR — NOTICE carried to distribution)",
    }
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "TROMR_EXPORT_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "tromr_export",
                "result": (
                    "pass"
                    if accepted_replay_outcome in {"exact_bytes", "value_tolerance"}
                    else accepted_replay_outcome
                ),
                "tensors": len(out_state),
                "ws_folded": len(folded_names),
                "out": args.out,
                "manifest": manifest_path,
                "sha256": manifest["model_safetensors_sha256"],
                "expected_sha256": args.expected_output_sha256,
                "expected_match": expected_match,
                "accepted_replay_outcome": accepted_replay_outcome,
            }
        )
    )
    if accepted_replay_outcome == "mismatch":
        print(
            f"FATAL: regenerated SHA-256 {output_sha256} satisfies neither the exact "
            "output pin nor the accepted-value tolerance contract; manifest retained "
            "for diagnosis",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
