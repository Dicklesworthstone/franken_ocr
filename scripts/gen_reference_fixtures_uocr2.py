#!/usr/bin/env python3
"""Unlimited-OCR-2 reference oracle (bead U2-A6, plan §7.1). OFFLINE TOOLING ONLY.

franken_ocr's Rust engine never runs this. It drives Baidu's HF reference
(`baidu/Unlimited-OCR-2` at HF commit 5632329a, trust_remote_code) through
`model.infer()` with explicit evaluation parameters and dumps every seam the v2
parity ladder needs (L0-L4, plan §7.2).

Primary oracle: float32 on CPU. The v1 precedent (`gen_reference_fixtures.py
--device cpu --cpu-patch --f32`, fixture set `native_f32`) reached L4
token-exact parity. Here `.cuda()` and the cuda autocast become no-ops, the
model loads in float32 (an exact upcast of the bf16 shards), and
`Tensor.to(torch.bfloat16)` is rewritten to float32, so the hard-coded image
casts in `_uocr_build_inputs` (MU:561,566) stay f32. On CPU the DASH router
takes `_forward_dense` for prefill and `_decode_attend` for decode; no part of
the reference is ported (OQ-V2-4). `--dtype bf16` runs the same unpatched
bf16 graph on CPU, for the cross-precision ledger only.

What is captured, without editing Baidu's code (module-level functions are
wrapped; module hooks do the rest):
  * prompt ids, preprocessed image tensors, seq mask, tile grid (L0);
  * page2d coordinates and the main/router cos-sin tables (L1);
  * per layer, for the selected prefill rows: the layer input, the router's
    post-RoPE iq (ik for every row), kbar, block scores, the page-aware
    candidate mask and the fallback flag, the entmax top-M input and prior,
    topi/bias/valid, the attention output before and after o_proj (L1-L2);
  * the final hidden state, the raw last-position logits (L3);
  * the generated ids and text, and per step the n-gram/guard effect (L4);
  * the same per-layer seams for decode steps 1..63 and +-8 steps around every
    generated <PAGE>.
Selected prefill rows are every row when the prompt has <= 1024 rows. Longer
prompts keep BOS, every <PAGE>, the first 160 and last 32 image rows of each
page (the 97-128 fallback queries of pages 2..N are among the first 160) and
the instruction rows.

Environment (an isolated venv, never shared with the v1 oracle):
    uv venv --python 3.10 <venv>
    uv pip install --python <venv>/bin/python torch==2.13.0 torchvision==0.28.0 \\
        transformers==4.52.1 tokenizers==0.21.4 huggingface-hub==0.36.0 \\
        addict einops easydict numpy safetensors pillow
    <venv>/bin/hf download baidu/Unlimited-OCR-2 \\
        --revision 5632329a32e103d5f1add270681e0f5c1233c6e4 --local-dir <model-dir>

Run (each fixture set once per dtype and thread count; see PROVENANCE.md):
    <venv>/bin/python -I scripts/gen_reference_fixtures_uocr2.py \\
        --model-dir <model-dir> --corpus ~/.cache/franken_ocr/corpus/pages \\
        --out "$FOCR_FIXTURES_DIR/uocr2_oracle" --sets S1,S2,S3,S4 \\
        --dtype f32 --threads 12 --tag t12 --commit-dir tests/fixtures/uocr2_oracle

Writes, per run, under <out>/<dtype>/<tag>/ (the v1 FixtureLoader layout):
    <fixture>_reference.json             ids, text, counters, activation manifest,
                                         provenance
    activations/<fixture>/<stage>.npy    C-order little-endian arrays
    runs.ndjson                          one line per fixture run
and with --commit-dir (f32 only): <fixture>.ids.txt, <fixture>.md and the
fixture's entry in ORACLE.json. Every model file is checked against
docs/truth-pack/SOURCE_HASHES.md before loading.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess  # nosec B404 - only `git rev-parse` for provenance, shell=False
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_HASHES = REPO_ROOT / "docs/truth-pack/SOURCE_HASHES.md"
MULTI_PAGE = REPO_ROOT / "tests/fixtures/multi_page"
SCRIPT_REL = "scripts/gen_reference_fixtures_uocr2.py"

PIN_PYTHON = "3.10"
PIN_TORCH = "2.13.0"
PIN_TRANSFORMERS = "4.52.1"
PIN_TOKENIZERS = "0.21.4"
PIN_HF_HUB = "0.36.0"
HF_REPO = "baidu/Unlimited-OCR-2"
HF_COMMIT = "5632329a32e103d5f1add270681e0f5c1233c6e4"
# The files infer() reads. README.md, the paper and the SGLang wheels are not needed.
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "configuration_deepseek_v2.py",
    "conversation.py",
    "dash_attention.py",
    "deepencoder.py",
    "modeling_deepseekv2.py",
    "modeling_unlimitedocr.py",
    "_sparse_base_impl.py",
    "packing_patch.py",
    "sparse_gather_triton.py",
    "sparse_gather_e2e_triton.py",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)

IMAGE_ID = 128815
PAGE_ID = 128827
NGRAM_SIZE = 35
NGRAM_WINDOW_SINGLE = 128
NGRAM_WINDOW_MULTI = 1024
MAX_PAGE_TOKENS = 20000
SEED = 1337

HEAD_STEPS = 64
PAGE_RADIUS = 8
FALLBACK_IMAGE_ROWS = 160
TAIL_IMAGE_ROWS = 32
FULL_ROWS_MAX = 1024

# Switches that change the reference's semantics. The oracle runs their defaults.
SEMANTIC_ENV = (
    "DASH_STAGE0",
    "PAGE_SCOPE_PREFILL",
    "KL_CHUNK_RECOMPUTE",
    "KL_QUERY_STRIDE",
    "INDEXER_SCORE_TRITON",
    "WARMUP_MAIN_LOCAL_WINDOW",
    "DASH_SPARSITY_SCHED",
    "DASH_DEBUG_ANOMALY",
)
# Diagnostics only; turned off so they cannot cost time or print per step.
QUIET_ENV = {"SPARSE_DIAG_LOG": "0", "TIME_DIAG": "0", "CODE_STAMP_LOG": "0"}

# The guard's decision for one step (`_uocr_page_count_guard`).
GUARD_NONE, GUARD_EOS_ONLY, GUARD_CAP_FORCED, GUARD_EOS_BANNED = 0, 1, 2, 3


@dataclass(frozen=True)
class Case:
    fixture: str
    pages: tuple[str, ...]
    crop_mode: bool

    @property
    def mode(self) -> str:
        return "gundam" if self.crop_mode else "base"

    @property
    def ngram_window(self) -> int:
        return NGRAM_WINDOW_SINGLE if len(self.pages) == 1 else NGRAM_WINDOW_MULTI


def multi_page_case(fixture: str, meta: str) -> Case:
    pages = json.loads((MULTI_PAGE / meta).read_text())["pages"]
    return Case(fixture, tuple(pages), False)


# S4's second page: the first corpus page (sorted, other than page_0590) whose
# gundam tile grid is not square. `--gundam-grids` shows every corpus page is
# 1182x1837 and tiles as a 2x3 grid (2026-10-29), so that is page_0009.
S4_NON_SQUARE_PAGE = "page_0009.png"


def fixture_sets() -> dict[str, list[Case]]:
    return {
        "S1": [
            Case("S1-page_0590", ("page_0590.png",), False),
            Case("S1-page_0009", ("page_0009.png",), False),
            Case("S1-page_0014", ("page_0014.png",), False),
        ],
        "S2": [Case("S2-p9_p14_p19", ("page_0009.png", "page_0014.png", "page_0019.png"), False)],
        "S3": [
            multi_page_case("S3-p9_p14", "p9_p14_meta.json"),
            multi_page_case("S3-p10", "p10_meta.json"),
            multi_page_case("S3-p20", "p20_meta.json"),
        ],
        "S4": [
            Case("S4-page_0590-gundam", ("page_0590.png",), True),
            Case(f"S4-{S4_NON_SQUARE_PAGE.removesuffix('.png')}-gundam", (S4_NON_SQUARE_PAGE,), True),
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (exercised by --self-test without torch)
# ─────────────────────────────────────────────────────────────────────────────
def parse_source_hashes(text: str) -> dict[str, str]:
    """`{file: sha256}` from the Unlimited-OCR-2 section of SOURCE_HASHES.md."""
    start = text.index("# Unlimited-OCR-2")
    end = text.find("**Upstream license text**", start)
    section = text[start : end if end >= 0 else len(text)]
    return {f: h for h, f in re.findall(r"^\| `([0-9a-f]{64})` \| `([^`]+)` \|", section, re.M)}


def page_spans(ids: list[int]) -> list[tuple[int, list[int]]]:
    """`(marker_row, image_rows)` per page, in order. A page is a <PAGE> marker
    and the image rows that follow it."""
    spans: list[tuple[int, list[int]]] = []
    for row, tok in enumerate(ids):
        if tok == PAGE_ID:
            spans.append((row, []))
        elif tok == IMAGE_ID and spans:
            spans[-1][1].append(row)
    return spans


def select_rows(ids: list[int], full_max: int = FULL_ROWS_MAX) -> list[int]:
    """The prefill rows whose per-layer seams are dumped (see the module doc)."""
    n = len(ids)
    if n <= full_max:
        return list(range(n))
    rows = {0}
    last_image = max((r for r, t in enumerate(ids) if t == IMAGE_ID), default=n - 1)
    for marker, image_rows in page_spans(ids):
        rows.add(marker)
        rows.update(image_rows[:FALLBACK_IMAGE_ROWS])
        rows.update(image_rows[-TAIL_IMAGE_ROWS:])
    rows.update(range(last_image + 1, n))
    return sorted(rows)


def row_pages(ids: list[int]) -> list[int]:
    """The page each prefill row belongs to: the <PAGE> count so far minus one up
    to the last image row, and -1 for BOS and the instruction."""
    last_image = max((r for r, t in enumerate(ids) if t == IMAGE_ID), default=-1)
    out, cum = [], 0
    for row, tok in enumerate(ids):
        cum += tok == PAGE_ID
        out.append(cum - 1 if row <= last_image else -1)
    return out


class StepKeeper:
    """Which decode steps keep their per-layer dumps: the first `head` steps and
    `radius` steps either side of every step that generated <PAGE>.

    Step k is the k-th logits call: step 0 reads the prefill, step k >= 1 the
    decode forward of generated token k-1. A <PAGE> made at step j is seen when
    step j+1 starts, so the last radius+1 unkept steps wait in a ring.
    """

    def __init__(self, head: int = HEAD_STEPS, radius: int = PAGE_RADIUS):
        self.head = head
        self.radius = radius
        self.ring: collections.deque[tuple[int, Any]] = collections.deque()
        self.kept: dict[int, Any] = {}
        self.keep_until = -1

    def finish(self, step: int, record: Any) -> None:
        if step < self.head or step <= self.keep_until:
            self.kept[step] = record
            return
        self.ring.append((step, record))
        while len(self.ring) > self.radius + 1:
            self.ring.popleft()

    def page_generated_at(self, step: int) -> None:
        for s, record in self.ring:
            if s >= step - self.radius:
                self.kept[s] = record
        self.ring.clear()
        self.keep_until = max(self.keep_until, step + self.radius)


def ids_text(model_id: str, fixture: str, ids: list[int]) -> str:
    """The committed id-file format of tests/fixtures/uocr2_v1_goldens."""
    lines = [f"# {model_id} generated token ids: {fixture} ({len(ids)} ids)"]
    for i in range(0, len(ids), 16):
        lines.append(" ".join(str(t) for t in ids[i : i + 16]))
    return "\n".join(lines) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids: list[int]) -> str:
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("ascii"))


def self_test() -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(json.dumps({"test": "uocr2_oracle_self_test", "case": name, "result": "pass" if cond else "fail"}))
        ok &= cond

    text = (
        "# Unlimited-OCR (v1)\n| `" + "1" * 64 + "` | `config.json` | 1 | v1 |\n"
        "# Unlimited-OCR-2 (v2) source hashes\n| `" + "a" * 64 + "` | `config.json` | 1 | v2 |\n"
        "| `" + "b" * 64 + "` | `model-00001-of-00002.safetensors` | 2 | shard |\n"
        "**Upstream license text** x\n| `" + "c" * 64 + "` | `LICENSE` | 3 | lic |\n"
    )
    hashes = parse_source_hashes(text)
    check("source_hashes_reads_only_the_v2_tables", hashes == {"config.json": "a" * 64, "model-00001-of-00002.safetensors": "b" * 64})

    one = [0, PAGE_ID] + [IMAGE_ID] * 273 + [10, 11, 12]
    check("a_short_prompt_keeps_every_row", select_rows(one) == list(range(len(one))))
    many = [0]
    for _ in range(5):
        many += [PAGE_ID] + [IMAGE_ID] * 273
    many += [10, 11, 12]
    rows = select_rows(many, full_max=100)
    spans = page_spans(many)
    want = {0, len(many) - 3, len(many) - 2, len(many) - 1}
    for marker, image_rows in spans:
        want |= {marker, *image_rows[:160], *image_rows[-32:]}
    check("a_long_prompt_keeps_bos_markers_page_heads_tails_and_instruction", rows == sorted(want))
    check("five_pages_have_five_spans_of_273", [len(s[1]) for s in spans] == [273] * 5)
    pages = row_pages(many)
    check(
        "row_pages_number_pages_from_zero_and_leave_bos_and_instruction_unpaged",
        pages[0] == -1 and pages[1] == 0 and pages[2 + 273] == 1 and pages[-1] == -1 and pages[-4] == 4,
    )

    keeper = StepKeeper(head=64, radius=8)
    for step in range(200):
        keeper.finish(step, step)
        if step + 1 in (101, 151):  # step 100 and step 150 generated <PAGE>
            keeper.page_generated_at(step)
    want_steps = set(range(64)) | set(range(92, 109)) | set(range(142, 159))
    check("steps_kept_are_the_head_and_eight_either_side_of_each_page", set(keeper.kept) == want_steps)
    keeper = StepKeeper(head=4, radius=8)
    for step in range(12):
        keeper.finish(step, step)
    keeper.page_generated_at(11)
    check("a_page_near_the_head_keeps_only_existing_steps", set(keeper.kept) == set(range(12)))

    body = ids_text("m", "f", list(range(20)))
    check("ids_text_matches_the_v1_golden_layout", body.splitlines()[0] == "# m generated token ids: f (20 ids)" and body.splitlines()[2] == "16 17 18 19")
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# Stack, model files, patches
# ─────────────────────────────────────────────────────────────────────────────
def check_stack() -> dict[str, str]:
    import tokenizers
    import torch
    import transformers
    import huggingface_hub

    got = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    want = {
        "python": PIN_PYTHON,
        "torch": PIN_TORCH,
        "transformers": PIN_TRANSFORMERS,
        "tokenizers": PIN_TOKENIZERS,
        "huggingface_hub": PIN_HF_HUB,
    }
    bad = [f"{k}=={got[k]} (need {want[k]})" for k in want if got[k] != want[k]]
    if bad:
        raise SystemExit(
            "gen_reference_fixtures_uocr2: pinned-stack mismatch: " + ", ".join(bad)
            + ". Recreate the venv from docs/truth-pack/PINNED_SOURCES.md (Reference stack)."
        )
    got["python_full"] = platform.python_version()
    return got


def verify_model_dir(model_dir: Path) -> dict[str, str]:
    """SHA-256 of every required file, each checked against SOURCE_HASHES.md."""
    pins = parse_source_hashes(SOURCE_HASHES.read_text())
    out: dict[str, str] = {}
    for name in REQUIRED_FILES:
        want = pins.get(name)
        if want is None:
            raise SystemExit(f"SOURCE_HASHES.md pins no `{name}`")
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"{path}: missing (the oracle needs the full HF snapshot at {HF_COMMIT[:8]})")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
        out[name] = got
    return out


def install_cpu_patches(torch: Any) -> None:
    """`.cuda()` is identity and the cuda autocast a no-op (v1's --cpu-patch)."""
    torch.Tensor.cuda = lambda self, *a, **k: self
    orig_autocast = torch.autocast

    class AutocastShim(contextlib.AbstractContextManager):
        def __init__(self, device_type: str, *a: Any, **k: Any):
            self._cm = contextlib.nullcontext() if device_type == "cuda" else orig_autocast(device_type, *a, **k)

        def __enter__(self) -> Any:
            return self._cm.__enter__()

        def __exit__(self, *exc: Any) -> Any:
            return self._cm.__exit__(*exc)

    torch.autocast = AutocastShim
    torch.cuda.is_available = lambda: False


def install_f32_patches(torch: Any) -> None:
    """`Tensor.to(torch.bfloat16)` becomes float32 (v1's --f32)."""
    orig_to = torch.Tensor.to

    def to_no_bf16(self: Any, *a: Any, **k: Any) -> Any:
        a = tuple(torch.float32 if x is torch.bfloat16 else x for x in a)
        if k.get("dtype") is torch.bfloat16:
            k = dict(k)
            k["dtype"] = torch.float32
        return orig_to(self, *a, **k)

    torch.Tensor.to = to_no_bf16


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", SCRIPT_REL], capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────
class Capture:
    """Records one infer() call's seams through hooks and module-function wrappers."""

    def __init__(self, torch: Any, np: Any, model: Any, mu: Any, md: Any):
        self.torch, self.np, self.model, self.mu, self.md = torch, np, model, mu, md
        self.decoder = model.model
        self.layers = list(self.decoder.layers)
        self.handles: list[Any] = []
        self.restore: list[tuple[Any, str, Any]] = []
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        self.inputs: dict[str, Any] = {}
        self.prefill: dict[str, Any] = {}
        self.rows: list[int] = []
        self.rows_t: Any = None
        self.pages: list[int] = []
        self.phase = "idle"
        self.layer = -1
        self.step = -1
        self.cur: dict[str, Any] | None = None
        self.keeper = StepKeeper()
        self.flags: dict[str, list[int]] = {k: [] for k in ("ngram_banned", "guard", "finite_after", "raw_argmax")}
        self.infer_params: dict[int, dict[str, Any]] = {}
        self.fallback_rows: dict[int, dict[int, int]] = {}
        self.no_remote_rows: dict[int, int] = {}
        self.route_replica_mismatches = 0
        self.kbar_replica_mismatches = 0
        self.stage0: set[str] = set()

    def arr(self, t: Any) -> Any:
        t = t.detach()
        if t.dtype == self.torch.bool:
            return t.cpu().numpy().astype(self.np.uint8)
        if t.dtype in (self.torch.int64, self.torch.int32):
            return t.cpu().numpy().astype(self.np.int32)
        return t.float().cpu().numpy().copy()

    def put(self, stage: str, value: Any) -> None:
        if self.phase == "prefill":
            self.prefill[stage] = value
        elif self.phase == "decode" and self.cur is not None:
            self.cur["layers"][stage] = value

    def rows_of(self, t: Any) -> Any:
        """Selected prefill rows of a [B, S, ...] tensor, or the one decode row."""
        if self.phase == "prefill":
            return self.arr(t[0].index_select(0, self.rows_t))
        return self.arr(t[0, -1])

    # -- hook plumbing -------------------------------------------------------
    def wrap(self, owner: Any, name: str, make: Any) -> None:
        orig = getattr(owner, name)
        self.restore.append((owner, name, orig))
        setattr(owner, name, make(orig))

    def install(self) -> None:
        torch = self.torch
        self.handles.append(self.decoder.register_forward_pre_hook(self.on_decoder, with_kwargs=True))
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self.layer_input(i), with_kwargs=True))
            attn = layer.self_attn
            self.handles.append(attn.register_forward_pre_hook(self.attn_enter(i)))
            self.handles.append(attn.register_forward_hook(self.attn_exit(i)))
            self.handles.append(attn.o_proj.register_forward_pre_hook(self.o_proj_input(i)))
        self.handles.append(self.decoder.norm.register_forward_hook(self.final_norm))
        self.handles.append(self.model.lm_head.register_forward_hook(self.lm_head))
        attn0 = self.layers[0].self_attn
        rope2d = getattr(attn0, "rotary_emb_2d_main", None)
        if isinstance(rope2d, torch.nn.Module):
            self.handles.append(rope2d.register_forward_hook(self.main_rope_out))
        self.wrap(type(attn0), "_main_rope_coords", self.wrap_coords)
        md, mu = self.md, self.mu
        self.wrap(md, "_dash_indexer_rope", self.wrap_router_rope)
        self.wrap(md, "_router_qk", self.wrap_router_qk)
        self.wrap(md, "_block_scores_infer", self.wrap_block_scores)
        self.wrap(md, "_infer_route_select", self.wrap_route_select)
        self.wrap(mu, "_uocr_build_inputs", self.wrap_build_inputs)
        self.wrap(mu, "_uocr_no_repeat_ngram", self.wrap_ngram)
        self.wrap(mu, "_uocr_page_count_guard", self.wrap_guard)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for owner, name, orig in reversed(self.restore):
            setattr(owner, name, orig)
        self.restore.clear()

    # -- inputs (L0) ---------------------------------------------------------
    def wrap_build_inputs(self, orig: Any) -> Any:
        def inner(*a: Any, **k: Any) -> Any:
            input_ids, imgs, seq_mask, spatial = orig(*a, **k)
            ids = [int(t) for t in input_ids.tolist()]
            self.inputs = {"prompt_ids": ids, "seq_mask": self.arr(seq_mask)}
            if imgs is not None:
                self.inputs["images_crop"] = self.arr(imgs[0])
                self.inputs["images_crop_dtype"] = str(imgs[0].dtype)
                self.inputs["images_ori"] = self.arr(imgs[1])
                self.inputs["images_ori_dtype"] = str(imgs[1].dtype)
                self.inputs["spatial"] = [[int(x) for x in r] for r in spatial.tolist()]
            self.rows = select_rows(ids)
            self.rows_t = self.torch.tensor(self.rows, dtype=self.torch.long)
            self.pages = row_pages(ids)
            return input_ids, imgs, seq_mask, spatial

        return inner

    # -- forward phases ------------------------------------------------------
    def on_decoder(self, _module: Any, _args: Any, kwargs: Any) -> None:
        ids = kwargs.get("input_ids")
        if ids is None:
            raise RuntimeError("the oracle expects the decoder to be called with input_ids")
        if ids.shape[1] > 1:
            self.phase = "prefill"
            self.step = 0
            self.cur = {"step": 0, "consumed": -1, "position": ids.shape[1] - 1, "layers": {}}
            return
        consumed = int(ids[0, 0])
        self.close_step()
        if consumed == PAGE_ID:
            self.keeper.page_generated_at(self.step)
        self.phase = "decode"
        self.step += 1
        pos = kwargs.get("position_ids")
        self.cur = {"step": self.step, "consumed": consumed, "position": int(pos[0, 0]) if pos is not None else -1, "layers": {}}

    def close_step(self) -> None:
        if self.cur is not None:
            self.keeper.finish(self.cur["step"], self.cur)
            self.cur = None

    def layer_input(self, i: int) -> Any:
        def hook(_m: Any, args: Any, kwargs: Any) -> None:
            hs = args[0] if args else kwargs["hidden_states"]
            self.put(f"L{i:02d}.hidden_in", self.rows_of(hs))

        return hook

    def attn_enter(self, i: int) -> Any:
        def hook(_m: Any, _args: Any) -> None:
            self.layer = i

        return hook

    def attn_exit(self, i: int) -> Any:
        def hook(_m: Any, _args: Any, out: Any) -> None:
            self.put(f"L{i:02d}.attn_out", self.rows_of(out[0]))
            self.layer = -1

        return hook

    def o_proj_input(self, i: int) -> Any:
        def hook(_m: Any, args: Any) -> None:
            self.put(f"L{i:02d}.attn_pre_o", self.rows_of(args[0]))

        return hook

    def final_norm(self, _m: Any, _args: Any, out: Any) -> None:
        self.put("final_hidden", self.rows_of(out))

    def lm_head(self, _m: Any, _args: Any, out: Any) -> None:
        if self.cur is None:
            raise RuntimeError("lm_head ran outside a step")
        logits = out.float()[0, -1]
        self.cur["logits"] = self.arr(logits)
        self.flags["raw_argmax"].append(int(self.torch.argmax(logits)))

    # -- page2d coordinates and RoPE tables (L1) ----------------------------
    def wrap_coords(self, orig: Any) -> Any:
        def inner(attn: Any, q_len: Any, past_kv: Any) -> Any:
            coords = orig(attn, q_len, past_kv)
            if attn.layer_idx == 0 and coords is not None:
                self.put("coords_tok", self.arr(coords[0][0]))
                self.put("coords_page", self.arr(coords[1][0]))
            return coords

        return inner

    def main_rope_out(self, _m: Any, _args: Any, out: Any) -> None:
        cos, sin = out
        self.put("main_cos", self.arr(cos[0]))
        self.put("main_sin", self.arr(sin[0]))

    def wrap_router_rope(self, orig: Any) -> Any:
        def inner(attn: Any, *a: Any, **k: Any) -> Any:
            pe = orig(attn, *a, **k)
            if attn.layer_idx == 0 and pe is not None:
                cos, sin = pe
                self.put("router_cos", self.arr(cos[0]))
                self.put("router_sin", self.arr(sin[0]))
            return pe

        return inner

    # -- router (L1-L2) ------------------------------------------------------
    def wrap_router_qk(self, orig: Any) -> Any:
        def inner(router: Any, hidden_states: Any, pe: Any) -> Any:
            idx_q, idx_k = orig(router, hidden_states, pe)
            i = self.layer
            self.put(f"L{i:02d}.iq", self.rows_of(idx_q))
            if self.phase == "prefill":
                self.put(f"L{i:02d}.ik", self.arr(idx_k[0]))
            else:
                self.put(f"L{i:02d}.ik", self.arr(idx_k[0, -1]))
            return idx_q, idx_k

        return inner

    def wrap_block_scores(self, orig: Any) -> Any:
        md, torch = self.md, self.torch
        F = torch.nn.functional

        def inner(router: Any, idx_q: Any, idx_k: Any, q_pos: Any, nblk: int) -> Any:
            sc = orig(router, idx_q, idx_k, q_pos, nblk)
            self.stage0.add(str(router.stage0))
            i = self.layer
            if router.stage0 == "head_cls":
                # kbar is local to the reference; recompute it with the reference's own
                # lines, then confirm the recomputation reproduces its scores bit for bit.
                B, S, d = idx_k.shape
                bs = router.block_size
                inv_sqrt_d = 1.0 / (d ** 0.5)
                Sp = nblk * bs
                blk_ids = torch.arange(nblk, device=idx_k.device)
                if Sp > S:
                    kk = F.pad(idx_k, (0, 0, 0, Sp - S))
                    real = F.pad(idx_k.new_ones(B, S, dtype=torch.bool), (0, Sp - S), value=False)
                else:
                    kk = idx_k[:, :Sp]
                    real = idx_k.new_ones(B, Sp, dtype=torch.bool)
                kk = kk.view(B, nblk, bs, d)
                vmask = real.view(B, nblk, bs)
                hc = router.head_cls.to(kk.dtype).view(1, 1, 1, d)
                aw = ((kk * hc).sum(-1) * inv_sqrt_d).clamp(-md._ACT_CLAMP, md._ACT_CLAMP)
                neg = torch.finfo(aw.dtype).min
                row_max = torch.where(vmask, aw, aw.new_full((), neg)).amax(dim=-1, keepdim=True)
                row_max = torch.where(vmask.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max))
                e = torch.exp(aw - row_max) * vmask.to(aw.dtype)
                w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                k_bar = torch.nan_to_num((w.unsqueeze(-1) * kk).sum(dim=2), nan=0.0, posinf=0.0, neginf=0.0)
                chunk_valid = vmask.any(dim=-1)
                out = []
                for c0 in range(0, idx_q.shape[1], router.q_chunk):
                    c1 = min(c0 + router.q_chunk, idx_q.shape[1])
                    z = torch.matmul(idx_q[:, c0:c1], k_bar.transpose(-1, -2)) * inv_sqrt_d
                    z = z.clamp(-md._ACT_CLAMP, md._ACT_CLAMP)
                    qblk = q_pos[:, c0:c1] // bs
                    keep = (blk_ids[None, None, :] <= (qblk[:, :, None] - router.local_blocks)) & chunk_valid[:, None, :]
                    out.append(z.masked_fill(~keep, md._SCORE_NEG))
                if not torch.equal(torch.cat(out, dim=1), sc):
                    self.kbar_replica_mismatches += 1
                if self.phase == "prefill":
                    self.put(f"L{i:02d}.kbar", self.arr(k_bar[0]))
                else:
                    self.put(f"L{i:02d}.kbar_last", self.arr(k_bar[0, -1]))
            self.put(f"L{i:02d}.scores", self.rows_of(sc))
            return sc

        return inner

    def wrap_route_select(self, orig: Any) -> Any:
        md, torch = self.md, self.torch

        def inner(block_scores: Any, block_cand: Any, p: Any) -> Any:
            topi, bias, valid = orig(block_scores, block_cand, p)
            i = self.layer
            self.infer_params.setdefault(i, {k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()})
            # The reference's own steps, kept to expose the intermediates; the outputs
            # must equal the reference's bit for bit.
            remote = block_scores > (md._SCORE_NEG * 0.5)
            cand = remote if block_cand is None else (remote & block_cand)
            fallback = ~cand.any(dim=-1) & remote.any(dim=-1)
            cand_used = torch.where(~cand.any(dim=-1, keepdim=True), remote, cand)
            x = torch.where(cand_used, block_scores * p["idx_scale"], block_scores.new_full((), md._ENTMAX_NEG))
            x = torch.nan_to_num(x, nan=md._ENTMAX_NEG, posinf=md._ENTMAX_NEG, neginf=md._ENTMAX_NEG)
            nblk = x.shape[-1]
            M = max(1, min(int(p["topm"]), nblk)) if int(p["topm"]) > 0 else nblk
            xv, xi = x.topk(M, dim=-1)
            prior = md._entmax15(xv, dim=-1)
            K = max(1, min(int(p["K"]), M))
            is_cand = cand_used.gather(-1, xi)
            r_xi, r_prior, r_valid = xi[..., :K], prior[..., :K], is_cand[..., :K]
            logp = torch.log(r_prior.clamp_min(p["floor"]))
            cnt = r_valid.sum(dim=-1, keepdim=True).clamp(min=1)
            mu = torch.where(r_valid, logp, torch.zeros_like(logp)).sum(dim=-1, keepdim=True) / cnt
            r_bias = torch.where(r_valid, (logp - mu) * (1.0 / p["sigma"]), torch.zeros_like(logp)).float()
            if not (torch.equal(r_xi, topi) and torch.equal(r_bias, bias) and torch.equal(r_valid, valid)):
                self.route_replica_mismatches += 1
            if self.phase == "prefill":
                per_page: dict[int, int] = collections.Counter()
                for row in fallback[0].nonzero().flatten().tolist():
                    per_page[self.pages[row]] += 1
                self.fallback_rows[i] = dict(sorted(per_page.items()))
                self.no_remote_rows[i] = int((~remote[0].any(dim=-1)).sum())
            cand_mask = remote if block_cand is None else block_cand.expand_as(remote)
            self.put(f"L{i:02d}.cand", self.rows_of(cand_mask))
            self.put(f"L{i:02d}.fallback", self.rows_of(fallback))
            self.put(f"L{i:02d}.topm_idx", self.rows_of(xi))
            self.put(f"L{i:02d}.topm_x", self.rows_of(xv))
            self.put(f"L{i:02d}.prior", self.rows_of(prior))
            self.put(f"L{i:02d}.topi", self.rows_of(topi))
            self.put(f"L{i:02d}.bias", self.rows_of(bias))
            self.put(f"L{i:02d}.valid", self.rows_of(valid))
            return topi, bias, valid

        return inner

    # -- logits processors (L4, DISC-U2-2) -----------------------------------
    def wrap_ngram(self, orig: Any) -> Any:
        torch = self.torch

        def inner(logits: Any, *a: Any, **k: Any) -> Any:
            before = int(torch.isneginf(logits).sum())
            out = orig(logits, *a, **k)
            self.flags["ngram_banned"].append(int(torch.isneginf(out).sum()) - before)
            return out

        return inner

    def wrap_guard(self, orig: Any) -> Any:
        torch = self.torch

        def inner(logits: Any, gen_ids: Any, expected_pages: int, page_id: int, eos_id: int, max_page_tokens: int, st: Any) -> Any:
            if len(self.flags["ngram_banned"]) < len(self.flags["raw_argmax"]):
                self.flags["ngram_banned"].append(0)  # the n-gram processor returned early
            out = orig(logits, gen_ids, expected_pages, page_id, eos_id, max_page_tokens, st)
            kind = GUARD_NONE
            if expected_pages > 0:
                mc = st["marker_cum"]
                if mc > expected_pages:
                    kind = GUARD_EOS_ONLY
                elif 0 < max_page_tokens <= st["page_len"]:
                    kind = GUARD_CAP_FORCED
                elif mc < expected_pages:
                    kind = GUARD_EOS_BANNED
            self.flags["guard"].append(kind)
            self.flags["finite_after"].append(int(torch.isfinite(out).sum()))
            return out

        return inner


# ─────────────────────────────────────────────────────────────────────────────
# One fixture
# ─────────────────────────────────────────────────────────────────────────────
def stack_steps(np: Any, rows: list[Any]) -> Any:
    """Stack per-step arrays. Rows that grow with the context (scores and the
    candidate mask over nblk, the top-M and top-K selections while nblk < M) are
    padded: -1 for ints, 0 for masks, 0.0 for floats. `decode.nblk`, `decode.topm`
    and `decode.k` give each step's real lengths."""
    if len({r.shape for r in rows}) == 1:
        return np.stack(rows)
    width = max(r.shape[0] for r in rows)
    fill = {"i": -1, "u": 0, "f": 0.0}[rows[0].dtype.kind]
    out = np.full((len(rows), width), fill, dtype=rows[0].dtype)
    for j, r in enumerate(rows):
        out[j, : r.shape[0]] = r
    return out


def write_activations(np: Any, out_dir: Path, arrays: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for stage, arr in sorted(arrays.items()):
        arr = np.ascontiguousarray(arr)
        path = out_dir / f"{stage}.npy"
        np.save(path, arr, allow_pickle=False)
        manifest[stage] = {
            "file": path.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha256_bytes(arr.tobytes()),
            "file_sha256": sha256_file(path),
        }
    return manifest


def collect_arrays(np: Any, cap: Capture, n_layers: int) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    inp = cap.inputs
    arrays["prompt_ids"] = np.asarray(inp["prompt_ids"], dtype=np.int32)
    arrays["images_seq_mask"] = inp["seq_mask"]
    if "images_ori" in inp:
        arrays["images_ori"] = inp["images_ori"]
        arrays["images_crop"] = inp["images_crop"]
        arrays["images_spatial_crop"] = np.asarray(inp["spatial"], dtype=np.int32)
    arrays["prefill.rows"] = np.asarray(cap.rows, dtype=np.int32)
    for stage, value in cap.prefill.items():
        arrays[f"prefill.{stage}"] = value

    steps = sorted(cap.keeper.kept)
    recs = [cap.keeper.kept[s] for s in steps]
    arrays["steps.index"] = np.asarray(steps, dtype=np.int32)
    arrays["steps.consumed"] = np.asarray([r["consumed"] for r in recs], dtype=np.int32)
    arrays["steps.position"] = np.asarray([r["position"] for r in recs], dtype=np.int32)
    arrays["steps.logits"] = np.stack([r["logits"] for r in recs])
    dec = [r for r in recs if r["step"] >= 1]
    if dec:
        arrays["decode.index"] = np.asarray([r["step"] for r in dec], dtype=np.int32)
        for stage in sorted(dec[0]["layers"]):
            arrays[f"decode.{stage}"] = stack_steps(np, [r["layers"][stage] for r in dec])
        for name, stage in (("nblk", "L00.scores"), ("topm", "L00.topm_idx"), ("k", "L00.topi")):
            arrays[f"decode.{name}"] = np.asarray([r["layers"][stage].shape[0] for r in dec], dtype=np.int32)
    for key, values in cap.flags.items():
        arrays[f"all_steps.{key}"] = np.asarray(values, dtype=np.int32)
    return arrays


def run_case(ctx: dict[str, Any], case: Case, args: argparse.Namespace) -> dict[str, Any]:
    torch, np, model, tok, cap = ctx["torch"], ctx["np"], ctx["model"], ctx["tok"], ctx["cap"]
    corpus: Path = args.corpus
    paths = [corpus / name for name in case.pages]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"{p}: input page missing")
    inputs = []
    from PIL import Image

    for p in paths:
        with Image.open(p) as im:
            inputs.append({"name": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size, "width": im.width, "height": im.height})

    params = {
        "prompt": "",
        "crop_mode": case.crop_mode,
        "no_repeat_ngram_size": NGRAM_SIZE,
        "ngram_window": case.ngram_window,
        "max_page_tokens": MAX_PAGE_TOKENS,
        "page_count_guard": True,
        "temperature": 0.0,
        "base_size": 1024,
        "image_size": 640,
        "max_crops": 16,
        "dash_topm": 16,
        "max_length": 131072,
        "max_new_tokens": args.max_new_tokens,
        "eval_mode": True,
    }
    cap.reset()
    torch.manual_seed(SEED)
    t0 = time.time()
    res = model.infer(
        tok,
        image_file=[str(p) for p in paths],
        crop_mode=case.crop_mode,
        no_repeat_ngram_size=NGRAM_SIZE,
        ngram_window=case.ngram_window,
        max_page_tokens=MAX_PAGE_TOKENS,
        page_count_guard=True,
        max_new_tokens=args.max_new_tokens,
        eval_mode=True,
        return_dict=True,
    )
    wall = time.time() - t0
    cap.close_step()

    gen = [int(t) for t in res["token_ids"]]
    text = res["text"]
    n_steps = len(cap.flags["raw_argmax"])
    if len(cap.flags["guard"]) != n_steps or len(cap.flags["ngram_banned"]) != n_steps:
        raise RuntimeError(f"step flags out of step: {[len(v) for v in cap.flags.values()]}")
    if cap.inputs.get("prompt_ids") != [int(t) for t in res["prompt_token_ids"]]:
        raise RuntimeError("captured prompt ids differ from infer()'s")
    hazard_steps = [s for s, n in enumerate(cap.flags["finite_after"]) if n == 0]
    forced = collections.Counter(cap.flags["guard"])

    run_dir = args.out / args.dtype / args.tag
    act_dir = run_dir / "activations" / case.fixture
    arrays = collect_arrays(np, cap, len(cap.layers))
    manifest = write_activations(np, act_dir, arrays)
    output_sha = sha256_bytes(text.encode("utf-8"))
    reference = {
        "schema_version": 1,
        "doc": case.fixture,
        "model_id": "unlimited-ocr-2",
        "mode": case.mode,
        "pages": inputs,
        "params": params,
        "prompt": res.get("prompt"),
        "prompt_token_ids": cap.inputs["prompt_ids"],
        "token_ids": gen,
        "token_ids_sha256": ids_sha256(gen),
        "decoded_text": text,
        "decoded_text_sha256": output_sha,
        "finish_reason": res["finish_reason"],
        "num_pages_out": res["num_pages_out"],
        "expected_pages": res["expected_pages"],
        "wall_seconds": round(wall, 2),
        "prefill_seconds": round(res["prefill_time"], 3) if res.get("prefill_time") else None,
        "steps": n_steps,
        "counters": {
            "disc_u2_2_all_neg_inf_steps": len(hazard_steps),
            "disc_u2_2_first_steps": hazard_steps[:16],
            "guard_eos_only_steps": forced.get(GUARD_EOS_ONLY, 0),
            "guard_cap_forced_steps": forced.get(GUARD_CAP_FORCED, 0),
            "guard_eos_banned_steps": forced.get(GUARD_EOS_BANNED, 0),
            "ngram_banning_steps": sum(1 for n in cap.flags["ngram_banned"] if n > 0),
            "raw_argmax_overridden_steps": sum(
                1 for s, a in enumerate(cap.flags["raw_argmax"]) if a != (gen[s] if s < len(gen) else int(tok.eos_token_id))
            ),
        },
        "routing": {
            "stage0": sorted(cap.stage0),
            "infer_params": {str(k): v for k, v in sorted(cap.infer_params.items())},
            "prefill_rows": len(cap.rows),
            "prompt_rows": len(cap.inputs["prompt_ids"]),
            "fallback_rows_per_page": {str(k): {str(p): n for p, n in v.items()} for k, v in sorted(cap.fallback_rows.items())},
            "no_remote_rows": {str(k): v for k, v in sorted(cap.no_remote_rows.items())},
            "route_replica_mismatches": cap.route_replica_mismatches,
            "kbar_replica_mismatches": cap.kbar_replica_mismatches,
            "kept_steps": len(cap.keeper.kept),
        },
        "activations": manifest,
        "provenance": ctx["provenance"],
    }
    if cap.route_replica_mismatches or cap.kbar_replica_mismatches:
        raise RuntimeError(
            f"{case.fixture}: the dump replicas diverged from the reference "
            f"(route {cap.route_replica_mismatches}, kbar {cap.kbar_replica_mismatches})"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{case.fixture}_reference.json").write_text(json.dumps(reference, indent=1, ensure_ascii=False) + "\n")
    line = {
        "schema_version": 1,
        "test": "uocr2_oracle",
        "event": "result",
        "result": "pass",
        "fixture": case.fixture,
        "mode": case.mode,
        "device": "cpu",
        "dtype": args.dtype,
        "threads": args.threads,
        "tag": args.tag,
        "wall_s": round(wall, 2),
        "tokens": len(gen),
        "pages": len(case.pages),
        "pages_out": res["num_pages_out"],
        "finish_reason": res["finish_reason"],
        "output_sha256": output_sha,
        "token_ids_sha256": ids_sha256(gen),
        "disc_u2_2_hits": len(hazard_steps),
    }
    print(json.dumps(line), flush=True)
    with (args.out / "runs.ndjson").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    if args.commit_dir is not None:
        commit_case(args.commit_dir, reference, line)
    return reference


def commit_case(commit_dir: Path, reference: dict[str, Any], line: dict[str, Any]) -> None:
    """The small committed twin of one fixture: ids, markdown and its ORACLE.json entry."""
    commit_dir.mkdir(parents=True, exist_ok=True)
    fixture = reference["doc"]
    (commit_dir / f"{fixture}.ids.txt").write_text(ids_text("unlimited-ocr-2 HF f32 CPU oracle", fixture, reference["token_ids"]))
    (commit_dir / f"{fixture}.md").write_text(reference["decoded_text"])
    path = commit_dir / "ORACLE.json"
    doc = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "fixtures": {}}
    doc["provenance"] = reference["provenance"]
    doc["fixtures"][fixture] = {
        "mode": reference["mode"],
        "pages": reference["pages"],
        "params": reference["params"],
        "prompt_token_count": len(reference["prompt_token_ids"]),
        "prompt_token_ids_sha256": ids_sha256(reference["prompt_token_ids"]),
        "token_count": len(reference["token_ids"]),
        "token_ids_sha256": reference["token_ids_sha256"],
        "decoded_text_sha256": reference["decoded_text_sha256"],
        "finish_reason": reference["finish_reason"],
        "num_pages_out": reference["num_pages_out"],
        "expected_pages": reference["expected_pages"],
        "counters": reference["counters"],
        "routing": reference["routing"],
        "run": {k: line[k] for k in ("dtype", "threads", "tag", "wall_s")},
    }
    doc["fixtures"] = dict(sorted(doc["fixtures"].items()))
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="model-free checks of the pure helpers")
    p.add_argument("--model-dir", type=Path, help="HF snapshot of baidu/Unlimited-OCR-2 at 5632329a")
    p.add_argument("--corpus", type=Path, default=Path.home() / ".cache/franken_ocr/corpus/pages")
    p.add_argument("--out", type=Path, help="dump root (FOCR_FIXTURES_DIR/uocr2_oracle)")
    p.add_argument("--sets", default="S1,S2,S3,S4", help="comma-separated fixture sets")
    p.add_argument("--only", default="", help="comma-separated fixture ids within --sets")
    p.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    p.add_argument("--threads", type=int, required=False, default=12)
    p.add_argument("--tag", default="", help="run tag (default t<threads>)")
    p.add_argument("--max-new-tokens", type=int, default=None, help="smoke runs only; refused with --commit-dir")
    p.add_argument("--commit-dir", type=Path, default=None, help="write the committed twin (f32 only)")
    p.add_argument("--gundam-grids", action="store_true", help="print each corpus page's gundam tile grid and exit")
    args = p.parse_args(argv)
    if not args.self_test:
        if args.model_dir is None or (args.out is None and not args.gundam_grids):
            p.error("--model-dir and --out are required")
        if args.commit_dir is not None and (args.dtype != "f32" or args.max_new_tokens is not None):
            p.error("--commit-dir takes only full-length f32 runs")
        args.tag = args.tag or f"t{args.threads}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    set_semantic = [k for k in SEMANTIC_ENV if k in os.environ]
    if set_semantic:
        raise SystemExit(f"refusing: reference switches set in the environment: {set_semantic}")
    os.environ.update(QUIET_ENV)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # transformers copies the remote code into a module cache: keep one per run beside
    # the dumps, so concurrent runs never race on it.
    cache_root = (args.out or args.model_dir) / ".hf_modules" / f"{args.dtype}-{args.tag}"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "modules")

    stack = check_stack()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(args.threads)
    det_error = None
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001 - recorded in provenance
        det_error = repr(exc)
    model_files = verify_model_dir(args.model_dir)
    install_cpu_patches(torch)
    if args.dtype == "f32":
        install_f32_patches(torch)
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, use_safetensors=True, torch_dtype=dtype).eval()
    load_s = time.time() - t0
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if param_dtypes != [str(dtype)]:
        raise SystemExit(f"parameters loaded as {param_dtypes}, not {dtype}")
    mu = sys.modules[type(model).__module__]
    md = sys.modules[mu.__name__.rsplit(".", 1)[0] + ".modeling_deepseekv2"]

    if args.gundam_grids:
        from PIL import Image

        for page in sorted(args.corpus.glob("page_*.png")):
            with Image.open(page) as im:
                rgb = im.convert("RGB")
                _, ratio = mu.dynamic_preprocess(rgb, min_num=2, max_num=16, image_size=640)
                print(json.dumps({"page": page.name, "width": rgb.width, "height": rgb.height, "grid": list(ratio)}))
        return 0

    provenance = {
        "oracle": "hf-reference-cpu",
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(Path(__file__)),
        "git_revision": git_revision(),
        "command_argv": [sys.executable, *sys.argv],
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "hf_repo": HF_REPO,
        "hf_commit": HF_COMMIT,
        "model_files_sha256": model_files,
        "pinned": {"python": PIN_PYTHON, "torch": PIN_TORCH, "transformers": PIN_TRANSFORMERS, "tokenizers": PIN_TOKENIZERS, "huggingface_hub": PIN_HF_HUB},
        "stack": stack,
        "platform": {"system": platform.platform(), "machine": platform.machine()},
        "device": "cpu",
        "dtype": args.dtype,
        "compute_dtype": "float32" if args.dtype == "f32" else "bfloat16",
        "shims": ["cpu-patch", "f32"] if args.dtype == "f32" else ["cpu-patch"],
        "threads": args.threads,
        "determinism": {"seed": SEED, "torch_manual_seed": True, "torch_deterministic_algorithms": det_error is None, "error": det_error},
        "env": dict(QUIET_ENV),
        "load_seconds": round(load_s, 1),
        "smoke": args.max_new_tokens is not None,
    }
    cap = Capture(torch, np, model, mu, md)
    cap.install()
    ctx = {"torch": torch, "np": np, "model": model, "tok": tok, "cap": cap, "provenance": provenance}
    args.out.mkdir(parents=True, exist_ok=True)
    sets = fixture_sets()
    only = {s for s in args.only.split(",") if s}
    cases = [c for name in args.sets.split(",") for c in sets[name] if not only or c.fixture in only]
    if only - {c.fixture for c in cases}:
        raise SystemExit(f"unknown fixtures: {sorted(only - {c.fixture for c in cases})}")
    try:
        with torch.no_grad():
            for case in cases:
                run_case(ctx, case, args)
    finally:
        cap.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
