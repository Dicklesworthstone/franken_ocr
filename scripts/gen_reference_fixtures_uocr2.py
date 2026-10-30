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

--windowed W (bead U2-D2a) instead serves one base-mode document (--doc) the
way the pinned SGLang wheel does with SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=W:
page windows, a lookahead page and copy-splice KV reuse, ported from the
wheel's source onto this HF model in f32 on CPU. It writes <out>/<tag>/run.json
and w<bi>/{window.json,rows.*.npy}; see the "Windowed long-document decoding"
section for the ports and WINDOWED_SUBSTITUTIONS for what replaces the server.
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

# The 20-page corpus (`PAGES.txt`), the S1-corpus references of U2-C4's
# single-page gate (plan §7.1 "S1-corpus"; base mode, one page each).
CORPUS_PAGES = (
    "page_0009.png", "page_0014.png", "page_0019.png", "page_0023.png", "page_0086.png",
    "page_0091.png", "page_0105.png", "page_0107.png", "page_0108.png", "page_0126.png",
    "page_0136.png", "page_0179.png", "page_0250.png", "page_0358.png", "page_0400.png",
    "page_0437.png", "page_0547.png", "page_0557.png", "page_0590.png", "page_0645.png",
)


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
        "S1C": [Case(f"S1C-{page.removesuffix('.png')}", (page,), False) for page in CORPUS_PAGES],
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


def self_test(windowed: bool = False) -> int:
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
    if windowed:
        windowed_self_test(check)
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
# Windowed long-document decoding (bead U2-D2a): ports of the SGLang wheel
# ─────────────────────────────────────────────────────────────────────────────
# `--windowed W` runs one base-mode document the way SGLang serves it with
# SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=W: contiguous page windows, one lookahead page
# and copy-splice KV reuse between windows. It is written only from the pinned
# SGLang wheel (read as text, never imported) and runs on the HF model above in
# f32 on CPU. Each port names its wheel function and `file:line`, relative to
# sglang/srt/ in WHEEL:
#   DS  entrypoints/openai/doc_split.py       SC  entrypoints/openai/serving_chat.py
#   DR  session/uocr_doc_reuse.py             DP  session/uocr_doc_prefetch.py
#   CLP sampling/custom_logit_processor.py    BE  layers/attention/unlimited_ocr_v2_backend.py
#   PS  layers/attention/unlimited_dash_ops/page_state.py
#   PR  layers/attention/unlimited_dash_ops/page2d_rope.py
#   SB  managers/schedule_batch.py            BRP managers/scheduler_components/batch_result_processor.py
#   OS  managers/scheduler_components/output_streamer.py
#   CV  parser/conversation.py                ENV environ.py
# MU and MD are the HF modeling_unlimitedocr.py and modeling_deepseekv2.py.
# The server machinery around the ports is replaced by its simplest sequential
# equivalent; WINDOWED_SUBSTITUTIONS lists each replacement and goes into every dump.
WHEEL = "sglang-0.5.20.dev10+g8d62f5fb9c"
WHEEL_SHA256 = "a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d"

WINDOWED_SUBSTITUTIONS = (
    "scheduler: one request at a time, windows in plan order; no batching, overlap scheduling, "
    "CUDA graphs or chunked prefill (README --max-prefill-tokens 2048; every window prompt here is shorter)",
    "prefetch (DP, ENV:1368 on): a follow-up window is built after its predecessor finishes with "
    "max_new = min(remaining - used, capacity), which is what DP:65-123 caps it to (used = output minus forced tokens, OS:493-499)",
    "KV pool, allocator, req_to_token rows, eviction: per-window tensors indexed by logical row. A copy "
    "(move_kv_cache, memory_pool.py:5705) gathers main K, V and the index key of the source rows. Allocation "
    "never fails unless forced (alloc_failed, splice_alloc_failed). A retained window keeps every row; each row a "
    "follow-up reads is checked against tail_ranges (DR:323-367) and counted in read_outside_kept",
    "retraction, aborts, client disconnects: none",
    "kbar (DR:443-457 _refresh_kbar and the BE:1218-1350 tail/all refresh plans): not cached. HF "
    "_block_scores_infer recomputes every block summary from the stored index keys on each call, the same "
    "values a full refresh gives; only blocks local_blocks or more behind the query are scored, and those are complete",
    "attention and routing math: HF reference functions (MD _router_qk, _block_scores_infer, "
    "_infer_route_select, _decode_attend, and DashAttention._forward_dense with a query-row offset) instead of "
    "SGLang's Triton route_prefill/route_decode kernels and its attend kernels",
    "page isolation and RoPE coordinates: the PS/PR PyTorch step paths ported for one request; the fused "
    "decode_meta_fused/decode_allow_fused are documented to match them (BE:940-975)",
    "index keys stored in f32 as HF keeps them",
    "prompt ids and pixels: HF _uocr_build_inputs (BOS + '<PAGE><image>'*n + 'Multi page parsing.', 273 image "
    "tokens per base page), the layout SGLang's chat template (CV:650-700, CV:924-938) and tokenize_with_images "
    "(configs/unlimited_ocr.py:404-494) build. Image placeholders keep the raw id 128815, not SGLang's pad "
    "values; DR is_image_token treats both alike",
    "vision for a reused window's extend: base-mode page features recomputed with HF's own module calls "
    "(the MU forward base branch); a straddling item contributes its last row only",
    "detokenizer: tokenizer.decode(skip_special_tokens=False) of the output without its stop token; the "
    "spliced text is removed as a decoded prefix (hide_forced_text, DR:1202-1214); not the server's fastokens backend",
    "streaming: ownership is computed once on the final window text; stream_owned_text's result is a prefix "
    "of every later call's (DS:260-275), so the text sent while streaming equals the last chunk's",
    "context: planning and window capacity use ctx = 131072 (task setting). The wheel plans with "
    "max_req_input_len = min(context_len - 1, pool - 1) - 5 <= 131066 (SC:378, tp_worker.py:544-557); the difference "
    "cannot matter at W <= 6, where fits = 478 either way and the page cap binds the capacity. "
    "page_capacity(131072) = 1773 coordinate buckets (PR:20-29)",
    "sampling: temperature 0 = argmax of the processed f32 logits",
)

# ENV defaults and the README server configuration (HF snapshot README.md, SGLang section).
UOCR_DOC_LOOKAHEAD_PAGES = 1  # ENV:1362
UOCR_DOC_LOOKAHEAD_KV_REUSE = True  # ENV:1365
UOCR_NGRAM_OUTPUT_ONLY = True  # ENV:1423
UOCR_MAX_PAGE_TOKENS = 20000  # ENV:1374 defaults to 0; the README server exports 20000
# SGLANG_UOCR_DOC_BASE_PAGE_WINDOW (ENV:1357, default 1, README 10) is --windowed W.
SERVER_CONTEXT_LEN = 131072
README_TOKENS_PER_PAGE = 8000  # README generate(): max_tokens = 8000 * pages
README_PROMPT_TAIL = "Multi page parsing."
EOS_ID = 1
STOP_IDS = frozenset({EOS_ID})
PER_PAGE_IMAGE_TOKENS = 273

# DS:16-35, DS:76
DS_PAGE_MARKER = "<PAGE>"
DS_CONTEXT_RESERVE_TOKENS = 64
DS_OVERLAP_PAGES = 0
DS_TOKENS_PER_PAGE = {"gundam": 1514, "large": 422, "base": 274, "small": 112, "tiny": 74}
DS_DEFAULT_TOKENS_PER_PAGE = 1514
DS_PAGE_CAP_SLACK_TOKENS = 256
DS_LIGHT_IMAGE_MODES = frozenset(DS_TOKENS_PER_PAGE) - {"gundam"}  # DS:334

# DR:22-37, DR:193, DR:197
DR_MM_PAD_SHIFT_VALUE = 1_000_000
DR_KEY_DOC_ID = "uocr_doc_id"
DR_KEY_KEEP = "uocr_doc_keep"
DR_KEY_REUSE = "uocr_doc_reuse"
DR_KEY_OVERLAP = "uocr_doc_overlap"
DR_KEY_PAGE_BASE = "uocr_doc_page_base"
DR_KEY_LOOKAHEAD = "uocr_doc_lookahead"
DR_TAIL_TOKENS = 128
DR_CONTEXT_PAGES = 1

# PR:20-29 page_capacity
PR_MIN_PAGE_TOKENS = 74


def pr_page_capacity(context_len: int) -> int:
    """PR:23-29 page_capacity."""
    return max(1, int(context_len)) // PR_MIN_PAGE_TOKENS + 2


# Reasons the hook can force. All but splice_alloc_failed make the window recompute.
FALLBACK_REASONS = {
    "no_entry": "DR:958-960 try_match_prefix: no retained window for the document",
    "page_mismatch": "DR:1049-1056 _build_reuse: entry.next_page != page_base",
    "layout_mismatch": "DR:1059-1067 _build_reuse: plan_tail_copy returned None",
    "no_text": "DR:1069-1077 _build_reuse: no text tail, or len(text)+2 >= max_new",
    "alloc_failed": "DR:1083-1094 _build_reuse: allocator.alloc returned None",
    "straddle": "DR:1095-1099 _build_reuse: _rewrite_prompt refused a straddling item",
    "planning_exception": "DR:963-971 try_match_prefix: _build_reuse raised",
    "pool_too_small": "DR:792-795 on_no_token -> _drop_reuse (DR:684-691)",
    "splice_alloc_failed": "DR:1249-1261 _splice_text: alloc_extend returned None -> marker only",
}


# -- doc_split.py ------------------------------------------------------------
def ds_normalize_mode(image_mode: str | None) -> str:
    """DS:39-40 _normalize_mode."""
    return str(image_mode).strip().lower() if image_mode else ""


def ds_tokens_per_page(image_mode: str | None) -> int:
    """DS:43-45 tokens_per_page."""
    return DS_TOKENS_PER_PAGE.get(ds_normalize_mode(image_mode), DS_DEFAULT_TOKENS_PER_PAGE)


def ds_completion_capacity(*, context_length: int, image_mode: str | None, input_pages: int) -> int:
    """DS:64-72 completion_capacity."""
    return context_length - DS_CONTEXT_RESERVE_TOKENS - ds_tokens_per_page(image_mode) * input_pages


def ds_requested_max_page_tokens(custom_params: dict | None) -> int:
    """DS:79-86 requested_max_page_tokens."""
    if not custom_params:
        return 0
    try:
        return max(0, int(custom_params.get("max_page_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def ds_effective_max_page_tokens(*, server_max_page_tokens: int, custom_params: dict | None) -> int:
    """DS:89-103 effective_max_page_tokens."""
    own = ds_requested_max_page_tokens(custom_params)
    if server_max_page_tokens <= 0:
        return own
    if own <= 0:
        return server_max_page_tokens
    return min(own, server_max_page_tokens)


def ds_window_output_cap(*, own_pages: int, max_page_tokens: int) -> int | None:
    """DS:106-114 window_output_cap."""
    if max_page_tokens <= 0:
        return None
    return own_pages * max_page_tokens + DS_PAGE_CAP_SLACK_TOKENS


def ds_plan_batches(
    *,
    n_pages: int,
    image_mode: str | None,
    context_length: int,
    max_completion_tokens: int,
    max_batch_pages: int | None = None,
    overlap_pages: int = 0,
) -> list[tuple[int, int, int]] | None:
    """DS:117-167 plan_batches."""
    if n_pages <= 1 or context_length <= DS_CONTEXT_RESERVE_TOKENS:
        return None
    if max_completion_tokens <= 0:
        return None
    if overlap_pages < 0:
        raise ValueError("overlap_pages must be non-negative")
    per_page = ds_tokens_per_page(image_mode)
    fits = (context_length - DS_CONTEXT_RESERVE_TOKENS - 1) // per_page
    if max_batch_pages is not None:
        fits = min(fits, max_batch_pages)
    if fits < 1 or overlap_pages >= fits:
        return None
    policy_forces_split = max_batch_pages is not None and n_pages > max_batch_pages
    needs_budget_clamp = max_completion_tokens > ds_completion_capacity(
        context_length=context_length, image_mode=image_mode, input_pages=n_pages
    )
    if fits >= n_pages and not policy_forces_split:
        return [(0, n_pages, 0)] if needs_budget_clamp else None
    batches: list[tuple[int, int, int]] = []
    emit_from = 0
    while emit_from < n_pages:
        lo = max(0, emit_from - overlap_pages) if emit_from else 0
        hi = min(lo + fits, n_pages)
        batches.append((lo, hi, emit_from))
        if hi >= n_pages:
            break
        emit_from = hi
    return batches


def ds_split_messages(messages: list[dict], lo: int, hi: int) -> list[dict]:
    """DS:170-206 split_messages (dict messages; the attribute branch is unused)."""
    out = []
    seen = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, (list, tuple)):
            out.append(msg)
            continue
        kept_parts, kept_images = [], 0
        for part in content:
            ptype = part.get("type")
            if ptype in ("image_url", "image"):
                if lo <= seen < hi:
                    kept_parts.append(part)
                    kept_images += 1
                seen += 1
            else:
                kept_parts.append(part)
        kept_parts = ds_retag_markers(kept_parts, kept_images)
        out.append({**msg, "content": kept_parts})
    return out


def ds_retag_markers(parts: list[dict], n_images: int) -> list[dict]:
    """DS:209-234 _retag_markers (dict parts)."""
    fixed = []
    for part in parts:
        if part.get("type") != "text":
            fixed.append(part)
            continue
        text = part.get("text")
        if "<image>" not in (text or ""):
            fixed.append(part)
            continue
        tail = (text or "").split("<image>")[-1]
        fixed.append({**part, "text": ("<PAGE><image>" * n_images) + tail})
    return fixed


def ds_select_batch_output(*, text: str, input_pages: int, context_pages: int, leads_with_body: bool = False) -> tuple[str, int, int]:
    """DS:237-257 select_batch_output."""
    if not 0 <= context_pages < input_pages:
        raise ValueError("Invalid document batch page range")
    head, *tail = text.split(DS_PAGE_MARKER)
    bodies = [head, *tail] if leads_with_body or head.strip() else list(tail)
    while len(bodies) > input_pages and not bodies[0].strip():
        bodies = bodies[1:]
    owned_bodies = bodies[context_pages : context_pages + input_pages - context_pages]
    owned = "".join(DS_PAGE_MARKER + body for body in owned_bodies)
    return owned, len(bodies), input_pages


def ds_stream_owned_text(*, text: str, input_pages: int, context_pages: int, leads_with_body: bool) -> str:
    """DS:260-275 stream_owned_text."""
    for n in range(len(DS_PAGE_MARKER) - 1, 0, -1):
        if text.endswith(DS_PAGE_MARKER[:n]):
            text = text[:-n]
            break
    head, *tail = text.split(DS_PAGE_MARKER)
    bodies = [head, *tail] if leads_with_body or head.strip() else list(tail)
    return "".join(DS_PAGE_MARKER + body for body in bodies[context_pages:input_pages])


def ds_window_incomplete(observed: int, expected: int) -> bool:
    """DS:278-286 window_incomplete."""
    return observed < expected


def ds_owned_page_bodies(owned: str) -> list[str]:
    """DS:289-291 owned_page_bodies."""
    return owned.split(DS_PAGE_MARKER)[1:]


def ds_page_guard_params(*, max_page_tokens: int, page_token_id: int | None, eos_token_id: int | None, custom_params: dict | None, stop_only: bool = False) -> dict:
    """DS:294-330 page_guard_params."""
    if page_token_id is None or eos_token_id is None:
        return {}
    if max_page_tokens <= 0:
        if not stop_only:
            return {}
        return {"page_token_id": int(page_token_id), "eos_token_id": int(eos_token_id), "page_stop_only": True}
    own = ds_requested_max_page_tokens(custom_params)
    if 0 < own <= max_page_tokens:
        return {}
    return {"max_page_tokens": int(max_page_tokens), "page_token_id": int(page_token_id), "eos_token_id": int(eos_token_id)}


def ds_page_limit_for_mode(image_mode: str | None, gundam_limit: int | None, base_limit: int | None) -> int | None:
    """DS:337-343 page_limit_for_mode."""
    if ds_normalize_mode(image_mode) in DS_LIGHT_IMAGE_MODES:
        return base_limit
    return gundam_limit


# -- serving_chat.py ---------------------------------------------------------
def sc_window_pages(reused: bool, *, lo: int, hi: int, emit_from: int) -> dict:
    """SC:313-322 _window_pages."""
    if reused:
        return {"input_pages": hi - emit_from, "context_pages": 0, "leads_with_body": True}
    return {"input_pages": hi - lo, "context_pages": emit_from - lo, "leads_with_body": False}


def sc_plan_doc_split(*, n_pages: int, window: int, max_tokens: int, context_length: int = SERVER_CONTEXT_LEN) -> list[tuple[int, int, int]] | None:
    """SC:374-453 _plan_doc_split for the README request (chat template unlimited-ocr,
    images_config crop_mode False -> image_mode "base", no doc_split_options) with
    SGLANG_UOCR_DOC_BASE_PAGE_WINDOW = window."""
    if max_tokens is None or max_tokens <= 0:
        return None
    image_mode = "base"
    max_batch_pages = ds_page_limit_for_mode(image_mode, None, window)
    return ds_plan_batches(
        n_pages=n_pages,
        image_mode=image_mode,
        context_length=int(context_length),
        max_completion_tokens=int(max_tokens),
        max_batch_pages=max_batch_pages,
        overlap_pages=DS_OVERLAP_PAGES,
    )


def sc_window_lookahead(plan: list[tuple[int, int, int]], bi: int) -> int:
    """SC:470-480 _window_lookahead."""
    if bi >= len(plan) - 1:
        return 0
    _, hi, _ = plan[bi]
    _, last_hi, _ = plan[-1]
    return max(0, min(UOCR_DOC_LOOKAHEAD_PAGES, last_hi - hi))


def sc_window_capacity(plan: list[tuple[int, int, int]], bi: int, custom_params: dict, *, context_length: int, image_mode: str) -> int:
    """SC:482-502 _window_capacity."""
    lo, hi, _ = plan[bi]
    capacity = ds_completion_capacity(context_length=context_length, image_mode=image_mode, input_pages=hi - lo + sc_window_lookahead(plan, bi))
    page_tokens = ds_effective_max_page_tokens(server_max_page_tokens=UOCR_MAX_PAGE_TOKENS, custom_params=custom_params)
    cap = ds_window_output_cap(own_pages=hi - lo, max_page_tokens=page_tokens)
    return capacity if cap is None else min(capacity, cap)


def sc_apply_page_cap(custom_params: dict, *, stop_only: bool = False) -> dict:
    """SC:535-576 _apply_page_cap. The README request already names the
    DeepseekOCRNoRepeatNGramLogitProcessor, so only the params can change."""
    params = ds_page_guard_params(
        max_page_tokens=UOCR_MAX_PAGE_TOKENS,
        page_token_id=PAGE_ID,
        eos_token_id=EOS_ID,
        custom_params=custom_params,
        stop_only=stop_only,
    )
    if not params:
        return custom_params
    return {**custom_params, **params}


def sc_window_custom_params(base: dict, *, plan: list[tuple[int, int, int]], bi: int, doc_reuse_id: str | None) -> dict:
    """SC:504-533 _window_request: the window's custom_params."""
    lo, hi, emit_from = plan[bi]
    lookahead = sc_window_lookahead(plan, bi)
    cp = {**base, DR_KEY_PAGE_BASE: lo, DR_KEY_LOOKAHEAD: lookahead}
    cp = sc_apply_page_cap(cp, stop_only=lookahead > 0 or (doc_reuse_id is not None and bi > 0))
    if doc_reuse_id is not None:
        cp = {
            **cp,
            DR_KEY_DOC_ID: doc_reuse_id,
            DR_KEY_KEEP: bi < len(plan) - 1,
            DR_KEY_REUSE: bi > 0,
            DR_KEY_OVERLAP: emit_from - lo,
        }
    return cp


def cv_prompt(messages: list[dict]) -> tuple[list[str], str]:
    """The unlimited-ocr chat template (CV:924-938, SeparatorStyle.UNLIMITED_OCR at
    CV:405-415) over one user message, as CV:650-700 assembles it: when the text
    already carries one <image> per image, no placeholder is prepended."""
    (msg,) = messages
    images = [p["page"] for p in msg["content"] if p["type"] == "image_url"]
    texts = [p["text"] for p in msg["content"] if p["type"] == "text"]
    if sum(t.count("<image>") for t in texts) != len(images):
        raise ValueError("the oracle builds only prompts with one <image> per page")
    return images, "".join(texts)


# -- uocr_doc_reuse.py -------------------------------------------------------
def dr_is_image_token(token: int, image_token_id: int = IMAGE_ID) -> bool:
    """DR:145-147 is_image_token."""
    return token >= DR_MM_PAD_SHIFT_VALUE or token == image_token_id


@dataclass(frozen=True)
class DrPromptPages:
    """DR:150-157 PromptPages."""

    prefix_len: int
    page_starts: tuple[int, ...]
    page_ends: tuple[int, ...]
    instr_start: int


def dr_parse_prompt(ids: list[int], page_token_id: int = PAGE_ID, image_token_id: int = IMAGE_ID) -> DrPromptPages | None:
    """DR:160-188 parse_prompt."""
    n = len(ids)
    pos = 0
    while pos < n and ids[pos] != page_token_id:
        if dr_is_image_token(ids[pos], image_token_id):
            return None
        pos += 1
    if pos >= n:
        return None
    prefix_len = pos
    starts: list[int] = []
    ends: list[int] = []
    while pos < n and ids[pos] == page_token_id:
        starts.append(pos)
        pos += 1
        img_start = pos
        while pos < n and dr_is_image_token(ids[pos], image_token_id):
            pos += 1
        if pos == img_start:
            return None
        ends.append(pos)
    for t in ids[pos:]:
        if dr_is_image_token(t, image_token_id):
            return None
    return DrPromptPages(prefix_len, tuple(starts), tuple(ends), pos)


@dataclass(frozen=True)
class DrTailCopyPlan:
    """DR:200-231 TailCopyPlan."""

    src_positions: tuple[int, ...]
    new_ids: list[int]
    copy_len: int
    kept_from: int
    offset_delta: int
    old_pages: int
    new_pages: int
    copied_pages: int = 0
    allow_straddle: bool = False


def dr_plan_tail_copy(
    old_ids: list[int],
    new_ids: list[int],
    overlap: int,
    page_token_id: int,
    image_token_id: int,
    page_size: int,
    old_lookahead: int = 0,
    copy_lookahead: bool = False,
) -> DrTailCopyPlan | None:
    """DR:234-300 plan_tail_copy."""
    old = dr_parse_prompt(old_ids, page_token_id, image_token_id)
    new = dr_parse_prompt(new_ids, page_token_id, image_token_id)
    if old is None or new is None:
        return None
    p_old, p_new = len(old.page_starts) - old_lookahead, len(new.page_starts)
    if p_old < 1 or overlap < 0 or overlap >= p_new or overlap > p_old:
        return None
    end = old.page_ends[p_old - 1]
    start = end - DR_TAIL_TOKENS - end % page_size
    if start <= old.page_starts[p_old - 1]:
        return None
    new_end = new.page_ends[overlap - 1] if overlap > 0 else new.prefix_len
    if overlap > 0 and list(old_ids[start:end]) != list(new_ids[new_end - (end - start) : new_end]):
        return None
    marker_src = old.page_starts[p_old - 1]
    copy_end, kept_from, copied, straddle = end, new_end, 0, False
    if copy_lookahead and old_lookahead > 0 and overlap == 0:
        la_start, la_end = old.page_starts[p_old], old.page_ends[p_old]
        if la_start == end and list(old_ids[la_start:la_end]) == list(new_ids[new.page_starts[0] : new.page_ends[0]]):
            copy_end, kept_from, copied = la_end, new.page_ends[0], 1
            if p_new == 1:
                copy_end, kept_from, straddle = la_end - 1, new.page_ends[0] - 1, True
    head = [page_token_id] + list(old_ids[start:copy_end])
    src = (marker_src,) + tuple(range(start, copy_end))
    return DrTailCopyPlan(
        src_positions=src,
        new_ids=head + list(new_ids[kept_from : new.instr_start]),
        copy_len=len(head),
        kept_from=kept_from,
        offset_delta=len(head) - kept_from,
        old_pages=p_old,
        new_pages=p_new,
        copied_pages=copied,
        allow_straddle=straddle,
    )


def dr_text_tail(output_ids: list[int], n: int, page_token_id: int) -> tuple[int, list[int]] | None:
    """DR:303-320 text_tail."""
    if not output_ids or n <= 0:
        return None
    last_marker = -1
    for i in range(len(output_ids) - 1, -1, -1):
        if output_ids[i] == page_token_id:
            last_marker = i
            break
    start = max(last_marker + 1, len(output_ids) - n)
    return start, list(output_ids[start:])


def dr_tail_ranges(
    input_ids: list[int],
    output_ids: list[int],
    text_ok: bool,
    page_token_id: int,
    image_token_id: int,
    page_size: int,
    allocated: int,
    lookahead: int = 0,
    keep_lookahead: bool = False,
) -> list[tuple[int, int]] | None:
    """DR:323-367 tail_ranges."""
    pages = dr_parse_prompt(input_ids, page_token_id, image_token_id)
    if pages is None or len(pages.page_starts) <= lookahead:
        return None
    last = len(pages.page_starts) - lookahead - 1
    end = pages.page_ends[last]
    spans = [
        (pages.page_starts[last], pages.page_starts[last] + 1),
        (max(0, end - DR_TAIL_TOKENS - end % page_size), end),
    ]
    if keep_lookahead and lookahead > 0:
        spans.append((pages.page_starts[last + 1], pages.page_ends[last + 1]))
    if text_ok:
        tail = dr_text_tail(output_ids, DR_TAIL_TOKENS, page_token_id)
        if tail is not None:
            start = len(input_ids) + tail[0]
            spans.append((start, start + len(tail[1])))
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        lo = lo // page_size * page_size
        hi = min(-(-hi // page_size) * page_size, allocated)
        if lo >= hi:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def dr_replay_page_isolation(ids: list[int], page_token_id: int, image_token_id: int, per_page: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """DR:370-389 replay_page_isolation."""
    img_page, marker_cum, img_run, prompt_pages = [], [], [], []
    mc = ir = pp = 0
    for t in ids:
        is_img = dr_is_image_token(t, image_token_id)
        if t == page_token_id:
            mc += 1
        if is_img:
            ir += 1
            pp = max(pp, mc)
            img_page.append(mc - 1 if mc > 0 else (ir - 1) // per_page)
        else:
            img_page.append(-1)
        marker_cum.append(mc)
        img_run.append(ir)
        prompt_pages.append(pp)
    return img_page, marker_cum, img_run, prompt_pages


def dr_replay_coords(ids: list[int], gen_start: int, page_token_id: int, image_token_id: int, max_pages: int) -> tuple[list[int], tuple[int, int, list[int]]]:
    """DR:392-415 replay_coords."""
    pages: list[int] = []
    cnt = [0] * max_pages
    mc = pp = 0
    for i, t in enumerate(ids):
        is_img = dr_is_image_token(t, image_token_id)
        if t == page_token_id:
            mc += 1
        if is_img:
            pp = max(pp, mc)
        in_prefill = i < gen_start and (is_img or mc > pp)
        resp = min(max(mc - pp, 0), pp)
        page = mc if in_prefill else resp
        page = min(max(page, 0), max_pages - 1)
        cnt[page] += 1
        pages.append(page)
    return pages, (mc, pp, cnt)


def dr_continuation_carry(new_ids: list[int], page_token_id: int, image_token_id: int, per_page: int, max_pages: int) -> tuple[tuple[int, int, list[int]], tuple[int, int, int]]:
    """DR:418-437 continuation_carry."""
    _, (mc, pp, cnt) = dr_replay_coords(new_ids, len(new_ids), page_token_id, image_token_id, max_pages)
    iso = dr_replay_page_isolation(new_ids, page_token_id, image_token_id, per_page)
    return (mc + DR_CONTEXT_PAGES, pp, cnt), (pp + DR_CONTEXT_PAGES, iso[2][-1], pp)


def dr_cut_at_extra_marker(out: list[int], page_token_id: int, pages: int) -> list[int]:
    """DR:807-820 _cut_at_extra_marker."""
    seen = 0
    for i, t in enumerate(out):
        if t == page_token_id:
            seen += 1
            if seen > pages:
                return out[:i]
    return out


def dr_next_page_after(input_ids: list[int], first_page: int, context_pages: int, page_token_id: int, image_token_id: int, lookahead: int = 0) -> int | None:
    """DR:1217-1234 next_page_after."""
    pages = dr_parse_prompt(input_ids, page_token_id, image_token_id)
    if pages is None:
        return None
    return first_page + len(pages.page_starts) - context_pages - lookahead


def mm_items_offset(ids: list[int], mm_token_id: int = IMAGE_ID) -> list[tuple[int, int]]:
    """multimodal/processors/base_processor.py:1521-1534 get_mm_items_offset:
    inclusive (start, end) of each contiguous run of image tokens."""
    out, start = [], None
    for i, t in enumerate(ids):
        if t == mm_token_id and start is None:
            start = i
        if t == mm_token_id and (i + 1 == len(ids) or ids[i + 1] != mm_token_id):
            out.append((start, i))
            start = None
    return out


# -- custom_logit_processor.py -----------------------------------------------
GUARD_KINDS = ("none", "forced_eos_extra_marker", "forced_cap", "banned_eos", "banned_page")


def clp_page_guard(req: Any, page_token_id: int, max_page_tokens: int, eos_id: int, stop_only: bool = False, lookahead: int = 0) -> tuple[int | None, int | None] | None:
    """CLP:194-262 _page_guard. `req` carries output_ids, origin_input_ids and the
    _uocr_* / _page_cap_* attributes under the same names."""
    output_ids = req.output_ids
    n_out = len(output_ids)
    end_on_marker = lookahead > 0 or getattr(req, "_uocr_end_on_marker", False)
    if stop_only and not end_on_marker:
        return None
    expected = getattr(req, "_page_cap_expected", None)
    if expected is None:
        expected = sum(1 for t in req.origin_input_ids if t == page_token_id) - getattr(req, "_uocr_context_pages", 0) - lookahead
        req._page_cap_expected = expected
    if expected <= 0:
        return None
    seen, marks, last = getattr(req, "_page_cap_state", (0, 0, -1))
    if seen > n_out:
        seen, marks, last = 0, 0, -1
    for i in range(seen, n_out):
        if output_ids[i] == page_token_id:
            marks += 1
            last = i
    req._page_cap_state = (n_out, marks, last)
    if marks > expected:
        return eos_id, None
    if stop_only:
        return None
    if 0 < max_page_tokens <= n_out - (last + 1):
        return (page_token_id if marks < expected else eos_id), None
    if marks < expected:
        return None, eos_id
    if end_on_marker:
        return None
    return None, page_token_id


def clp_process_row(logits: Any, req: Any, params: dict) -> str:
    """CLP:269-387 DeepseekOCRNoRepeatNGramLogitProcessor.__call__ for one row, in
    place on `logits` [V]. Returns the guard's action kind (GUARD_KINDS)."""
    sequence = req.output_ids if UOCR_NGRAM_OUTPUT_ONLY else req.origin_input_ids + req.output_ids
    try:
        rep_max = int(params.get("rep_max_pattern_size") or 0)
        rep_min_count = int(params.get("rep_min_count") or 0)
        eos_id = params.get("eos_token_id")
        eos_id = int(eos_id) if eos_id is not None else None
    except (TypeError, ValueError):
        rep_max = rep_min_count = 0
        eos_id = None
    if rep_max > 0 and rep_min_count >= 2:
        raise NotImplementedError("the README request sets no repetition stop (CLP:300-313)")
    kind = "none"
    try:
        max_page_tokens = int(params.get("max_page_tokens") or 0)
        page_token_id = params.get("page_token_id")
        page_token_id = int(page_token_id) if page_token_id is not None else None
    except (TypeError, ValueError):
        max_page_tokens, page_token_id = 0, None
    vocab = logits.shape[-1]
    if page_token_id is not None and eos_id is not None and 0 <= page_token_id < vocab and 0 <= eos_id < vocab:
        action = clp_page_guard(
            req, page_token_id, max_page_tokens, eos_id,
            stop_only=bool(params.get("page_stop_only")),
            lookahead=int(params.get(DR_KEY_LOOKAHEAD) or 0),
        )
        if action is not None:
            forced, banned = action
            if forced is not None:
                logits[:] = -float("inf")
                logits[forced] = 0.0
                return "forced_eos_extra_marker" if forced == eos_id and req._page_cap_state[1] > req._page_cap_expected else "forced_cap"
            if banned is not None:
                logits[banned] = -float("inf")
                kind = "banned_eos" if banned == eos_id else "banned_page"
    try:
        ngram_size = int(params.get("ngram_size") or 0)
        window_size = int(params.get("window_size") or 0)
    except (TypeError, ValueError):
        return kind
    if ngram_size <= 0 or window_size <= 0 or len(sequence) < ngram_size:
        return kind
    search_start = max(0, len(sequence) - window_size)
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return kind
    current_prefix = sequence[-(ngram_size - 1) :] if ngram_size > 1 else []
    banned_tokens: set[int] = set()
    for idx in range(search_start, search_end):
        ngram = sequence[idx : idx + ngram_size]
        if ngram_size == 1 or ngram[:-1] == current_prefix:
            banned_tokens.add(ngram[-1])
    whitelist = {int(t) for t in (params.get("whitelist_token_ids") or [])}
    banned_tokens.difference_update(whitelist)
    indices = [t for t in banned_tokens if 0 <= t < vocab]
    if indices:
        logits[indices] = -float("inf")
    return kind


# -- page_state.py / page2d_rope.py: one request's row, stepped sequentially ---
class PageIsolationRow:
    """PS:182-371 PageIsolationState for one request. Its slots are the request's
    logical KV rows (row r of the window == slot of position r). The stale-slot
    clearing of PS:295-312 has nothing to clear: rows past the end are unwritten."""

    def __init__(self, per_page: int = PER_PAGE_IMAGE_TOKENS):
        self.per_page = per_page
        self.img_page: list[int] = []
        self.marker_cum: list[int] = []
        self.img_run: list[int] = []
        self.prompt_pages: list[int] = []

    def write(self, row: int, img_page: int, marker_cum: int, img_run: int, prompt_pages: int) -> None:
        """A slot write (PS:288-292, DR:460-469 _write_page_state)."""
        if row > len(self.img_page):
            raise RuntimeError(f"page-state write past the end ({row} > {len(self.img_page)})")
        if row == len(self.img_page):
            self.img_page.append(img_page)
            self.marker_cum.append(marker_cum)
            self.img_run.append(img_run)
            self.prompt_pages.append(prompt_pages)
        else:
            self.img_page[row], self.marker_cum[row], self.img_run[row], self.prompt_pages[row] = img_page, marker_cum, img_run, prompt_pages

    def carry(self, start_pos: int) -> tuple[int, int, int]:
        """PS:200-216 _carry: read off the slot of the token before the step."""
        p = start_pos - 1
        if p < 0:
            return 0, 0, 0
        return self.marker_cum[p], self.img_run[p], self.prompt_pages[p]

    def step(self, ids: list[int], start_pos: int, is_gen: list[bool]) -> list[int]:
        """PS:218-313 step: write this step's slots, return q_page per token."""
        prev_m, prev_i, prev_pp = self.carry(start_pos)
        mc, ir, run_pp = prev_m, prev_i, 0
        q_pages = []
        for j, t in enumerate(ids):
            is_img = dr_is_image_token(t)
            if t == PAGE_ID:
                mc += 1
            if is_img:
                ir += 1
            img_page = (mc - 1 if mc > 0 else (ir - 1) // self.per_page) if is_img else -1
            if is_img:
                run_pp = max(run_pp, mc)
            pp = max(run_pp, prev_pp)
            prefill_page = mc - 1 if mc > pp else -1
            resp_page = mc - 1 - pp
            resp_page = resp_page if resp_page >= 0 else -1
            text_page = resp_page if is_gen[j] else prefill_page
            q_pages.append(img_page if is_img else text_page)
            self.write(start_pos + j, img_page, mc, ir, pp)
        return q_pages

    def bounds(self, nblk: int, bs: int) -> tuple[list[int], list[int]]:
        """PS:315-328 block_page_bounds over the request's logical blocks."""
        bmin, bmax = [-1] * nblk, [-1] * nblk
        for r, p in enumerate(self.img_page[: nblk * bs]):
            if p < 0:
                continue
            b = r // bs
            bmax[b] = max(bmax[b], p)
            bmin[b] = p if bmin[b] < 0 else min(bmin[b], p)
        return bmin, bmax

    def allow(self, torch: Any, q_pages: list[int], nblk: int, bs: int) -> Any:
        """PS:330-371 _block_page_rows + build_query_allow: [T, nblk] bool."""
        bmin, bmax = self.bounds(nblk, bs)
        bmin_t = torch.tensor(bmin, dtype=torch.int64)[None, :]
        bmax_t = torch.tensor(bmax, dtype=torch.int64)[None, :]
        qp = torch.tensor(q_pages, dtype=torch.int64)[:, None]
        img_blk = bmin_t >= 0
        same = img_blk & (bmin_t <= qp) & (qp <= bmax_t) & (qp >= 0)
        open_row = (qp < 0) | ~same.any(dim=-1, keepdim=True)
        return same | open_row


class PageCoordRow:
    """PR:533-620 PageCoordState.step for one request. The row is keyed by the
    request's first KV block (PR:412-438 _rows): reset when a step starts at
    position 0, kept otherwise. No speculative rewind (PR:551 acts on verify
    snapshots only)."""

    def __init__(self, max_pages: int):
        self.max_pages = max_pages
        self.live = False
        self.st_marker = 0
        self.st_pp = 0
        self.cnt = [0] * max_pages

    def seed(self, marker: int, pp: int, cnt: list[int]) -> None:
        """DR:472-487 _seed_coord_row."""
        self.live = True
        self.st_marker, self.st_pp, self.cnt = marker, pp, list(cnt)

    def step(self, ids: list[int], start_pos: int, is_gen: list[bool]) -> tuple[list[int], list[int]]:
        """PR:534-620: (tok_pos, page) per token; the carry persists."""
        if start_pos <= 0 or not self.live:
            self.live = True
            self.st_marker, self.st_pp, self.cnt = 0, 0, [0] * self.max_pages
        mc, st_pp, run_pp = self.st_marker, self.st_pp, 0
        cnt = list(self.cnt)
        toks, pages = [], []
        pp = st_pp
        for j, t in enumerate(ids):
            is_img = dr_is_image_token(t)
            if t == PAGE_ID:
                mc += 1
            if is_img:
                run_pp = max(run_pp, mc)
            pp = max(run_pp, st_pp)
            in_prefill = (not is_gen[j]) and (is_img or mc > pp)
            resp = min(max(mc - pp, 0), pp)
            page = min(max(mc if in_prefill else resp, 0), self.max_pages - 1)
            # tok = cnt_in[page] + rank inside this step's (request, page) bucket
            toks.append(cnt[page])
            cnt[page] += 1
            pages.append(page)
        self.cnt = cnt
        self.st_marker, self.st_pp = mc, pp
        return toks, pages


def be_shift_page(pages: list[int], page_base: int, page_mod: int = 2) -> list[int]:
    """BE:1001-1010 note_prompt_lens (base % page_mod) + BE:1012-1031
    _shift_page_coords (added where page > 0)."""
    base = page_base % page_mod if page_mod > 0 else 0
    return [p + base if p > 0 else p for p in pages]


@dataclass
class WinReq:
    """The Req fields the ports read (SB Req), under the wheel's attribute names."""

    rid: str
    pages: list[str]
    prompt: str
    custom_params: dict
    max_new_tokens: int
    origin_input_ids: list[int] = None  # type: ignore[assignment]
    output_ids: list[int] = None  # type: ignore[assignment]
    items: list[dict] = None  # type: ignore[assignment]
    finished_reason: str | None = None
    finished_len: int | None = None
    _uocr_page_base: int | None = None
    _uocr_context_pages: int = 0
    _uocr_end_on_marker: bool = False
    _uocr_forced_len: int = 0
    _uocr_orig_prompt: Any = None
    _uocr_text_plan: Any = None
    _uocr_source: Any = None
    _page_cap_expected: int | None = None
    _page_cap_state: tuple[int, int, int] = (0, 0, -1)
    imgs: Any = None  # HF _uocr_build_inputs pixels, for the vision replica

    def doc_params(self) -> dict | None:
        """DR:553-557 _doc_params."""
        cp = self.custom_params
        if not isinstance(cp, dict) or not cp.get(DR_KEY_DOC_ID):
            return None
        return cp


def dr_page_base(req: WinReq) -> int:
    """DR:109-122 _page_base."""
    if req._uocr_page_base is not None:
        return int(req._uocr_page_base)
    cp = req.custom_params
    if not isinstance(cp, dict):
        return 0
    return int(cp.get(DR_KEY_PAGE_BASE) or 0)


def sb_update_finish_state(req: WinReq) -> None:
    """SB:1843-1888 update_finish_state for one accepted token: stop token before
    the length cap; no stop strings, grammar or vocab-boundary cases here."""
    if req.finished_reason is not None:
        return
    if req.output_ids and req.output_ids[-1] in STOP_IDS:
        req.finished_reason = "matched_token"
        return
    if len(req.output_ids) >= req.max_new_tokens:
        req.finished_reason = "length"
        req.finished_len = req.max_new_tokens


def windowed_self_test(check: Any) -> None:
    """Model-free checks of the windowed ports (`--self-test --windowed W`)."""
    check("w_plan_7_pages_window_2", ds_plan_batches(n_pages=7, image_mode="base", context_length=131072, max_completion_tokens=56000, max_batch_pages=2) == [(0, 2, 0), (2, 4, 2), (4, 6, 4), (6, 7, 6)])
    check("w_plan_7_pages_window_3", sc_plan_doc_split(n_pages=7, window=3, max_tokens=56000) == [(0, 3, 0), (3, 6, 3), (6, 7, 6)])
    check("w_plan_fits_is_unsplit", sc_plan_doc_split(n_pages=3, window=3, max_tokens=24000) is None and sc_plan_doc_split(n_pages=1, window=1, max_tokens=8000) is None)
    plan = sc_plan_doc_split(n_pages=7, window=2, max_tokens=56000)
    check("w_lookahead_is_one_but_last", [sc_window_lookahead(plan, b) for b in range(4)] == [1, 1, 1, 0])
    base = {"ngram_size": 35, "window_size": 1024, "page_token_id": PAGE_ID, "eos_token_id": EOS_ID, "max_page_tokens": 20000}
    check("w_capacity_is_page_cap", sc_window_capacity(plan, 0, base, context_length=131072, image_mode="base") == 2 * 20000 + 256)
    cp = sc_window_custom_params(base, plan=plan, bi=1, doc_reuse_id="d")
    check("w_window_params", cp[DR_KEY_PAGE_BASE] == 2 and cp[DR_KEY_LOOKAHEAD] == 1 and cp[DR_KEY_REUSE] and cp[DR_KEY_KEEP] and "page_stop_only" not in cp)
    msgs = [{"role": "user", "content": [{"type": "image_url", "page": f"p{i}"} for i in range(7)] + [{"type": "text", "text": "<PAGE><image>" * 7 + README_PROMPT_TAIL}]}]
    imgs, prompt = cv_prompt(ds_split_messages(msgs, 4, 7))
    check("w_split_messages_retags", imgs == ["p4", "p5", "p6"] and prompt == "<PAGE><image>" * 3 + README_PROMPT_TAIL)

    def page_ids(n_pages: int, bos: bool = True, instr: bool = True) -> list[int]:
        ids = [0] if bos else []
        for _ in range(n_pages):
            ids += [PAGE_ID] + [IMAGE_ID] * PER_PAGE_IMAGE_TOKENS
        return ids + ([10, 11] if instr else [])

    old, new = page_ids(3), page_ids(1)
    tp = dr_plan_tail_copy(old, new, 0, PAGE_ID, IMAGE_ID, 32, old_lookahead=1, copy_lookahead=True)
    check("w_tail_copy_straddles_a_one_page_window", tp is not None and tp.allow_straddle and tp.copied_pages == 1 and tp.src_positions[:2] == (275, 416) and tp.copy_len == 407 and tp.kept_from == 274 and tp.new_ids == [PAGE_ID] + old[416:822] + [IMAGE_ID])
    tp2 = dr_plan_tail_copy(old, page_ids(2), 0, PAGE_ID, IMAGE_ID, 32, old_lookahead=1, copy_lookahead=True)
    check("w_tail_copy_two_page_window", tp2 is not None and not tp2.allow_straddle and tp2.kept_from == 275 and tp2.new_ids[tp2.copy_len] == PAGE_ID and len(tp2.new_ids) == tp2.copy_len + 274)
    check("w_mm_offsets_are_inclusive", mm_items_offset(page_ids(2)) == [(2, 274), (276, 548)])
    check("w_text_tail_after_last_marker", dr_text_tail([5, PAGE_ID, 6, 7], 128, PAGE_ID) == (2, [6, 7]) and dr_text_tail([5, PAGE_ID], 128, PAGE_ID) == (2, []))
    check("w_cut_at_extra_marker", dr_cut_at_extra_marker([PAGE_ID, 5, PAGE_ID, 6, PAGE_ID, 7], PAGE_ID, 2) == [PAGE_ID, 5, PAGE_ID, 6])
    check("w_next_page_after", dr_next_page_after(old, 4, 0, PAGE_ID, IMAGE_ID, lookahead=1) == 6 and dr_next_page_after(tp2.new_ids, 2, 1, PAGE_ID, IMAGE_ID, lookahead=1) == 3)
    gen = [PAGE_ID, 5, 6, PAGE_ID, 7, PAGE_ID, 8]
    seq = old + gen
    iso = PageIsolationRow()
    iso.step(old, 0, [False] * len(old))
    for j, t in enumerate(gen):
        iso.step([t], len(old) + j, [True])
    rp = dr_replay_page_isolation(seq, PAGE_ID, IMAGE_ID, 273)
    check("w_isolation_step_matches_replay", (iso.img_page, iso.marker_cum, iso.img_run, iso.prompt_pages) == rp)
    cr = PageCoordRow(64)
    t0, p0 = cr.step(old[:300], 0, [False] * 300)
    t1, p1 = cr.step(old[300:], 300, [False] * (len(old) - 300))
    tg, pg = [], []
    for j, t in enumerate(gen):
        a, b = cr.step([t], len(old) + j, [True])
        tg += a
        pg += b
    pages, carry = dr_replay_coords(seq, len(old), PAGE_ID, IMAGE_ID, 64)
    check("w_coord_step_matches_replay", p0 + p1 + pg == pages and (cr.st_marker, cr.st_pp, cr.cnt) == carry)
    check("w_coord_text_page_starts_at_its_prompt_count", tg[0] == 274 and pg[:3] == [1, 1, 1] and pg[3] == 2 and p0[0] == 0 and p0[1] == 1)
    gc, gi = dr_continuation_carry(tp2.new_ids, PAGE_ID, IMAGE_ID, 273, 64)
    check("w_continuation_carry", gc[0] == 4 and gc[1] == 3 and gi == (4, dr_replay_page_isolation(tp2.new_ids, PAGE_ID, IMAGE_ID, 273)[2][-1], 3))
    check("w_shift_page", be_shift_page([0, 1, 2], 3) == [0, 2, 3] and be_shift_page([0, 1, 2], 4) == [0, 1, 2])

    class R:
        pass

    r = R()
    r.origin_input_ids, r.output_ids = page_ids(3), []
    r._uocr_context_pages, r._uocr_end_on_marker = 0, False
    check("w_guard_bans_eos_while_pages_owed", clp_page_guard(r, PAGE_ID, 20000, EOS_ID, lookahead=1) == (None, EOS_ID) and r._page_cap_expected == 2)
    r.output_ids = [PAGE_ID, 5, PAGE_ID, 6]
    check("w_guard_lookahead_leaves_last_page_open", clp_page_guard(r, PAGE_ID, 20000, EOS_ID, lookahead=1) is None)
    r.output_ids = [PAGE_ID, 5, PAGE_ID, 6, PAGE_ID]
    check("w_guard_extra_marker_forces_eos", clp_page_guard(r, PAGE_ID, 20000, EOS_ID, lookahead=1) == (EOS_ID, None))
    r2 = R()
    r2.origin_input_ids, r2.output_ids = page_ids(2), [PAGE_ID, 5, PAGE_ID, 6]
    r2._uocr_context_pages, r2._uocr_end_on_marker = 0, False
    check("w_guard_masks_page_on_last_page_of_an_unsplit_doc", clp_page_guard(r2, PAGE_ID, 20000, EOS_ID) == (None, PAGE_ID))
    r2._page_cap_expected, r2._page_cap_state = None, (0, 0, -1)
    r2.output_ids = [PAGE_ID, 5]
    check("w_guard_cap_turns_the_page", clp_page_guard(r2, PAGE_ID, 1, EOS_ID) == (PAGE_ID, None))
    owned, obs, exp = ds_select_batch_output(text="<PAGE>a<PAGE>b<PAGE>", input_pages=2, context_pages=0)
    check("w_select_drops_the_lookahead_body", owned == "<PAGE>a<PAGE>b" and obs == 3 and exp == 2)
    owned, obs, exp = ds_select_batch_output(text="a<PAGE>b", **sc_window_pages(True, lo=2, hi=4, emit_from=2))
    check("w_reused_window_leads_with_body", owned == "<PAGE>a<PAGE>b" and ds_stream_owned_text(text="a<PAGE>b<PA", **sc_window_pages(True, lo=2, hi=4, emit_from=2)) == "<PAGE>a<PAGE>b")


# -- the model side ------------------------------------------------------------
# Row origins in a window's KV row (dumped per row).
ROW_PREFILL, ROW_COPIED, ROW_SPLICED, ROW_DECODE = 0, 1, 2, 3


@dataclass
class KVRow:
    """One request's KV row: per-layer main K/V (in an HF DynamicCache), the index
    keys (`cache._dash_idx_k`), the page state, the coordinate carry and per-row
    records for the dump."""

    cache: Any
    page_base: int
    gen_start: int
    iso: PageIsolationRow
    coord: PageCoordRow
    length: int = 0
    rows: dict = None  # type: ignore[assignment]

    def record(self, **cols: list) -> None:
        for k, v in cols.items():
            self.rows.setdefault(k, []).extend(v)


@dataclass
class DocEntry:
    """DR:522-537 DocEntry, holding the finished window's KV tensors."""

    input_ids: list[int]
    output_ids: list[int]
    text_ok: bool
    next_page: int
    lookahead: int
    kept: list[tuple[int, int]] | None
    keys: list[Any]
    values: list[Any]
    idx_keys: list[Any]
    rows: dict
    window: int


@dataclass
class TextPlan:
    """DR:539-550 TextPlan (source rows instead of slots)."""

    forced: list[int]
    src_rows: list[int]
    iso: tuple[int, int, int]
    coord: tuple[int, int, list[int]]


class WindowedOracle:
    """Drives the HF model through SGLang's windowed decoding of one document."""

    def __init__(self, torch: Any, np: Any, model: Any, tok: Any, mu: Any, md: Any):
        self.torch, self.np, self.model, self.tok, self.mu, self.md = torch, np, model, tok, mu, md
        self.decoder = model.model
        self.layers = list(self.decoder.layers)
        attn0 = self.layers[0].self_attn
        if any(getattr(l.self_attn, "dash_router", None) is None for l in self.layers):
            raise RuntimeError("the windowed oracle expects every decoder layer to be a DASH layer")
        router = attn0.dash_router
        self.bs = int(router.block_size)
        self.max_pages = pr_page_capacity(SERVER_CONTEXT_LEN)
        self.mask_neg = sys.modules[type(router).__module__]._MASK_NEG
        self.page_mod = int(getattr(attn0.rotary_emb_2d_main, "page_mod", 0) or 0)
        self.idx_page_mod = int(getattr(attn0.indexer.rotary_emb_2d, "page_mod", 0) or 0)
        if self.page_mod != 2 or self.idx_page_mod != 2:
            raise RuntimeError(f"page_mod main={self.page_mod} index={self.idx_page_mod}, expected 2 (BE:1006-1009)")
        self.docs: dict[str, DocEntry] = {}
        self.cur: dict[str, Any] | None = None
        self.counters: collections.Counter = collections.Counter()
        # Failure counters start at zero so every dump states them.
        self.counters.update({k: 0 for k in ("read_outside_kept", "copied_rope_parity_mismatch", "spliced_rope_parity_mismatch")})
        self.restore: list[tuple[Any, str, Any]] = []
        self.feature_cache: dict[tuple[int, int], Any] = {}

    # -- attention -----------------------------------------------------------
    def install(self) -> None:
        md = self.md
        orig_prefill, orig_decode = md._dash_attn_forward, md._dash_decode_forward

        def prefill(attn: Any, hidden_states: Any, attention_mask: Any, position_ids: Any, past_kv: Any) -> Any:
            if self.cur is None:
                return orig_prefill(attn, hidden_states, attention_mask, position_ids, past_kv)
            if attention_mask is not None:
                raise RuntimeError("the windowed oracle expects no padding mask")
            return self.attn(attn, hidden_states, position_ids, past_kv)

        def decode(attn: Any, hidden_states: Any, position_ids: Any, past_kv: Any) -> Any:
            if self.cur is None:
                return orig_decode(attn, hidden_states, position_ids, past_kv)
            return self.attn(attn, hidden_states, position_ids, past_kv)

        for name, fn, orig in (("_dash_attn_forward", prefill, orig_prefill), ("_dash_decode_forward", decode, orig_decode)):
            self.restore.append((md, name, orig))
            setattr(md, name, fn)

    def remove(self) -> None:
        for owner, name, orig in reversed(self.restore):
            setattr(owner, name, orig)
        self.restore.clear()

    def attn(self, attn: Any, hidden_states: Any, position_ids: Any, past_kv: Any) -> Any:
        """One DASH layer on this step's rows: MD _dash_attn_forward (MD:146-213) and
        _dash_decode_forward (MD:484-529), with the RoPE coordinates and the
        candidate mask taken from the ports (BE:976-987) instead of HF's builders,
        and the prefill attend generalised to rows past a cached prefix."""
        torch, md, cur = self.torch, self.md, self.cur
        idx, router = attn.indexer, attn.dash_router
        cfg = attn.config
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, attn.head_dim
        bsz, q_len, _ = hidden_states.size()
        q = attn.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)
        if cur["check_hf"] and attn.layer_idx == 0:
            self.check_hf_layer0(attn, q_len, past_kv, position_ids)
        tok_pos, rope_page = cur["tok"], cur["rope_page"]
        # BE:494-509 cos_sin: main and index RoPE both read md.coords, shifted by BE:1012-1031.
        cos, sin = attn.rotary_emb_2d_main(v, tok_pos, rope_page)
        q, k = md._llama_apply_rotary_pos_emb(q, k, cos, sin)
        k_all, v_all = past_kv.update(k, v, attn.layer_idx)
        idx_pe = md._dash_indexer_rope(attn, idx, router, v, position_ids, (tok_pos, rope_page) if idx.rope_mode == "page2d" else None, (cos, sin))
        p = router._infer_params
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            idx_q, idx_k_new = md._router_qk(router, hidden_states, idx_pe)
            if not hasattr(past_kv, "_dash_idx_k"):
                past_kv._dash_idx_k = {}
            prev = past_kv._dash_idx_k.get(attn.layer_idx)
            idx_k = idx_k_new if prev is None else torch.cat([prev, idx_k_new], dim=1)
            past_kv._dash_idx_k[attn.layer_idx] = idx_k
            S = k_all.shape[2]
            if idx_k.shape[1] != S:
                raise RuntimeError(f"index keys ({idx_k.shape[1]}) and main keys ({S}) out of step")
            nblk = -(-S // router.block_size)
            sc = md._block_scores_infer(router, idx_q, idx_k, position_ids, nblk)
            cand = cur["allow"][None, :, :nblk] if router.image_only else None
            topi, bias, valid = md._infer_route_select(sc, cand, p)
            if q_len == 1:
                out = md._decode_attend(q, k_all, v_all, topi, bias, valid, position_ids, router.block_size, router.local_window, router._scale)
        if q_len > 1:
            out = self.dense_extend(router, q, md._llama_repeat_kv(k_all, attn.num_key_value_groups), md._llama_repeat_kv(v_all, attn.num_key_value_groups), topi, bias, valid, S - q_len, nblk)
        out = out.to(hidden_states.dtype).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return attn.o_proj(out), None, past_kv, None

    def dense_extend(self, router: Any, q: Any, k: Any, v: Any, topi: Any, blk_bias: Any, valid: Any, past: int, nblk: int) -> Any:
        """DashAttention._forward_dense (dash_attention.py:1230-1261) with the query
        rows offset by `past` cached rows; at past == 0 it is that function."""
        torch = self.torch
        neg = self.mask_neg
        B, H, Lq, D = q.shape
        S = k.shape[2]
        kidx = torch.arange(S)
        key_block = kidx // router.block_size
        src = torch.where(valid, blk_bias, blk_bias.new_full((), neg))
        logp = torch.full((B, Lq, nblk), neg, dtype=blk_bias.dtype).scatter(-1, topi, src)
        chunks = []
        for qs in range(0, Lq, router.q_chunk):
            qe = min(qs + router.q_chunk, Lq)
            qidx = torch.arange(past + qs, past + qe)
            base = logp[:, qs:qe].index_select(-1, key_block)[:, None]
            causal = (kidx[None, :] <= qidx[:, None])[None, None]
            local = ((kidx[None, :] <= qidx[:, None]) & (kidx[None, :] > (qidx[:, None] - router.local_window)))[None, None]
            sel = base > (neg * 0.5)
            keep = causal & (local | sel)
            mask = torch.where(local, torch.zeros_like(base), base).masked_fill(~keep, neg)
            mask = mask.clamp_min(neg).to(q.dtype)
            chunks.append(torch.nn.functional.scaled_dot_product_attention(q[:, :, qs:qe, :], k, v, attn_mask=mask, is_causal=False, scale=router._scale))
        return torch.cat(chunks, dim=2)

    def check_hf_layer0(self, attn: Any, q_len: int, past_kv: Any, position_ids: Any) -> None:
        """In a fresh window HF's own builders apply: its coordinates (MU forward ->
        _build_page_rope_pos / _main_rope_coords) and candidate masks (MD
        _dash_block_cand, MU _uocr_dash_decode_allow) must equal the ports'."""
        torch, md, cur = self.torch, self.md, self.cur
        hf = attn._main_rope_coords(q_len, past_kv)
        if hf is None:
            self.counters["hf_check_coords_none"] += 1
        elif not (torch.equal(hf[0].to(torch.int64), cur["tok"]) and torch.equal(hf[1].to(torch.int64), cur["page"])):
            self.counters["hf_check_coords_mismatch"] += 1
        else:
            self.counters["hf_check_coords_equal"] += 1
        nblk = cur["allow"].shape[-1]
        if q_len > 1:
            idx = attn.indexer
            itm, ipi, qpi = md._dash_page_ids(attn, idx, attn._sparse_source_input_ids, q_len)
            hf_allow = md._dash_block_cand(attn, idx, attn.dash_router, position_ids, itm, ipi, qpi, nblk, 1, q_len)
            same = hf_allow is not None and torch.equal(hf_allow[0], cur["allow"])
        else:
            hf_allow = attn._dash_dec_allow
            same = hf_allow is not None and torch.equal(hf_allow[0, :nblk], cur["allow"][0])
        self.counters["hf_check_allow_equal" if same else "hf_check_allow_mismatch"] += 1

    # -- one forward over a window's rows ------------------------------------
    def forward_rows(self, kv: KVRow, ids: list[int], *, start: int, decode: bool, check_hf: bool, embeds: Any = None, vision: dict | None = None) -> Any:
        """Advance the page state and coordinates over rows [start, start+len(ids))
        in the order BE:976-987 builds them (page_state.step, build_query_allow,
        coord_state.step, then the page shift), then run the HF decoder on them.
        Returns the last row's final hidden state."""
        torch = self.torch
        n = len(ids)
        if start != kv.length:
            raise RuntimeError(f"rows must be appended in order ({start} != {kv.length})")
        # PS:56-65 generated_mask: decode tokens; extend rows at or past gen_start (BE:1047-1068).
        is_gen = [decode or (start + j) >= kv.gen_start for j in range(n)]
        q_pages = kv.iso.step(ids, start, is_gen)
        nblk = -(-(start + n) // self.bs)
        allow = kv.iso.allow(torch, q_pages, nblk, self.bs)
        toks, pages = kv.coord.step(ids, start, is_gen)
        rope_pages = be_shift_page(pages, kv.page_base, self.page_mod)
        self.cur = {
            "tok": torch.tensor([toks], dtype=torch.int64),
            "page": torch.tensor([pages], dtype=torch.int64),
            "rope_page": torch.tensor([rope_pages], dtype=torch.int64),
            "allow": allow,
            "check_hf": check_hf,
        }
        kv.record(ids=list(ids), origin=[ROW_DECODE if decode else ROW_PREFILL] * n, src_row=[-1] * n, q_page=q_pages, bucket=pages, tok_pos=toks, rope_page=rope_pages, frame_bucket=pages, frame_rope_page=rope_pages)
        kw: dict[str, Any] = {}
        if vision is not None:
            kw = vision
        try:
            # With embeddings given (an extend over a copied prefix) no ids go in:
            # MU forward then sets none of HF's own page-state attributes.
            out = self.decoder(
                input_ids=None if embeds is not None else torch.tensor([ids], dtype=torch.long),
                inputs_embeds=embeds,
                past_key_values=kv.cache,
                use_cache=True,
                position_ids=torch.arange(start, start + n, dtype=torch.long)[None],
                return_dict=True,
                **kw,
            )
        finally:
            self.cur = None
        kv.cache = out.past_key_values
        kv.length = start + n
        return out.last_hidden_state[:, -1:]

    def logits(self, h: Any) -> Any:
        return self.model.lm_head(h).float()[0, -1]

    # -- vision for an extend over a copied prefix ---------------------------
    def page_features(self, req: WinReq, page: int) -> Any:
        """The base-mode (no crop) branch of the MU UnlimitedOCRModel.forward vision
        path for one page: sam -> clip -> concat -> projector -> image_newline ->
        view_seperator, i.e. the 273 embeddings of that page's image tokens."""
        key = (id(req), page)
        if key in self.feature_cache:
            return self.feature_cache[key]
        torch, dm = self.torch, self.decoder
        image_ori_p = req.imgs[1][page : page + 1]
        with torch.no_grad():
            g1 = dm.sam_model(image_ori_p)
            g2 = dm.vision_model(image_ori_p, g1)
            gf = torch.cat((g2[:, 1:], g1.flatten(2).permute(0, 2, 1)), dim=-1)
            gf = dm.projector(gf)
            _, hw, n_dim = gf.shape
            h = w = int(hw ** 0.5)
            gf = gf.view(h, w, n_dim)
            gf = torch.cat([gf, dm.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
            gf = gf.view(-1, n_dim)
            feats = torch.cat([gf, dm.view_seperator[None, :]], dim=0)
        self.feature_cache[key] = feats
        return feats

    def extend_embeds(self, req: WinReq, start: int) -> Any:
        """Input embeddings of prompt rows [start, len): token embeddings with the
        image rows of every multimodal item (after DR:980-1035) scattered in."""
        torch = self.torch
        ids = req.origin_input_ids[start:]
        emb = self.decoder.embed_tokens(torch.tensor([ids], dtype=torch.long)).clone()
        covered = 0
        for item in req.items:
            for s, e in item["offsets"]:
                lo, hi = max(s, start), e + 1
                if lo >= hi:
                    continue
                feats = self.page_features(req, item["page"])
                if feats.shape[0] != e - s + 1:
                    raise RuntimeError(f"item of {e - s + 1} rows has {feats.shape[0]} features")
                emb[0, lo - start : hi - start] = feats[lo - s : hi - s]
                covered += hi - lo
        if covered != sum(1 for t in ids if t == IMAGE_ID):
            raise RuntimeError(f"extend has {sum(1 for t in ids if t == IMAGE_ID)} image rows, items cover {covered}")
        return emb

    # -- reuse: DR try_match_prefix / _build_reuse / _rewrite_prompt ----------
    def release_source(self, req: WinReq) -> None:
        """DR:615-621 _release_source."""
        req._uocr_source = None

    def restore_prompt(self, req: WinReq) -> None:
        """DR:662-681 _restore_prompt."""
        saved = req._uocr_orig_prompt
        req._uocr_orig_prompt = None
        req._uocr_page_base = None
        req._uocr_context_pages = 0
        req._uocr_end_on_marker = False
        if saved is None:
            return
        req.origin_input_ids, req.items = saved

    def rewrite_prompt(self, req: WinReq, plan: DrTailCopyPlan) -> bool:
        """DR:980-1035 _rewrite_prompt."""
        items = req.items
        if any(not item["offsets"] for item in items):
            return False
        straddling = [item for item in items if any(start < plan.kept_from <= end for start, end in item["offsets"])]
        if straddling and not (plan.allow_straddle and len(straddling) == 1 and straddling[0]["offsets"][-1][1] == plan.kept_from):
            return False
        req._uocr_orig_prompt = (list(req.origin_input_ids), [dict(item) for item in items])
        kept = []
        for item in items:
            if item not in straddling and all(end < plan.kept_from for _, end in item["offsets"]):
                continue
            kept.append({**item, "offsets": [(s + plan.offset_delta, e + plan.offset_delta) for s, e in item["offsets"]]})
        req.items = kept
        req.origin_input_ids = list(plan.new_ids)
        return True

    def try_match_prefix(self, req: WinReq, force: str | None, info: dict) -> dict | None:
        """DR:941-977 try_match_prefix (no retraction, no output yet)."""
        cp = req.doc_params()
        if cp is None or not cp.get(DR_KEY_REUSE):
            return None
        entry = self.docs.pop(cp[DR_KEY_DOC_ID], None)
        if force == "no_entry":
            entry = None
        if entry is None:
            info["fallback"] = "no_entry"
            return None
        req._uocr_source = entry
        try:
            if force == "planning_exception":
                raise RuntimeError("forced planning exception")
            result, reason = self.build_reuse(req, entry, int(cp.get(DR_KEY_OVERLAP) or 0), int(cp.get(DR_KEY_PAGE_BASE) or 0) + int(cp.get(DR_KEY_OVERLAP) or 0), force)
        except RuntimeError as exc:
            if force != "planning_exception":
                raise
            result, reason = None, f"planning_exception ({exc})"
        if result is None:
            self.release_source(req)  # DR:642-645 on_req_dropped
            info["fallback"] = reason
            return None
        if force == "pool_too_small":
            # DR:755-795 on_no_token -> _drop_reuse (DR:684-691)
            self.release_source(req)
            self.restore_prompt(req)
            req._uocr_text_plan = None
            info["fallback"] = "pool_too_small"
            return None
        return result

    def build_reuse(self, req: WinReq, entry: DocEntry, overlap: int, page_base: int, force: str | None) -> tuple[dict | None, str | None]:
        """DR:1038-1161 _build_reuse."""
        page_id, image_id, per_page, bs, max_pages = PAGE_ID, IMAGE_ID, PER_PAGE_IMAGE_TOKENS, self.bs, self.max_pages
        if entry.next_page != page_base or force == "page_mismatch":
            return None, "page_mismatch"
        new_ids = list(req.origin_input_ids)
        plan = dr_plan_tail_copy(entry.input_ids, new_ids, overlap, page_id, image_id, bs, old_lookahead=entry.lookahead, copy_lookahead=UOCR_DOC_LOOKAHEAD_KV_REUSE)
        if plan is None or force == "layout_mismatch":
            return None, "layout_mismatch"
        text = dr_text_tail(entry.output_ids, DR_TAIL_TOKENS, page_id) if entry.text_ok else None
        max_new = req.max_new_tokens
        if text is not None and len(text[1]) + 2 >= (max_new or 0):
            text = None
        if text is None or force == "no_text":
            return None, "no_text"
        forced = text[1] + [page_id]
        n_copy_text = len(forced) - 1
        if force == "alloc_failed":
            return None, "alloc_failed"
        if force == "straddle" or not self.rewrite_prompt(req, plan):
            return None, "straddle"
        new_ids = plan.new_ids
        req._uocr_page_base = page_base - DR_CONTEXT_PAGES
        req._uocr_context_pages = DR_CONTEXT_PAGES
        req._uocr_end_on_marker = True
        copy_len = plan.copy_len
        old_len = len(entry.input_ids)
        _, prefix_carry = dr_replay_coords(new_ids[:copy_len], copy_len, page_id, image_id, max_pages)
        iso = dr_replay_page_isolation(new_ids, page_id, image_id, per_page)
        gen_coord, gen_iso = dr_continuation_carry(new_ids, page_id, image_id, per_page, max_pages)
        start = old_len + text[0]
        req._uocr_text_plan = TextPlan(forced=forced, src_rows=list(range(start, start + n_copy_text)), iso=gen_iso, coord=gen_coord)
        return {
            "plan": plan,
            "entry": entry,
            "values": tuple(v[:copy_len] for v in iso),
            "prefix_carry": prefix_carry,
        }, None

    def gather_rows(self, entry: DocEntry, rows: list[int]) -> tuple[list[Any], list[Any], list[Any]]:
        """move_kv_cache (memory_pool.py:5705): main K, V and the index key of rows."""
        torch = self.torch
        for r in rows:
            self.counters["retained_rows_read"] += 1
            if entry.kept is not None and not any(lo <= r < hi for lo, hi in entry.kept):
                self.counters["read_outside_kept"] += 1
            if r >= entry.keys[0].shape[2]:
                raise RuntimeError(f"row {r} past the retained window ({entry.keys[0].shape[2]} rows)")
        idx = torch.tensor(rows, dtype=torch.long)
        return (
            [k.index_select(2, idx) for k in entry.keys],
            [v.index_select(2, idx) for v in entry.values],
            [ik.index_select(1, idx) for ik in entry.idx_keys],
        )

    def new_cache(self, keys: list[Any] | None = None, values: list[Any] | None = None, idx_keys: list[Any] | None = None) -> Any:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        cache._dash_idx_k = {}
        if keys is not None:
            for l in range(len(self.layers)):
                cache.update(keys[l], values[l], l)
                cache._dash_idx_k[l] = idx_keys[l]
        return cache

    def cache_tensors(self, cache: Any) -> tuple[list[Any], list[Any], list[Any]]:
        keys, values = [], []
        for l in range(len(self.layers)):
            if hasattr(cache, "key_cache"):
                keys.append(cache.key_cache[l])
                values.append(cache.value_cache[l])
            else:
                keys.append(cache.layers[l].keys)
                values.append(cache.layers[l].values)
        return keys, values, [cache._dash_idx_k[l] for l in range(len(self.layers))]

    def copy_prefix(self, kv: KVRow, match: dict) -> None:
        """DR:500-516 _apply_segment for the copied prefix, queued at DR:1131-1141
        and run before the extend's metadata (BE:816): KV rows, page-state values
        (replayed over the rewritten prompt), kbar (recomputed, see the notes) and
        the coordinate row seeded with the prefix carry."""
        plan, entry = match["plan"], match["entry"]
        src = list(plan.src_positions)
        keys, values, idx_keys = self.gather_rows(entry, src)
        kv.cache = self.new_cache(keys, values, idx_keys)
        for r, vals in enumerate(zip(*match["values"])):
            kv.iso.write(r, *vals)
        kv.coord.seed(*match["prefix_carry"])
        kv.length = plan.copy_len
        er = entry.rows
        # The copied rows' coordinates in this window's frame (DR:392-415 replay_coords
        # over the rewritten prompt), against the source rotation they keep.
        frame, _ = dr_replay_coords(list(plan.new_ids[: plan.copy_len]), plan.copy_len, PAGE_ID, IMAGE_ID, self.max_pages)
        frame_rope = be_shift_page(frame, kv.page_base, self.page_mod)
        kept_rope = [er["rope_page"][r] for r in src]
        self.counters["copied_rows"] += len(src)
        self.counters["copied_rope_parity_mismatch"] += sum(1 for a, b in zip(frame_rope, kept_rope) if a % self.page_mod != b % self.page_mod)
        kv.record(
            ids=list(plan.new_ids[: plan.copy_len]),
            origin=[ROW_COPIED] * plan.copy_len,
            src_row=src,
            q_page=[-2] * plan.copy_len,
            bucket=[er["bucket"][r] for r in src],
            tok_pos=[er["tok_pos"][r] for r in src],
            rope_page=kept_rope,
            frame_bucket=frame,
            frame_rope_page=frame_rope,
        )
        if [er["ids"][r] for r in src][1:] != list(plan.new_ids[1 : plan.copy_len]) or er["ids"][src[0]] != PAGE_ID:
            raise RuntimeError("copied rows do not hold the planned ids")

    def splice_text(self, req: WinReq, kv: KVRow, entry: DocEntry, plan: TextPlan, force: str | None, info: dict) -> None:
        """DR:1237-1298 _splice_text, from DR:1164-1199 after_prefill_forward."""
        seq_len = kv.length
        if seq_len != len(req.origin_input_ids):
            raise RuntimeError("after_prefill_forward: seq_len != prompt length")
        n = len(plan.src_rows)
        if n > 0 and force == "splice_alloc_failed":
            n = 0
            info["splice"] = "splice_alloc_failed: marker only"
        if n > 0:
            keys, values, idx_keys = self.gather_rows(entry, plan.src_rows)
            for l in range(len(self.layers)):
                kv.cache.update(keys[l], values[l], l)
                kv.cache._dash_idx_k[l] = self.torch.cat([kv.cache._dash_idx_k[l], idx_keys[l]], dim=1)
            for j in range(n):
                kv.iso.write(seq_len + j, -1, *plan.iso)
            kv.coord.seed(*plan.coord)
            er = entry.rows
            # Their bucket in this frame is the response page the seeded carry implies
            # (PR:589-596 with marker = plan.coord[0], pp = plan.coord[1]).
            marker, pp = plan.coord[0], plan.coord[1]
            frame = [min(max(marker - pp, 0), pp)] * n
            frame_rope = be_shift_page(frame, kv.page_base, self.page_mod)
            kept_rope = [er["rope_page"][r] for r in plan.src_rows]
            self.counters["spliced_rows"] += n
            self.counters["spliced_rope_parity_mismatch"] += sum(1 for a, b in zip(frame_rope, kept_rope) if a % self.page_mod != b % self.page_mod)
            kv.record(
                ids=[er["ids"][r] for r in plan.src_rows],
                origin=[ROW_SPLICED] * n,
                src_row=list(plan.src_rows),
                q_page=[-2] * n,
                bucket=[er["bucket"][r] for r in plan.src_rows],
                tok_pos=[er["tok_pos"][r] for r in plan.src_rows],
                rope_page=kept_rope,
                frame_bucket=frame,
                frame_rope_page=frame_rope,
            )
            if [er["ids"][r] for r in plan.src_rows] != plan.forced[:-1]:
                raise RuntimeError("spliced rows do not hold the forced text")
            kv.length += n
            info["splice_rows"] = [seq_len, seq_len + n]
            info["splice_src_rows"] = [plan.src_rows[0], plan.src_rows[-1] + 1]
        else:
            # DR:490-497 _seed_continuation on the prompt's own last slot.
            kv.iso.write(seq_len - 1, -1, *plan.iso)
            kv.coord.seed(*plan.coord)
            info["splice_rows"] = [seq_len, seq_len]
            info.setdefault("splice", "marker only (empty text tail)")
        forced = plan.forced if n > 0 else plan.forced[-1:]
        req.output_ids.extend(forced)
        req._uocr_forced_len = len(forced)
        req.max_new_tokens += len(forced)
        info["forced_ids"] = list(forced)
        info["seed_iso"] = list(plan.iso)
        info["seed_coord"] = [plan.coord[0], plan.coord[1], {str(i): c for i, c in enumerate(plan.coord[2]) if c}]

    # -- DR:823-938 on_cache_finished_req + _trim_to_tails ---------------------
    def on_cache_finished_req(self, req: WinReq, kv: KVRow, window: int) -> dict:
        cp = req.doc_params()
        if cp is None:
            return {"retained": False, "why": "not a document window"}
        self.release_source(req)
        text_ok = req.finished_reason == "matched_token"
        if not cp.get(DR_KEY_KEEP) or req.finished_reason is None or not text_ok:
            return {"retained": False, "why": "keep=False" if not cp.get(DR_KEY_KEEP) else f"finish={req.finished_reason}"}
        committed = kv.length
        out = list(req.output_ids)
        if req.finished_len is not None:
            out = out[: req.finished_len]
        while out and out[-1] in STOP_IDS:
            out.pop()
        lookahead = int(cp.get(DR_KEY_LOOKAHEAD) or 0)
        if lookahead or req._uocr_end_on_marker:
            own_pages = sum(1 for t in req.origin_input_ids if t == PAGE_ID) - req._uocr_context_pages - lookahead
            out = dr_cut_at_extra_marker(out, PAGE_ID, own_pages)
        out = out[: max(0, committed - len(req.origin_input_ids))]
        overlap = int(cp.get(DR_KEY_OVERLAP) or 0)
        next_page = dr_next_page_after(req.origin_input_ids, int(cp.get(DR_KEY_PAGE_BASE) or 0) + overlap, req._uocr_context_pages or overlap, PAGE_ID, IMAGE_ID, lookahead=lookahead)
        if next_page is None:
            return {"retained": False, "why": "next_page_after is None"}
        keys, values, idx_keys = self.cache_tensors(kv.cache)
        kept = dr_tail_ranges(list(req.origin_input_ids), out, text_ok, PAGE_ID, IMAGE_ID, self.bs, committed, lookahead=lookahead, keep_lookahead=UOCR_DOC_LOOKAHEAD_KV_REUSE)
        entry = DocEntry(list(req.origin_input_ids), out, text_ok, next_page, lookahead, kept, keys, values, idx_keys, kv.rows, window)
        self.docs.pop(cp[DR_KEY_DOC_ID], None)
        self.docs[cp[DR_KEY_DOC_ID]] = entry
        return {"retained": True, "next_page": next_page, "output_ids_kept": len(out), "kept_ranges": kept, "committed": committed}

    # -- one request -----------------------------------------------------------
    def run_request(self, req: WinReq, force: str | None = None, check_hf: bool = True) -> dict:
        """One window (or the unsplit document) from prompt to finish."""
        torch, mu = self.torch, self.mu
        from PIL import Image

        t0 = time.time()
        info: dict[str, Any] = {"rid": req.rid, "pages": [Path(p).name for p in req.pages], "prompt": req.prompt, "custom_params": {k: v for k, v in req.custom_params.items()}, "max_new_tokens_sent": req.max_new_tokens, "forced_reason": force}
        images = []
        for p in req.pages:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        ids_t, imgs, seq_mask, spatial = mu._uocr_build_inputs(self.tok, req.prompt, images, crop_mode=False, base_size=1024, image_size=640, max_crops=16)
        req.origin_input_ids = [int(t) for t in ids_t.tolist()]
        req.items = [{"page": j, "offsets": [off]} for j, off in enumerate(mm_items_offset(req.origin_input_ids))]
        req.imgs, req.output_ids = imgs, []
        if len(req.items) != len(req.pages):
            raise RuntimeError("one multimodal item per page expected")
        info["prompt_ids_sent"] = list(req.origin_input_ids)
        dash_state = None
        cfg = self.model.config
        saved_cfg = (getattr(cfg, "_ring_window", None), getattr(cfg, "sliding_window", None))
        steps: collections.Counter = collections.Counter()
        try:
            # MU:1153-1157: the reference's decode-time configuration.
            cfg._ring_window = None
            cfg.sliding_window = None
            dash_state = self.md.configure_dash_inference(self.decoder, True, topm=16)
            match = self.try_match_prefix(req, force, info)
            page_base = dr_page_base(req)
            kv = KVRow(cache=None, page_base=page_base, gen_start=len(req.origin_input_ids), iso=PageIsolationRow(), coord=PageCoordRow(self.max_pages), rows={})
            info["page_base"] = {"custom_param": req.custom_params.get(DR_KEY_PAGE_BASE), "effective": page_base, "rope_shift": page_base % self.page_mod}
            if match is None:
                kv.cache = self.new_cache()
                vision = dict(images=[(imgs[0], imgs[1], spatial)], images_seq_mask=seq_mask.unsqueeze(0), images_spatial_crop=spatial)
                h = self.forward_rows(kv, req.origin_input_ids, start=0, decode=False, check_hf=check_hf, vision=vision)
                info["reused"] = False
            else:
                plan = match["plan"]
                self.copy_prefix(kv, match)
                embeds = self.extend_embeds(req, plan.copy_len)
                h = self.forward_rows(kv, req.origin_input_ids[plan.copy_len :], start=plan.copy_len, decode=False, check_hf=False, embeds=embeds)
                info["reused"] = True
                info["reuse_from_window"] = match["entry"].window
                info["tail_copy_plan"] = {
                    "src_positions": [plan.src_positions[0], plan.src_positions[1], plan.src_positions[-1] + 1],
                    "src_positions_note": "marker row, then rows [second, third) of the previous window",
                    "copy_len": plan.copy_len,
                    "kept_from": plan.kept_from,
                    "offset_delta": plan.offset_delta,
                    "old_pages": plan.old_pages,
                    "new_pages": plan.new_pages,
                    "copied_pages": plan.copied_pages,
                    "allow_straddle": plan.allow_straddle,
                }
                info["prompt_ids_rewritten"] = list(req.origin_input_ids)
                info["items_after_rewrite"] = req.items
            # The prefill step's sample (BRP:347-366).
            logits = self.logits(h)
            kind = clp_process_row(logits, req, req.custom_params)
            nxt = int(torch.argmax(logits))
            override = None
            if req._uocr_text_plan is not None:
                # DR:1164-1199 after_prefill_forward: the sample is dropped (BRP:358-361),
                # the text spliced and <PAGE> forced as the next decode input (DR:1296).
                plan_t = req._uocr_text_plan
                req._uocr_text_plan = None
                info["dropped_prefill_sample"] = nxt
                self.splice_text(req, kv, match["entry"], plan_t, force, info)
                override = plan_t.forced[-1]
            else:
                steps[kind] += 1
                req.output_ids.append(nxt)
            sb_update_finish_state(req)
            while req.finished_reason is None:
                inp = override if override is not None else req.output_ids[-1]
                override = None
                h = self.forward_rows(kv, [inp], start=kv.length, decode=True, check_hf=check_hf and match is None)
                logits = self.logits(h)
                kind = clp_process_row(logits, req, req.custom_params)
                steps[kind] += 1
                req.output_ids.append(int(torch.argmax(logits)))
                sb_update_finish_state(req)
        finally:
            if dash_state is not None:
                self.md.configure_dash_inference(self.decoder, False, state=dash_state)
            cfg._ring_window, cfg.sliding_window = saved_cfg
        wall = time.time() - t0
        forced_len = req._uocr_forced_len
        out = list(req.output_ids)
        gen = out[forced_len:]
        visible = gen[:-1] if gen and gen[-1] in STOP_IDS and req.finished_reason == "matched_token" else gen
        full = self.tok.decode(out[: len(out) - (len(gen) - len(visible))], skip_special_tokens=False)
        pre = self.tok.decode(out[:forced_len], skip_special_tokens=False) if forced_len else ""
        text = full[len(pre) :] if full.startswith(pre) else self.tok.decode(visible, skip_special_tokens=False)
        info.update(
            finish_reason=req.finished_reason,
            output_ids=out,
            forced_len=forced_len,
            generated_ids=visible,
            generated_ids_sha256=ids_sha256(visible),
            completion_tokens=len(out) - forced_len,
            text=text,
            max_new_tokens_final=req.max_new_tokens,
            guard_steps=dict(steps),
            page_cap_expected=req._page_cap_expected,
            context_pages=req._uocr_context_pages,
            end_on_marker=bool(req.custom_params.get(DR_KEY_LOOKAHEAD) or 0) or req._uocr_end_on_marker,
            kv_rows=kv.length,
            wall_seconds=round(wall, 2),
        )
        info["kv"] = kv
        return info

    # -- the document (SC:1043-1189, sequential) ------------------------------
    def run_document(self, pages: list[Path], window: int, force: dict[int, str], log: Any) -> dict:
        n_pages = len(pages)
        budget = README_TOKENS_PER_PAGE * n_pages
        base = {
            "ngram_size": NGRAM_SIZE,
            "window_size": NGRAM_WINDOW_MULTI if n_pages > 1 else NGRAM_WINDOW_SINGLE,
            "page_token_id": PAGE_ID,
            "eos_token_id": EOS_ID,
            "max_page_tokens": MAX_PAGE_TOKENS,
        }
        messages = [{"role": "user", "content": [{"type": "image_url", "page": str(p)} for p in pages] + [{"type": "text", "text": "<PAGE><image>" * n_pages + README_PROMPT_TAIL}]}]
        plan = sc_plan_doc_split(n_pages=n_pages, window=window, max_tokens=budget)
        doc: dict[str, Any] = {"pages": [p.name for p in pages], "window": window, "budget": budget, "plan": plan, "windows": []}
        if plan is None:
            # SC:349-356: unsplit, the page guard per _apply_page_cap.
            imgs, prompt = cv_prompt(messages)
            req = WinReq("unsplit", imgs, prompt, sc_apply_page_cap(dict(base)), budget)
            info = self.run_request(req, None)
            info["window"] = 0
            info["lo_hi_emit"] = [0, n_pages, 0]
            info["lookahead"] = 0
            # An unsplit request answers with its whole text (no ownership selection).
            info["owned"] = info["emitted"] = info["text"]
            info["bodies"] = ds_owned_page_bodies(info["text"])
            doc["windows"].append(info)
            doc["text"] = info["text"]
            doc["finish"] = "stop" if info["finish_reason"] == "matched_token" else "length"
            log(info)
            return doc
        doc_reuse_id = "uocr-doc-oracle" if len(plan) > 1 else None
        remaining = budget
        finishes, incomplete, emitted_all = [], False, []
        for bi, (lo, hi, emit_from) in enumerate(plan):
            if remaining <= 0:
                break
            lookahead = sc_window_lookahead(plan, bi)
            capacity = sc_window_capacity(plan, bi, base, context_length=SERVER_CONTEXT_LEN, image_mode="base")
            if capacity <= 0:
                raise RuntimeError(f"doc-split batch {bi} has no completion capacity")
            cp = sc_window_custom_params(base, plan=plan, bi=bi, doc_reuse_id=doc_reuse_id)
            imgs, prompt = cv_prompt(ds_split_messages(messages, lo, hi + lookahead))
            req = WinReq(f"w{bi}", imgs, prompt, cp, min(remaining, capacity))
            info = self.run_request(req, force.get(bi))
            info["window"] = bi
            info["lo_hi_emit"] = [lo, hi, emit_from]
            info["lookahead"] = lookahead
            info["capacity"] = capacity
            remaining -= info["completion_tokens"]
            reused = info["forced_len"] > 0  # DR:42-48 window_reused
            wp = sc_window_pages(reused, lo=lo, hi=hi, emit_from=emit_from)
            sent = ds_stream_owned_text(text=info["text"], **wp)
            owned, observed, expected = ds_select_batch_output(text=info["text"], **wp)
            emitted = owned if owned.startswith(sent) else sent
            info.update(owned=owned, emitted=emitted, ownership_moved=not owned.startswith(sent), bodies=ds_owned_page_bodies(owned), observed_pages=observed, expected_pages=expected)
            incomplete = incomplete or ds_window_incomplete(observed, expected)
            finishes.append("stop" if info["finish_reason"] == "matched_token" else "length")
            emitted_all.append(emitted)
            info["retention"] = self.on_cache_finished_req(req, info["kv"], bi)
            doc["windows"].append(info)
            log(info)
        if doc_reuse_id is not None:
            self.docs.pop(doc_reuse_id, None)  # DR:648-654 release_doc
        doc["text"] = "".join(emitted_all)
        doc["finish"] = "length" if any(f != "stop" for f in finishes) or len(finishes) < len(plan) or incomplete else "stop"
        return doc


def resolve_doc(spec: str) -> tuple[str, list[str]]:
    """`--doc FIXTURE[:N]`: the first N pages of a fixture's page list."""
    name, _, n = spec.partition(":")
    cases = {c.fixture: c for cases in fixture_sets().values() for c in cases}
    if name not in cases:
        raise SystemExit(f"--doc: unknown fixture {name}")
    pages = list(cases[name].pages)
    if n:
        pages = pages[: int(n)]
    return f"{name}:{len(pages)}" if n else name, pages


def write_window_dump(np: Any, out_dir: Path, info: dict) -> dict:
    """w<bi>/window.json plus the per-row arrays of the window's KV row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kv: KVRow = info.pop("kv")
    rows = dict(kv.rows)
    rows["img_page"] = kv.iso.img_page
    rows["marker_cum"] = kv.iso.marker_cum
    rows["img_run"] = kv.iso.img_run
    rows["prompt_pages"] = kv.iso.prompt_pages
    lengths = {k: len(v) for k, v in rows.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"row records out of step: {lengths}")
    arrays = {f"rows.{k}": np.asarray(v, dtype=np.int32) for k, v in rows.items()}
    manifest = write_activations(np, out_dir, arrays)
    info["rows_manifest"] = {k: {"shape": m["shape"], "sha256": m["sha256"]} for k, m in manifest.items()}
    info["rows_legend"] = {
        "origin": {"0": "prefilled prompt row", "1": "copied prefix row", "2": "spliced text row", "3": "decode input"},
        "q_page": "page-isolation query page of the row when it was a query (-1 unconstrained, -2 copied, not a query here)",
        "bucket": "PageCoordState page bucket (window-local); copied rows keep their source row's",
        "tok_pos": "PageCoordState token coordinate; copied rows keep their source row's",
        "rope_page": "the RoPE page coordinate the row's K and index key carry: bucket + page_base % 2 where bucket > 0 (BE:1012-1031); copied and spliced rows keep their source row's",
        "frame_bucket": "the row's bucket in this window's frame: as bucket for rows computed here; for copied rows replay_coords over the rewritten prompt (DR:392-415); for spliced rows the response bucket the seeded carry implies",
        "frame_rope_page": "frame_bucket + page_base % 2 where frame_bucket > 0; equal to rope_page modulo page_mod (2) on every row (counters *_rope_parity_mismatch)",
        "img_page/marker_cum/img_run/prompt_pages": "page-state slot values at the end of the window",
    }
    (out_dir / "window.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    return info


def run_windowed(args: argparse.Namespace, torch: Any, np: Any, model: Any, tok: Any, mu: Any, md: Any, provenance: dict) -> int:
    """The --windowed entry point: one document, optional self-test checks."""
    doc_name, page_names = resolve_doc(args.doc)
    pages = [args.corpus / p for p in page_names]
    for p in pages:
        if not p.is_file():
            raise SystemExit(f"{p}: input page missing")
    force: dict[int, str] = {}
    for spec in args.force_fallback:
        bi, _, reason = spec.partition(":")
        if reason not in FALLBACK_REASONS:
            raise SystemExit(f"--force-fallback {spec}: reason must be one of {sorted(FALLBACK_REASONS)}")
        force[int(bi)] = reason
    run_dir = args.out / args.tag
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir}: exists and is not empty (pick another --tag)")
    run_dir.mkdir(parents=True, exist_ok=True)
    eng = WindowedOracle(torch, np, model, tok, mu, md)
    eng.install()
    t0 = time.time()

    def log(info: dict) -> None:
        print(json.dumps({"event": "window", "window": info["window"], "lo_hi_emit": info["lo_hi_emit"], "reused": info.get("reused"), "fallback": info.get("fallback"), "tokens": len(info["generated_ids"]), "finish": info["finish_reason"], "wall_s": info["wall_seconds"], "since_start_s": round(time.time() - t0, 1)}), flush=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "test": "uocr2_windowed_oracle",
        "doc": doc_name,
        "window": args.windowed,
        "threads": args.threads,
        "forced_fallbacks": {str(k): {"reason": v, "wheel": FALLBACK_REASONS[v]} for k, v in force.items()},
        "wheel": {"name": WHEEL, "sha256": WHEEL_SHA256},
        "server_config": {
            "SGLANG_UOCR_EANABLE_DOC_LIMIT": 1,
            "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": args.windowed,
            "SGLANG_UOCR_DOC_LOOKAHEAD_PAGES": UOCR_DOC_LOOKAHEAD_PAGES,
            "SGLANG_UOCR_DOC_LOOKAHEAD_KV_REUSE": UOCR_DOC_LOOKAHEAD_KV_REUSE,
            "SGLANG_UOCR_MAX_PAGE_TOKENS": UOCR_MAX_PAGE_TOKENS,
            "SGLANG_UOCR_NGRAM_OUTPUT_ONLY": UOCR_NGRAM_OUTPUT_ONLY,
            "context_length": SERVER_CONTEXT_LEN,
            "request": {"max_tokens": f"{README_TOKENS_PER_PAGE} * pages", "temperature": 0, "images_config": {"crop_mode": False}, "ngram_size": NGRAM_SIZE, "window_size": "1024 if pages > 1 else 128", "max_page_tokens": MAX_PAGE_TOKENS},
        },
        "substitutions": list(WINDOWED_SUBSTITUTIONS),
        "provenance": provenance,
    }
    checks: dict[str, Any] = {}
    wanted = {c for c in args.windowed_check.split(",") if c}
    try:
        with torch.no_grad():
            torch.manual_seed(SEED)
            doc = eng.run_document(pages, args.windowed, force, log)
            doc_wall = time.time() - t0
            infos = [write_window_dump(np, run_dir / f"w{info['window']}", info) for info in doc["windows"]]
            result["plan"] = doc["plan"]
            result["document"] = {"text": doc["text"], "text_sha256": sha256_bytes(doc["text"].encode("utf-8")), "finish": doc["finish"], "bodies": ds_owned_page_bodies(doc["text"]), "wall_seconds": round(doc_wall, 1)}
            result["windows"] = [
                {k: info.get(k) for k in ("window", "lo_hi_emit", "lookahead", "reused", "fallback", "splice", "page_base", "finish_reason", "forced_len", "completion_tokens", "generated_ids_sha256", "kv_rows", "observed_pages", "expected_pages", "ownership_moved", "guard_steps", "wall_seconds", "retention")}
                | {"tokens": len(info["generated_ids"])}
                for info in infos
            ]
            if "degenerate" in wanted:
                checks["degenerate"] = degenerate_check(eng, args, pages, doc_name, infos)
            if "extend" in wanted:
                checks["extend_split"] = extend_split_check(eng, pages)
            if "recompute" in wanted:
                checks["recompute"] = recompute_check(eng, pages, args.windowed, force, infos)
    finally:
        eng.remove()
    result["counters"] = dict(eng.counters)
    result["checks"] = checks
    result["wall_seconds"] = round(time.time() - t0, 1)
    (run_dir / "run.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    summary = {"event": "result", "test": "uocr2_windowed_oracle", "doc": doc_name, "window": args.windowed, "tag": args.tag, "windows": len(result["windows"]), "tokens": [w["tokens"] for w in result["windows"]], "text_sha256": result["document"]["text_sha256"], "counters": result["counters"], "checks": {k: v.get("pass") for k, v in checks.items()}, "wall_s": result["wall_seconds"]}
    print(json.dumps(summary), flush=True)
    failed = [k for k, v in checks.items() if not v.get("pass")]
    bad_counters = {k: v for k, v in eng.counters.items() if k in ("hf_check_coords_mismatch", "hf_check_coords_none", "hf_check_allow_mismatch", "read_outside_kept", "copied_rope_parity_mismatch", "spliced_rope_parity_mismatch") and v}
    return 1 if failed or bad_counters else 0


def degenerate_check(eng: WindowedOracle, args: argparse.Namespace, pages: list[Path], doc_name: str, infos: list[dict]) -> dict:
    """W >= N: the oracle's own loop (one unsplit window) against HF infer_multi."""
    if len(infos) != 1 or infos[0]["lookahead"] != 0 or infos[0].get("reused"):
        return {"pass": False, "why": "not a single unsplit window"}
    mine = infos[0]["generated_ids"]
    captured: dict[str, Any] = {}
    model = eng.model
    orig = model._uocr_generate

    def grab(*a: Any, **k: Any) -> Any:
        res = orig(*a, **k)
        captured.update(res)
        return res

    model._uocr_generate = grab
    t0 = time.time()
    try:
        eng.torch.manual_seed(SEED)
        model.infer_multi(eng.tok, image_files=[str(p) for p in pages], no_repeat_ngram_size=NGRAM_SIZE, ngram_window=NGRAM_WINDOW_MULTI if len(pages) > 1 else NGRAM_WINDOW_SINGLE, max_page_tokens=MAX_PAGE_TOKENS, page_count_guard=True, eval_mode=True)
    finally:
        del model._uocr_generate
    hf_wall = time.time() - t0
    hf = [int(t) for t in captured["token_ids"]]
    first_diff = next((i for i, (a, b) in enumerate(zip(mine, hf)) if a != b), None if len(mine) == len(hf) else min(len(mine), len(hf)))
    out = {"pass": mine == hf, "oracle_tokens": len(mine), "hf_tokens": len(hf), "first_difference": first_diff, "oracle_wall_s": infos[0]["wall_seconds"], "hf_infer_multi_wall_s": round(hf_wall, 1), "hf_finish": captured.get("finish_reason"), "oracle_finish": infos[0]["finish_reason"]}
    committed = REPO_ROOT / "tests/fixtures/uocr2_oracle" / f"{doc_name}.ids.txt"
    if committed.is_file():
        ids = [int(t) for line in committed.read_text().splitlines() if not line.startswith("#") for t in line.split()]
        out["committed_fixture"] = str(committed.relative_to(REPO_ROOT))
        out["equals_committed_fixture"] = ids == mine
    return out


def extend_split_check(eng: WindowedOracle, pages: list[Path]) -> dict:
    """The generalised extend: one prompt prefilled whole, and prefilled as rows
    [0, P) then [P, S), must give the same logits (and the native HF vision path
    the same embeddings as the replica the extend uses)."""
    torch, mu = eng.torch, eng.mu
    from PIL import Image

    images = []
    for p in pages:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    prompt = "<PAGE><image>" * len(pages) + README_PROMPT_TAIL
    ids_t, imgs, seq_mask, spatial = mu._uocr_build_inputs(eng.tok, prompt, images, crop_mode=False, base_size=1024, image_size=640, max_crops=16)
    ids = [int(t) for t in ids_t.tolist()]
    req = WinReq("extend-check", [str(p) for p in pages], prompt, {}, 1)
    req.origin_input_ids, req.imgs = ids, imgs
    req.items = [{"page": j, "offsets": [off]} for j, off in enumerate(mm_items_offset(ids))]
    cfg = eng.model.config
    saved_cfg = (getattr(cfg, "_ring_window", None), getattr(cfg, "sliding_window", None))
    cfg._ring_window = None
    cfg.sliding_window = None
    state = eng.md.configure_dash_inference(eng.decoder, True, topm=16)
    seen: dict[str, Any] = {}
    hook = eng.layers[0].register_forward_pre_hook(lambda _m, a, k: seen.__setitem__("emb", (a[0] if a else k["hidden_states"]).detach().clone()), with_kwargs=True)

    def fresh() -> KVRow:
        return KVRow(cache=eng.new_cache(), page_base=0, gen_start=len(ids), iso=PageIsolationRow(), coord=PageCoordRow(eng.max_pages), rows={})

    out: dict[str, Any] = {"prompt_rows": len(ids)}
    try:
        kv = fresh()
        h_native = eng.forward_rows(kv, ids, start=0, decode=False, check_hf=False, vision=dict(images=[(imgs[0], imgs[1], spatial)], images_seq_mask=seq_mask.unsqueeze(0), images_spatial_crop=spatial))
        native_emb = seen["emb"]
        hook.remove()
        emb = eng.extend_embeds(req, 0)
        out["embeds_bitwise_equal"] = bool(torch.equal(emb, native_emb))
        out["embeds_max_abs_diff"] = float((emb - native_emb).abs().max())
        ref = eng.logits(h_native)
        kv_r = fresh()
        ref_mine = eng.logits(eng.forward_rows(kv_r, ids, start=0, decode=False, check_hf=False, embeds=emb))
        out["whole_prefill_replica_vs_native_logits_max_abs_diff"] = float((ref_mine - ref).abs().max())
        splits = {}
        for P in (333, 416, len(ids) - 1):
            kv2 = fresh()
            eng.forward_rows(kv2, ids[:P], start=0, decode=False, check_hf=False, embeds=emb[:, :P])
            lg = eng.logits(eng.forward_rows(kv2, ids[P:], start=P, decode=False, check_hf=False, embeds=emb[:, P:]))
            k_whole = eng.cache_tensors(kv_r.cache)
            k_split = eng.cache_tensors(kv2.cache)
            splits[str(P)] = {
                "logits_max_abs_diff": float((lg - ref_mine).abs().max()),
                "argmax_equal": int(torch.argmax(lg)) == int(torch.argmax(ref_mine)),
                "rows_equal": kv2.rows == kv_r.rows,
                "keys_max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(k_whole[0], k_split[0])),
                "index_keys_max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(k_whole[2], k_split[2])),
            }
        out["splits"] = splits
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception:  # noqa: BLE001 - already removed
                pass
        eng.md.configure_dash_inference(eng.decoder, False, state=state)
        cfg._ring_window, cfg.sliding_window = saved_cfg
    tol = 1e-3
    out["tolerance_logits"] = tol
    out["pass"] = out["embeds_bitwise_equal"] and all(s["argmax_equal"] and s["rows_equal"] and s["logits_max_abs_diff"] <= tol for s in out["splits"].values())
    return out


def recompute_check(eng: WindowedOracle, pages: list[Path], window: int, force: dict[int, str], infos: list[dict]) -> dict:
    """A forced-recompute window against an unwindowed run of the oracle's loop
    configured as the wheel configures a recomputed window: page_base = lo
    (SC:514, DR:109-122, BE:1001-1031), expected = own pages via the lookahead
    (CLP:229-237 with _uocr_context_pages = 0 after DR:662-681), end_on_marker =
    lookahead > 0 (CLP:227), guard then n-gram on output ids (CLP:315-385, ENV:1423)."""
    out: dict[str, Any] = {"windows": {}}
    ok = True
    n_pages = len(pages)
    for bi, reason in sorted(force.items()):
        if reason == "splice_alloc_failed":
            continue
        info = next((i for i in infos if i["window"] == bi), None)
        if info is None or info.get("reused") or not info.get("fallback"):
            out["windows"][str(bi)] = {"pass": False, "why": "window did not recompute"}
            ok = False
            continue
        lo, hi, _ = info["lo_hi_emit"]
        la = info["lookahead"]
        base = {"ngram_size": NGRAM_SIZE, "window_size": NGRAM_WINDOW_MULTI if n_pages > 1 else NGRAM_WINDOW_SINGLE, "page_token_id": PAGE_ID, "eos_token_id": EOS_ID, "max_page_tokens": MAX_PAGE_TOKENS}
        cp = sc_apply_page_cap({**base, DR_KEY_PAGE_BASE: lo, DR_KEY_LOOKAHEAD: la}, stop_only=True)
        ref_req = WinReq(f"recompute-ref-w{bi}", [str(p) for p in pages[lo : hi + la]], "<PAGE><image>" * (hi + la - lo) + README_PROMPT_TAIL, cp, info["max_new_tokens_sent"])
        ref = eng.run_request(ref_req, None)
        ref.pop("kv")
        a, b = info["output_ids"], ref["output_ids"]

        def upto_lookahead_marker(ids: list[int]) -> list[int]:
            seen = 0
            for i, t in enumerate(ids):
                if t == PAGE_ID:
                    seen += 1
                    if seen == hi - lo + 1:
                        return ids[: i + 1]
            return ids

        ca, cb = upto_lookahead_marker(a), upto_lookahead_marker(b)
        same = ca == cb and info["prompt_ids_sent"] == ref["prompt_ids_sent"]
        ok &= same
        out["windows"][str(bi)] = {
            "pass": same,
            "reason": reason,
            "compared_tokens": len(ca),
            "ends_on_lookahead_marker": bool(ca) and ca[-1] == PAGE_ID and ca.count(PAGE_ID) == hi - lo + 1,
            "whole_output_equal": a == b,
            "window_tokens": len(a),
            "reference_tokens": len(b),
            "window_wall_s": info["wall_seconds"],
            "reference_wall_s": ref["wall_seconds"],
            "reference_config": {"page_base": ref["page_base"], "page_cap_expected": ref["page_cap_expected"], "end_on_marker": ref["end_on_marker"], "context_pages": ref["context_pages"], "custom_params": ref["custom_params"]},
        }
    out["pass"] = ok and bool(out["windows"])
    return out


def compare_windowed_runs(run_dirs: list[Path]) -> int:
    """`--windowed-compare A,B,...`: model-free determinism check of finished
    runs: the plan, every window's ids, text, coordinates and splice, and the
    document text must be identical (the first run is the reference)."""

    def digest(run_dir: Path) -> dict[str, Any]:
        run = json.loads((run_dir / "run.json").read_text())
        out: dict[str, Any] = {"plan": run["plan"], "document_text_sha256": run["document"]["text_sha256"], "finish": run["document"]["finish"]}
        for w in run["windows"]:
            win = json.loads((run_dir / f"w{w['window']}" / "window.json").read_text())
            out[f"w{w['window']}"] = {
                "output_ids_sha256": ids_sha256(win["output_ids"]),
                "prompt_ids_sha256": ids_sha256(win.get("prompt_ids_rewritten") or win["prompt_ids_sent"]),
                "text_sha256": sha256_bytes(win["text"].encode("utf-8")),
                "reused": win.get("reused"),
                "fallback": win.get("fallback"),
                "splice_rows": win.get("splice_rows"),
                "rows": {k: v["sha256"] for k, v in sorted(win["rows_manifest"].items())},
            }
        return out

    ref = digest(run_dirs[0])
    ok = True
    for other in run_dirs[1:]:
        got = digest(other)
        diffs = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
        ok &= not diffs
        print(json.dumps({"test": "uocr2_windowed_determinism", "reference": str(run_dirs[0]), "run": str(other), "result": "pass" if not diffs else "fail", "differing": diffs}))
    return 0 if ok else 1


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
    w = p.add_argument_group("windowed long-document decoding (bead U2-D2a); dumps go to <out>/<tag>/")
    w.add_argument("--windowed", type=int, default=None, metavar="W", help="serve --doc as SGLang does with SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=W (f32 only)")
    w.add_argument("--doc", default="", help="FIXTURE[:N]: the document, the first N pages of a fixture's page list (e.g. S3-p10:7)")
    w.add_argument("--force-fallback", action="append", default=[], metavar="BI:REASON", help=f"make window BI fall back for REASON, one of {', '.join(sorted(FALLBACK_REASONS))} (repeatable)")
    w.add_argument("--windowed-check", default="", help="comma-separated self-checks to run after the document: degenerate, extend, recompute")
    w.add_argument("--windowed-compare", default="", metavar="RUN_DIR,RUN_DIR,...", help="model-free: compare finished --windowed runs for determinism and exit")
    args = p.parse_args(argv)
    if args.windowed_compare:
        return args
    if args.windowed is not None:
        if args.windowed < 1:
            p.error("--windowed takes W >= 1")
        if not args.self_test:
            if not args.doc:
                p.error("--windowed needs --doc")
            if args.dtype != "f32" or args.commit_dir is not None or args.max_new_tokens is not None:
                p.error("--windowed runs f32 only, without --commit-dir or --max-new-tokens")
            unknown = {c for c in args.windowed_check.split(",") if c} - {"degenerate", "extend", "recompute"}
            if unknown:
                p.error(f"--windowed-check: unknown checks {sorted(unknown)}")
    elif args.doc or args.force_fallback or args.windowed_check:
        p.error("--doc, --force-fallback and --windowed-check need --windowed")
    if not args.self_test:
        if args.model_dir is None or (args.out is None and not args.gundam_grids):
            p.error("--model-dir and --out are required")
        if args.commit_dir is not None and (args.dtype != "f32" or args.max_new_tokens is not None):
            p.error("--commit-dir takes only full-length f32 runs")
        args.tag = args.tag or f"t{args.threads}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.windowed_compare:
        return compare_windowed_runs([Path(d) for d in args.windowed_compare.split(",") if d])
    if args.self_test:
        return self_test(windowed=args.windowed is not None)
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
        "oracle": "hf-reference-cpu" if args.windowed is None else "windowed-sglang-on-hf-reference-cpu",
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
    if args.windowed is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        return run_windowed(args, torch, np, model, tok, mu, md, provenance)
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
