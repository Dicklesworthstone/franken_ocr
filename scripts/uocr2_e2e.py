#!/usr/bin/env python3
"""The Unlimited-OCR-2 end-to-end script (bead U2-T1a, plan §7.7).

Drives the real `focr` binary on real inputs, one named step per user-facing
scenario, and writes one NDJSON line per step in the docs/TEST_LOGGING.md
contract, so a failure is diagnosable from the log alone. No mocks.

A step that needs an artifact skips with success when it is absent. Decoding
steps beyond what the token golden gate covers run only with FOCR_E2E_LONG=1,
and `focr pull` only with FOCR_E2E_NETWORK=1:

    scripts/uocr2_e2e.sh                      # fast steps: h refused, j, k, l
    FOCR_E2E_LONG=1 scripts/uocr2_e2e.sh      # every step, required at milestone exits

Artifacts: FOCR_E2E_V2_MODEL and FOCR_E2E_V1_MODEL, each beside its
tokenizer.json, or the newest match in FOCR_MODEL_DIR and
~/.cache/franken_ocr/models. Pages: FOCR_CORPUS_DIR (default
~/.cache/franken_ocr/corpus/pages).

Each run gets its own HOME and FOCR_MODEL_PATH=/nonexistent, so nothing can
fall back to another model, and steps name their artifact with --model. The
only `FOCR_*` the caller's shell passes through is FOCR_THREADS.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TEST = "uocr2_e2e"
LOG_SCHEMA_VERSION = 1
FALLBACK_TARGET = "/nonexistent"
V1_GOLDENS = ROOT / "tests" / "fixtures" / "uocr2_v1_goldens"
V2_GOLDENS = ROOT / "tests" / "fixtures" / "uocr2_v2_goldens"
CJK_PAGES = ROOT / "tests" / "fixtures" / "cjk_stream"
CENSUS = ROOT / "src" / "native_engine"
SINGLE_PAGES = ("page_0009", "page_0014")
S2_PAGES = ("page_0009.png", "page_0014.png", "page_0019.png")
# Corpus pages the v2 reference labels `image`.
FIGURE_PAGES = ("page_0105.png", "page_0136.png")
FIGURE_NAME = re.compile(r"page\d+[lr]?_figure_\d+(\.[a-z]+)?$")
CATALOG = ROOT / "models" / "manifest-v2.json"
STDERR_TAIL_LINES = 40
DIFF_HEAD_LINES = 20


class Skipped(Exception):
    """The step cannot run here, and that is a success."""


class Failed(Exception):
    """The step ran and something was wrong."""

    def __init__(self, message: str, diff: list[str] | None = None) -> None:
        super().__init__(message)
        self.diff = diff or []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_input(path: Path) -> str:
    """A file's digest, or a directory's over its files' names and digests."""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(path.iterdir()):
        digest.update(child.name.encode())
        digest.update(sha256_file(child).encode())
    return digest.hexdigest()


@dataclass
class Run:
    argv: list[str]
    env: dict[str, str]
    exit_code: int
    stdout: str
    stderr: str
    stdout_path: Path
    stderr_path: Path
    peak_rss_bytes: int | None

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.stdout.splitlines() if line.strip()]

    def decoded_pages(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.events()
            if event.get("event") == "page" and event.get("status") == "decoded"
        ]


@dataclass
class Suite:
    binary: Path
    work: Path
    log: Path
    corpus: Path
    v1: Path | None
    v2: Path | None
    long: bool
    network: bool
    seq: int = 0
    failures: list[str] = field(default_factory=list)
    digests: dict[Path, str] = field(default_factory=dict)
    binary_sha256: str = ""
    git_sha: str = ""

    def artifact_sha256(self, path: Path) -> str:
        if path not in self.digests:
            self.digests[path] = sha256_file(path)
        return self.digests[path]


class Step:
    """One scenario's runs, and what they prove."""

    def __init__(self, suite: Suite, case: str) -> None:
        self.suite = suite
        self.case = case
        self.dir = suite.work / case
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[Run] = []
        self.outputs: list[Path] = []
        self.inputs: list[Path] = []
        self.artifact: Path | None = None
        self.model_id: str | None = None

    def need_v1(self) -> Path:
        return self.need(self.suite.v1, "v1")

    def need_v2(self) -> Path:
        return self.need(self.suite.v2, "unlimited-ocr-2")

    def need(self, artifact: Path | None, name: str) -> Path:
        if artifact is None:
            raise Skipped(f"weights absent: no {name} artifact")
        self.artifact = artifact
        return artifact

    def need_long(self) -> None:
        if not self.suite.long:
            raise Skipped("decodes real pages: set FOCR_E2E_LONG=1")

    def corpus_page(self, name: str) -> Path:
        page = self.suite.corpus / name
        if not page.is_file():
            raise Skipped(f"no corpus page {page}")
        return page

    def document(self, name: str, pages: list[Path]) -> Path:
        """A directory of `pages` in order, as one document."""
        directory = self.dir / name
        directory.mkdir(exist_ok=True)
        for k, page in enumerate(pages):
            target = directory / f"{k:04}_{page.name}"
            if not target.exists():
                shutil.copyfile(page, target)
        self.inputs.append(directory)
        return directory

    def focr(self, *args: object, env: dict[str, str] | None = None) -> Run:
        ordinal = len(self.runs)
        stdout_path = self.dir / f"run{ordinal}.stdout"
        stderr_path = self.dir / f"run{ordinal}.stderr"
        home = self.dir / "home"
        home.mkdir(exist_ok=True)
        run_env = {k: v for k, v in os.environ.items() if not k.startswith("FOCR_")}
        run_env.update(
            HOME=str(home),
            FOCR_MODEL_PATH=FALLBACK_TARGET,
            FOCR_NO_RESIDENT="1",
        )
        if "FOCR_THREADS" in os.environ:
            run_env["FOCR_THREADS"] = os.environ["FOCR_THREADS"]
        run_env.update(env or {})
        argv = [str(self.suite.binary), *map(str, args)]
        timed = argv
        if Path("/usr/bin/time").is_file():
            timed = ["/usr/bin/time", "-lp" if sys.platform == "darwin" else "-v", *argv]
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            code = subprocess.run(timed, env=run_env, stdout=out, stderr=err, check=False).returncode
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
        kib = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
        peak = int(rss.group(1)) if rss else int(kib.group(1)) * 1024 if kib else None
        run = Run(argv, run_env, code, stdout, stderr, stdout_path, stderr_path, peak)
        self.runs.append(run)
        self.outputs.append(stdout_path)
        return run

    def check(self, ok: bool, message: str, diff: list[str] | None = None) -> None:
        if not ok:
            raise Failed(message, diff)

    def exits(self, run: Run, code: int) -> None:
        self.check(run.exit_code == code, f"exit {run.exit_code}, wanted {code}")

    def same(self, ours: str, golden: str, what: str) -> None:
        diff = list(
            difflib.unified_diff(
                golden.splitlines(), ours.splitlines(), "golden", "ours", lineterm=""
            )
        )
        self.check(not diff, f"{what} differs from its golden", diff)


def record(suite: Suite, step: Step, event: str, result: str, elapsed_us: int, **extra: Any) -> None:
    """The step's one NDJSON line."""
    suite.seq += 1
    last = step.runs[-1] if step.runs else None
    peaks = [run.peak_rss_bytes for run in step.runs if run.peak_rss_bytes is not None]
    line: dict[str, Any] = {
        "schema_version": LOG_SCHEMA_VERSION,
        "ts": int(time.time() * 1000),
        "test": TEST,
        "case": step.case,
        "run_seq": suite.seq,
        "event": event,
        "result": result,
        "elapsed_us": elapsed_us,
        "native_path_ran": bool(step.runs),
        "fallback_target": FALLBACK_TARGET,
        "git_sha": suite.git_sha,
        "binary_sha256": suite.binary_sha256,
        "runs": [
            {
                "cmd": run.argv,
                "exit_code": run.exit_code,
                "peak_rss_bytes": run.peak_rss_bytes,
                "stdout_path": str(run.stdout_path),
                "stderr_path": str(run.stderr_path),
            }
            for run in step.runs
        ],
        "input_sha256": {str(path): sha256_input(path) for path in step.inputs},
        "outputs": {str(path): sha256_file(path) for path in step.outputs if path.is_file()},
    }
    if last is not None:
        line.update(
            cmd=last.argv,
            env={
                key: value
                for key, value in last.env.items()
                if key.startswith("FOCR_") or key == "RAYON_NUM_THREADS"
            },
            threads=last.env.get("FOCR_THREADS", "default"),
            exit_code=last.exit_code,
            peak_rss_bytes=max(peaks, default=None),
            stdout_path=str(last.stdout_path),
            stderr_path=str(last.stderr_path),
        )
    if step.artifact is not None:
        line["artifact_sha256"] = suite.artifact_sha256(step.artifact)
    if step.model_id is not None:
        line["model_id"] = step.model_id
    line.update(extra)
    with suite.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def run_step(suite: Suite, case: str, body: Callable[[Step], None]) -> None:
    step = Step(suite, case)
    started = time.monotonic()
    elapsed = lambda: int((time.monotonic() - started) * 1e6)  # noqa: E731
    try:
        body(step)
    except Skipped as why:
        record(suite, step, "skip", "skip_no_model", elapsed(), reason=str(why))
        print(f"E2E   SKIP  {case}: {why}", file=sys.stderr)
        return
    except Failed as failure:
        last = step.runs[-1] if step.runs else None
        code = last.exit_code if last else -1
        record(
            suite,
            step,
            "error",
            "fail",
            elapsed(),
            diag={"error_kind": "e2e_step", "focr_exit_code": code, "message": str(failure)},
            stderr_tail=last.stderr.splitlines()[-STDERR_TAIL_LINES:] if last else [],
            diff_head=failure.diff[:DIFF_HEAD_LINES],
        )
        suite.failures.append(f"{case}: {failure}")
        print(f"E2E   FAIL  {case}: {failure}", file=sys.stderr)
        return
    record(suite, step, "result", "pass", elapsed())
    print(f"E2E   PASS  {case} ({elapsed() / 1e6:.1f}s)", file=sys.stderr)


# ── the steps ────────────────────────────────────────────────────────────────


def step_a_single_page(step: Step) -> None:
    """A single page, base, equals the v2 golden."""
    step.need_long()
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    for name in SINGLE_PAGES:
        page = step.corpus_page(f"{name}.png")
        step.inputs.append(page)
        run = step.focr("ocr", page, "--model", v2, "--crop-mode", "base")
        step.exits(run, 0)
        golden = (V2_GOLDENS / f"{name}.md").read_text(encoding="utf-8")
        step.same(run.stdout.rstrip("\n"), golden.rstrip("\n"), name)


def step_b_three_pages(step: Step) -> None:
    """Three pages: three bodies, page events in order, `pages[]` of 3."""
    step.need_long()
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    document = step.document("s2", [step.corpus_page(name) for name in S2_PAGES])
    run = step.focr("ocr", document, "--multi-page", "--model", v2, "--robot")
    step.exits(run, 0)
    pages = [event["source_page"] for event in run.decoded_pages()]
    step.check(pages == [1, 2, 3], f"page events {pages}")
    out = step.dir / "s2.json"
    step.outputs.append(out)
    run = step.focr("ocr", document, "--multi-page", "--model", v2, "--json", "-o", out)
    step.exits(run, 0)
    value = json.loads(out.read_text(encoding="utf-8"))
    step.check(len(value["pages"]) == 3, f"pages[] holds {len(value['pages'])}")
    golden = (V2_GOLDENS / "s2.md").read_text(encoding="utf-8")
    step.same(value["markdown"].rstrip("\n"), golden.rstrip("\n"), "s2")


def write_pdf(pages: list[Path], path: Path) -> None:
    """One PDF page per non-interlaced 8-bit RGB PNG, its IDAT stream embedded as is."""
    objects: list[bytes] = []
    kids = " ".join(f"{3 + 3 * k} 0 R" for k in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode())
    for k, page in enumerate(pages):
        data = page.read_bytes()
        width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", data[16:29])
        if (depth, color, interlace) != (8, 2, 0):
            raise Failed(f"{page.name} is not non-interlaced 8-bit RGB")
        idat = bytearray()
        at = 8
        while at < len(data):
            length, kind = struct.unpack(">I4s", data[at : at + 8])
            if kind == b"IDAT":
                idat += data[at + 8 : at + 8 + length]
            at += 12 + length
        first = 3 + 3 * k
        content = f"q {width} 0 0 {height} 0 0 cm /Page Do Q".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Contents {first + 1} 0 R "
            f"/Resources << /XObject << /Page {first + 2} 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)
        )
        objects.append(
            (
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                f"/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {width} >> "
                f"/Length {len(idat)} >>\nstream\n"
            ).encode()
            + bytes(idat)
            + b"\nendstream"
        )
    with path.open("wb") as out:
        out.write(b"%PDF-1.5\n")
        offsets = []
        for number, body in enumerate(objects, 1):
            offsets.append(out.tell())
            out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        xref = out.tell()
        out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
        for offset in offsets:
            out.write(b"%010d 00000 n \n" % offset)
        out.write(
            b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref)
        )


def step_c_twenty_pages(step: Step) -> None:
    """A 20-page PDF in one pass (T1b has the windowed run): bodies in order."""
    step.need_long()
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    pages = sorted(step.suite.corpus.glob("*.png"))[:20]
    if len(pages) < 20:
        raise Skipped(f"{step.suite.corpus} holds {len(pages)} pages, not 20")
    document = step.dir / "doc.pdf"
    write_pdf(pages, document)
    step.inputs.append(document)
    run = step.focr(
        "ocr", document, "--multi-page", "--page-window", "off", "--model", v2, "--robot"
    )
    step.exits(run, 0)
    decoded = [event["source_page"] for event in run.decoded_pages()]
    step.check(decoded == list(range(1, 21)), f"page events {decoded}")
    last = run.events()[-1]
    step.check(last.get("event") == "run_complete", f"ended on {last.get('event')}")
    step.check(last.get("pages_out") == 20, f"pages_out {last.get('pages_out')}")
    step.check(last.get("windows") == 0, f"windows {last.get('windows')}")


def step_g_v1_regression(step: Step) -> None:
    """v1's output is byte-identical to its goldens."""
    step.need_long()
    v1 = step.need_v1()
    step.model_id = "unlimited-ocr"
    for name in SINGLE_PAGES:
        page = step.corpus_page(f"{name}.png")
        step.inputs.append(page)
        run = step.focr("ocr", page, "--model", v1)
        step.exits(run, 0)
        golden = (V1_GOLDENS / f"{name}.md").read_text(encoding="utf-8")
        step.same(run.stdout.rstrip("\n"), golden.rstrip("\n"), name)


def step_h_figures_refused(step: Step) -> None:
    """A model that merges pages (or none) refuses per-page figures."""
    v1 = step.suite.v1
    if v1 is not None:
        step.artifact = v1
        step.model_id = "unlimited-ocr"
    run = step.focr(
        "ocr", CJK_PAGES, "--multi-page", "--json", "--extract-figures",
        "--figures-dir", step.dir / "figures", "--model", v1 or FALLBACK_TARGET,
    )  # fmt: skip
    step.exits(run, 2)
    step.check("--extract-figures" in run.stderr, "the refusal does not name the flag")


def step_h_figures(step: Step) -> None:
    """Document figures are named per page and listed in the JSON."""
    step.need_long()
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    document = step.document("pages", [step.corpus_page(name) for name in FIGURE_PAGES])
    figures = step.dir / "figures"
    out = step.dir / "figures.json"
    step.outputs.append(out)
    run = step.focr(
        "ocr", document, "--multi-page", "--model", v2, "--json", "-o", out,
        "--figures-dir", figures,
    )  # fmt: skip
    step.exits(run, 0)
    value = json.loads(out.read_text(encoding="utf-8"))
    step.check(len(value["pages"]) == len(FIGURE_PAGES), "one pages[] entry per page")
    written = sorted(path.name for path in figures.iterdir()) if figures.is_dir() else []
    step.check(bool(written), "no figure was written")
    bad = [name for name in written if not FIGURE_NAME.match(name)]
    step.check(not bad, f"misnamed figures {bad}")
    listed = sorted(Path(figure["path"]).name for figure in value.get("figures", []))
    step.check(listed == written, f"the JSON lists {listed}, the directory holds {written}")
    step.outputs.extend(figures / name for name in written)


def step_i_pull(step: Step) -> None:
    """`focr pull unlimited-ocr-2` installs the artifact and its tokenizer."""
    if not step.suite.network:
        raise Skipped("downloads the published artifact: set FOCR_E2E_NETWORK=1")
    run = step.focr("pull", "unlimited-ocr-2")
    step.exits(run, 0)
    installed = list((step.dir / "home").rglob("unlimited-ocr-2*.focrq"))
    step.check(len(installed) == 1, f"installed {installed}")
    sidecar = installed[0].with_name("tokenizer.json")
    step.check(sidecar.is_file(), f"no tokenizer beside {installed[0]}")
    step.artifact = installed[0]
    step.model_id = "unlimited-ocr-2"


def write_v2_stand_in(directory: Path) -> Path:
    """A sparse safetensors checkpoint with v2's tensor census: admission reads
    only its directory, and a foreign sidecar stops an admitted run before a
    weight byte is read (exit 7)."""
    census = json.loads((CENSUS / "unlimited_ocr_manifest.json").read_text(encoding="utf-8"))
    delta = json.loads((CENSUS / "unlimited_ocr_2_manifest.json").read_text(encoding="utf-8"))
    tensors = dict(sorted({**census["tensors"], **delta["added_tensors"]}.items()))
    entries: dict[str, Any] = {}
    offset = 0
    for name, tensor in tensors.items():
        length = {"F32": 4, "F16": 2, "BF16": 2}[tensor["source_dtype"]]
        for extent in tensor["shape"]:
            length *= extent
        entries[name] = {
            "dtype": tensor["source_dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    aranges: list[tuple[int, int]] = []
    for name, rule in delta["int64_tensors"].items():
        count = 1
        for extent in rule["shape"]:
            count *= extent
        entries[name] = {
            "dtype": "I64",
            "shape": rule["shape"],
            "data_offsets": [offset, offset + count * 8],
        }
        aranges.append((offset, count))
        offset += count * 8
    header = json.dumps(entries, separators=(",", ":")).encode()
    path = directory / "unlimited-ocr-2-stand-in.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.truncate(8 + len(header) + offset)
        for start, count in aranges:
            handle.seek(8 + len(header) + start)
            handle.write(struct.pack(f"<{count}q", *range(count)))
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def step_j_levers(step: Step) -> None:
    """Refused levers exit 2 before weights load; an ignored one warns."""
    stand_in = write_v2_stand_in(step.dir)
    image = CJK_PAGES / "page_1.png"
    step.inputs.append(image)
    for lever, value in (
        ("FOCR_PREFILL_CHUNK", "64"),
        ("FOCR_DECODE_INT8", "1"),
        ("FOCR_MOE_SCORE_ORDER", "1"),
        ("FOCR_BATCH_SPINE", "1"),
    ):
        run = step.focr("ocr", image, "--model", stand_in, env={lever: value})
        step.exits(run, 2)
        step.check(lever in run.stderr, f"the refusal does not name {lever}")
    run = step.focr("ocr", image, "--model", stand_in, env={"FOCR_QKV_FUSED": "0"})
    step.exits(run, 7)
    step.check("FOCR_QKV_FUSED" in run.stderr, "the ignored lever was not reported")
    step.check("tokenizer.json has sha256" in run.stderr, "stopped short of the sidecar check")


def step_j_ignored_lever(step: Step) -> None:
    """On real weights an ignored lever reaches the run's JSON warnings."""
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    image = CJK_PAGES / "page_1.png"
    step.inputs.append(image)
    run = step.focr("ocr", image, "--model", v2, "--json", env={"FOCR_QKV_FUSED": "0"})
    step.exits(run, 0)
    warnings = json.loads(run.stdout).get("warnings", [])
    ignored = [w for w in warnings if w.get("kind") == "ignored_lever"]
    step.check(
        any(w.get("env") == "FOCR_QKV_FUSED" for w in ignored), f"warnings {warnings}"
    )


def step_k_cjk_streaming(step: Step) -> None:
    """A split multi-byte character does not stop `page` events."""
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    document = step.document("cjk", sorted(CJK_PAGES.glob("*.png")))
    run = step.focr("ocr", document, "--multi-page", "--model", v2, "--robot")
    step.exits(run, 0)
    decoded = run.decoded_pages()
    step.check(len(decoded) == 2, f"{len(decoded)} page events")
    bodies = "".join(event.get("text", "") for event in decoded)
    step.check(any(ord(ch) > 0x2e7f for ch in bodies), "no CJK text was streamed")


def step_l_v1_reporting(step: Step) -> None:
    """A v1 document cut at its positions says so, and names a remedy that works."""
    v1 = step.need_v1()
    step.model_id = "unlimited-ocr"
    pages = [step.corpus_page(name) for name in S2_PAGES]
    step.inputs.extend(pages)
    batch = ("ocr-batch", *pages, "--multi-page", "--json", "--model", v1)
    # The pull is the remedy exactly when the catalog can serve it (U2-E4).
    published = "unlimited-ocr-2" in CATALOG.read_text(encoding="utf-8")
    remedy = "focr pull unlimited-ocr-2"
    probe = step.focr(*batch, env={"FOCR_TEST_MAX_POSITIONS": "1"})
    too_long = probe.stdout + probe.stderr
    found = re.search(r"prefix is (\d+)", too_long)
    step.check(found is not None, "no prompt length in the over-budget error")
    step.check((remedy in too_long) == published, f"catalog published={published}: {too_long}")
    budget = int(found.group(1)) + 48
    run = step.focr(*batch, env={"FOCR_TEST_MAX_POSITIONS": str(budget)})
    step.exits(run, 0)
    value = json.loads(run.stdout)
    step.check(value.get("finish_reason") == "length", f"finish {value.get('finish_reason')}")
    warning = next(
        (w for w in value.get("warnings", []) if w.get("kind") == "v1_truncated"), None
    )
    step.check(warning is not None, f"warnings {value.get('warnings')}")
    step.check(warning.get("cause") == "positions", f"cause {warning.get('cause')}")
    step.check(warning.get("expected") == 3, f"expected {warning.get('expected')}")
    got = warning.get("got")
    step.check(isinstance(got, int) and got < 3, f"got {got}")
    named = remedy in warning.get("recovery", "")
    step.check(named == published, f"catalog published={published}, recovery {warning}")
    line = f"[focr] warning: stopped after {got} of 3 pages"
    step.check(line in run.stderr, "the warning did not reach stderr")


def step_r_batch(step: Step) -> None:
    """`ocr-batch` on v2 equals its solo runs, and `--multi-page` its document."""
    step.need_long()
    v2 = step.need_v2()
    step.model_id = "unlimited-ocr-2"
    pages = [step.corpus_page(name) for name in S2_PAGES]
    step.inputs.extend(pages)
    run = step.focr("ocr-batch", *pages, "--model", v2, "--json")
    step.exits(run, 0)
    results = json.loads(run.stdout)["results"]
    step.check(len(results) == 3, f"{len(results)} results")
    for page, result in zip(pages, results):
        solo = step.focr("ocr", page, "--model", v2)
        step.exits(solo, 0)
        step.same(result["markdown"].rstrip("\n"), solo.stdout.rstrip("\n"), page.name)
    run = step.focr("ocr-batch", *pages, "--multi-page", "--model", v2, "--json")
    step.exits(run, 0)
    golden = (V2_GOLDENS / "s2.md").read_text(encoding="utf-8")
    step.same(json.loads(run.stdout)["markdown"].rstrip("\n"), golden.rstrip("\n"), "s2")


STEPS: dict[str, Callable[[Step], None]] = {
    "a_single_page": step_a_single_page,
    "b_three_pages": step_b_three_pages,
    "c_twenty_pages": step_c_twenty_pages,
    "g_v1_regression": step_g_v1_regression,
    "h_figures_refused": step_h_figures_refused,
    "h_figures": step_h_figures,
    "i_pull": step_i_pull,
    "j_levers": step_j_levers,
    "j_ignored_lever": step_j_ignored_lever,
    "k_cjk_streaming": step_k_cjk_streaming,
    "l_v1_reporting": step_l_v1_reporting,
    "r_batch": step_r_batch,
}


def find_artifact(override: str, wanted: Callable[[str], bool]) -> Path | None:
    """The named artifact, or the newest match in the model directories."""
    named = os.environ.get(override)
    if named:
        return Path(named) if Path(named).is_file() else None
    roots = [Path(p) for p in os.environ.get("FOCR_MODEL_DIR", "").split(":") if p]
    roots.append(Path.home() / ".cache" / "franken_ocr" / "models")
    found = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.focrq")
        if wanted(path.name)
    ]
    return max(found, key=lambda path: path.stat().st_mtime, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", required=True, type=Path, help="the focr binary")
    parser.add_argument("--work", required=True, type=Path, help="kept after the run")
    parser.add_argument("--only", help="comma-separated steps, by name")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    corpus = os.environ.get("FOCR_CORPUS_DIR") or str(
        Path.home() / ".cache" / "franken_ocr" / "corpus" / "pages"
    )
    suite = Suite(
        binary=args.bin.resolve(),
        work=args.work.resolve(),
        log=args.work.resolve() / "uocr2_e2e.ndjson",
        corpus=Path(corpus),
        v1=find_artifact(
            "FOCR_E2E_V1_MODEL",
            lambda name: name.startswith("unlimited-ocr.") and "wasm" not in name,
        ),
        v2=find_artifact(
            "FOCR_E2E_V2_MODEL", lambda name: name.startswith("unlimited-ocr-2")
        ),
        long=os.environ.get("FOCR_E2E_LONG") == "1",
        network=os.environ.get("FOCR_E2E_NETWORK") == "1",
    )
    suite.log.write_text("", encoding="utf-8")
    suite.binary_sha256 = sha256_file(suite.binary)
    suite.git_sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps {sorted(unknown)}")
    for case, body in STEPS.items():
        if only is None or case in only:
            run_step(suite, case, body)

    checked = subprocess.run(  # ubs:ignore[python.taint.command] argv list, shell=False; the only non-constant element is the e2e log path from the --work developer CLI arg, not request input
        [sys.executable, "-I", str(ROOT / "scripts" / "check_test_logs.py"), str(suite.log)],
        capture_output=True,
        text=True,
        check=False,
    )
    if checked.returncode != 0:
        suite.failures.append(f"log validation: {checked.stdout[-400:]}{checked.stderr[-400:]}")
    print(f"E2E log: {suite.log}", file=sys.stderr)
    for failure in suite.failures:
        print(f"E2E FAILED {failure}", file=sys.stderr)
    return 1 if suite.failures else 0


if __name__ == "__main__":
    sys.exit(main())
