#!/usr/bin/env python3
"""What a program linking franken_ocr really gets (bd-hfdt, docs/library-output.md).

Builds a separate crate that depends on this one with only the `library` feature
and resolves its own lockfile, then checks:

  * its dependency tree holds none of the crates `tests/library_contract.rs` forbids;
  * the contract names the version in `src/lib.rs` and every field the goldens hold;
  * with a model: it reads a clean, a table and a turned page with a bare environment
    and an empty HOME, TMPDIR and working directory, and leaves all three empty;
  * with a `focr` binary too: the same pages read the same on focr's own lock. Run
    this one again when `wide` or `safe_arch` moves.

A step that lacks what it needs is listed under `skipped` with the reason. The last
line of output is one JSON object with `ok`, `assertion_count`, `assertion_failures`
and `skipped`, and the exit code is 0 only when `ok` is true.

  --features F       the feature to link (default `library`); `native` must fail
  FOCR_E2E_ROOT      the checkout to link (default this one); its siblings must build
  FOCR_E2E_WORK      where runs are kept (default target/hfdt_admission)
  FOCR_E2E_V2_MODEL  an Unlimited-OCR-2 `.focrq`, with `tokenizer.json` beside it
  FOCR_E2E_PAGES     holds page_0014.png and page_0590.png
                     (default ~/.cache/franken_ocr/corpus/pages)
  FOCR_BIN           the focr binary to compare with
  CARGO              the cargo binary (default cargo)
  FOCR_E2E_BUILD     the cargo subcommand that builds (default build)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "docs" / "library-output.md"

CONSUMER = """\
#![forbid(unsafe_code)]

use std::path::PathBuf;

use franken_ocr::native_engine::OcrModel;
use franken_ocr::page_source::PageSource;
use franken_ocr::{DocumentOptions, OUTPUT_CONTRACT_VERSION};

fn main() {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let model = args.next().expect("a model path");
    let pages: Vec<PathBuf> = args.collect();
    let model = OcrModel::load(&model).expect("load the model");
    let mut source = PageSource::images(pages).expect("open the pages");
    let document = model
        .recognize_document(&mut source, &DocumentOptions::default(), &mut |_| {}, None)
        .expect("read the document");
    let pages: Vec<_> = document.pages.iter().map(franken_ocr::PageResult::to_json).collect();
    let report = serde_json::json!({
        "contract": OUTPUT_CONTRACT_VERSION,
        "pages": pages,
        "skipped": document.skipped.len(),
    });
    println!("{report}");
}
"""


class Report:
    def __init__(self) -> None:
        self.count = 0
        self.failures: list[str] = []
        self.skipped: list[dict[str, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.count += 1
        print(json.dumps({"assertion": name, "pass": passed, "detail": detail[:500]}), flush=True)
        if not passed:
            self.failures.append(name)
        return passed

    def skip(self, step: str, reason: str) -> None:
        print(json.dumps({"skipped": step, "reason": reason}), flush=True)
        self.skipped.append({"step": step, "reason": reason})

    def finish(self) -> int:
        ok = not self.failures
        print(
            json.dumps(
                {
                    "ok": ok,
                    "assertion_count": self.count,
                    "assertion_failures": self.failures,
                    "skipped": self.skipped,
                }
            )
        )
        return 0 if ok else 1


def forbidden_crates() -> list[str]:
    source = (REPO / "tests" / "library_contract.rs").read_text(encoding="utf-8")
    block = re.search(r"const FORBIDDEN: \[&str; \d+\] = \[(.*?)\];", source, re.S)
    if block is None:
        raise SystemExit("tests/library_contract.rs: no FORBIDDEN list")
    return re.findall(r'"([^"]+)"', block.group(1))


def check_contract_doc(report: Report) -> None:
    source = (REPO / "src" / "lib.rs").read_text(encoding="utf-8")
    version = re.search(r"OUTPUT_CONTRACT_VERSION: u32 = (\d+);", source)
    contract = CONTRACT.read_text(encoding="utf-8") if CONTRACT.is_file() else ""
    report.check("the contract exists", bool(contract), str(CONTRACT))
    report.check(
        "the contract names the crate's version",
        version is not None and f"`franken_ocr::OUTPUT_CONTRACT_VERSION` = {version.group(1)}" in contract,
    )
    fields: set[str] = set()
    for golden in sorted((REPO / "tests" / "fixtures" / "golden").glob("library_*.golden")):
        for line in golden.read_text(encoding="utf-8").splitlines():
            fields.update(json.loads(line))
    fields.discard("contract")
    fields.discard("spans")
    missing = sorted(field for field in fields if f"`{field}`" not in contract)
    report.check("the goldens hold fields", bool(fields))
    report.check("the contract names every span field", not missing, ",".join(missing))


def write_consumer(root: Path, run: Path, feature: str) -> Path:
    crate = run / "consumer"
    (crate / "src").mkdir(parents=True)
    (crate / "src" / "main.rs").write_text(CONSUMER, encoding="utf-8")
    (crate / "Cargo.toml").write_text(
        f"""[package]
name = "hfdt_consumer"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
franken_ocr = {{ path = "{root}", default-features = false, features = ["{feature}"] }}
serde_json = "1"
""",
        encoding="utf-8",
    )
    shutil.copy(REPO / "rust-toolchain.toml", crate / "rust-toolchain.toml")
    return crate


def build_consumer(report: Report, crate: Path, target: Path) -> Path | None:
    cargo = os.environ.get("CARGO", "cargo")
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    resolved = subprocess.run(
        [cargo, "tree", "--offline", "--edges", "normal", "--prefix", "none"],
        cwd=crate,
        capture_output=True,
        text=True,
        env=env,
    )
    if not report.check("the consumer resolves its own lock", resolved.returncode == 0, resolved.stderr[-400:]):
        return None
    tree = {line.split()[0] for line in resolved.stdout.splitlines() if line.strip()}
    inherited = sorted(tree.intersection(forbidden_crates()))
    report.check("the consumer inherits no forbidden crate", not inherited, ",".join(inherited))
    versions = re.findall(r"^(wide|safe_arch) v(\S+)", resolved.stdout, re.M)
    print(json.dumps({"resolved": sorted(set(versions))}), flush=True)
    built = subprocess.run(
        [cargo, os.environ.get("FOCR_E2E_BUILD", "build"), "--offline", "--release"],
        cwd=crate,
        capture_output=True,
        text=True,
        env=env,
    )
    if not report.check("the consumer builds", built.returncode == 0, built.stderr[-400:]):
        return None
    return target / "release" / "hfdt_consumer"


def turn(page: Path, out: Path) -> bool:
    shutil.copy(page, out)
    if shutil.which("sips"):
        command = ["sips", "-r", "90", str(out)]
    elif shutil.which("magick"):
        command = ["magick", str(page), "-rotate", "90", str(out)]
    else:
        return False
    return subprocess.run(command, capture_output=True).returncode == 0


def prepare_pages(report: Report, run: Path) -> list[Path] | None:
    corpus = os.environ.get("FOCR_E2E_PAGES", str(Path.home() / ".cache/franken_ocr/corpus/pages"))
    clean, table = Path(corpus) / "page_0014.png", Path(corpus) / "page_0590.png"
    if not clean.is_file() or not table.is_file():
        report.skip("read", f"no page_0014.png and page_0590.png in {corpus}")
        return None
    pages = run / "pages"
    pages.mkdir()
    named = [pages / "1_clean.png", pages / "2_table.png", pages / "3_rotated.png"]
    shutil.copy(clean, named[0])
    shutil.copy(table, named[1])
    if not turn(clean, named[2]):
        report.skip("read", "neither sips nor magick can turn a page")
        return None
    return named


def read_pages(report: Report, consumer: Path, model: Path, pages: list[Path], run: Path) -> list[dict] | None:
    sandbox = run / "sandbox"
    dirs = [sandbox / "home", sandbox / "tmp", sandbox / "cwd"]
    for directory in dirs:
        directory.mkdir(parents=True)
    done = subprocess.run(
        [str(consumer), str(model), *map(str, pages)],
        cwd=dirs[2],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(dirs[0]),
            "TMPDIR": str(dirs[1]),
            "FOCR_MMAP": "1",
        },
    )
    if not report.check("the consumer reads the pages", done.returncode == 0, done.stderr[-400:]):
        return None
    left = [str(path) for path in sandbox.rglob("*") if path.is_file()]
    report.check("the read leaves no file behind", not left, ",".join(left[:5]))
    result = json.loads(done.stdout)
    source = (REPO / "src" / "lib.rs").read_text(encoding="utf-8")
    version = int(re.search(r"OUTPUT_CONTRACT_VERSION: u32 = (\d+);", source).group(1))
    report.check("the run reports the contract version", result["contract"] == version)
    read = result["pages"]
    if not report.check("every page was read", len(read) == 3 and result["skipped"] == 0):
        return None
    report.check("every page has spans", all(page["layout"] for page in read))
    clean, table, turned = read
    report.check("the table page has a table span", any(span["label"] == "table" for span in table["layout"]))
    report.check(
        "the turned page's frame turned",
        (turned["width"], turned["height"]) == (clean["height"], clean["width"]),
        f"{clean['width']}x{clean['height']} then {turned['width']}x{turned['height']}",
    )
    contract = CONTRACT.read_text(encoding="utf-8")
    missing = sorted({key for page in read for key in page if f"`{key}`" not in contract})
    report.check("the contract names every page field", not missing, ",".join(missing))
    return read


def compare_with_focr(report: Report, focr: str, model: Path, pages: list[Path], read: list[dict], run: Path) -> None:
    out = run / "focr.json"
    tmp = run / "focr_tmp"
    tmp.mkdir()
    done = subprocess.run(
        [focr, "ocr", str(pages[0].parent), "--multi-page", "--model", "unlimited-ocr-2", "--json", "-o", str(out)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FOCR_NO_RESIDENT": "1",
            "FOCR_MMAP": "1",
            "FOCR_MODEL_DIR": str(model.parent),
            "FOCR_RUN_STORE": str(run / "runs.db"),
            "TMPDIR": str(tmp),
        },
    )
    if not report.check("focr reads the same pages", done.returncode == 0, done.stderr[-400:]):
        return
    theirs = json.loads(out.read_text(encoding="utf-8"))["pages"]
    report.check("focr read as many pages", len(theirs) == len(read))
    for number, (mine, other) in enumerate(zip(read, theirs), start=1):
        same = all(mine.get(key) == other.get(key) for key in ("text", "layout", "tokens", "finish"))
        report.check(f"page {number} reads the same on both locks", same)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="library")
    args = parser.parse_args()

    report = Report()
    root = Path(os.environ.get("FOCR_E2E_ROOT", str(REPO)))
    work = Path(os.environ.get("FOCR_E2E_WORK", str(REPO / "target" / "hfdt_admission")))
    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=work))

    check_contract_doc(report)
    crate = write_consumer(root, run, args.features)
    consumer = build_consumer(report, crate, work / "target")
    if consumer is None:
        return report.finish()

    model = os.environ.get("FOCR_E2E_V2_MODEL")
    if not model:
        report.skip("read", "FOCR_E2E_V2_MODEL is not set")
        return report.finish()
    pages = prepare_pages(report, run)
    if pages is None:
        return report.finish()
    read = read_pages(report, consumer, Path(model), pages, run)
    if read is None:
        return report.finish()

    focr = os.environ.get("FOCR_BIN")
    if not focr:
        report.skip("compare", "FOCR_BIN is not set")
        return report.finish()
    compare_with_focr(report, focr, Path(model), pages, read, run)
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
