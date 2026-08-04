#!/usr/bin/env python3
"""Synthesize the GOT `--format` validation corpus (bd-3kix phase 1, model-free).

OFFLINE TOOLING ONLY — no model, no network. Renders SMALL, clean, synthetic
images for each GOT-OCR2 specialized `OCR with format:` modality, each with a
ground-truth sidecar holding the exact SOURCE the image was rendered from:

    tests/fixtures/got/format_corpus/
        formula.png   + formula.gt.txt    (LaTeX source, matplotlib mathtext)
        table.png     + table.gt.json     (headers + rows, PIL bordered grid)
        chart.png     + chart.gt.json     (labels + values, matplotlib bars)
        molecule.png  + molecule.gt.txt   (SMILES, RDKit 2D depiction)
        music.png     + music.gt.krn      (**kern source, Verovio engraving)
        manifest.json                     (generator versions + asset sha256s)

The Rust format-mode smoke gates (`src/native_engine/got.rs` tests, env-gated on
`FOCR_GOT_MODEL`) run `recognize(..., format=true)` on these and assert lenient
structural markers; exact CER budgets are phase 2 (see bd-3kix), scored with
``scripts/got_cer.py`` against the sidecars once outputs are eyeballed.

Everything is deterministic: fixed data (no randomness — seeds are pinned anyway
for belt-and-braces), fixed figure sizes/DPI/fonts (matplotlib's bundled DejaVu),
and the matplotlib `Software` PNG comment stripped, so reruns on the same library
versions are byte-identical (see manifest.json sha256s).

Usage (uv venv, per repo tooling doctrine — system python lacks matplotlib):
    uv venv /private/tmp/focr_corpus_venv --python 3.12
    uv pip install --python /private/tmp/focr_corpus_venv/bin/python \
        matplotlib pillow rdkit verovio cairosvg
    # macOS: cairosvg needs homebrew cairo for the verovio SVG->PNG step
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
        /private/tmp/focr_corpus_venv/bin/python scripts/gen_got_format_corpus.py

molecule.png (rdkit) and music.png (verovio+cairosvg) are OPTIONAL: if their
libraries are unavailable the script prints a SKIP note and still exits 0 as
long as the three core assets (formula/table/chart) rendered.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "got" / "format_corpus"

SEED = 3141  # no randomness is actually used; pinned for belt-and-braces.

# ---------------------------------------------------------------------------
# Ground-truth sources (the single point of truth for every asset)
# ---------------------------------------------------------------------------

# Mathtext-renderable AND GOT-plausible: fraction, integral, super/subscripts.
FORMULA_LATEX = r"E = mc^2 + \frac{1}{2}\int_0^1 x^2 \, dx"

TABLE = {
    "headers": ["Item", "Qty", "Price"],
    "rows": [
        ["Widget", "4", "2.50"],
        ["Gadget", "17", "3.75"],
        ["Sprocket", "42", "1.20"],
    ],
}

CHART = {
    "title": "Widget output",
    "ylabel": "units",
    "labels": ["A", "B", "C", "D"],
    "values": [3, 7, 5, 9],
}

SMILES_ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Two 4/4 bars, C-major quarter-note scale, treble clef (GrandStaff-style kern).
KERN_MUSIC = """**kern
*clefG2
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
4cc
==
*-
"""


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # headless + deterministic
    import matplotlib.pyplot as plt

    # Fixed, bundled fonts -> identical glyphs everywhere.
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    matplotlib.rcParams["svg.hashsalt"] = str(SEED)
    return matplotlib, plt


# `metadata={"Software": None}` strips the matplotlib version comment chunk so
# the PNG bytes depend only on the pixels (rerun-stable on a pinned version).
_PNG_META = {"Software": None}


def gen_formula(path: Path) -> None:
    """formula.png — rendered LaTeX equation, white bg, 600x200."""
    _, plt = _matplotlib()
    fig = plt.figure(figsize=(6, 2), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.5,
        f"${FORMULA_LATEX}$",
        ha="center",
        va="center",
        fontsize=26,
        color="black",
    )
    fig.savefig(path, dpi=100, facecolor="white", metadata=_PNG_META)
    plt.close(fig)
    (path.parent / "formula.gt.txt").write_text(
        FORMULA_LATEX + "\n", encoding="utf-8"
    )


def gen_table(path: Path) -> None:
    """table.png — clean bordered data grid drawn with PIL + DejaVu."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    dejavu_dir = Path(font_manager.__file__).parent / "mpl-data" / "fonts" / "ttf"
    font = ImageFont.truetype(str(dejavu_dir / "DejaVuSans.ttf"), 22)
    bold = ImageFont.truetype(str(dejavu_dir / "DejaVuSans-Bold.ttf"), 22)

    col_w, row_h, x0, y0 = 160, 44, 20, 20
    n_cols = len(TABLE["headers"])
    n_rows = 1 + len(TABLE["rows"])
    img = Image.new("RGB", (2 * x0 + n_cols * col_w, 2 * y0 + n_rows * row_h), "white")
    d = ImageDraw.Draw(img)

    for r in range(n_rows + 1):  # horizontal rules
        y = y0 + r * row_h
        d.line([(x0, y), (x0 + n_cols * col_w, y)], fill="black", width=2)
    for c in range(n_cols + 1):  # vertical rules
        x = x0 + c * col_w
        d.line([(x, y0), (x, y0 + n_rows * row_h)], fill="black", width=2)

    grid = [TABLE["headers"], *TABLE["rows"]]
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            f = bold if r == 0 else font
            cx = x0 + c * col_w + col_w // 2
            cy = y0 + r * row_h + row_h // 2
            d.text((cx, cy), cell, font=f, fill="black", anchor="mm")

    img.save(path)
    (path.parent / "table.gt.json").write_text(
        json.dumps(TABLE, indent=2) + "\n", encoding="utf-8"
    )


def gen_chart(path: Path) -> None:
    """chart.png — labeled bar chart, few points, big fonts, values on bars."""
    _, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
    bars = ax.bar(CHART["labels"], CHART["values"], color="#4878d0", edgecolor="black")
    ax.set_title(CHART["title"], fontsize=22)
    ax.set_ylabel(CHART["ylabel"], fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_ylim(0, max(CHART["values"]) + 2)
    ax.bar_label(bars, fontsize=18, padding=2)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white", metadata=_PNG_META)
    plt.close(fig)
    (path.parent / "chart.gt.json").write_text(
        json.dumps(CHART, indent=2) + "\n", encoding="utf-8"
    )


def gen_molecule(path: Path) -> None:
    """molecule.png — RDKit 2D depiction of aspirin (optional)."""
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.rdDepictor import Compute2DCoords, SetPreferCoordGen

    SetPreferCoordGen(True)  # canonical CoordGen layout, deterministic
    mol = Chem.MolFromSmiles(SMILES_ASPIRIN)
    Compute2DCoords(mol)
    Draw.MolToImage(mol, size=(400, 400)).save(path)
    (path.parent / "molecule.gt.txt").write_text(
        SMILES_ASPIRIN + "\n", encoding="utf-8"
    )


def gen_music(path: Path) -> None:
    """music.png — Verovio engraving of the 2-bar **kern staff (optional)."""
    import cairosvg  # needs a native cairo (macOS: DYLD_FALLBACK_LIBRARY_PATH)
    import verovio

    tk = verovio.toolkit()
    tk.setOptions(
        {
            "inputFrom": "humdrum",  # **kern is a Humdrum representation
            "scale": 60,
            "pageWidth": 1400,
            "adjustPageHeight": True,
            "adjustPageWidth": True,
            "header": "none",
            "footer": "none",
        }
    )
    if not tk.loadData(KERN_MUSIC):
        raise RuntimeError("verovio rejected the kern source")
    svg = tk.renderToSVG(1)
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(path),
        output_width=800,
        background_color="white",
    )
    (path.parent / "music.gt.krn").write_text(KERN_MUSIC, encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod, key in [
        ("matplotlib", "matplotlib"),
        ("PIL", "pillow"),
        ("rdkit", "rdkit"),
        ("cairosvg", "cairosvg"),
    ]:
        try:
            v[key] = __import__(mod).__version__
        except Exception:
            v[key] = None
    try:
        import verovio

        v["verovio"] = verovio.toolkit().getVersion()
    except Exception:
        v["verovio"] = None
    return v


def main() -> int:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    core = [("formula.png", gen_formula), ("table.png", gen_table), ("chart.png", gen_chart)]
    optional = [("molecule.png", gen_molecule), ("music.png", gen_music)]

    generated, skipped = [], []
    for name, fn in core + optional:
        target = OUT_DIR / name
        try:
            fn(target)
            generated.append(name)
            print(f"  wrote {target.relative_to(REPO_ROOT)}")
        except Exception as e:  # noqa: BLE001 — record + decide fatality below
            if (name, fn) in core:
                print(f"FATAL: {name}: {e}", file=sys.stderr)
                return 1
            skipped.append(name)
            print(f"  SKIP {name}: {type(e).__name__}: {e}", file=sys.stderr)

    manifest = {
        "generator": "scripts/gen_got_format_corpus.py",
        "seed": SEED,
        "versions": _versions(),
        "sha256": {
            n: hashlib.sha256((OUT_DIR / n).read_bytes()).hexdigest() for n in generated
        },
        "skipped": skipped,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["versions"], indent=2))
    if skipped:
        print(f"skipped (optional): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
