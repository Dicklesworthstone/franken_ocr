#!/usr/bin/env python3
"""The bd-kd75 probe images, styled as tests/fixtures/got/format_corpus/chart.png.

Thirty-six seeded four-bar charts in three shapes, and four formulas in the corpus's
sans-serif and in Computer Modern. Writes the PNGs and truth.json to the given
directory. Readings depend on the exact pixels: images.sha256 pins them, on
matplotlib 3.11.2.

    gen_probe.py <output directory>
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLES = ["Widget output", "Quarterly sales", "Plant yield", "Site visits", "Error counts", "Team budget"]
SHAPES = (("chart", 20261031, (6, 4)), ("wide", 20261101, (8, 4.5)), ("tall", 20261102, (4.5, 6)))
FORMULAS = {
    "energy": r"E = mc^2 + \frac{1}{2}\int_0^1 x^2 \, dx",
    "pythagoras": r"a^2 + b^2 = c^2",
    "sum": r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
    "derivative": r"\frac{d}{dx} \sin x = \cos x",
}


def charts(out: Path, truth: dict) -> None:
    for prefix, seed, size in SHAPES:
        rng = random.Random(seed)
        for n in range(12):
            labels = ["A", "B", "C", "D"]
            values = [rng.randint(1, 12) for _ in labels]
            title = TITLES[n % len(TITLES)]
            fig, ax = plt.subplots(figsize=size, dpi=100, facecolor="white")
            bars = ax.bar(labels, values, color="#4878d0", edgecolor="black")
            ax.set_title(title, fontsize=22)
            ax.set_ylabel("units", fontsize=18)
            ax.tick_params(axis="both", labelsize=18)
            ax.set_ylim(0, max(values) + 2)
            ax.bar_label(bars, fontsize=18, padding=2)
            fig.tight_layout()
            name = f"{prefix}{n:02}"
            fig.savefig(out / f"{name}.png", dpi=100, facecolor="white")
            plt.close(fig)
            truth["charts"][name] = {"title": title, "values": dict(zip(labels, map(str, values)))}


def formulas(out: Path, truth: dict) -> None:
    for fontset, tag in (("dejavusans", "sans"), ("cm", "cm")):
        matplotlib.rcParams["mathtext.fontset"] = fontset
        for key, latex in FORMULAS.items():
            fig = plt.figure(figsize=(6, 2), dpi=100, facecolor="white")
            fig.text(0.5, 0.5, f"${latex}$", fontsize=28, ha="center", va="center", color="black")
            name = f"formula_{key}_{tag}"
            fig.savefig(out / f"{name}.png", dpi=100, facecolor="white")
            plt.close(fig)
            truth["formulas"][name] = latex


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    truth = {"charts": {}, "formulas": {}}
    charts(out, truth)
    formulas(out, truth)
    (out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(len(truth["charts"]), "charts,", len(truth["formulas"]), "formulas")


if __name__ == "__main__":
    main()
