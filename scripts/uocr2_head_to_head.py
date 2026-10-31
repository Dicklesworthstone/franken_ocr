#!/usr/bin/env python3
"""U2-C5a: ground-truth head-to-head of unlimited-ocr (v1) and unlimited-ocr-2 (v2).

Every rule here (sets, seeds, normalization, metrics, statistics) was
pre-registered on bead bd-uocr2-epic-cgux.21 before the first evaluation run,
and `docs/UOCR2_EVIDENCE.md` repeats it. The inputs are downloads, so run this
with `python3 -I` and keep the data directory apart from the scripts.

Subcommands, in order:

    select   ground-truth files + PDFs -> items.json (the seeded sample, ids only)
    fetch    items.json -> page images and text-layer ground truth under DATA
    run      the focr cells -> raw outputs + runs.ndjson (wall, peak RSS, exit)
    score    -> scores.ndjson, one line per item and cell
    report   -> summary.json + report.md (bootstrap intervals, categories, cost)
    selftest the fast edit distance and TEDS against slow references

Nothing here is a gate by itself: `report` prints what the pre-registered
statistics say, losing cells included.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

SEED_OMNI = 20261030
SEED_MARATHON = 20261031
SEED_NONBAIDU = 20261101
SEED_BOOTSTRAP = 20261030
SINGLE_PAGES = 320
MARATHON_SPANS = 20
CHUNK = 40
BOOTSTRAP = 10_000
DELTA = 0.005
TAU = 0.02
CATEGORY_MIN_N = 15
FURNITURE_NED = 0.25
RASTER_DPI = 150
USER_AGENT = (
    "franken_ocr-c5a-eval/1.0 (https://github.com/Dicklesworthstone/franken_ocr; "
    "ground-truth evaluation)"
)

OMNI_REPO = "opendatalab/OmniDocBench"
OMNI_REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
MARATHON_REPO = "baidu/MarathonDocBench"
MARATHON_REVISION = "fa05fccbe82ee6ec628006a1d58bb153954c3ee3"

# The non-Baidu documents, in the pre-registered order (the span generator
# draws in this order).
NON_BAIDU = [
    ("scotus_22-451", "https://www.supremecourt.gov/opinions/23pdf/22-451_7m58.pdf",
     "12f5ea075004886774c25e7831ea1608fd0f831f0113e83bb0e85811c0a4bb6e", "en",
     "US Supreme Court opinion, public domain"),
    ("scotus_23-719", "https://www.supremecourt.gov/opinions/23pdf/23-719_19m2.pdf",
     "f3015ab4890996a0cb1f1cb3e943cf27d2d6a58ced20f8dbdc3d06e79c15d07a", "en",
     "US Supreme Court opinion, public domain"),
    ("scotus_21-476", "https://www.supremecourt.gov/opinions/22pdf/21-476_c185.pdf",
     "eb177a0fbe75459e489530e9bc75b5aad8e1e45ffb515a3b06ff0573fb3a9676", "en",
     "US Supreme Court opinion, public domain"),
    ("scotus_20-1199", "https://www.supremecourt.gov/opinions/22pdf/20-1199_hgdj.pdf",
     "23f5bdc9eed69c1ed5f76097a9f3abda0df5f6b4d5fd08e0dcc4ba5d4cc1dbd1", "en",
     "US Supreme Court opinion, public domain"),
    ("scotus_19-1392", "https://www.supremecourt.gov/opinions/21pdf/19-1392_6j37.pdf",
     "24e1b14e521fd6cc653ddbe1fbb0c41bcdd7fda6c8f32d0867d23b4415a1e8b0", "en",
     "US Supreme Court opinion, public domain"),
    ("scotus_17-1618", "https://www.supremecourt.gov/opinions/19pdf/17-1618_hfci.pdf",
     "7ed4ed6f1919e99b728b8612626a0f37cb713a023c1734e88b938592e6a3b1b8", "en",
     "US Supreme Court opinion, public domain"),
    ("nist_fips197upd1", "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf",
     "62c86eb567f13edb8f71826e985da870b04ef6381634f303cdb16e84d47becd1", "en",
     "NIST publication, US government work, public domain"),
    ("nist_ir7298r2", "https://nvlpubs.nist.gov/nistpubs/ir/2013/NIST.IR.7298r2.pdf",
     "5b8c8fde0e534d4568f9dd4c6b9c6a36ac6539e5c4296af5a27de55c69f1a393", "en",
     "NIST publication, US government work, public domain"),
    ("arxiv_1706.03762v7", "https://arxiv.org/pdf/1706.03762v7",
     "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697", "en",
     "arXiv paper; local evaluation only, not redistributed"),
    ("wiki_en_ocr",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Optical_character_recognition",
     "5e90c384a43ee804b82b51444f31aba1834187415ede686db16c3d492115ee03", "en",
     "Wikipedia PDF render, CC BY-SA 4.0; local evaluation only"),
    ("wiki_zh_printing",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E5%8D%B0%E5%88%B7%E6%9C%AF",
     "07bb287fe537b9c2fcf7b5c27a880ee0a7fd9a9d9926065866006ee45e7bd38e", "zh",
     "Wikipedia PDF render, CC BY-SA 4.0; local evaluation only"),
    ("wiki_zh_papermaking",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E9%80%A0%E7%BA%B8%E6%9C%AF",
     "597590fbeb489fc6dbc9b154aaf45ceb7853e689fe3bcdf5aee93cdc9248108f", "zh",
     "Wikipedia PDF render, CC BY-SA 4.0; local evaluation only"),
]

FURNITURE = {
    "header", "footer", "page_number", "header_image", "footer_image", "abandon",
    "formula_number",
}
TEXT_CATEGORIES = {
    "text_block", "title", "figure_caption", "table_caption", "table_footnote",
    "figure_footnote", "equation_caption", "equation_explanation", "reference",
    "page_footnote", "footnote", "code_txt", "code_txt_caption",
}
MODELS = {"v1": "unlimited-ocr", "v2": "unlimited-ocr-2"}
DOC_CELLS = ["v2_doc", "v1_multi", "v1_pp"]


# --------------------------------------------------------------------------
# Edit distance and normalization
# --------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Unit-cost edit distance, bit-parallel (Myers 1999, Hyyrö 2001).

    Python integers are arbitrary width, so the longer string is the bit
    pattern and the loop runs over the shorter one. `selftest` checks it
    against the dynamic program.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if not b:
        return m
    peq: dict[str, int] = {}
    bit = 1
    for ch in a:
        peq[ch] = peq.get(ch, 0) | bit
        bit <<= 1
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    vp, vn, score = mask, 0, m
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | vn
        xh = (((eq & vp) + vp) ^ vp) | eq
        hp = vn | (~(xh | vp) & mask)
        hn = vp & xh
        if hp & high:
            score += 1
        elif hn & high:
            score -= 1
        hp = ((hp << 1) | 1) & mask
        hn = (hn << 1) & mask
        vp = hn | (~(xv | hp) & mask)
        vn = hp & xv
    return score


def levenshtein_reference(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def ned(a: str, b: str) -> float:
    """Normalized edit distance of two already-normalized strings."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return levenshtein(a, b) / longest


_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_HEADING = re.compile(r"^\s*#+\s*")
_BULLET = re.compile(r"^\s*[-*+•]\s+")
_WHITESPACE = re.compile(r"\s+")
_LATEX_SPACING = ("\\left", "\\right", "\\qquad", "\\quad", "\\,", "\\;", "\\!")


def normalize(text: str, formula: bool = False) -> str:
    """N(s) of the pre-registration, applied to both sides."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("<PAGE>", "")
    text = _IMAGE_REF.sub("", text)
    text = html.unescape(text)
    text = _TAG.sub("", text)
    for delimiter in ("\\(", "\\)", "\\[", "\\]", "$"):
        text = text.replace(delimiter, "")
    lines = []
    for line in text.split("\n"):
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        lines.append(line)
    text = "\n".join(lines)
    for marker in ("**", "__", "`"):
        text = text.replace(marker, "")
    if formula:
        for command in _LATEX_SPACING:
            text = text.replace(command, "")
    return _WHITESPACE.sub("", text)


# --------------------------------------------------------------------------
# TEDS (exact tree edit distance, Zhang–Shasha)
# --------------------------------------------------------------------------


class TableNode:
    __slots__ = ("tag", "span", "text", "children")

    def __init__(self, tag: str, span=(1, 1), text: str = ""):
        self.tag = tag
        self.span = span
        self.text = text
        self.children: list[TableNode] = []


def _span(cell, name: str) -> int:
    try:
        return max(1, int(str(cell.get(name, "1")).strip() or "1"))
    except ValueError:
        return 1


def table_tree_from_html(fragment: str) -> TableNode | None:
    import lxml.html  # noqa: PLC0415 - only the table metrics need lxml

    try:
        root = lxml.html.fromstring(fragment)
    except Exception:  # noqa: BLE001 - malformed model output is an empty table
        return None
    table = root if root.tag == "table" else root.find(".//table")
    if table is None:
        return None
    tree = TableNode("table")
    for row in table.iter("tr"):
        row_node = TableNode("tr")
        for cell in row:
            if not isinstance(cell.tag, str) or cell.tag not in ("td", "th"):
                continue
            span = (_span(cell, "colspan"), _span(cell, "rowspan"))
            row_node.children.append(TableNode("td", span, normalize(cell.text_content())))
        tree.children.append(row_node)
    return tree


def table_tree_from_pipes(lines: list[str]) -> TableNode:
    tree = TableNode("table")
    for line in lines:
        cells = line.strip().strip("|").split("|")
        if all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells):
            continue
        row = TableNode("tr")
        row.children = [TableNode("td", (1, 1), normalize(c)) for c in cells]
        tree.children.append(row)
    return tree


def _postorder(root: TableNode):
    nodes: list[TableNode] = []
    leftmost: list[int] = []

    def walk(node: TableNode) -> int:
        first = None
        for child in node.children:
            child_leftmost = walk(child)
            if first is None:
                first = child_leftmost
        nodes.append(node)
        index = len(nodes) - 1
        leftmost.append(index if first is None else first)
        return leftmost[index]

    walk(root)
    seen: dict[int, int] = {}
    for index, left in enumerate(leftmost):
        seen[left] = index
    return nodes, leftmost, sorted(seen.values())


def _rename_cost(a: TableNode, b: TableNode) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag == "td":
        if a.span != b.span:
            return 1.0
        return ned(a.text, b.text)
    return 0.0


def tree_edit_distance(t1: TableNode, t2: TableNode) -> float:
    n1, l1, k1 = _postorder(t1)
    n2, l2, k2 = _postorder(t2)
    treedist = [[0.0] * len(n2) for _ in n1]
    for i in k1:
        for j in k2:
            ioff, joff = l1[i] - 1, l2[j] - 1
            rows, cols = i - l1[i] + 2, j - l2[j] + 2
            fd = [[0.0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, rows):
                node_x = x + ioff
                for y in range(1, cols):
                    node_y = y + joff
                    if l1[node_x] == l1[i] and l2[node_y] == l2[j]:
                        value = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[x - 1][y - 1] + _rename_cost(n1[node_x], n2[node_y]),
                        )
                        fd[x][y] = value
                        treedist[node_x][node_y] = value
                    else:
                        p = l1[node_x] - 1 - ioff
                        q = l2[node_y] - 1 - joff
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[p][q] + treedist[node_x][node_y],
                        )
    return treedist[len(n1) - 1][len(n2) - 1]


def _size(node: TableNode) -> int:
    return 1 + sum(_size(child) for child in node.children)


def teds(t1: TableNode, t2: TableNode) -> float:
    return 1.0 - tree_edit_distance(t1, t2) / max(_size(t1), _size(t2))


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

_HTML_TABLE = re.compile(r"<table\b.*?(?:</table>|\Z)", re.IGNORECASE | re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_SINGLE_DOLLAR_LINE = re.compile(r"^\s*\$[^$\n]+\$\s*$", re.MULTILINE)


def _pipe_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in markdown.split("\n"):
        if line.lstrip().startswith("|"):
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []
    if len(current) >= 2:
        blocks.append(current)
    return blocks


def predicted_tables(markdown: str) -> list[TableNode]:
    tables = []
    for match in _HTML_TABLE.finditer(markdown):
        tree = table_tree_from_html(match.group(0))
        if tree is not None:
            tables.append(tree)
    for block in _pipe_blocks(markdown):
        tables.append(table_tree_from_pipes(block))
    return tables


def predicted_display_math(markdown: str) -> list[str]:
    blocks = [m.group(0) for m in _DISPLAY_MATH.finditer(markdown)]
    rest = _DISPLAY_MATH.sub("\n", markdown)
    blocks += [m.group(0) for m in _SINGLE_DOLLAR_LINE.finditer(rest)]
    return blocks


def strip_tables_and_math(markdown: str) -> str:
    text = _HTML_TABLE.sub("\n", markdown)
    lines = [line for line in text.split("\n") if not line.lstrip().startswith("|")]
    text = "\n".join(lines)
    text = _DISPLAY_MATH.sub("\n", text)
    return _SINGLE_DOLLAR_LINE.sub("", text)


def remove_furniture(lines: list[str], furniture: list[str]) -> list[str]:
    """Each furniture string removes at most one line: the unremoved line
    closest to it, when its NED is at most 0.25."""
    normalized = [normalize(line) for line in lines]
    removed = [False] * len(lines)
    for item in furniture:
        target = normalize(item)
        if not target:
            continue
        best, best_index = None, None
        for index, candidate in enumerate(normalized):
            if removed[index] or not candidate:
                continue
            longest = max(len(candidate), len(target))
            if abs(len(candidate) - len(target)) > FURNITURE_NED * longest:
                continue
            distance = ned(candidate, target)
            if best is None or distance < best:
                best, best_index = distance, index
        if best is not None and best <= FURNITURE_NED:
            removed[best_index] = True
    return [line for line, gone in zip(lines, removed) if not gone]


def prediction_views(markdown: str, furniture: list[str]) -> dict:
    text_lines = remove_furniture(strip_tables_and_math(markdown).split("\n"), furniture)
    all_lines = remove_furniture(markdown.split("\n"), furniture)
    return {
        "text": normalize("\n".join(text_lines)),
        "all": normalize("\n".join(all_lines)),
        "tables": predicted_tables(markdown),
        "formula": normalize("\n".join(predicted_display_math(markdown)), formula=True),
    }


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def _box_key(det: dict):
    poly = det.get("poly") or [0, 0]
    xs, ys = poly[0::2], poly[1::2]
    return (min(ys) if ys else 0, min(xs) if xs else 0)


def ordered_dets(dets: list[dict]) -> list[dict]:
    live = [d for d in dets if not d.get("ignore")]
    ordered = sorted((d for d in live if d.get("order") is not None), key=lambda d: d["order"])
    rest = sorted((d for d in live if d.get("order") is None), key=_box_key)
    return ordered + rest


def layout_views(dets: list[dict]) -> dict:
    text_parts, all_parts, furniture, tables, formulas = [], [], [], [], []
    for det in ordered_dets(dets):
        category = det.get("category_type")
        if category in FURNITURE:
            if det.get("text"):
                furniture.append(det["text"])
        elif category in TEXT_CATEGORIES:
            text = det.get("text") or ""
            text_parts.append(text)
            all_parts.append(text)
        elif category == "table":
            fragment = det.get("html") or ""
            tables.append(fragment)
            all_parts.append(fragment)
        elif category == "equation_isolated":
            latex = det.get("latex") or ""
            formulas.append(latex)
            all_parts.append(latex)
    return {
        "text": normalize("\n".join(text_parts)),
        "all": normalize("\n".join(all_parts)),
        "furniture": furniture,
        "tables": tables,
        "formula": normalize("\n".join(formulas), formula=True),
        "formulas": len(formulas),
    }


_PAGE_NUMBER = re.compile(r"^[0-9ivxlcdmIVXLCDM-]{1,6}$")


def text_layer_views(pages: list[str]) -> list[dict]:
    """Per-page all view and furniture of a born-digital PDF's text layer.

    A running header or footer is an edge line (first or last three non-empty
    lines) whose digit-free normalized form has at least 5 characters and
    recurs on at least 3 pages; a bare page number is furniture among the first
    or last two lines. (Amended before the first run: recurrence alone took
    table cells and repeated mid-page lines for headers.)
    """
    page_lines = [[line for line in page.split("\n") if line.strip()] for page in pages]

    def key(line: str) -> str:
        return re.sub(r"\d", "", normalize(line))

    def edge(index: int, count: int, width: int) -> bool:
        return index < width or index >= count - width

    recurrence = collections.Counter()
    for lines in page_lines:
        recurrence.update({
            key(line) for index, line in enumerate(lines)
            if len(key(line)) >= 5 and edge(index, len(lines), 3)
        })
    views = []
    for lines in page_lines:
        content, furniture = [], []
        for index, line in enumerate(lines):
            running = len(key(line)) >= 5 and edge(index, len(lines), 3) and recurrence[key(line)] >= 3
            if running or (edge(index, len(lines), 2) and _PAGE_NUMBER.fullmatch(normalize(line))):
                furniture.append(line)
            else:
                content.append(line)
        views.append({"all": normalize("\n".join(content)), "furniture": furniture})
    return views


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


def _masked(page: dict) -> bool:
    return any(
        (d.get("category_type") or "").endswith("_mask") or d.get("category_type") == "need_mask"
        for d in page["layout_dets"]
        if not d.get("ignore")
    )


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    frame = sum(sizes.values())
    quotas = {k: total * n / frame for k, n in sizes.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    order = sorted(sizes, key=lambda k: (-(quotas[k] - counts[k]), k))
    for k in order[: total - sum(counts.values())]:
        counts[k] += 1
    return counts


def _pdf_pages(path: Path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
    return int(re.search(r"^Pages:\s+(\d+)", out.stdout, re.MULTILINE).group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_select(args: argparse.Namespace) -> int:
    with open(args.omni_json, encoding="utf-8") as handle:
        omni = json.load(handle)
    frame = [p for p in omni if not _masked(p)]
    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for page in frame:
        strata[page["page_info"]["page_attribute"]["data_source"]].append(page)
    counts = _allocate({k: len(v) for k, v in strata.items()}, SINGLE_PAGES)
    rng = random.Random(SEED_OMNI)
    singles = []
    for source in sorted(strata):
        pages = sorted(strata[source], key=lambda p: p["page_info"]["image_path"])
        for page in rng.sample(pages, counts[source]):
            attr = page["page_info"]["page_attribute"]
            cats = [d["category_type"] for d in page["layout_dets"] if not d.get("ignore")]
            singles.append({
                "id": "omni-" + Path(page["page_info"]["image_path"]).stem,
                "image_path": page["page_info"]["image_path"],
                "data_source": attr.get("data_source"),
                "language": attr.get("language"),
                "layout": attr.get("layout"),
                "special_issue": attr.get("special_issue") or [],
                "tables": cats.count("table"),
                "formulas": cats.count("equation_isolated"),
            })

    with open(args.marathon_json, encoding="utf-8") as handle:
        marathon = json.load(handle)
    pages_of = {Path(d["page_info"]["image_path"]).stem: len(d["page_info"]["images_list"]) for d in marathon}
    rng = random.Random(SEED_MARATHON)
    documents = []
    for doc_id in rng.sample(sorted(pages_of), MARATHON_SPANS):
        n = pages_of[doc_id]
        length = rng.randint(5, 10)
        start = rng.randint(1, n - length + 1)
        documents.append({
            "id": f"marathon-{doc_id}-p{start}-{start + length - 1}",
            "source": "marathon",
            "doc_id": doc_id,
            "pages": list(range(start, start + length)),
            "document_pages": n,
            "language": "zh+en",
        })

    rng = random.Random(SEED_NONBAIDU)
    pdf_dir = Path(args.pdf_dir)
    for name, url, sha, language, licence in NON_BAIDU:
        path = pdf_dir / f"{name}.pdf"
        actual = _sha256(path)
        if actual != sha:
            print(f"{path}: sha256 {actual}, expected {sha}", file=sys.stderr)
            return 2
        n = _pdf_pages(path)
        length = rng.randint(5, min(10, n))
        start = rng.randint(1, n - length + 1)
        documents.append({
            "id": f"nonbaidu-{name}-p{start}-{start + length - 1}",
            "source": "nonbaidu",
            "name": name,
            "url": url,
            "sha256": sha,
            "licence": licence,
            "pages": list(range(start, start + length)),
            "document_pages": n,
            "language": language,
        })

    items = {
        "schema_version": 1,
        "bead": "bd-uocr2-epic-cgux.21",
        "seeds": {"omni": SEED_OMNI, "marathon": SEED_MARATHON, "nonbaidu": SEED_NONBAIDU},
        "sources": {
            "omnidocbench": {
                "repo": OMNI_REPO, "revision": OMNI_REVISION,
                "ground_truth": "OmniDocBench.json",
                "ground_truth_sha256": _sha256(Path(args.omni_json)),
                "pages": len(omni), "frame_pages": len(frame),
                "licence": "research use only, non-commercial; local evaluation, nothing redistributed",
            },
            "marathondocbench": {
                "repo": MARATHON_REPO, "revision": MARATHON_REVISION,
                "ground_truth": "MarathonDocBench.json",
                "ground_truth_sha256": _sha256(Path(args.marathon_json)),
                "documents": len(marathon), "licence": "Apache-2.0", "builder": "Baidu",
            },
            "nonbaidu": {
                "ground_truth": "PDF text layer (pdftotext -enc UTF-8), born-digital",
                "raster": f"pdftoppm -r {RASTER_DPI} -png",
            },
        },
        "single_pages": singles,
        "documents": documents,
    }
    Path(args.out).write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    nonbaidu = sum(d["source"] == "nonbaidu" for d in documents)
    print(f"{len(singles)} single pages from {len(frame)}; {len(documents)} documents, "
          f"{nonbaidu} non-Baidu ({nonbaidu / len(documents):.1%})")
    return 0


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(dest)


def _hf(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def cmd_fetch(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    data = Path(args.data)
    for item in items["single_pages"]:
        _download(_hf(OMNI_REPO, OMNI_REVISION, f"images/{item['image_path']}"),
                  data / "omni" / item["image_path"])
    for doc in items["documents"]:
        pages_dir = data / "docs" / doc["id"]
        pages_dir.mkdir(parents=True, exist_ok=True)
        if doc["source"] == "marathon":
            for index, page in enumerate(doc["pages"], 1):
                cached = data / "marathon" / doc["doc_id"] / f"page_{page}.png"
                _download(_hf(MARATHON_REPO, MARATHON_REVISION,
                              f"images/{doc['doc_id']}/page_{page}.png"), cached)
                target = pages_dir / f"p{index}.png"
                if not target.exists():
                    shutil.copyfile(cached, target)
        else:
            pdf = Path(args.pdf_dir) / f"{doc['name']}.pdf"
            if _sha256(pdf) != doc["sha256"]:
                print(f"{pdf}: sha256 mismatch", file=sys.stderr)
                return 2
            for index, page in enumerate(doc["pages"], 1):
                target = pages_dir / f"p{index}.png"
                if not target.exists():
                    subprocess.run(
                        ["pdftoppm", "-r", str(RASTER_DPI), "-png", "-f", str(page), "-l",
                         str(page), "-singlefile", str(pdf), str(target.with_suffix(""))],
                        check=True,
                    )
            layer = data / "text_layer" / f"{doc['name']}.json"
            if not layer.exists():
                out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
                                     capture_output=True, check=True)
                pages = out.stdout.decode("utf-8").split("\f")
                layer.parent.mkdir(parents=True, exist_ok=True)
                layer.write_text(json.dumps(pages[: doc["document_pages"]], ensure_ascii=False),
                                 encoding="utf-8")
    print("fetched", len(items["single_pages"]), "pages and", len(items["documents"]), "documents")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _parse_time(stderr: str) -> dict:
    wall = re.search(r"([\d.]+) real", stderr)
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return {
        "wall_s": float(wall.group(1)) if wall else None,
        "peak_rss_bytes": int(rss.group(1)) if rss else None,
    }


def _loadavg() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _done(runs_path: Path) -> set[str]:
    if not runs_path.exists():
        return set()
    return {json.loads(line)["run"] for line in runs_path.read_text().splitlines() if line.strip()}


def _timed(argv: list[str], stdout_path: Path, timeout: int, env: dict) -> dict:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    load = _loadavg()
    try:
        with open(stdout_path, "wb") as out:
            proc = subprocess.run(["/usr/bin/time", "-l", *argv], stdout=out,
                                  stderr=subprocess.PIPE, timeout=timeout, env=env)
        stderr = proc.stderr.decode("utf-8", "replace")
        code = proc.returncode
    except subprocess.TimeoutExpired as expired:
        stderr = (expired.stderr or b"").decode("utf-8", "replace")
        code = "timeout"
    stdout_path.with_suffix(".stderr").write_text(stderr, encoding="utf-8")
    return {"started": started, "loadavg_at_start": load, "exit": code, **_parse_time(stderr)}


def cmd_run(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    data = Path(args.data)
    runs_path = data / "runs.ndjson"
    done = _done(runs_path)
    env = dict(os.environ, FOCR_NO_RESIDENT="1", FOCR_MODEL_DIR=args.model_dir)
    artifacts = {}
    for key, path in (("v1", args.v1_artifact), ("v2", args.v2_artifact)):
        artifacts[key] = {"path": path, "sha256": _sha256(Path(path))}
    meta = data / "artifacts.json"
    if not meta.exists():
        meta.write_text(json.dumps({"focr": args.focr, "artifacts": artifacts}, indent=1) + "\n")

    def record(entry: dict) -> None:
        with open(runs_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if args.part in ("all", "single"):
        pages = items["single_pages"]
        chunks = [pages[i : i + CHUNK] for i in range(0, len(pages), CHUNK)]
        for k, chunk in enumerate(chunks):
            order = ["v1", "v2"] if k % 2 == 0 else ["v2", "v1"]
            for key in order:
                run = f"single/{key}/chunk{k}"
                if run in done:
                    continue
                images = [str(data / "omni" / item["image_path"]) for item in chunk]
                out = data / "runs" / "single" / key / f"chunk{k}.json"
                argv = [args.focr, "ocr-batch", "--model", MODELS[key], "--json", *images]
                result = _timed(argv, out, args.timeout, env)
                record({"run": run, "cell": f"{key}_single", "model": MODELS[key],
                        "artifact_sha256": artifacts[key]["sha256"],
                        "items": [item["id"] for item in chunk], "output": str(out), **result})
                print(run, result["exit"], result["wall_s"], flush=True)

    if args.part in ("all", "docs"):
        for index, doc in enumerate(items["documents"]):
            shift = index % len(DOC_CELLS)
            for cell in DOC_CELLS[shift:] + DOC_CELLS[:shift]:
                run = f"docs/{doc['id']}/{cell}"
                if run in done:
                    continue
                key = cell[:2]
                out = data / "runs" / "docs" / doc["id"] / f"{cell}.json"
                argv = [args.focr, "ocr", str(data / "docs" / doc["id"]), "--model",
                        MODELS[key], "--json", "-o", str(out)]
                if cell != "v1_pp":
                    argv.insert(5, "--multi-page")
                result = _timed(argv, out.with_suffix(".stdout"), args.timeout, env)
                record({"run": run, "cell": cell, "model": MODELS[key],
                        "artifact_sha256": artifacts[key]["sha256"], "items": [doc["id"]],
                        "pages": len(doc["pages"]), "output": str(out), **result})
                print(run, result["exit"], result["wall_s"], flush=True)
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _score_single(item: dict, gt: dict, markdown: str | None) -> dict:
    failed = markdown is None
    pred = prediction_views(markdown or "", gt["furniture"])
    result = {"failed": failed}
    if gt["text"]:
        result["cer_text"] = 1.0 if failed else ned(pred["text"], gt["text"])
    result["ed_all"] = 1.0 if failed else ned(pred["all"], gt["all"])
    if len(gt["tables"]) == 1:
        truth = table_tree_from_html(gt["tables"][0])
        scores = [teds(truth, t) for t in pred["tables"]] if truth is not None and not failed else []
        result["teds"] = max(scores, default=0.0)
    if gt["formulas"]:
        result["formula_ed"] = 1.0 if failed else ned(pred["formula"], gt["formula"])
    result["pred_chars"] = len(pred["all"])
    result["gt_chars"] = len(gt["all"])
    return result


def _doc_truth(doc: dict, marathon: dict, data: Path) -> list[dict]:
    if doc["source"] == "marathon":
        dets = marathon[doc["doc_id"]]["layout_dets"]
        by_page = collections.defaultdict(list)
        for det in dets:
            by_page[det.get("page_id")].append(det)
        return [layout_views(by_page.get(page, [])) for page in doc["pages"]]
    pages = json.loads((data / "text_layer" / f"{doc['name']}.json").read_text(encoding="utf-8"))
    views = text_layer_views(pages)
    return [views[page - 1] for page in doc["pages"]]


def _page_bodies(output: dict) -> list[str] | None:
    pages = output.get("pages")
    if isinstance(pages, list) and pages and all("text" in p for p in pages):
        return [p.get("text") or "" for p in pages]
    return None


def cmd_score(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    data = Path(args.data)
    runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
    with open(args.omni_json, encoding="utf-8") as handle:
        omni = {p["page_info"]["image_path"]: p for p in json.load(handle)}
    with open(args.marathon_json, encoding="utf-8") as handle:
        marathon = {Path(d["page_info"]["image_path"]).stem: d for d in json.load(handle)}
    out = open(data / "scores.ndjson", "w", encoding="utf-8")  # noqa: SIM115

    single_runs = {}
    for run in runs:
        if run["cell"].endswith("_single"):
            single_runs[(run["cell"], run["output"])] = run
    by_item: dict[tuple[str, str], tuple[str | None, float | None]] = {}
    for (cell, path), run in single_runs.items():
        output = _load_json(Path(path)) if run["exit"] == 0 else None
        results = {r["image"]: r for r in (output or {}).get("results", [])}
        for item in items["single_pages"]:
            if item["id"] not in run["items"]:
                continue
            image = str(data / "omni" / item["image_path"])
            entry = results.get(image)
            markdown = entry.get("markdown") if entry and entry.get("ok") else None
            by_item[(cell, item["id"])] = (markdown, entry.get("seconds") if entry else None)
    for item in items["single_pages"]:
        gt = layout_views(omni[item["image_path"]]["layout_dets"])
        for cell in ("v1_single", "v2_single"):
            if (cell, item["id"]) not in by_item:
                continue
            markdown, seconds = by_item[(cell, item["id"])]
            line = {"item": item["id"], "cell": cell, "seconds": seconds,
                    **_score_single(item, gt, markdown)}
            out.write(json.dumps(line) + "\n")

    for doc in items["documents"]:
        truth = _doc_truth(doc, marathon, data)
        gt_all = "".join(page["all"] for page in truth)
        furniture = [f for page in truth for f in page["furniture"]]
        for run in runs:
            if run["items"] != [doc["id"]] or run["cell"] not in DOC_CELLS:
                continue
            output = _load_json(Path(run["output"])) if run["exit"] in (0, 5) else None
            failed = output is None or not output.get("markdown")
            markdown = "" if failed else output["markdown"]
            pred = prediction_views(markdown, furniture)
            line = {"item": doc["id"], "cell": run["cell"], "failed": failed,
                    "exit": run["exit"], "wall_s": run["wall_s"], "pages": len(doc["pages"]),
                    "ed_all": 1.0 if failed else ned(pred["all"], gt_all),
                    "pred_chars": len(pred["all"]), "gt_chars": len(gt_all)}
            bodies = None if failed else _page_bodies(output)
            if bodies is not None and len(bodies) == len(truth):
                line["page_ed_all"] = [
                    ned(prediction_views(body, page["furniture"])["all"], page["all"])
                    for body, page in zip(bodies, truth)
                ]
            out.write(json.dumps(line) + "\n")
    out.close()
    print("wrote", data / "scores.ndjson")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _bootstrap(deltas: list[float]) -> dict:
    import numpy as np  # noqa: PLC0415

    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    picks = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": len(values), "mean": float(values.mean()), "low": float(low), "high": float(high)}


def _paired(scores, cell_a: str, cell_b: str, metric: str, keep=lambda item: True):
    """Per-item (a − b) for items where both cells carry the metric."""
    a = {s["item"]: s for s in scores if s["cell"] == cell_a and metric in s}
    b = {s["item"]: s for s in scores if s["cell"] == cell_b and metric in s}
    return [a[i][metric] - b[i][metric] for i in sorted(a.keys() & b.keys()) if keep(i)]


def cmd_report(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    data = Path(args.data)
    scores = [json.loads(line) for line in (data / "scores.ndjson").read_text().splitlines() if line.strip()]
    runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
    single = {item["id"]: item for item in items["single_pages"]}
    docs = {doc["id"]: doc for doc in items["documents"]}
    summary: dict = {"delta": DELTA, "tau": TAU, "bootstrap": BOOTSTRAP, "seed": SEED_BOOTSTRAP}

    # Single page: CER_text(v2) − CER_text(v1); TEDS(v1) − TEDS(v2).
    primary = _bootstrap(_paired(scores, "v2_single", "v1_single", "cer_text"))
    primary["non_inferior"] = primary["high"] < DELTA
    tables = _bootstrap(_paired(scores, "v1_single", "v2_single", "teds"))
    tables["non_inferior"] = tables["high"] < DELTA
    summary["single_page"] = {"cer_text": primary, "teds": tables,
                              "ed_all": _bootstrap(_paired(scores, "v2_single", "v1_single", "ed_all"))}
    for cell in ("v1_single", "v2_single"):
        rows = [s for s in scores if s["cell"] == cell]
        summary["single_page"][cell] = {
            "pages": len(rows),
            "failed": sum(s["failed"] for s in rows),
            "mean_cer_text": _mean([s["cer_text"] for s in rows if "cer_text" in s]),
            "mean_ed_all": _mean([s["ed_all"] for s in rows]),
            "mean_teds": _mean([s["teds"] for s in rows if "teds" in s]),
            "mean_formula_ed": _mean([s["formula_ed"] for s in rows if "formula_ed" in s]),
        }

    def single_category(name, keep, metric="cer_text", worse=("v2_single", "v1_single")):
        deltas = _paired(scores, worse[0], worse[1], metric, lambda i: keep(single[i]))
        row = _bootstrap(deltas) if deltas else {"n": 0}
        row["category"] = name
        row["metric"] = metric
        if row["n"] >= CATEGORY_MIN_N:
            row["regression"] = row["mean"] > TAU
        return row

    categories = []
    for source in sorted({i["data_source"] for i in single.values()}):
        categories.append(single_category(f"data_source={source}", lambda i, s=source: i["data_source"] == s))
    for language in ("english", "simplified_chinese", "en_ch_mixed"):
        categories.append(single_category(f"language={language}", lambda i, l=language: i["language"] == l))
    categories.append(single_category(
        "CJK", lambda i: i["language"] in ("simplified_chinese", "en_ch_mixed", "traditional_chinese")))
    for layout in sorted({i["layout"] for i in single.values()}):
        categories.append(single_category(f"layout={layout}", lambda i, l=layout: i["layout"] == l))
    for issue, names in (("watermark", ("watermark", "with_watermark")),
                         ("colorful_backgroud", ("colorful_backgroud",)),
                         ("fuzzy_scan", ("fuzzy_scan",))):
        categories.append(single_category(
            f"special_issue={issue}", lambda i, n=names: any(x in i["special_issue"] for x in n)))
    categories.append(single_category("tables (TEDS)", lambda i: True, "teds", ("v1_single", "v2_single")))
    categories.append(single_category("formulas (FormulaED)", lambda i: True, "formula_ed"))
    summary["single_page_categories"] = categories

    # Documents: ED_all(v1 cell) − ED_all(v2).
    doc_summary = {}
    for baseline in ("v1_multi", "v1_pp"):
        row = _bootstrap(_paired(scores, baseline, "v2_doc", "ed_all"))
        row["v2_wins"] = row["low"] > 0
        doc_summary[f"vs_{baseline}"] = row
        cats = []
        for name, keep in (
            ("marathon", lambda d: d["source"] == "marathon"),
            ("non-Baidu (born-digital)", lambda d: d["source"] == "nonbaidu"),
            ("en", lambda d: d["language"] == "en"),
            ("zh-containing", lambda d: d["language"] != "en"),
        ):
            deltas = _paired(scores, baseline, "v2_doc", "ed_all", lambda i, k=keep: k(docs[i]))
            cat = _bootstrap(deltas) if deltas else {"n": 0}
            cat["category"] = name
            if cat["n"] >= CATEGORY_MIN_N:
                cat["regression"] = -cat["mean"] > TAU
            cats.append(cat)
        doc_summary[f"vs_{baseline}_categories"] = cats
    for cell in DOC_CELLS:
        rows = [s for s in scores if s["cell"] == cell]
        doc_summary[cell] = {
            "documents": len(rows),
            "failed": sum(s["failed"] for s in rows),
            "exit_nonzero": sum(s["exit"] != 0 for s in rows),
            "mean_ed_all": _mean([s["ed_all"] for s in rows]),
        }
    page_pairs = _paired_pages(scores)
    doc_summary["per_page_v1_multi_minus_v2"] = _bootstrap(page_pairs) if page_pairs else {"n": 0}
    summary["documents"] = doc_summary

    # Cost: wall seconds per completed page and peak RSS, per cell.
    cost = {}
    for cell in ("v1_single", "v2_single", *DOC_CELLS):
        cell_runs = [r for r in runs if r["cell"] == cell]
        pages = sum(len(r["items"]) if cell.endswith("_single") else r["pages"] for r in cell_runs)
        wall = sum(r["wall_s"] or 0 for r in cell_runs)
        rss = max((r["peak_rss_bytes"] or 0 for r in cell_runs), default=0)
        cost[cell] = {"runs": len(cell_runs), "pages": pages, "wall_s": wall,
                      "wall_s_per_page": wall / pages if pages else None,
                      "peak_rss_gib": rss / 2**30}
        if cell.endswith("_single"):
            seconds = [s["seconds"] for s in scores if s["cell"] == cell and s.get("seconds")]
            cost[cell]["decode_s_per_page"] = _mean(seconds)
    summary["cost"] = cost
    (data / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (data / "report.md").write_text(_markdown(summary), encoding="utf-8")
    print(_markdown(summary))
    return 0


def _paired_pages(scores) -> list[float]:
    v1 = {s["item"]: s for s in scores if s["cell"] == "v1_multi" and "page_ed_all" in s}
    v2 = {s["item"]: s for s in scores if s["cell"] == "v2_doc" and "page_ed_all" in s}
    deltas = []
    for item in sorted(v1.keys() & v2.keys()):
        deltas += [a - b for a, b in zip(v1[item]["page_ed_all"], v2[item]["page_ed_all"])]
    return deltas


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(row: dict) -> str:
    if not row.get("n"):
        return "n = 0"
    return f"n = {row['n']}, mean {row['mean']:+.4f}, 95% [{row['low']:+.4f}, {row['high']:+.4f}]"


def _markdown(summary: dict) -> str:
    sp = summary["single_page"]
    lines = ["# U2-C5a head-to-head", "", "## Single page (OmniDocBench)", ""]
    lines.append(f"- CER_text(v2) − CER_text(v1): {_fmt(sp['cer_text'])}; non-inferior at δ = {DELTA}: "
                 f"{sp['cer_text'].get('non_inferior')}")
    lines.append(f"- TEDS(v1) − TEDS(v2), single-table pages: {_fmt(sp['teds'])}; non-inferior: "
                 f"{sp['teds'].get('non_inferior')}")
    lines.append(f"- ED_all(v2) − ED_all(v1): {_fmt(sp['ed_all'])}")
    for cell in ("v1_single", "v2_single"):
        lines.append(f"- {cell}: {json.dumps(sp[cell])}")
    lines += ["", "| category | metric | n | mean Δ (v2 worse > 0) | 95% interval | regression (τ = 0.02) |",
              "|---|---|---|---|---|---|"]
    for row in summary["single_page_categories"]:
        if row["n"]:
            lines.append(f"| {row['category']} | {row['metric']} | {row['n']} | {row['mean']:+.4f} | "
                         f"[{row['low']:+.4f}, {row['high']:+.4f}] | {row.get('regression', 'n < 15')} |")
    doc = summary["documents"]
    lines += ["", "## Documents of 5–10 pages", ""]
    for baseline in ("v1_multi", "v1_pp"):
        lines.append(f"- ED_all({baseline}) − ED_all(v2_doc): {_fmt(doc[f'vs_{baseline}'])}; "
                     f"v2 wins: {doc[f'vs_{baseline}'].get('v2_wins')}")
        for cat in doc[f"vs_{baseline}_categories"]:
            lines.append(f"  - {cat['category']}: {_fmt(cat)}; regression: {cat.get('regression', 'n < 15')}")
    for cell in DOC_CELLS:
        lines.append(f"- {cell}: {json.dumps(doc[cell])}")
    lines.append(f"- per page, v1_multi − v2_doc (both split): {_fmt(doc['per_page_v1_multi_minus_v2'])}")
    lines += ["", "## Cost", "", "| cell | pages | wall s/page | peak RSS GiB |", "|---|---|---|---|"]
    for cell, row in summary["cost"].items():
        per = f"{row['wall_s_per_page']:.1f}" if row["wall_s_per_page"] else "—"
        lines.append(f"| {cell} | {row['pages']} | {per} | {row['peak_rss_gib']:.2f} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def cmd_selftest(_args: argparse.Namespace) -> int:
    rng = random.Random(7)
    alphabet = "abcde漢字 \n"
    for _ in range(3000):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 90)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 90)))
        assert levenshtein(a, b) == levenshtein_reference(a, b), (a, b)
    long_a = "".join(rng.choice(alphabet) for _ in range(700))
    long_b = long_a[:300] + "xyz" + long_a[350:]
    assert levenshtein(long_a, long_b) == levenshtein_reference(long_a, long_b)

    assert normalize("# Title\n- item **bold** $x^2$ \\(y\\)") == "Titleitemboldx^2y"
    assert normalize("if x < y and y > z") == "ifx<yandy>z"
    assert normalize("ＡＢ，") == "AB,"
    assert normalize("<td>a&amp;b</td>") == "a&b"

    table = "<table><tr><td>a</td><td colspan='2'>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    truth = table_tree_from_html(table)
    assert teds(truth, table_tree_from_html(table)) == 1.0
    one_cell = table_tree_from_html("<table><tr><td>a</td></tr></table>")
    # 7 nodes vs 3: delete 1 tr and 3 tds = 4 → 1 − 4/7.
    assert abs(teds(truth, one_cell) - (1 - 4 / 7)) < 1e-9
    changed = table_tree_from_html(table.replace(">d<", ">e<"))
    assert abs(teds(truth, changed) - (1 - 1 / 7)) < 1e-9
    pipes = table_tree_from_pipes(["| a | b |", "|---|---|", "| c | d |"])
    assert _size(pipes) == 7

    furniture = remove_furniture(["CIVIL HISTORY, 1603-1649.", "body text", "4"], ["4", "CIVIL HISTORY, 1603–1649."])
    assert furniture == ["body text"], furniture
    views = text_layer_views(["Cite as: 1 U.S. 2\nbody one\n1", "Cite as: 1 U.S. 3\nbody two\n2",
                              "Cite as: 1 U.S. 4\nbody three\n3"])
    assert views[0]["all"] == "bodyone" and len(views[0]["furniture"]) == 2, views[0]
    # A short line and a mid-page line that recur are content, not headers.
    middle = "Header Line\na\nb\nB\nSOURCE: SP 800-53\nc\nd\ne\nf"
    views = text_layer_views([middle, middle, middle])
    assert views[0]["furniture"] == ["Header Line"], views[0]["furniture"]
    md = "intro\n\\[ E = mc^2 \\]\n<table><tr><td>x</td></tr></table>\nend"
    assert normalize(strip_tables_and_math(md)) == "introend"
    assert normalize("\n".join(predicted_display_math(md)), formula=True) == "E=mc^2"
    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select")
    p.add_argument("--omni-json", required=True)
    p.add_argument("--marathon-json", required=True)
    p.add_argument("--pdf-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("fetch")
    p.add_argument("--items", required=True)
    p.add_argument("--pdf-dir", required=True)
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("run")
    p.add_argument("--items", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--focr", required=True)
    p.add_argument("--model-dir", required=True, help="FOCR_MODEL_DIR for both models")
    p.add_argument("--v1-artifact", required=True)
    p.add_argument("--v2-artifact", required=True)
    p.add_argument("--part", choices=["all", "single", "docs"], default="all")
    p.add_argument("--timeout", type=int, default=3 * 3600)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score")
    p.add_argument("--items", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--omni-json", required=True)
    p.add_argument("--marathon-json", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report")
    p.add_argument("--items", required=True)
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
