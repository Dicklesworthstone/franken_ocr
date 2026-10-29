#!/usr/bin/env python3
"""Render the two Chinese-text pages behind the CJK multi-page streaming test (U2-X1).

The pages mix common characters, which the v1 byte-level BPE vocab holds as
single tokens, with rarer ones it spells as several byte tokens (checked
against tokenizer.json by the test that uses them). A streamed decode therefore
passes through prefixes that end inside a character, which is the case U2-X1
fixes.

    python3 -I scripts/gen_cjk_stream_fixture.py --font "/System/Library/Fonts/Hiragino Sans GB.ttc" \
        --out tests/fixtures/cjk_stream

Needs Pillow and a font covering every character; the script refuses a font
that would draw a missing-glyph box. The output PNGs are committed, so the
test never needs Python or the font.
"""

import argparse
import pathlib

from PIL import Image, ImageDraw, ImageFont

PAGES = [
    [
        "第一页：饕餮盛宴",
        "魑魅魍魉，犇骉鱻羴。",
        "囍字常见于婚礼请柬。",
        "龘靐齉爨是罕见的汉字。",
    ],
    [
        "第二页：垚与赑屃",
        "三土为垚，三水为淼。",
        "鱻字由三个鱼字组成。",
        "麤犇猋皆为古字。",
    ],
]

SIZE = 1024
FONT_PX = 56
MARGIN = 96
LINE = 120


def glyph(font, ch):
    img = Image.new("L", (2 * FONT_PX, 2 * FONT_PX), 0)
    ImageDraw.Draw(img).text((0, 0), ch, font=font, fill=255)
    return img.tobytes()


def missing(font, ch):
    """Whether `font` draws `ch` as its missing-glyph box (a private-use char's)."""
    return glyph(font, ch) == glyph(font, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    font = ImageFont.truetype(args.font, FONT_PX)
    gaps = sorted({c for page in PAGES for line in page for c in line if missing(font, c)})
    if gaps:
        raise SystemExit(f"font lacks glyphs: {''.join(gaps)}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(PAGES, start=1):
        img = Image.new("L", (SIZE, SIZE), 255)
        draw = ImageDraw.Draw(img)
        for row, text in enumerate(lines):
            draw.text((MARGIN, MARGIN + row * LINE), text, font=font, fill=0)
        img.save(out / f"page_{i}.png", optimize=True)
        print(out / f"page_{i}.png")


if __name__ == "__main__":
    main()
