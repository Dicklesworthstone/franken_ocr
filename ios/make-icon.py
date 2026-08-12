#!/usr/bin/env python3
"""Draw the FrankenOCR app icon: the site's brand mark at 1024x1024.

An emerald gradient rounded square, a heavy monospace "O", and the two
Frankenstein bolt studs at opposite corners — the same lockup the website
header and every panel use. Regenerate with:

    python3 ios/make-icon.py
"""
from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
# The site's gradient: linear-gradient(140deg, #04351f, #34d399)
DARK = (0x04, 0x35, 0x1F)
LIGHT = (0x34, 0xD3, 0x99)
INK = (0x04, 0x14, 0x0D)
BOLT_CROSS = (0x0F, 0x17, 0x2A)


def gradient(size: int) -> Image.Image:
    """A 140-degree linear gradient, evaluated per pixel along the axis."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    # 140deg in CSS runs top-left-ish to bottom-right-ish; project each pixel
    # onto that axis and normalize to [0, 1].
    for y in range(size):
        for x in range(size):
            t = (x * 0.64 + y * 0.77) / (size * 1.41)
            t = min(1.0, max(0.0, t))
            # Ease the ramp so the dark end actually reads as dark. A linear
            # projection spends most of its range in the middle green and the
            # icon loses the depth the site's mark has.
            t = t * t * (3 - 2 * t)
            px[x, y] = (
                int(DARK[0] + (LIGHT[0] - DARK[0]) * t),
                int(DARK[1] + (LIGHT[1] - DARK[1]) * t),
                int(DARK[2] + (LIGHT[2] - DARK[2]) * t),
            )
    return img


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def bolt(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    """One stud: a metal disc with a crossed slot."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0x64, 0x74, 0x8B))
    draw.ellipse(
        [cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6], fill=(0x33, 0x41, 0x55)
    )
    arm = int(r * 0.62)
    width = max(3, r // 7)
    draw.line([cx - arm, cy - arm, cx + arm, cy + arm], fill=BOLT_CROSS, width=width)
    draw.line([cx - arm, cy + arm, cx + arm, cy - arm], fill=BOLT_CROSS, width=width)


def main() -> None:
    img = gradient(SIZE)
    draw = ImageDraw.Draw(img)

    font = load_font(560)
    text = "O"
    # `anchor="mm"` centers on the glyph's own middle, which is what the eye
    # reads as centered — a bbox-based placement is thrown off by the font's
    # ascent/descent padding.
    draw.text((SIZE / 2, SIZE / 2), text, font=font, fill=INK, anchor="mm")

    # Studs at the top-left / bottom-right corners, inset so the iOS icon mask
    # (which rounds the corners hard) cannot clip them.
    r = 62
    bolt(draw, 168, 168, r)
    bolt(draw, SIZE - 168, SIZE - 168, r)

    out = "Assets.xcassets/AppIcon.appiconset/icon-1024.png"
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
