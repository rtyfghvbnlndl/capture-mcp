"""Pure image helpers: region parsing, grid overlay, letterbox crop, change detection, encoding.

Everything here works on numpy RGB frames (Pillow + numpy, no OpenCV) and has no device dependency,
so it is unit-testable.

Coordinate convention used everywhere a model can see it:
  * grid cells: columns are letters A.., rows are numbers 1.. ("C3"), ranges "B2:D4"
  * normalized: "x1,y1,x2,y2" in 0..1000 relative to the full (cropped) screen
Internally a Region stores fractions 0..1.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import av
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REGION_HELP = (
    'Examples: "C3" (one grid cell), "B2:D4" (cell range), '
    '"100,200,500,600" (x1,y1,x2,y2 in 0-1000 of the full screen), or "" for the full screen.'
)


class RegionError(ValueError):
    pass


@dataclass(frozen=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def is_full(self) -> bool:
        return self.x1 <= 0 and self.y1 <= 0 and self.x2 >= 1 and self.y2 >= 1

    def to_pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.x1 * w))
        y1 = int(round(self.y1 * h))
        x2 = int(round(self.x2 * w))
        y2 = int(round(self.y2 * h))
        # guarantee at least 1 px
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        return x1, y1, min(x2, w), min(y2, h)

    def norm1000(self) -> str:
        return f"{round(self.x1 * 1000)},{round(self.y1 * 1000)},{round(self.x2 * 1000)},{round(self.y2 * 1000)}"


FULL = Region(0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------- grid names

def col_name(i: int) -> str:
    return chr(ord("A") + i)


def cell_name(col: int, row: int) -> str:
    return f"{col_name(col)}{row + 1}"


def grid_spec_text(cols: int, rows: int) -> str:
    return f"{cols} columns A-{col_name(cols - 1)} x {rows} rows 1-{rows}"


def cell_region(col: int, row: int, cols: int, rows: int) -> Region:
    return Region(col / cols, row / rows, (col + 1) / cols, (row + 1) / rows)


def region_to_cells(r: Region, cols: int, rows: int) -> str:
    """Smallest cell range covering a region, e.g. "C2:E4" or "C3"."""
    tol = 0.05  # in cell units: a region overlapping a neighbour cell by a sliver does not count
    c1 = min(cols - 1, max(0, int(r.x1 * cols + tol)))
    r1 = min(rows - 1, max(0, int(r.y1 * rows + tol)))
    c2 = min(cols - 1, max(0, int(np.ceil(r.x2 * cols - tol)) - 1))
    r2 = min(rows - 1, max(0, int(np.ceil(r.y2 * rows - tol)) - 1))
    c2, r2 = max(c1, c2), max(r1, r2)
    a, b = cell_name(c1, r1), cell_name(c2, r2)
    return a if a == b else f"{a}:{b}"


# ---------------------------------------------------------------- region parsing

_CELL = r"([A-Za-z])\s*(\d{1,2})"
_CELL_RE = re.compile(rf"^\s*{_CELL}\s*$")
_RANGE_RE = re.compile(rf"^\s*{_CELL}\s*(?::|-|~|to|,|\s)\s*{_CELL}\s*$", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_cell(letter: str, num: str, cols: int, rows: int) -> tuple[int, int]:
    c = ord(letter.upper()) - ord("A")
    r = int(num) - 1
    if not (0 <= c < cols) or not (0 <= r < rows):
        raise RegionError(
            f'Cell "{letter.upper()}{num}" is outside the grid ({grid_spec_text(cols, rows)}). {REGION_HELP}'
        )
    return c, r


def _from_numbers(nums: Sequence[float]) -> Region:
    if len(nums) != 4:
        raise RegionError(f"Expected 4 numbers x1,y1,x2,y2 but got {len(nums)}. {REGION_HELP}")
    x1, y1, x2, y2 = (float(n) for n in nums)
    if any(v < 0 for v in (x1, y1, x2, y2)):
        raise RegionError(f"Coordinates must not be negative. {REGION_HELP}")
    if max(x1, y1, x2, y2) <= 1.0 and any(v != int(v) for v in (x1, y1, x2, y2)):
        scale = 1.0  # fractions 0..1
    elif max(x1, y1, x2, y2) <= 1000:
        scale = 1000.0
    else:
        raise RegionError(
            "Coordinates are normalized 0-1000 (not pixels); values above 1000 are not allowed. " + REGION_HELP
        )
    x1, x2 = sorted((x1 / scale, x2 / scale))
    y1, y2 = sorted((y1 / scale, y2 / scale))
    if x2 - x1 < 0.002 or y2 - y1 < 0.002:
        raise RegionError(f"Region is too small (x1==x2 or y1==y2). {REGION_HELP}")
    return Region(x1, y1, x2, y2)


def parse_region(value, cols: int, rows: int) -> Region:
    """Lenient parser for whatever a small model sends as a region."""
    if value is None:
        return FULL
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return FULL
        if all(isinstance(v, (int, float)) for v in value):
            return _from_numbers(value)
        value = " ".join(str(v) for v in value)
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in value for k in keys):
            return _from_numbers([value[k] for k in keys])
        raise RegionError(f"Unsupported region object. {REGION_HELP}")
    s = str(value).strip().strip("\"'`[](){}").strip()
    if s.lower() in ("", "all", "full", "fullscreen", "full screen", "screen", "none", "null", "whole", "全屏"):
        return FULL
    m = _CELL_RE.match(s)
    if m:
        c, r = _parse_cell(m.group(1), m.group(2), cols, rows)
        return cell_region(c, r, cols, rows)
    m = _RANGE_RE.match(s)
    if m:
        c1, r1 = _parse_cell(m.group(1), m.group(2), cols, rows)
        c2, r2 = _parse_cell(m.group(3), m.group(4), cols, rows)
        c1, c2 = sorted((c1, c2))
        r1, r2 = sorted((r1, r2))
        return Region(c1 / cols, r1 / rows, (c2 + 1) / cols, (r2 + 1) / rows)
    if not re.search(r"[A-Za-z]", s):
        nums = [float(n) for n in _NUM_RE.findall(s)]
        if nums:
            return _from_numbers(nums)
    raise RegionError(f'Cannot parse region "{value}". {REGION_HELP}')


# ---------------------------------------------------------------- geometry ops

def crop(frame: np.ndarray, r: Region) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = r.to_pixels(w, h)
    return frame[y1:y2, x1:x2]


def _pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(img))


def scale(img: np.ndarray, w: int, h: int, interp: str = "AREA", gray: bool = False) -> np.ndarray:
    """Resize (and optionally convert to gray) with FFmpeg's swscale: SIMD-optimised, ~3.5x faster than
    Pillow for the same filter, which matters on low-power CPUs."""
    src = np.ascontiguousarray(img)
    w, h = max(1, w), max(1, h)
    try:
        frame = av.VideoFrame.from_ndarray(src, format="rgb24" if src.ndim == 3 else "gray")
        out = frame.reformat(width=w, height=h, format="gray" if gray else frame.format.name, interpolation=interp)
        return out.to_ndarray()
    except Exception:  # swscale rejects some tiny/odd sizes; Pillow handles anything
        p = _pil(src).convert("L") if gray and src.ndim == 3 else _pil(src)
        return np.asarray(p.resize((w, h), Image.Resampling.LANCZOS))


def _gray(img: np.ndarray) -> np.ndarray:
    return scale(img, img.shape[1], img.shape[0], "POINT", gray=True) if img.ndim == 3 else img


def fit_long_side(img: np.ndarray, long_side: int, max_upscale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    factor = min(long_side / max(w, h), max_upscale)
    if abs(factor - 1.0) < 1e-3:
        return img
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    # Lanczos keeps thin UI text legible when shrinking; bicubic when enlarging zoomed crops
    return scale(img, nw, nh, "LANCZOS" if factor < 1 else "BICUBIC")


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def draw_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Semi-transparent grid lines plus a translucent label in each cell's top-left corner
    (translucent so labels do not completely hide text underneath; zoom has no grid anyway)."""
    # Blends only the line and label pixels (a full-image alpha composite was the slowest step).
    out = np.array(img, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    yellow = np.array([255, 255, 0], np.float32)

    def blend(region: np.ndarray, color: np.ndarray, alpha: float) -> None:
        region[...] = (region * (1 - alpha) + color * alpha).astype(np.uint8)

    thick = max(1, round(min(w, h) / 600))
    for c in range(1, cols):
        x = round(c * w / cols)
        blend(out[:, x:x + thick], yellow, 0.55)
    for r in range(1, rows):
        y = round(r * h / rows)
        blend(out[y:y + thick, :], yellow, 0.55)
    font = _font(max(11, round(min(w, h) / 40)))
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    for r in range(rows):
        for c in range(cols):
            label = cell_name(c, r)
            x0, y0 = round(c * w / cols) + 2, round(r * h / rows) + 2
            _, _, tx, ty = measure.textbbox((3, 1), label, font=font)
            x1, y1 = min(w, x0 + tx + 4), min(h, y0 + ty + 4)
            box = out[y0:y1, x0:x1]
            blend(box, np.zeros(3, np.float32), 0.7)  # translucent dark box
            patch = Image.fromarray(box)
            ImageDraw.Draw(patch).text((3, 1), label, font=font, fill=(255, 255, 0))
            out[y0:y1, x0:x1] = np.asarray(patch)
    return out


def contact_sheet(tiles: Sequence[tuple[np.ndarray | None, str]], tile_w: int = 480, tile_h: int = 270) -> np.ndarray:
    """One image with a labelled thumbnail per device (one image costs far fewer tokens than N images)."""
    n = max(1, len(tiles))
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    bar = 34
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + bar)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    font, big = _font(20), _font(22)
    for i, (img, label) in enumerate(tiles):
        x0 = (i % cols) * tile_w
        y0 = (i // cols) * (tile_h + bar)
        d.rectangle([x0, y0, x0 + tile_w - 1, y0 + bar - 1], fill=(0, 0, 0))
        d.text((x0 + 8, y0 + 6), label[:40], font=font, fill=(255, 255, 0))
        area_y = y0 + bar
        if img is None:
            d.text((x0 + 170, area_y + tile_h // 2 - 12), "no picture", font=big, fill=(180, 180, 180))
        else:
            th = _pil(img)
            th.thumbnail((tile_w, tile_h), Image.Resampling.BILINEAR, reducing_gap=2.0)
            sheet.paste(th, (x0 + (tile_w - th.width) // 2, area_y + (tile_h - th.height) // 2))
        d.rectangle([x0, y0, x0 + tile_w - 1, area_y + tile_h - 1], outline=(90, 90, 90))
    return np.asarray(sheet)


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    _pil(img).save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


# ---------------------------------------------------------------- frame analysis

def letterbox_box(frame: np.ndarray, black: int = 24, tol: float = 0.02) -> tuple[int, int, int, int] | None:
    """Detect symmetric black letter/pillar-boxing. Returns (x1, y1, x2, y2) or None.

    Requiring symmetry keeps dark UI near one edge (a dark taskbar, a terminal) from being cropped,
    which would silently shift the coordinate system between screenshots.
    """
    h, w = frame.shape[:2]
    gray = _gray(frame)
    rows_dark = gray.max(axis=1) < black
    cols_dark = gray.max(axis=0) < black
    if rows_dark.all():
        return None

    def run(mask: np.ndarray) -> int:
        idx = np.argmin(mask)  # first non-dark
        return int(idx) if not mask[idx] else len(mask)

    top, bottom = run(rows_dark), run(rows_dark[::-1])
    left, right = run(cols_dark), run(cols_dark[::-1])
    # each pair must be symmetric (within tol), otherwise treat that axis as uncropped
    if abs(top - bottom) > tol * h or min(top, bottom) < 0.01 * h:
        top = bottom = 0
    if abs(left - right) > tol * w or min(left, right) < 0.01 * w:
        left = right = 0
    if top == bottom == left == right == 0:
        return None
    x1, y1, x2, y2 = left, top, w - right, h - bottom
    if (x2 - x1) * (y2 - y1) < 0.4 * w * h:
        return None
    return x1, y1, x2, y2


def signal_state(frame: np.ndarray) -> tuple[str, str]:
    """Classify an RGB frame as OK or blank. Returns (state, human description)."""
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    # Count pixels that differ clearly from the dominant color. A std-based test would also flag a
    # mostly-white page with a single line of text as blank; this one only needs ~0.02% of pixels
    # (about one short word at 1080p): calling real content "no signal" is the worse mistake.
    small = scale(frame, 480, 270, "AREA").astype(np.int16)
    med = np.median(small.reshape(-1, 3), axis=0)
    deviating = float((np.abs(small - med).max(axis=2) > 12).mean())
    if deviating < 0.0002:
        r, g, b = (int(v) for v in med)
        mean = (r + g + b) / 3
        if mean < 20:
            desc = "screen is completely black"
        else:
            desc = f"screen is a single flat color (RGB {r},{g},{b})"
        return "BLANK", desc + ": no HDMI signal, source monitor asleep/screensaver, or capture card idle screen"
    return "OK", "signal OK"


def diff_prep(f: np.ndarray) -> np.ndarray:
    """Small blurred grayscale version of a frame for change detection (compute once per frame)."""
    h, w = f.shape[:2]
    k = 320 / max(w, h)
    g = scale(f, round(w * k), round(h * k), "AREA", gray=True)
    return np.asarray(Image.fromarray(g).filter(ImageFilter.GaussianBlur(0.8))).astype(np.int16)


def diff_ratio(a: np.ndarray, b: np.ndarray, r: Region = FULL, pixel_thresh: int = 12) -> tuple[float, Region | None]:
    """Fraction of pixels (inside region) that changed between two frames, and the bounding box of changes.

    Frames are downsampled and blurred first so capture-card compression noise does not count.
    The threshold is low on purpose: a light-gray dialog over a white window differs by only ~15 levels.
    """
    if a.shape != b.shape:
        return 1.0, FULL
    return diff_prepped(diff_prep(a), diff_prep(b), r, pixel_thresh)


def diff_prepped(ga: np.ndarray, gb: np.ndarray, r: Region = FULL, pixel_thresh: int = 12) -> tuple[float, Region | None]:
    """diff_ratio on outputs of diff_prep, so a baseline frame is prepared only once."""
    if ga.shape != gb.shape:
        return 1.0, FULL
    h, w = ga.shape
    x1, y1, x2, y2 = r.to_pixels(w, h)
    mask = np.abs(ga - gb)[y1:y2, x1:x2] > pixel_thresh
    if mask.size == 0:
        return 0.0, None
    ratio = float(mask.mean())
    if ratio == 0:
        return 0.0, None
    ys, xs = np.nonzero(mask)
    bbox = Region(
        (x1 + xs.min()) / w, (y1 + ys.min()) / h, (x1 + xs.max() + 1) / w, (y1 + ys.max() + 1) / h
    )
    return ratio, bbox


def iter_lines(items: Iterable[str], limit: int) -> list[str]:
    out = []
    for i, s in enumerate(items):
        if i >= limit:
            out.append(f"... (truncated, more than {limit} lines; use a smaller region)")
            break
        out.append(s)
    return out
