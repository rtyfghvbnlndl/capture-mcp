"""Pure image helpers: region parsing, grid overlay, letterbox crop, change detection, encoding.

Everything here works on numpy BGR frames and has no device dependency, so it is unit-testable.

Coordinate convention used everywhere a model can see it:
  * grid cells: columns are letters A.., rows are numbers 1.. ("C3"), ranges "B2:D4"
  * normalized: "x1,y1,x2,y2" in 0..1000 relative to the full (cropped) screen
Internally a Region stores fractions 0..1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

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


def fit_long_side(img: np.ndarray, long_side: int, max_upscale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(w, h)
    scale = min(scale, max_upscale)
    if abs(scale - 1.0) < 1e-3:
        return img
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def draw_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Semi-transparent grid lines plus a high-contrast label in each cell's top-left corner."""
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    thick = max(1, round(min(w, h) / 600))
    for c in range(1, cols):
        x = round(c * w / cols)
        cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 255), thick)
    for r in range(1, rows):
        y = round(r * h / rows)
        cv2.line(overlay, (0, y), (w - 1, y), (0, 255, 255), thick)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Labels are translucent so they do not completely hide text underneath (zoom has no grid anyway).
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.35, min(w, h) / 1300)
    fthick = max(1, round(fscale * 2))
    labels = out.copy()
    for r in range(rows):
        for c in range(cols):
            label = cell_name(c, r)
            (tw, th), base = cv2.getTextSize(label, font, fscale, fthick)
            x0 = round(c * w / cols) + 2
            y0 = round(r * h / rows) + 2
            cv2.rectangle(labels, (x0, y0), (x0 + tw + 4, y0 + th + base + 3), (0, 0, 0), -1)
            cv2.putText(labels, label, (x0 + 2, y0 + th + 2), font, fscale, (0, 255, 255), fthick, cv2.LINE_AA)
    cv2.addWeighted(labels, 0.7, out, 0.3, 0, out)
    return out


def contact_sheet(tiles: Sequence[tuple[np.ndarray | None, str]], tile_w: int = 480, tile_h: int = 270) -> np.ndarray:
    """One image with a labelled thumbnail per device (one image costs far fewer tokens than N images)."""
    n = max(1, len(tiles))
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    bar = 34
    sheet = np.full((rows * (tile_h + bar), cols * tile_w, 3), 40, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (img, label) in enumerate(tiles):
        x0 = (i % cols) * tile_w
        y0 = (i // cols) * (tile_h + bar)
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_w - 1, y0 + bar - 1), (0, 0, 0), -1)
        cv2.putText(sheet, label[:40], (x0 + 8, y0 + 24), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        area_y = y0 + bar
        if img is None:
            cv2.putText(sheet, "no picture", (x0 + 150, area_y + tile_h // 2), font, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        else:
            th = fit_long_side(img, tile_w, max_upscale=1.0)
            if th.shape[0] > tile_h:
                th = cv2.resize(th, (round(th.shape[1] * tile_h / th.shape[0]), tile_h), interpolation=cv2.INTER_AREA)
            h, w = th.shape[:2]
            ox, oy = x0 + (tile_w - w) // 2, area_y + (tile_h - h) // 2
            sheet[oy:oy + h, ox:ox + w] = th
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_w - 1, area_y + tile_h - 1), (90, 90, 90), 1)
    return sheet


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------- frame analysis

def letterbox_box(frame: np.ndarray, black: int = 24, tol: float = 0.02) -> tuple[int, int, int, int] | None:
    """Detect symmetric black letter/pillar-boxing. Returns (x1, y1, x2, y2) or None.

    Requiring symmetry keeps dark UI near one edge (a dark taskbar, a terminal) from being cropped,
    which would silently shift the coordinate system between screenshots.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
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
    """Classify a frame as OK or blank. Returns (state, human description)."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Count pixels that differ clearly from the dominant color. A std-based test would also flag a
    # mostly-white page with a single line of text as blank; this one only needs ~0.05% of pixels.
    small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA).astype(np.int16)
    med = np.median(small.reshape(-1, 3), axis=0)
    deviating = float((np.abs(small - med).max(axis=2) > 12).mean())
    if deviating < 0.0005:
        b, g, r = (int(v) for v in med)
        mean = (r + g + b) / 3
        if mean < 20:
            desc = "screen is completely black"
        else:
            desc = f"screen is a single flat color (RGB {r},{g},{b})"
        return "BLANK", desc + ": no HDMI signal, source monitor asleep/screensaver, or capture card idle screen"
    return "OK", "signal OK"


def diff_ratio(a: np.ndarray, b: np.ndarray, r: Region = FULL, pixel_thresh: int = 12) -> tuple[float, Region | None]:
    """Fraction of pixels (inside region) that changed between two frames, and the bounding box of changes.

    Frames are downsampled and blurred first so capture-card compression noise does not count.
    The threshold is low on purpose: a light-gray dialog over a white window differs by only ~15 levels.
    """
    if a.shape != b.shape:
        return 1.0, FULL

    def prep(f: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        h, w = g.shape
        scale = 320 / max(w, h)
        g = cv2.resize(g, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (3, 3), 0)

    ga, gb = prep(a), prep(b)
    h, w = ga.shape
    x1, y1, x2, y2 = r.to_pixels(w, h)
    mask = cv2.absdiff(ga, gb)[y1:y2, x1:x2] > pixel_thresh
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
