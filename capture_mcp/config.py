"""Configuration from environment variables (MCP clients pass these via the "env" block)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DETAIL_SIZES = {"low": 768, "normal": 1280, "high": 1920}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _grid(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[x×*,]\s*(\d+)\s*", raw or "")
    if not m:
        return 8, 5
    cols, rows = int(m.group(1)), int(m.group(2))
    # Columns are letters A..Z, rows are numbers; keep it readable for small models.
    return max(1, min(cols, 26)), max(1, min(rows, 20))


@dataclass
class Config:
    device: str = ""            # index ("1") or name substring ("USB Video"); empty = auto
    backend: str = "auto"       # auto | dshow | msmf | v4l2 | any
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"        # empty = leave driver default
    max_side: int = 1280        # default long side of returned images ("normal")
    jpeg_quality: int = 80
    grid_cols: int = 8
    grid_rows: int = 5
    autocrop: bool = True       # crop letterbox black borders
    save_dir: str = ""          # save every returned image here (debugging)
    idle_release_s: int = 0     # release device after N idle seconds; 0 = never. Keep 0 for capture cards:
                                # releasing one can make the source PC think its display was unplugged
    text_only: bool = False     # no image tools at all: for models without vision, read_screen only
    ocr_engine: str = "auto"    # auto | windows | rapid | none
    ocr_lang: str = ""          # Windows OCR language tag, e.g. "zh-Hans-CN"; empty = best installed
    ocr_threads: int = 4        # RapidOCR CPU threads
    ocr_scale: float = 1.25     # Windows OCR: enlarge before OCR (measured best accuracy/time trade-off)
    ocr_det_size: int = 960     # RapidOCR: text detection on a screen scaled to this long side (0 = full); measured: same accuracy, ~2x faster
    ocr_graphics: bool = True   # also report large non-text areas (images, charts)
    text_max_chars: int = 8000  # cap on read_screen output

    @classmethod
    def from_env(cls) -> "Config":
        cols, rows = _grid(os.environ.get("CAPTURE_GRID", "8x5"))
        return cls(
            device=os.environ.get("CAPTURE_DEVICE", "").strip(),
            backend=os.environ.get("CAPTURE_BACKEND", "auto").strip().lower() or "auto",
            width=_int("CAPTURE_WIDTH", 1920),
            height=_int("CAPTURE_HEIGHT", 1080),
            fps=_int("CAPTURE_FPS", 30),
            fourcc=os.environ.get("CAPTURE_FOURCC", "MJPG").strip().upper()[:4],
            max_side=_int("CAPTURE_MAX_SIDE", 1280),
            jpeg_quality=max(30, min(_int("CAPTURE_JPEG_QUALITY", 80), 95)),
            grid_cols=cols,
            grid_rows=rows,
            autocrop=_bool("CAPTURE_AUTOCROP", True),
            save_dir=os.environ.get("CAPTURE_SAVE_DIR", "").strip(),
            idle_release_s=_int("CAPTURE_IDLE_RELEASE_S", 0),
            text_only=_bool("CAPTURE_TEXT_ONLY", False),
            ocr_engine=os.environ.get("CAPTURE_OCR_ENGINE", "auto").strip().lower() or "auto",
            ocr_lang=os.environ.get("CAPTURE_OCR_LANG", "").strip(),
            ocr_threads=max(1, _int("CAPTURE_OCR_THREADS", min(4, os.cpu_count() or 4))),
            ocr_scale=_float("CAPTURE_OCR_SCALE", 1.25),
            ocr_det_size=_int("CAPTURE_OCR_DET_SIZE", 960),
            ocr_graphics=_bool("CAPTURE_OCR_GRAPHICS", True),
            text_max_chars=max(500, _int("CAPTURE_TEXT_MAX_CHARS", 8000)),
        )

    def detail_size(self, detail: str) -> int:
        """Map the low/normal/high enum to a long-side pixel size. Unknown values -> normal."""
        d = (detail or "").strip().lower()
        if d == "normal":
            return self.max_side
        return DETAIL_SIZES.get(d, self.max_side)
