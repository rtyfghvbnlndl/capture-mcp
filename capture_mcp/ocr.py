"""OCR engines and the screen reader that turns a frame into layout-preserving text.

Engines (CAPTURE_OCR_ENGINE=auto|windows|rapid|none):
  * windows - the OCR built into Windows 10/11 (Windows.Media.Ocr via pywinrt). Native, no model download,
              fast on low-power CPUs. Needs the OCR language pack (Chinese/English are usually present).
  * rapid   - RapidOCR (PP-OCR models on onnxruntime). Works on Linux/macOS/Windows.
  auto picks windows when it is usable, else rapid.

ScreenReader adds what matters on slow machines: an unchanged screen returns the cached result, and a
partly changed screen is re-recognised only where it changed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import imaging, layout
from .config import Config
from .layout import Word

log = logging.getLogger(__name__)


class OcrUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- engines

class WindowsEngine:
    name = "windows"

    def __init__(self, lang: str = "", scale: float = 1.0):
        self.scale = max(0.5, min(scale, 3.0))
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.ocr import OcrEngine
        except ImportError as e:
            raise OcrUnavailable("pywinrt OCR packages not installed (pip install capture-mcp[ocr])") from e
        self._engine = None
        tried = []
        candidates = [lang] if lang else []
        candidates += ["zh-Hans-CN", "zh-Hans", "zh-CN", "zh-Hant-TW", "en-US", "en"]
        for tag in candidates:
            tried.append(tag)
            try:
                if OcrEngine.is_language_supported(Language(tag)):
                    self._engine = OcrEngine.try_create_from_language(Language(tag))
                    if self._engine is not None:
                        self.lang = tag
                        break
            except Exception:  # noqa: BLE001 - invalid tag
                continue
        if self._engine is None:
            self._engine = OcrEngine.try_create_from_user_profile_languages()
            self.lang = "user-profile"
        if self._engine is None:
            raise OcrUnavailable(f"no Windows OCR language installed (tried {tried}); add one in Settings > "
                                 "Time & language > Language (\"Optical character recognition\")")
        self.max_dim = int(OcrEngine.max_image_dimension)

    def recognize(self, img_bgr: np.ndarray) -> list[Word]:
        import asyncio

        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        h, w = gray.shape
        scale = min(self.scale, self.max_dim / max(w, h))
        if abs(scale - 1.0) > 1e-3:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            h, w = gray.shape
        writer = DataWriter()
        writer.write_bytes(np.ascontiguousarray(gray).tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.GRAY8, w, h)

        async def run():
            return await self._engine.recognize_async(bitmap)

        result = asyncio.run(run())
        words = []
        for line in result.lines:
            for wd in line.words:
                r = wd.bounding_rect
                words.append(Word(wd.text, r.x / scale, r.y / scale, (r.x + r.width) / scale,
                                  (r.y + r.height) / scale))
        return words


class RapidEngine:
    name = "rapid"

    def __init__(self, threads: int, det_size: int = 0):
        try:
            from rapidocr import RapidOCR  # type: ignore
        except ImportError as e:
            raise OcrUnavailable("RapidOCR not installed (pip install capture-mcp[ocr-rapid])") from e
        from rapidocr import ModelType, OCRVersion  # type: ignore

        params = {
            "Global.use_cls": False,             # screen text is never rotated
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            # Detection dominates the cost. The bundled PP-OCRv6 "small" detector needs ~8 s for a 1080p
            # screen (measured); PP-OCRv4 "mobile" does it ~10x faster with the same results on screen
            # text. RapidOCR downloads it (~5 MB) on first use: the installer runs --ocr-test for that.
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            # The bundled PP-OCRv6 small recogniser is kept: faster and better on mixed Chinese/English.
        }
        if det_size > 0:  # detect on a downscaled screen (long side = det_size); recognition stays sharp
            params["Det.limit_type"] = "max"
            params["Det.limit_side_len"] = det_size
        self._engine = RapidOCR(params=params)

    def recognize(self, img_bgr: np.ndarray) -> list[Word]:
        res = self._engine(img_bgr, use_cls=False)
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return []
        words = []
        for box, txt, score in zip(boxes, res.txts, res.scores):
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            words.append(Word(str(txt), min(xs), min(ys), max(xs), max(ys), float(score)))
        return words


def create_engine(cfg: Config):
    choice = (cfg.ocr_engine or "auto").lower()
    if choice == "none":
        raise OcrUnavailable("OCR disabled (CAPTURE_OCR_ENGINE=none)")
    errors = []
    order = ["windows", "rapid"] if choice == "auto" else [choice]
    for name in order:
        if name == "windows" and sys.platform != "win32":
            errors.append("windows: only on Windows")
            continue
        try:
            if name == "windows":
                return WindowsEngine(cfg.ocr_lang, cfg.ocr_scale)
            if name == "rapid":
                return RapidEngine(cfg.ocr_threads, cfg.ocr_det_size)
            errors.append(f"unknown engine {name!r}")
        except OcrUnavailable as e:
            errors.append(f"{name}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise OcrUnavailable("; ".join(errors))


# ---------------------------------------------------------------- screen reader

@dataclass
class Reading:
    lines: list[str]
    words: list[Word]
    engine: str
    seconds: float
    mode: str                    # "full" | "incremental" | "unchanged" | "region"
    graphics: list[str] = field(default_factory=list)   # grid ranges of non-text areas


class ScreenReader:
    """Frame -> layout text, with result caching and incremental re-recognition."""

    INCREMENTAL_MAX_AREA = 0.4
    PREP_SIZE = 960  # change detection resolution: half of 1080p, so one changed digit is still seen

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._engine = None
        self._error: str | None = None
        self._lock = threading.Lock()           # engine creation
        self._read_lock = threading.Lock()      # the cache below: read_screen and wait_for_change may overlap
        self._prep: np.ndarray | None = None    # diff_prep of the frame the cache belongs to
        self._shape: tuple | None = None
        self._words: list[Word] = []            # full-screen words of that frame (screen pixels)
        self._previous: dict[str, list[str]] = {}  # region key -> last rendered lines (for changes_only)

    # engine lifecycle ------------------------------------------------
    def engine(self):
        with self._lock:
            if self._engine is None and self._error is None:
                try:
                    self._engine = create_engine(self.cfg)
                    # warm-up: the first recognition loads/compiles models
                    self._engine.recognize(np.full((64, 256, 3), 255, np.uint8))
                    log.info("OCR engine ready: %s", self._engine.name)
                except OcrUnavailable as e:
                    self._error = str(e)
            if self._engine is None:
                raise OcrUnavailable(self._error or "no OCR engine")
            return self._engine

    def warm_up_in_background(self) -> None:
        threading.Thread(target=lambda: self._try_engine(), name="ocr-warmup", daemon=True).start()

    def _try_engine(self) -> None:
        try:
            self.engine()
        except OcrUnavailable as e:
            log.info("OCR unavailable: %s", e)

    def available(self) -> bool:
        try:
            self.engine()
            return True
        except OcrUnavailable:
            return False

    # recognition -----------------------------------------------------
    def _recognize(self, img: np.ndarray, x0: int = 0, y0: int = 0) -> list[Word]:
        eng = self.engine()
        out = []
        for w in eng.recognize(img):
            if w.text.strip():
                out.append(Word(w.text, w.x1 + x0, w.y1 + y0, w.x2 + x0, w.y2 + y0, w.conf))
        return out

    def _full_screen_words(self, frame: np.ndarray) -> tuple[list[Word], str]:
        prep = imaging.diff_prep(frame, self.PREP_SIZE)
        if self._prep is not None and self._shape == frame.shape:
            boxes = imaging.changed_boxes(self._prep, prep)
            if boxes == []:
                return self._words, "unchanged"
            if boxes is not None:
                H, W = frame.shape[:2]
                rects = [self._expand(b, W, H) for b in boxes]
                area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rects) / (W * H)
                if area <= self.INCREMENTAL_MAX_AREA:
                    kept = [w for w in self._words if not any(_center_in(w, r) for r in rects)]
                    for x1, y1, x2, y2 in rects:
                        kept += self._recognize(frame[y1:y2, x1:x2], x1, y1)
                    self._prep, self._words = prep, kept
                    return kept, "incremental"
        words = self._recognize(frame)
        self._prep, self._shape, self._words = prep, frame.shape, words
        return words, "full"

    def _expand(self, r: imaging.Region, W: int, H: int) -> tuple[int, int, int, int]:
        """Grow a changed area so it never cuts through a word (margin + every cached word it touches), and
        give the recogniser some context: tiny crops are misread far more often than the same text in a
        bigger picture."""
        x1, y1, x2, y2 = r.to_pixels(W, H)
        min_w, min_h, m = 320, 120, 24
        for _ in range(4):
            before = (x1, y1, x2, y2)
            # 1) every cached word the area touches is re-read whole
            for w in self._words:
                if w.x2 > x1 and w.x1 < x2 and w.y2 > y1 and w.y1 < y2:
                    x1, y1 = min(x1, int(w.x1)), min(y1, int(w.y1))
                    x2, y2 = max(x2, int(w.x2) + 1), max(y2, int(w.y2) + 1)
            # 2) then a context margin on all sides and a minimum size
            x1, y1, x2, y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
            if x2 - x1 < min_w:
                cx = (x1 + x2) // 2
                x1, x2 = max(0, cx - min_w // 2), min(W, cx + min_w // 2)
            if y2 - y1 < min_h:
                cy = (y1 + y2) // 2
                y1, y2 = max(0, cy - min_h // 2), min(H, cy + min_h // 2)
            # the margin may now touch more words: repeat until stable (words re-read whole + margin)
            touching = any(w.x2 > x1 and w.x1 < x2 and w.y2 > y1 and w.y1 < y2 and not
                           (w.x1 >= x1 + m - 1 and w.x2 <= x2 - m + 1 and w.y1 >= y1 + m - 1 and w.y2 <= y2 - m + 1)
                           for w in self._words)
            if (x1, y1, x2, y2) == before or not touching:
                break
        return x1, y1, x2, y2

    def read(self, frame: np.ndarray, region: imaging.Region = imaging.FULL) -> Reading:
        """frame: BGR screen (letterbox already cropped). Returns rendered lines in screen coordinates."""
        with self._read_lock:
            return self._read(frame, region)

    def _read(self, frame: np.ndarray, region: imaging.Region) -> Reading:
        H, W = frame.shape[:2]
        t0 = time.monotonic()
        if region.is_full:
            words, mode = self._full_screen_words(frame)
        else:
            x1, y1, x2, y2 = region.to_pixels(W, H)
            prep = imaging.diff_prep(frame, self.PREP_SIZE)
            if self._prep is not None and self._shape == frame.shape and imaging.changed_boxes(self._prep, prep) == []:
                words, mode = self._words, "unchanged"
            else:
                words, mode = self._recognize(frame[y1:y2, x1:x2], x1, y1), "region"
            words = [w for w in words if x1 <= (w.x1 + w.x2) / 2 <= x2 and y1 <= w.cy <= y2]
        seconds = time.monotonic() - t0
        if not words:
            lines = []
            node = None
        else:
            node = layout.xy_cut(words)
            layout.annotate(node, frame, rgb=False)
            lines = layout.render(node, W, H, self.cfg.grid_cols, self.cfg.grid_rows)
        graphics = []
        if self.cfg.ocr_graphics:
            gx = imaging.graphic_regions(frame, [(w.x1, w.y1, w.x2, w.y2) for w in words])
            graphics = [imaging.region_to_cells(g, self.cfg.grid_cols, self.cfg.grid_rows) for g in gx
                        if g.x2 > region.x1 and g.x1 < region.x2 and g.y2 > region.y1 and g.y1 < region.y2]
        return Reading(lines, words, self.engine().name, seconds, mode, graphics)

    # changes_only bookkeeping ------------------------------------------
    def remember(self, key: str, lines: list[str]) -> list[str] | None:
        prev = self._previous.get(key)
        self._previous[key] = lines
        return prev


def _center_in(w: Word, r: tuple[int, int, int, int]) -> bool:
    cx = (w.x1 + w.x2) / 2
    return r[0] <= cx <= r[2] and r[1] <= w.cy <= r[3]


def default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 4))
