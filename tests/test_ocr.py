"""ScreenReader caching/incremental logic (fake engine) and real-engine integration (skipped if unavailable)."""

import time

import cv2
import numpy as np
import pytest

from capture_mcp import ocr
from capture_mcp.config import Config
from capture_mcp.layout import Word

W, H = 1920, 1080


class FakeScreen:
    """A frame plus the words drawn on it. The fake engine 'recognises' exactly the words that lie fully
    inside the image it is given, so the reader's crop bookkeeping is what gets tested."""

    def __init__(self):
        self.img = np.full((H, W, 3), 255, np.uint8)
        self.words: list[Word] = []

    def put(self, text, x, y, w=None, h=18):
        w = w or 10 * len(text)
        self.words = [o for o in self.words if not (o.x1 < x + w and o.x2 > x and o.y1 < y + h and o.y2 > y)]
        self.img[y - 4:y + h + 4, x - 4:x + w + 4] = 255
        self.img[y + 3:y + h - 3, x:x + w] = (30, 30, 30)  # something for the change detector to see
        for i, ch in enumerate(text):  # glyph-like, text-dependent high-contrast pixels
            cx = x + int(i * w / max(1, len(text)))
            if ord(ch) % 3:
                self.img[y + 6:y + h - 6, cx + 2:cx + 5] = 255
        self.words.append(Word(text, x, y, x + w, y + h))


def fake_reader(screen: FakeScreen, calls: list):
    reader = ocr.ScreenReader(Config())

    class Eng:
        name = "fake"

        def recognize(self, img):
            return []

    reader._engine = Eng()

    def recognize(img, x0=0, y0=0):
        h, w = img.shape[:2]
        calls.append((x0, y0, w, h))
        return [o for o in screen.words if o.x1 >= x0 and o.y1 >= y0 and o.x2 <= x0 + w and o.y2 <= y0 + h]

    reader._recognize = recognize
    return reader


def texts(rd):
    return sorted(w.text for w in rd.words)


def test_unchanged_screen_is_served_from_cache():
    s, calls = FakeScreen(), []
    for i in range(10):
        s.put(f"line {i}", 100, 100 + i * 40)
    r = fake_reader(s, calls)
    assert r.read(s.img).mode == "full"
    n = len(calls)
    rd = r.read(s.img.copy())
    assert rd.mode == "unchanged" and len(calls) == n


def test_incremental_read_equals_full_read():
    s, calls = FakeScreen(), []
    for i in range(12):
        s.put(f"row {i} text", 200, 100 + i * 40)
    s.put("12:00", 1800, 1050, w=60)
    r = fake_reader(s, calls)
    r.read(s.img)
    s.put("12:01", 1800, 1050, w=60)            # clock ticks
    s.put("row 5 CHANGED", 200, 300, w=170)     # one line edited
    rd = r.read(s.img)
    assert rd.mode == "incremental"
    full = fake_reader(s, [])._full_screen_words(s.img)[0]
    assert texts(rd) == sorted(w.text for w in full)
    # only the changed areas were re-read, not the whole screen
    assert all(w * h < 0.25 * W * H for _, _, w, h in calls[1:])


def test_large_change_falls_back_to_full_read():
    s, calls = FakeScreen(), []
    s.put("before", 100, 100)
    r = fake_reader(s, calls)
    r.read(s.img)
    s.img[:] = 0
    s.words = []
    s.put("after", 100, 100)
    s.img[200:900, 200:1700] = np.random.default_rng(0).integers(0, 255, (700, 1500, 3), dtype=np.uint8)
    assert r.read(s.img).mode == "full"


def test_region_read_only_returns_words_inside():
    s, calls = FakeScreen(), []
    s.put("left", 100, 100)
    s.put("right", 1500, 100)
    r = fake_reader(s, calls)
    from capture_mcp.imaging import Region
    rd = r.read(s.img, Region(0, 0, 0.5, 1))
    assert texts(rd) == ["left"]


# ---------------------------------------------------------------- real engine

def _engine_or_skip():
    import screens

    if not screens.have_fonts():
        pytest.skip("no CJK font to render test screens")
    try:
        return ocr.ScreenReader(Config.from_env()), screens
    except Exception as e:  # pragma: no cover
        pytest.skip(str(e))


@pytest.mark.parametrize("name", ["notepad", "mail", "table", "dialog"])
def test_real_engine_reads_layout(name):
    reader, screens = _engine_or_skip()
    try:
        reader.engine()
    except ocr.OcrUnavailable as e:
        pytest.skip(f"no OCR engine: {e}")
    img, expect = screens.ALL[name]()
    t = time.perf_counter()
    rd = reader.read(img)
    dt = time.perf_counter() - t
    got = "".join("".join(rd.lines).split())
    found = [e for e in expect if "".join(e.split()) in got]
    assert len(found) >= len(expect) // 2, f"{name}: only {found} of {expect}\n" + "\n".join(rd.lines)
    assert dt < 5, f"{name}: OCR took {dt:.1f}s"
    if name == "mail":
        assert any(l.startswith("── pane") for l in rd.lines)  # sidebar and message are separate panes
    if name == "table":
        assert any(" | " in l for l in rd.lines)
    if name == "dialog":
        assert rd.graphics  # the picture is reported as a non-text area
