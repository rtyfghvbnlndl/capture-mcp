"""Regression test: a screenshot must show the screen as it is *now*, not as it was at the previous call.

OpenCV's DirectShow backend (the Windows default) keeps the first frame that arrives after the last
retrieve() and drops every later frame until someone retrieves again; grab() does not wait at all.
FakeDShowCap reproduces exactly that, with the frame number encoded in the pixels.
"""

import math
import threading
import time

import numpy as np
import pytest

from capture_mcp import camera, devices
from capture_mcp.config import Config

FPS = 30.0


class FakeDShowCap:
    def __init__(self):
        self.t0 = time.time()
        self.last_retrieve = self.t0
        self.lock = threading.Lock()
        self.grabs = 0
        rng = np.random.default_rng(0)
        self.base = rng.integers(40, 220, (180, 320, 3), dtype=np.uint8)  # non-blank picture

    def _frame(self, tick: int) -> np.ndarray:
        img = self.base.copy()
        img[0, 0, 0], img[0, 0, 1] = tick % 256, tick // 256
        return img

    def grab(self):  # DirectShow: returns immediately, never waits for a frame
        self.grabs += 1
        return True

    def retrieve(self):
        with self.lock:
            # first frame produced after the previous retrieve() is held; later ones were dropped
            first = math.floor((self.last_retrieve - self.t0) * FPS) + 1
            due = self.t0 + first / FPS
            now = time.time()
            if due > now:
                time.sleep(due - now)
            self.last_retrieve = time.time()
            return True, self._frame(first)

    def read(self):
        self.grab()
        return self.retrieve()

    def release(self):
        pass


def frame_time(cap: FakeDShowCap, img: np.ndarray) -> float:
    tick = int(img[0, 0, 0]) + 256 * int(img[0, 0, 1])
    return cap.t0 + tick / FPS


@pytest.fixture
def grabber(monkeypatch):
    cap = FakeDShowCap()
    dev = devices.DeviceInfo(1, "USB Video", 700)
    monkeypatch.setattr(devices, "list_devices", lambda api, probe_fallback=True: [dev])
    monkeypatch.setattr(camera, "open_capture", lambda d, cfg: (cap, {"width": 320, "height": 180}))
    cfg = Config.from_env()
    cfg.autocrop = False
    g = camera.Grabber(cfg)
    g.selected = dev
    yield g, cap
    g.close()


def test_frame_is_fresh_after_idle(grabber):
    g, cap = grabber
    g.get_frame()
    for _ in range(3):
        time.sleep(0.5)  # the model looks at the image, thinks, then asks again
        t_request = time.time()
        f = g.get_frame()
        # the frame must have been captured after the request (allow one frame interval of slack)
        assert frame_time(cap, f.image) >= t_request - 1.5 / FPS, (
            f"stale frame: captured {t_request - frame_time(cap, f.image):.2f}s before the request"
        )


def test_reader_does_not_busy_spin(grabber):
    g, cap = grabber
    g.get_frame()
    before = cap.grabs
    time.sleep(0.5)
    # a reader paced by real frames makes ~15 calls in 0.5s; a spinning one makes hundreds of thousands
    assert cap.grabs - before < 200


def test_idle_reader_stops_decoding_and_still_returns_fresh_frames(grabber, monkeypatch):
    g, cap = grabber
    monkeypatch.setattr(camera, "ACTIVE_S", 0.3)
    g.get_frame()
    time.sleep(0.5)  # let the active window expire
    before = cap.grabs
    time.sleep(1.0)
    assert cap.grabs - before <= 1  # idle: at most a health-check read
    t_request = time.time()
    f = g.get_frame()
    assert frame_time(cap, f.image) >= t_request - 1.5 / FPS


def test_idle_release_and_transparent_reopen(grabber, monkeypatch):
    g, cap = grabber
    monkeypatch.setattr(camera, "ACTIVE_S", 0.2)
    monkeypatch.setattr(camera, "IDLE_CHECK_S", 0.1)
    g.cfg.idle_release_s = 1
    g.get_frame()
    time.sleep(1.6)
    assert g.status == "closed" and g._cap is None  # device released while idle
    t_request = time.time()
    f = g.get_frame()  # reopens on demand
    assert g.status == "open"
    assert frame_time(cap, f.image) >= t_request - 1.5 / FPS


def test_slow_open_is_not_mistaken_for_idleness(monkeypatch):
    """Opening a webcam can take ~6s; that must not count as idle time and release the device."""
    cap = FakeDShowCap()
    dev = devices.DeviceInfo(1, "USB Video", 700)

    def slow_open(d, cfg):
        time.sleep(1.5)
        return cap, {"width": 320, "height": 180}

    monkeypatch.setattr(devices, "list_devices", lambda api, probe_fallback=True: [dev])
    monkeypatch.setattr(camera, "open_capture", slow_open)
    cfg = Config.from_env()
    cfg.autocrop, cfg.idle_release_s = False, 1
    g = camera.Grabber(cfg)
    g.selected = dev
    try:
        f = g.get_frame(timeout=3)
        assert f.image is not None and g.status == "open"
    finally:
        g.close()
