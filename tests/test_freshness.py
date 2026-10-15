"""Grabber behaviour against a fake device that streams real MJPG packets at 30 fps.

Covers the bugs fixed so far: a screenshot must show the screen as it is *now* (not as it was at the
previous call), the reader must not spin, a vanished stream must reconnect, and slow opens must not count
as idle time.
"""

import io
import threading
import time
import zlib

import numpy as np
import pytest
from PIL import Image

from capture_mcp import camera, devices, source
from capture_mcp.config import Config

FPS = 30.0


class FakeCard:
    """Produces one JPEG packet per frame interval; remembers when each packet was produced."""

    def __init__(self, fail_after: int | None = None, static: bool = False):
        self.t0 = time.time()
        self.produced = 0
        self.born: dict[int, float] = {}  # crc32 of packet -> production time
        self.fail_after = fail_after
        self.static = static
        self.opens = 0
        rng = np.random.default_rng(0)
        self.base = rng.integers(40, 220, (180, 320, 3), dtype=np.uint8)

    def _jpeg(self, tick: int) -> bytes:
        img = self.base.copy()
        if not self.static:
            img[:16, :64] = (tick * 7) % 256  # changes the picture every frame
            img[16:32, :64] = (tick // 37) % 256
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=90)
        return buf.getvalue()

    def packets(self):
        n = 0
        while True:
            if self.fail_after is not None and n >= self.fail_after:
                raise OSError("device unplugged")
            due = self.t0 + (self.produced + 1) / FPS
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            self.produced += 1
            data = self._jpeg(self.produced)
            self.born.setdefault(zlib.crc32(data), time.time())
            n += 1
            yield data

    def open(self, dev, cfg):
        self.opens += 1
        it = self.packets()

        class Container:
            def close(self):
                pass

        so = source.Opened(Container(), None, "mjpeg", "", 320, 180,
                           {"width": 320, "height": 180, "fps": FPS, "format": "mjpeg", "backend": "fake"})
        return camera.Opened(so, it, camera.Decoder("mjpeg", "", 320, 180), next(it))


def make_grabber(monkeypatch, card: FakeCard, **cfg_overrides):
    dev = devices.DeviceInfo(1, "USB Video", "fake", path="p1")
    monkeypatch.setattr(source, "list_devices", lambda backend="auto": [dev])
    monkeypatch.setattr(camera, "open_device", card.open)
    cfg = Config.from_env()
    cfg.autocrop = False
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    g = camera.Grabber(cfg)
    g.selected = dev
    return g


@pytest.fixture
def card_and_grabber(monkeypatch):
    card = FakeCard()
    g = make_grabber(monkeypatch, card)
    yield card, g
    g.close()


def test_frame_is_fresh_after_idle(card_and_grabber):
    card, g = card_and_grabber
    g.get_frame()
    for _ in range(3):
        time.sleep(0.5)  # the model looks at the image, thinks, then asks again
        t_request = time.time()
        f = g.get_frame()
        born = card.born[f.raw_digest]
        assert born >= t_request, f"stale frame: produced {t_request - born:.2f}s before the request"


def test_reader_paced_by_device_not_spinning(card_and_grabber):
    card, g = card_and_grabber
    g.get_frame()
    before = card.produced
    time.sleep(1.0)
    assert 20 <= card.produced - before <= 40  # ~30 packets/s, read as they come


def test_stream_loss_reconnects(monkeypatch):
    card = FakeCard(fail_after=10)
    g = make_grabber(monkeypatch, card)
    try:
        g.get_frame()
        time.sleep(1.0)  # stream dies after 10 packets; reader reconnects
        f = g.get_frame(timeout=5)
        assert f.image is not None and card.opens >= 2
    finally:
        g.close()


def test_identical_pictures_have_equal_digests(monkeypatch):
    card = FakeCard(static=True)
    g = make_grabber(monkeypatch, card)
    try:
        a, b = g.get_frame(), g.get_frame()
        assert a.raw_digest == b.raw_digest and a.t_wall < b.t_wall
    finally:
        g.close()


def test_device_stays_open_by_default(card_and_grabber):
    card, g = card_and_grabber
    g.get_frame()
    time.sleep(1.5)
    assert g.status == "open" and card.opens == 1


def test_optional_idle_release_and_reopen(monkeypatch):
    card = FakeCard()
    g = make_grabber(monkeypatch, card, idle_release_s=1)
    try:
        g.get_frame()
        time.sleep(1.6)
        assert g.status == "closed"
        t_request = time.time()
        f = g.get_frame()
        assert g.status == "open" and card.born[f.raw_digest] >= t_request
    finally:
        g.close()


def test_slow_open_is_not_mistaken_for_idleness(monkeypatch):
    card = FakeCard()
    fast_open = card.open

    def slow_open(dev, cfg):
        time.sleep(1.5)
        return fast_open(dev, cfg)

    g = make_grabber(monkeypatch, card, idle_release_s=1)
    monkeypatch.setattr(camera, "open_device", slow_open)
    try:
        f = g.get_frame(timeout=3)
        assert f.image is not None and g.status == "open"
    finally:
        g.close()


def test_decoder_handles_raw_nv12():
    w, h = 64, 32
    y = np.full((h, w), 128, np.uint8)
    uv = np.full((h // 2, w), 128, np.uint8)
    rgb = camera.Decoder("rawvideo", "nv12", w, h).decode(np.concatenate([y, uv]).tobytes())
    assert rgb.shape == (h, w, 3) and abs(int(rgb.mean()) - 128) <= 2


def test_unchanged_packet_skips_decoding(monkeypatch):
    card = FakeCard(static=True)
    g = make_grabber(monkeypatch, card)
    try:
        base = g.get_frame()
        calls = []
        real = camera.Decoder.decode
        monkeypatch.setattr(camera.Decoder, "decode", lambda self, p: calls.append(1) or real(self, p))
        assert g.get_frame(unless_digest=base.raw_digest) is None
        assert calls == []  # identical packet: never decoded
    finally:
        g.close()
