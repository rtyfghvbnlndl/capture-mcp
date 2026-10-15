"""Capture-card reader built on FFmpeg packets.

A background thread keeps reading packets from the device exactly as the device sends them (MJPG from a
capture card, or a raw format if that is all it offers) and keeps only the latest one. Nothing is decoded
while streaming, so keeping the card active costs about a memory copy per frame. A screenshot waits for a
packet that arrived *after* the request and decodes just that one.

The device stays open while the server runs (by default): releasing a capture card can make the source
PC think its display was unplugged.
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from dataclasses import dataclass

import av
import numpy as np

from . import devices, settings, source
from .config import Config
from .imaging import letterbox_box, signal_state

log = logging.getLogger(__name__)

WARMUP_S = 2.5              # black frames right after opening are retried this long
FIRST_PACKET_S = 6.0        # an opened device must deliver its first packet within this time


class CaptureError(RuntimeError):
    pass


class _Released(Exception):
    """The reader released the device for idleness while a request was starting."""


@dataclass
class Frame:
    image: np.ndarray           # RGB, letterbox already cropped if enabled
    raw_size: tuple[int, int]   # (w, h) as delivered by the card
    crop_box: tuple[int, int, int, int] | None
    t_wall: float               # time.time() when the packet arrived
    signal: str                 # "OK" | "BLANK"
    signal_desc: str
    raw_digest: int = 0         # crc32 of the packet: equal digests = identical picture (HDMI is digital)


class Decoder:
    """Turns one packet into an RGB array. A fresh codec context per call: MJPG frames are independent,
    and reusing a context can return the previous frame again (PyAV issue #107)."""

    def __init__(self, codec: str, pix_fmt: str, width: int, height: int):
        self.codec, self.pix_fmt, self.width, self.height = codec, pix_fmt, width, height

    def decode(self, packet) -> np.ndarray:
        ctx = av.CodecContext.create(self.codec, "r")
        if self.codec == "rawvideo":
            ctx.width, ctx.height, ctx.pix_fmt = self.width, self.height, self.pix_fmt
        pkt = av.Packet(packet) if isinstance(packet, (bytes, bytearray)) else packet
        frames = list(ctx.decode(pkt)) or list(ctx.decode(None))
        if not frames:
            raise CaptureError("could not decode the frame from the capture card")
        return frames[0].to_ndarray(format="rgb24")


def _packets(container, stream):
    """Yield non-empty packets; demux() also yields empty flush packets."""
    for pkt in container.demux(stream):
        if pkt.size:
            yield pkt


@dataclass
class Opened:
    stream: source.Opened
    packets: object             # iterator of packets
    decoder: Decoder
    first: object               # first packet (already read while verifying the device works)


def open_device(dev: devices.DeviceInfo, cfg: Config) -> Opened:
    """Open dev and make sure it actually streams."""
    try:
        so = source.open_stream(dev, cfg)
    except Exception as e:
        raise CaptureError(
            f"Cannot open video device {dev.label}: {e}. It may be in use by another program "
            "(OBS, Camera app, browser tab, another MCP instance) or it was unplugged. "
            "Call list_devices to see what is available."
        ) from None
    it = _packets(so.container, so.stream)
    result: dict = {}

    def first():
        try:
            result["pkt"] = next(it)
        except Exception as e:  # noqa: BLE001 - reported below
            result["err"] = e

    t = threading.Thread(target=first, daemon=True)
    t.start()
    t.join(FIRST_PACKET_S)
    if "pkt" not in result:
        try:
            so.container.close()
        except Exception:
            pass
        why = result.get("err") or "timeout"
        raise CaptureError(
            f"Device {dev.label} opened but delivers no frames ({why}). It may be held by another program, "
            "or the format is unsupported: try CAPTURE_FORMAT=nv12 or a lower CAPTURE_WIDTH/HEIGHT."
        )
    return Opened(so, it, Decoder(so.codec, so.pix_fmt, so.width, so.height), result["pkt"])


@dataclass
class Probe:
    device: devices.DeviceInfo
    actual: dict
    image: np.ndarray | None    # RGB frame (uncropped) or None on error
    signal: str                 # "OK" | "BLANK" | "ERROR"
    desc: str


def probe(dev: devices.DeviceInfo, cfg: Config) -> Probe:
    """Open a device briefly, look at one frame (waiting out black warm-up frames), close it."""
    try:
        op = open_device(dev, cfg)
    except CaptureError as e:
        return Probe(dev, {}, None, "ERROR", str(e))
    try:
        img, state, desc = None, "ERROR", "no frame"
        t_end = time.monotonic() + WARMUP_S
        next_decode = 0.0
        for pkt in [op.first]:
            img = op.decoder.decode(pkt)
            state, desc = signal_state(img)
        while state != "OK" and time.monotonic() < t_end:
            pkt = next(op.packets)
            if time.monotonic() >= next_decode:  # decode a few frames, not all of them
                img = op.decoder.decode(pkt)
                state, desc = signal_state(img)
                next_decode = time.monotonic() + 0.3
        return Probe(dev, op.stream.info, img, state, desc)
    except Exception as e:  # noqa: BLE001
        return Probe(dev, op.stream.info, None, "ERROR", str(e))
    finally:
        try:
            op.stream.container.close()
        except Exception:
            pass


class Grabber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = source.backend_name(cfg.backend)
        self.device: devices.DeviceInfo | None = None
        self._op: Opened | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._open_lock = threading.Lock()
        self._cond = threading.Condition()
        self._packet = None          # latest packet
        self._packet_t = 0.0         # wall time it arrived
        self._decoder: Decoder | None = None
        self._last_use = time.monotonic()
        self._opened_at = 0.0
        self.status = "closed"       # closed | open | reconnecting | error
        self.last_error = ""
        self.selected: devices.DeviceInfo | None = None   # chosen at runtime via select_device
        self.source = ""
        self.actual: dict = {}

    def list_devices(self) -> list[devices.DeviceInfo]:
        return source.list_devices(self.backend)

    # ------------------------------------------------------------ open / close

    def _open_device(self, dev: devices.DeviceInfo) -> Opened:
        op = open_device(dev, self.cfg)
        self.actual = dict(op.stream.info)
        self._decoder = op.decoder
        self._opened_at = time.monotonic()
        with self._cond:
            self._packet, self._packet_t = op.first, time.time()
        return op

    def _resolve(self) -> tuple[devices.DeviceInfo, str]:
        """Which device to open, and why. Precedence: agent's runtime choice > CAPTURE_DEVICE > saved > auto."""
        devs = self.list_devices()
        if self.selected is not None:
            again = devices.match_saved(self.selected.to_saved(), devs) if self.selected.named else None
            return again or self.selected, "selected by agent"
        if self.cfg.device:
            return devices.pick(self.cfg.device, devs), "CAPTURE_DEVICE env"
        saved = settings.load().get("device")
        if saved:
            dev = devices.match_saved(saved, devs)
            if dev:
                return dev, "remembered from an earlier select_device"
            log.warning("remembered device %s not present, falling back to auto", saved.get("name"))
        best = devices.pick("", devs)
        # Several capture-card-like devices (e.g. two "USB Video"): prefer one that actually has a picture.
        cards = [d for d in devs if d.kind() == "capture card (likely)"]
        if len(cards) > 1:
            for d in cards:
                if probe(d, self.cfg).signal == "OK":
                    return d, "auto-detected: the only capture card with a picture"
        return best, "auto-detected"

    def planned(self) -> str:
        """Describe which device the next capture would open, without opening it."""
        try:
            dev, why = self._resolve()
            return f"{dev.label} ({why})"
        except (LookupError, CaptureError) as e:
            return f"none ({e})"

    def open(self) -> None:
        """Open the chosen device (idempotent) and start the reader thread."""
        with self._open_lock:
            if self._op is not None and self._thread and self._thread.is_alive():
                return
            try:
                dev, why = self._resolve()
                op = self._open_device(dev)
            except (CaptureError, LookupError) as e:
                self.status, self.last_error = "error", str(e)
                raise CaptureError(str(e)) from None
            self._start(dev, op, why)

    def _start(self, dev: devices.DeviceInfo, op: Opened, why: str) -> None:
        self.device, self._op, self.source = dev, op, why
        self.status, self.last_error = "open", ""
        self._last_use = time.monotonic()  # opening can take seconds; that is not idle time
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture-reader", daemon=True)
        self._thread.start()
        log.info("opened %s (%s) %s", dev.label, why, self.actual)

    def _close_container(self, op: Opened | None) -> None:
        if op is not None:
            try:
                op.stream.container.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._open_lock:
            self._stop.set()
            t = self._thread
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=3)  # the reader notices _stop at the next packet (one frame interval)
            self._thread = None
            op, self._op = self._op, None
            self._close_container(op)
            self.status = "closed"
            with self._cond:
                self._packet = None
                self._cond.notify_all()

    def select(self, dev: devices.DeviceInfo) -> None:
        """Switch to dev. If it cannot be opened, go back to the previous device and raise."""
        prev, prev_sel, prev_source = self.device, self.selected, self.source
        self.close()
        try:
            with self._open_lock:
                op = self._open_device(dev)
                self.selected = dev
                self._start(dev, op, "selected by agent")
        except CaptureError:
            self.selected = prev_sel
            if prev is not None:
                try:
                    with self._open_lock:
                        self._start(prev, self._open_device(prev), prev_source)
                except CaptureError as e:
                    self.status, self.last_error = "error", str(e)
            raise

    # ------------------------------------------------------------ reader thread

    def _loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            op = self._op
            if op is None:
                # reconnect after unplug / driver hiccup; the card may come back under another index
                self.status = "reconnecting"
                try:
                    assert self.device is not None
                    present = self.list_devices()
                    dev = devices.match_saved(self.device.to_saved(), present) if self.device.named else None
                    self.device = dev or self.device
                    self._op = self._open_device(self.device)
                    self.status, self.last_error, backoff = "open", "", 0.5
                    log.info("reconnected")
                except CaptureError as e:
                    self.last_error = str(e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 5)
                continue

            if self._release_if_idle(op):
                return
            try:
                pkt = next(op.packets)
            except Exception as e:  # noqa: BLE001 - StopIteration, FFmpeg/OS errors: the stream is gone
                if self._stop.is_set():
                    return
                log.warning("capture stream ended (%s), reconnecting", e)
                self.last_error = f"capture device stopped delivering frames ({e or 'end of stream'})"
                self._op = None
                self._close_container(op)
                continue
            with self._cond:
                self._packet, self._packet_t = pkt, time.time()
                self._cond.notify_all()

    def _release_if_idle(self, op: Opened) -> bool:
        """Optional (CAPTURE_IDLE_RELEASE_S > 0): give the device back after a period without requests.
        Decided under the same lock a request takes, so a request cannot slip in unnoticed."""
        if self.cfg.idle_release_s <= 0:
            return False
        with self._cond:
            if time.monotonic() - self._last_use <= self.cfg.idle_release_s:
                return False
            log.info("idle for %ss, releasing device", self.cfg.idle_release_s)
            self._op = None
            self._close_container(op)
            self.status = "closed"
            self._cond.notify_all()
            return True

    # ------------------------------------------------------------ public API

    def get_frame(self, timeout: float = 4.0, unless_digest: int = 0) -> Frame | None:
        """Return the first frame that arrived after this call started.

        unless_digest: if that frame's packet is byte-identical to the one with this digest, return None
        without decoding it (wait_for_change polls a static screen almost for free this way).
        """
        for _ in range(2):
            if self._op is None or not (self._thread and self._thread.is_alive()):
                if self.status == "reconnecting" and self._thread and self._thread.is_alive():
                    raise CaptureError(f"Capture device disconnected, reconnecting... ({self.last_error})")
                self.open()
            try:
                frame = self._fetch(timeout, unless_digest)
                if frame is None:
                    return None
                break
            except _Released:
                continue  # released for idleness just as we asked: reopen once
        else:
            raise CaptureError("Could not get a frame from the capture card: device was released while reading")
        # Right after opening, cards (and webcams) deliver black frames while locking onto the signal;
        # retry briefly instead of telling the model "no signal".
        while frame.signal != "OK" and time.monotonic() - self._opened_at < WARMUP_S:
            time.sleep(0.2)
            frame = self._fetch(timeout)
        return frame

    def _fetch(self, timeout: float, unless_digest: int = 0) -> Frame | None:
        with self._cond:
            if self._op is None and self.status == "closed":
                raise _Released
            t_request = time.time()
            self._last_use = time.monotonic()

            def released() -> bool:
                return self._op is None and self.status == "closed"

            ok = self._cond.wait_for(
                lambda: self._stop.is_set() or released() or self._packet_t > t_request, timeout
            )
            if released() and not self._stop.is_set():
                raise _Released
            pkt, t, decoder = self._packet, self._packet_t, self._decoder
        if not ok or pkt is None or decoder is None:
            msg = self.last_error or "no frame received in time"
            raise CaptureError(f"Could not get a frame from the capture card: {msg}")
        data = bytes(pkt)
        digest = zlib.crc32(data)
        if unless_digest and digest == unless_digest:
            return None
        try:
            img = decoder.decode(data)
        except Exception as e:  # noqa: BLE001
            raise CaptureError(f"Could not decode the frame from the capture card: {e}") from None
        return self._postprocess(img, t, digest)

    def _postprocess(self, img: np.ndarray, t: float, digest: int = 0) -> Frame:
        h, w = img.shape[:2]
        state, desc = signal_state(img)
        box = None
        if self.cfg.autocrop and state == "OK":
            box = letterbox_box(img)
            if box:
                x1, y1, x2, y2 = box
                img = img[y1:y2, x1:x2]
        return Frame(img, (w, h), box, t, state, desc, digest)

    def describe(self) -> dict:
        d = self.device
        return {
            "status": self.status,
            "device": d.label if d else None,
            "chosen_by": self.source or None,
            **self.actual,
            "last_error": self.last_error or None,
        }
