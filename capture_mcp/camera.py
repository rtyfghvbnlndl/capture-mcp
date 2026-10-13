"""Capture-card reader.

A background thread reads (grab + retrieve) every frame at the device's own pace and keeps only the
latest one; callers wait for a frame that was read *after* their request.
While nobody asks for frames (ACTIVE_S after the last request) the thread stops decoding and only
does an occasional health-check read, so an idle server costs almost no CPU.

Reading every frame, not just grabbing, matters: OpenCV's DirectShow backend (the Windows default)
returns from grab() immediately and holds the first frame that arrives after the last retrieve(),
dropping all later ones until the next retrieve(). Decoding only on demand therefore handed out the
frame from just after the *previous* screenshot (the "first screenshot is old" bug), and the grab()
loop busy-spun a CPU core.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import devices, settings
from .config import Config
from .imaging import letterbox_box, signal_state

log = logging.getLogger(__name__)

FAILS_BEFORE_RECONNECT = 30
WARMUP_S = 2.5
ACTIVE_S = 5.0              # keep reading at full rate this long after the last request
IDLE_CHECK_S = 5.0          # while idle, one read this often to notice an unplugged card
FRESH_FALLBACK_READS = 4    # backends that never block: trust the Nth read after the request


class CaptureError(RuntimeError):
    pass


class _Released(Exception):
    """The reader released the device for idleness while a request was starting."""


@dataclass
class Frame:
    image: np.ndarray           # BGR, letterbox already cropped if enabled
    raw_size: tuple[int, int]   # (w, h) as delivered by the card
    crop_box: tuple[int, int, int, int] | None
    t_wall: float               # time.time() when grabbed
    signal: str                 # "OK" | "BLANK"
    signal_desc: str


def open_capture(dev: devices.DeviceInfo, cfg: Config) -> tuple[cv2.VideoCapture, dict]:
    """Open a device with the configured format and wait for the first frame."""
    cap = cv2.VideoCapture(dev.index, dev.api)
    if not cap.isOpened():
        cap.release()
        raise CaptureError(
            f"Cannot open video device {dev.label}. It may be in use by another program "
            "(OBS, Camera app, browser tab, another MCP instance) or it was unplugged. "
            "Call list_devices to see what is available."
        )
    if cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc.ljust(4)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.monotonic() + 5
    got = False
    while time.monotonic() < deadline:
        if cap.grab():
            got = True
            break
        time.sleep(0.05)
    if not got:
        cap.release()
        raise CaptureError(
            f"Device {dev.label} opened but delivers no frames. It may be held by another "
            "program, or the format is unsupported: try CAPTURE_FOURCC= (empty) or "
            "CAPTURE_BACKEND=msmf, or a lower CAPTURE_WIDTH/HEIGHT."
        )
    for _ in range(3):
        cap.grab()
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 1),
        "fourcc": "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4)).rstrip(chr(0)) if fcc > 0 else "?",
        "backend": devices.backend_name(dev.api),
    }
    return cap, actual


@dataclass
class Probe:
    device: devices.DeviceInfo
    actual: dict
    image: np.ndarray | None    # last frame (raw, uncropped) or None on error
    signal: str                 # "OK" | "BLANK" | "ERROR"
    desc: str


def probe(dev: devices.DeviceInfo, cfg: Config) -> Probe:
    """Open a device briefly, grab one frame (waiting out black warm-up frames), release it."""
    try:
        cap, actual = open_capture(dev, cfg)
    except CaptureError as e:
        return Probe(dev, {}, None, "ERROR", str(e))
    try:
        t_end = time.monotonic() + WARMUP_S
        img = None
        state, desc = "ERROR", "no frame"
        while True:
            ok, frame = cap.read()
            if ok and frame is not None:
                img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                state, desc = signal_state(img)
                if state == "OK":
                    break
            if time.monotonic() > t_end:
                break
            time.sleep(0.1)
        return Probe(dev, actual, img, state, desc)
    finally:
        cap.release()


class Grabber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api = devices.backend_api(cfg.backend)
        self.device: devices.DeviceInfo | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._open_lock = threading.Lock()
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_t = 0.0         # wall time the latest read finished
        self._frame_start = 0.0     # wall time the latest read started
        self._frame_waited = False  # did that read block for a new frame (vs. return a buffered one)?
        self._reads = 0
        self._active_until = 0.0    # monotonic; reader decodes continuously until then
        self._wake = threading.Event()
        self._last_use = time.monotonic()
        self._opened_at = 0.0
        self.status = "closed"      # closed | open | reconnecting | error
        self.last_error = ""
        self.selected: devices.DeviceInfo | None = None   # chosen at runtime via select_device
        self.source = ""
        self.actual: dict = {}

    # ------------------------------------------------------------ open / close

    def _open_device(self, dev: devices.DeviceInfo) -> cv2.VideoCapture:
        cap, self.actual = open_capture(dev, self.cfg)
        self._opened_at = time.monotonic()
        return cap

    def _resolve(self) -> tuple[devices.DeviceInfo, str]:
        """Which device to open, and why. Precedence: agent's runtime choice > CAPTURE_DEVICE > saved > auto."""
        devs = devices.list_devices(self.api)
        if self.selected is not None:
            again = devices.match_saved(self.selected.to_saved(), devs) if self.selected.named else None
            return again or self.selected, "selected by agent"
        if self.cfg.device:
            return devices.pick(self.cfg.device, devs, self.api), "CAPTURE_DEVICE env"
        saved = settings.load().get("device")
        if saved:
            dev = devices.match_saved(saved, devs)
            if dev:
                return dev, "remembered from an earlier select_device"
            log.warning("remembered device %s not present, falling back to auto", saved.get("name"))
        best = devices.pick("", devs, self.api)
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
            if self._cap is not None and self._thread and self._thread.is_alive():
                return
            try:
                dev, source = self._resolve()
                cap = self._open_device(dev)
            except (CaptureError, LookupError) as e:
                self.status, self.last_error = "error", str(e)
                raise CaptureError(str(e)) from None
            self._start(dev, cap, source)

    def _start(self, dev: devices.DeviceInfo, cap: cv2.VideoCapture, source: str) -> None:
        self.device, self._cap, self.source = dev, cap, source
        self.status, self.last_error = "open", ""
        self._last_use = time.monotonic()  # opening can take seconds; that is not idle time
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture-reader", daemon=True)
        self._thread.start()
        log.info("opened %s (%s, %s) %s", dev.index, dev.name, source, self.actual)

    def close(self) -> None:
        with self._open_lock:
            self._stop.set()
            self._wake.set()
            t = self._thread
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=3)
            self._thread = None
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.status = "closed"
            with self._cond:
                self._frame = None
                self._cond.notify_all()

    def select(self, dev: devices.DeviceInfo) -> None:
        """Switch to dev. If it cannot be opened, go back to the previous device and raise."""
        prev, prev_sel, prev_source = self.device, self.selected, self.source
        self.close()
        try:
            with self._open_lock:
                cap = self._open_device(dev)
                self.selected = dev
                self._start(dev, cap, "selected by agent")
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
        fails = 0
        backoff = 0.5
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                # reconnect after unplug / driver hiccup
                self.status = "reconnecting"
                try:
                    assert self.device is not None
                    # the card may come back under a different index after replugging
                    present = devices.list_devices(self.api, probe_fallback=False)
                    dev = devices.match_saved(self.device.to_saved(), present) if self.device.named else None
                    self.device = dev or self.device
                    self._cap = self._open_device(self.device)
                    self.status, self.last_error, fails, backoff = "open", "", 0, 0.5
                    log.info("reconnected")
                except CaptureError as e:
                    self.last_error = str(e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 5)
                continue

            if self._release_if_idle(cap):
                return

            if time.monotonic() >= self._active_until:
                # Idle: no decoding. Wake up on a request, or time out and do one health-check read.
                if self._wake.wait(IDLE_CHECK_S):
                    self._wake.clear()
                    continue

            t_start = time.monotonic()
            t_start_wall = time.time()
            ok, img = cap.read()
            if not ok or img is None:
                fails += 1
                if fails >= FAILS_BEFORE_RECONNECT:
                    log.warning("capture lost, reconnecting")
                    self.last_error = "capture device stopped delivering frames (unplugged?)"
                    cap.release()
                    self._cap = None
                    fails = 0
                else:
                    time.sleep(0.03)
                continue
            fails = 0
            spent = time.monotonic() - t_start
            fps = self.actual.get("fps") or self.cfg.fps or 30
            with self._cond:
                self._frame, self._frame_t, self._frame_start = img, time.time(), t_start_wall
                # A read that returns almost instantly handed back a frame that was already buffered
                # (possibly from long ago); one that blocked waited for the camera's next frame.
                self._frame_waited = spent >= 0.3 / fps
                self._reads += 1
                self._cond.notify_all()
            # never spin faster than the frame rate on backends that do not block
            min_interval = 0.5 / fps
            if spent < min_interval:
                time.sleep(min_interval - spent)

    def _release_if_idle(self, cap: cv2.VideoCapture) -> bool:
        """An open device keeps the driver decoding (DirectShow: ~50% of a core at 1080p), so give it
        back after idle_release_s. Decided under the same lock a request takes, so a request can never
        slip in between the decision and the release unnoticed."""
        if self.cfg.idle_release_s <= 0:
            return False
        with self._cond:
            now = time.monotonic()
            if now < self._active_until or now - self._last_use <= self.cfg.idle_release_s:
                return False
            log.info("idle for %ss, releasing device", self.cfg.idle_release_s)
            cap.release()
            self._cap = None
            self.status = "closed"
            self._cond.notify_all()
            return True

    # ------------------------------------------------------------ public API

    def get_frame(self, timeout: float = 4.0) -> Frame:
        """Return a frame read after this call started (at most one frame interval old)."""
        for _ in range(2):
            if self._cap is None or not (self._thread and self._thread.is_alive()):
                if self.status == "reconnecting" and self._thread and self._thread.is_alive():
                    raise CaptureError(f"Capture device disconnected, reconnecting... ({self.last_error})")
                self.open()
            try:
                frame = self._fetch(timeout)
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

    def _fetch(self, timeout: float) -> Frame:
        with self._cond:
            if self._cap is None and self.status == "closed":
                raise _Released
            t_request = time.time()
            reads_before = self._reads
            self._last_use = time.monotonic()
            self._active_until = self._last_use + ACTIVE_S
            self._wake.set()

            def released() -> bool:
                return self._cap is None and self.status == "closed"

            def fresh() -> bool:
                if self._stop.is_set() or released():
                    return True
                if self._frame_start < t_request:
                    return False
                return self._frame_waited or self._reads - reads_before >= FRESH_FALLBACK_READS

            ok = self._cond.wait_for(fresh, timeout)
            if released() and not self._stop.is_set():
                raise _Released
            img, t = self._frame, self._frame_t
        if not ok or img is None:
            msg = self.last_error or "no frame received in time"
            raise CaptureError(f"Could not get a frame from the capture card: {msg}")
        return self._postprocess(img, t)

    def _postprocess(self, img: np.ndarray, t: float) -> Frame:
        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        state, desc = signal_state(img)
        box = None
        if self.cfg.autocrop and state == "OK":
            box = letterbox_box(img)
            if box:
                x1, y1, x2, y2 = box
                img = img[y1:y2, x1:x2]
        return Frame(img, (w, h), box, t, state, desc)

    def describe(self) -> dict:
        d = self.device
        return {
            "status": self.status,
            "device": d.label if d else None,
            "chosen_by": self.source or None,
            **self.actual,
            "last_error": self.last_error or None,
        }
