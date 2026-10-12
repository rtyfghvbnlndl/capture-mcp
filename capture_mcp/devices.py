"""Enumerate video devices and pick the capture card (not the laptop webcam)."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

import cv2

log = logging.getLogger(__name__)

# Names typical of HDMI capture cards (MS2109/MS2130 clones usually show up as "USB Video").
PREFER = re.compile(r"capture|hdmi|usb video|usb3 video|cam link|ezcap|ugreen|elgato|avermedia|magewell|video grabber|uvc", re.I)
AVOID = re.compile(r"integrated|front|rear|\bir\b|infrared|facetime|virtual|obs|droidcam|webcam", re.I)


@dataclass
class DeviceInfo:
    index: int          # value to pass to cv2.VideoCapture
    name: str
    api: int            # cv2.CAP_* backend
    named: bool = True  # False when names could not be enumerated
    path: str = ""      # OS device path: stable across reboots while the card stays on the same USB port
    vid: int = 0
    pid: int = 0

    @property
    def label(self) -> str:
        return f'{self.index}: "{self.name}"'

    def kind(self) -> str:
        """Guess from the name; lets the agent (and auto mode) tell capture cards from webcams."""
        if not self.named:
            return "unknown"
        if PREFER.search(self.name):
            return "capture card (likely)"
        if AVOID.search(self.name):
            return "webcam (likely)"
        return "unknown"

    def to_saved(self) -> dict:
        return {"name": self.name, "index": self.index, "path": self.path, "vid": self.vid, "pid": self.pid}


def backend_api(name: str) -> int:
    name = (name or "auto").lower()
    table = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "any": cv2.CAP_ANY,
    }
    if name in table:
        return table[name]
    if sys.platform == "win32":
        return cv2.CAP_DSHOW  # DSHOW honours FOURCC=MJPG reliably and opens fast
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def backend_name(api: int) -> str:
    try:
        return cv2.videoio_registry.getBackendName(api)
    except Exception:
        return str(api)


def list_devices(api: int, probe_fallback: bool = True) -> list[DeviceInfo]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        cams = enumerate_cameras(api)
        return [
            DeviceInfo(
                c.index, c.name or f"camera #{c.index}", c.backend or api,
                path=getattr(c, "path", "") or "", vid=getattr(c, "vid", 0) or 0, pid=getattr(c, "pid", 0) or 0,
            )
            for c in cams
        ]
    except ImportError:
        log.info("cv2-enumerate-cameras not installed; device names unavailable")
    except Exception as e:  # pragma: no cover - platform specific
        log.warning("device enumeration failed: %s", e)
    if not probe_fallback:
        return []
    # Fallback: probe indices. Opening each device briefly is slow but works everywhere.
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i, api)
        try:
            if cap.isOpened():
                found.append(DeviceInfo(i, f"camera #{i} (name unknown)", api, named=False))
        finally:
            cap.release()
    return found


def pick(spec: str, devices: list[DeviceInfo], api: int) -> DeviceInfo:
    """Resolve CAPTURE_DEVICE (index / name substring / empty=auto) to a device."""
    spec = (spec or "").strip()
    if spec.lstrip("-").isdigit():
        idx = int(spec)
        for d in devices:
            if d.index == idx:
                return d
        return DeviceInfo(idx, f"camera #{idx}", api, named=False)
    if spec:
        for d in devices:
            if spec.lower() in d.name.lower():
                return d
        names = ", ".join(d.label for d in devices) or "none found"
        raise LookupError(f'No video device name contains "{spec}". Available: {names}')
    if not devices:
        raise LookupError("No video devices found. Is the capture card plugged in?")
    if all(d.named for d in devices):
        scored = sorted(
            devices,
            key=lambda d: (bool(PREFER.search(d.name)), not AVOID.search(d.name), d.index),
            reverse=True,
        )
        best = scored[0]
        if AVOID.search(best.name) and not PREFER.search(best.name):
            raise LookupError(
                "No capture card recognized, only webcam-like devices: "
                + ", ".join(d.label for d in devices)
                + ". Plug in the capture card, or call list_devices and then select_device."
            )
        return best
    # No names: index 0 is usually the built-in webcam on laptops, so prefer the highest index.
    best = max(devices, key=lambda d: d.index)
    log.warning("device names unknown; guessing index %d. Set CAPTURE_DEVICE to be sure.", best.index)
    return best


def match_saved(saved: dict, devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Find a previously selected device again. Indices shift when USB devices come and go, and cheap
    cards all share the name "USB Video", so match on the most specific identity available."""
    if not saved:
        return None
    if saved.get("path"):
        for d in devices:
            if d.path and d.path == saved["path"]:
                return d
    if saved.get("vid") and saved.get("pid"):
        same = [d for d in devices if (d.vid, d.pid) == (saved["vid"], saved["pid"])]
        if len(same) == 1:
            return same[0]
    same = [d for d in devices if d.name == saved.get("name")]
    if len(same) == 1:
        return same[0]
    if same:  # several identical names: prefer the old index
        return next((d for d in same if d.index == saved.get("index")), same[0])
    return None
