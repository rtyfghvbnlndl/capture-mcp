"""Device identity and selection: which of the video devices is the capture card (not the laptop webcam)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Names typical of HDMI capture cards (MS2109/MS2130 clones usually show up as "USB Video").
PREFER = re.compile(r"capture|hdmi|usb video|usb3 video|cam link|ezcap|ugreen|elgato|avermedia|magewell|video grabber|uvc", re.I)
AVOID = re.compile(r"integrated|front|rear|\bir\b|infrared|facetime|virtual|obs|droidcam|webcam", re.I)


@dataclass
class DeviceInfo:
    index: int          # position in the device list (what the agent passes to select_device)
    name: str
    backend: str = ""   # "dshow" | "v4l2" | "avfoundation"
    named: bool = True  # False when the OS gave no usable name
    path: str = ""      # dshow PnP alternative name / /dev/videoN / avfoundation index: unique and stable
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


def pick(spec: str, devices: list[DeviceInfo]) -> DeviceInfo:
    """Resolve CAPTURE_DEVICE / select_device input (index / name substring / empty=auto) to a device."""
    spec = (spec or "").strip()
    if spec.lstrip("-").isdigit():
        idx = int(spec)
        for d in devices:
            if d.index == idx:
                return d
        names = ", ".join(d.label for d in devices) or "none found"
        raise LookupError(f"No video device with index {idx}. Available: {names}")
    if spec:
        for d in devices:
            if spec.lower() in d.name.lower():
                return d
        names = ", ".join(d.label for d in devices) or "none found"
        raise LookupError(f'No video device name contains "{spec}". Available: {names}')
    if not devices:
        raise LookupError("No video devices found. Is the capture card plugged in?")
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
