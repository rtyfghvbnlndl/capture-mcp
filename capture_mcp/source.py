"""Video device access through FFmpeg (PyAV): enumerate devices, pick a capture format, open a stream.

The point of going through FFmpeg instead of OpenCV is format control. A capture card's native format is
usually MJPG; FFmpeg can hand us those compressed packets untouched, so keeping the card streaming costs
little more than a memory copy, and we decode only the frame a screenshot actually needs. OpenCV's
Windows backends always convert every frame to RGB (measured: 30-50% of a core at 1080p).

Format preference: mjpeg > nv12 > other raw formats, at the configured size if offered.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
from dataclasses import dataclass, field

import av
import av.logging

from .config import Config
from .devices import DeviceInfo

log = logging.getLogger(__name__)

# raw formats in order of preference (cheapest for the OS camera stack / smallest to copy first)
RAW_PREFERENCE = ["nv12", "yuyv422", "uyvy422", "yuv420p", "bgr24", "rgb24"]


def backend_name(cfg_backend: str = "auto") -> str:
    b = (cfg_backend or "auto").lower()
    if b in ("dshow", "v4l2", "avfoundation"):
        return b
    if sys.platform == "win32":
        return "dshow"
    if sys.platform == "darwin":
        return "avfoundation"
    return "v4l2"


def _ffmpeg_log(url: str, fmt: str, options: dict) -> str:
    """Run an FFmpeg 'list' command and return what it printed to its log."""
    av.logging.set_level(av.logging.INFO)
    try:
        with av.logging.Capture(True) as records:
            try:
                av.open(url, format=fmt, options=options).close()
            except Exception:  # list_* commands always "fail" after printing
                pass
    finally:
        av.logging.set_level(av.logging.ERROR)
    return "".join(r[2] for r in records)


# ---------------------------------------------------------------- device listing

_DSHOW_DEV = re.compile(r'^\s*"(?P<name>[^"]+)"\s*\((?P<kind>[^)]*)\)\s*$')
_DSHOW_ALT = re.compile(r'^\s*Alternative name\s+"(?P<alt>[^"]+)"')
_VIDPID = re.compile(r"vid_([0-9a-f]{4}).*?pid_([0-9a-f]{4})", re.I)


def parse_dshow_devices(text: str) -> list[DeviceInfo]:
    devs: list[DeviceInfo] = []
    current: DeviceInfo | None = None
    for line in text.splitlines():
        m = _DSHOW_DEV.match(line)
        if m:
            current = None
            if "video" in m.group("kind"):
                current = DeviceInfo(len(devs), m.group("name"), "dshow")
                devs.append(current)
            continue
        m = _DSHOW_ALT.match(line)
        if m and current is not None:
            current.path = m.group("alt")
            vp = _VIDPID.search(current.path)
            if vp:
                current.vid, current.pid = int(vp.group(1), 16), int(vp.group(2), 16)
    return devs


def _v4l2_devices(sys_root: str = "/sys/class/video4linux") -> list[DeviceInfo]:
    devs = []
    for d in sorted(glob.glob(os.path.join(sys_root, "video*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        try:
            # each UVC device exposes a metadata node too; only index 0 delivers video
            if open(os.path.join(d, "index")).read().strip() not in ("0", ""):
                continue
            name = open(os.path.join(d, "name")).read().strip()
        except OSError:
            continue
        info = DeviceInfo(len(devs), name, "v4l2", path="/dev/" + os.path.basename(d))
        try:
            dev_dir = os.path.realpath(os.path.join(d, "device"))
            for up in (dev_dir, os.path.dirname(dev_dir)):
                if os.path.exists(os.path.join(up, "idVendor")):
                    info.vid = int(open(os.path.join(up, "idVendor")).read(), 16)
                    info.pid = int(open(os.path.join(up, "idProduct")).read(), 16)
                    break
        except (OSError, ValueError):
            pass
        devs.append(info)
    return devs


_AVF_DEV = re.compile(r"^\s*\[(\d+)\]\s+(.+?)\s*$")


def parse_avfoundation_devices(text: str) -> list[DeviceInfo]:
    devs, in_video = [], False
    for line in text.splitlines():
        if "video devices" in line.lower():
            in_video = True
            continue
        if "audio devices" in line.lower():
            in_video = False
            continue
        m = _AVF_DEV.match(line.split("]", 1)[-1] if line.startswith("[AVFoundation") else line)
        if in_video and m:
            devs.append(DeviceInfo(len(devs), m.group(2), "avfoundation", path=m.group(1)))
    return devs


def list_devices(cfg_backend: str = "auto") -> list[DeviceInfo]:
    b = backend_name(cfg_backend)
    try:
        if b == "dshow":
            return parse_dshow_devices(_ffmpeg_log("dummy", "dshow", {"list_devices": "true"}))
        if b == "avfoundation":
            return parse_avfoundation_devices(_ffmpeg_log("", "avfoundation", {"list_devices": "true"}))
        return _v4l2_devices()
    except Exception as e:  # pragma: no cover - platform specific
        log.warning("device enumeration failed: %s", e)
        return []


# ---------------------------------------------------------------- formats

@dataclass(frozen=True)
class Format:
    kind: str           # "mjpeg" or a raw pixel format such as "nv12"
    width: int
    height: int
    max_fps: float = 0.0

    @property
    def compressed(self) -> bool:
        return self.kind == "mjpeg"


_DSHOW_FMT = re.compile(
    r"(?:vcodec=(?P<codec>\w+)|pixel_format=(?P<pix>\w+))\s+min s=(?P<w1>\d+)x(?P<h1>\d+)\s+fps=(?P<f1>[\d.]+)"
    r"\s+max s=(?P<w2>\d+)x(?P<h2>\d+)\s+fps=(?P<f2>[\d.]+)"
)


def parse_dshow_formats(text: str) -> list[Format]:
    out = set()
    for m in _DSHOW_FMT.finditer(text):
        kind = m.group("codec") or m.group("pix")
        out.add(Format(kind, int(m.group("w2")), int(m.group("h2")), float(m.group("f2"))))
        out.add(Format(kind, int(m.group("w1")), int(m.group("h1")), float(m.group("f1"))))
    return sorted(out, key=lambda f: (f.kind, f.width * f.height))


_V4L2_FMT = re.compile(r"(Compressed|Raw)\s*:\s*(\w+)\s*:[^:]*:\s*(.*)$")


def parse_v4l2_formats(text: str) -> list[Format]:
    out = []
    for line in text.splitlines():
        m = _V4L2_FMT.search(line)
        if not m:
            continue
        kind = m.group(2)
        for size in re.findall(r"(\d+)x(\d+)", m.group(3)):
            out.append(Format(kind, int(size[0]), int(size[1])))
    return out


def list_formats(dev: DeviceInfo) -> list[Format]:
    try:
        if dev.backend == "dshow":
            return parse_dshow_formats(_ffmpeg_log(dshow_url(dev), "dshow", {"list_options": "true"}))
        if dev.backend == "v4l2":
            return parse_v4l2_formats(_ffmpeg_log(dev.path, "v4l2", {"list_formats": "all"}))
    except Exception as e:  # pragma: no cover
        log.warning("format listing failed for %s: %s", dev.label, e)
    return []


def choose_format(formats: list[Format], cfg: Config) -> Format | None:
    """Best format: preferred kind first, then the configured size (or the largest not above it)."""
    if not formats:
        return None
    wanted = (cfg.format or "auto").lower()
    kinds = ["mjpeg"] + RAW_PREFERENCE if wanted in ("", "auto") else [wanted]
    target = cfg.width * cfg.height
    for kind in kinds:
        cands = [f for f in formats if f.kind == kind]
        if not cands:
            continue
        exact = [f for f in cands if (f.width, f.height) == (cfg.width, cfg.height)]
        if exact:
            return max(exact, key=lambda f: f.max_fps)
        below = [f for f in cands if f.width * f.height <= target]
        if below:  # largest size not above the target
            return max(below, key=lambda f: (f.width * f.height, f.max_fps))
        return min(cands, key=lambda f: (f.width * f.height, -f.max_fps))  # else the smallest above it
    return None


# ---------------------------------------------------------------- open

def dshow_url(dev: DeviceInfo) -> str:
    # The alternative (PnP) name is unique even when two cards are both called "USB Video".
    return f"video={dev.path or dev.name}"


@dataclass
class Opened:
    container: object
    stream: object
    codec: str          # "mjpeg" or "rawvideo"
    pix_fmt: str        # raw pixel format ("" for mjpeg)
    width: int
    height: int
    info: dict = field(default_factory=dict)


def open_stream(dev: DeviceInfo, cfg: Config) -> Opened:
    """Open dev with the best available format. Raises OSError/av.FFmpegError on failure."""
    fmt = choose_format(list_formats(dev), cfg)
    fps = min(cfg.fps, int(fmt.max_fps)) if fmt and fmt.max_fps else cfg.fps
    w, h = (fmt.width, fmt.height) if fmt else (cfg.width, cfg.height)
    opts = {"video_size": f"{w}x{h}", "framerate": str(fps)}
    if dev.backend == "dshow":
        url, f = dshow_url(dev), "dshow"
        opts["rtbufsize"] = "32M"
        if fmt and fmt.compressed:
            opts["vcodec"] = "mjpeg"
        elif fmt:
            opts["pixel_format"] = fmt.kind
    elif dev.backend == "v4l2":
        url, f = dev.path, "v4l2"
        if fmt:
            opts["input_format"] = fmt.kind
    else:
        url, f = f"{dev.path}:none", "avfoundation"
    container = av.open(url, format=f, options=opts, timeout=(8.0, 5.0))
    stream = container.streams.video[0]
    cc = stream.codec_context
    codec = cc.name
    pix = "" if codec == "mjpeg" else (cc.format.name if cc.format else "")
    info = {
        "width": cc.width,
        "height": cc.height,
        "fps": round(float(stream.average_rate or fps), 1),
        "format": "mjpeg" if codec == "mjpeg" else pix,
        "backend": f,
    }
    return Opened(container, stream, codec, pix, cc.width, cc.height, info)
