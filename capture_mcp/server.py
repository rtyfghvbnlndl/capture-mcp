"""MCP tools. Designed for small (quantized) vision models:

* few tools, flat optional parameters, lenient parsing, error messages that contain a correct example
* one image per call plus a short text header, so context does not explode
* a labelled grid instead of pixel coordinates
* blank / no-signal frames are reported as text, never sent as an image to be hallucinated about
"""

from __future__ import annotations

import base64
import logging
import os
import time
from datetime import datetime
from typing import Annotated, Any

import anyio
from mcp.types import ImageContent, TextContent
from pydantic import Field, WithJsonSchema

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from . import devices, imaging, settings
from .camera import CaptureError, Frame, Grabber, probe
from .config import Config

log = logging.getLogger(__name__)

cfg = Config.from_env()
grabber = Grabber(cfg)

INSTRUCTIONS = """\
Tools to look at another computer's screen through an HDMI capture card (view only, no mouse/keyboard).
How to use them well:
1. Call screenshot first. The image has a grid: columns A,B,C... and rows 1,2,3...; each cell is labelled in its top-left corner.
2. If text is small or you are not sure what it says, call zoom with the cell(s), e.g. zoom(region="C3") or zoom(region="B2:D3"). Do not guess unreadable text.
3. To wait for something to load or change, call wait_for_change instead of taking screenshots repeatedly.
4. If a tool says the screen is blank / no signal, tell the user; do not describe an imaginary screen.
5. If a tool reports a device error, or the picture is a camera view instead of a computer screen, call list_devices
   and then select_device with the capture card's index. The choice is remembered for next time.
"""

mcp = MCPServer("capture-card", instructions=INSTRUCTIONS)

Content = list[ImageContent | TextContent]


def tool(description: str):
    # unstructured: the result is the image + text blocks themselves, no JSON output schema
    return mcp.tool(description=description, structured_output=False)

# Advertise a plain string in the schema (simplest for small models and client grammars) but accept
# lists/dicts/numbers too, since quantized models often send [x1,y1,x2,y2] anyway.
RegionArg = Annotated[
    Any,
    WithJsonSchema(
        {
            "type": "string",
            "description": 'Grid cell "C3", cell range "B2:D4", or "x1,y1,x2,y2" in 0-1000. Empty = full screen.',
        }
    ),
]
DetailArg = Annotated[
    str,
    Field(description='Image size: "low" (768px), "normal" (default) or "high" (1920px). Higher = sharper but more tokens.'),
]


# ---------------------------------------------------------------- helpers

def _text(s: str) -> TextContent:
    return TextContent(type="text", text=s)


def _image(img, tag: str) -> tuple[ImageContent, str]:
    data = imaging.encode_jpeg(img, cfg.jpeg_quality)
    note = ""
    if cfg.save_dir:
        try:
            os.makedirs(cfg.save_dir, exist_ok=True)
            path = os.path.join(cfg.save_dir, f"{datetime.now():%Y%m%d-%H%M%S-%f}-{tag}.jpg")
            with open(path, "wb") as f:
                f.write(data)
            note = f"\nsaved: {path}"
        except OSError as e:
            note = f"\n(save failed: {e})"
    # validate via the camelCase key so this works with both mcp 1.x (mimeType) and 2.x (mime_type)
    img = ImageContent.model_validate({"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/jpeg"})
    return img, note


def _detail(detail: str) -> tuple[int, str]:
    d = (detail or "normal").strip().lower()
    if d not in ("low", "normal", "high"):
        return cfg.detail_size("normal"), f' (unknown detail "{detail}", used "normal")'
    return cfg.detail_size(d), ""


def _when(f: Frame) -> str:
    age = max(0, int((time.time() - f.t_wall) * 1000))
    return f"{datetime.fromtimestamp(f.t_wall):%H:%M:%S.%f}"[:-3] + f" (age {age} ms)"


def _size_note(f: Frame) -> str:
    h, w = f.image.shape[:2]
    s = f"screen {w}x{h}"
    if f.crop_box:
        s += f" (black borders cropped from {f.raw_size[0]}x{f.raw_size[1]})"
    return s


def _blank(f: Frame) -> Content:
    return [_text(f"NO IMAGE: {f.signal_desc}. Captured {_when(f)}. "
                  "Do not describe screen contents; tell the user the screen is blank or has no signal.")]


async def _frame() -> Frame:
    return await anyio.to_thread.run_sync(grabber.get_frame)


def _err(e: Exception) -> Content:
    return [_text(f"ERROR: {e}")]


# ---------------------------------------------------------------- tools

@tool(description="Take a screenshot of the other computer's screen (截图). Returns one image with a labelled grid (A1, B2, ...).")
async def screenshot(detail: DetailArg = "normal", grid: bool = True) -> Content:
    try:
        f = await _frame()
    except CaptureError as e:
        return _err(e)
    if f.signal != "OK":
        return _blank(f)
    size, dnote = _detail(detail)
    img = imaging.fit_long_side(f.image, size)
    if grid:
        img = imaging.draw_grid(img, cfg.grid_cols, cfg.grid_rows)
    content, save = _image(img, "screenshot")
    h, w = img.shape[:2]
    meta = f"{_size_note(f)}, image {w}x{h}{dnote}, captured {_when(f)}"
    if grid:
        meta += (f"\ngrid: {imaging.grid_spec_text(cfg.grid_cols, cfg.grid_rows)}; label is in each cell's top-left corner. "
                 'Use zoom(region="C3") or zoom(region="B2:D4") to read small text.')
    return [content, _text(meta + save)]


@tool(description='Zoom into part of the screen to read small text (放大). region: grid cell "C3", range "B2:D4", or "x1,y1,x2,y2" (0-1000).')
async def zoom(region: RegionArg = "", detail: DetailArg = "high") -> Content:
    try:
        r = imaging.parse_region(region, cfg.grid_cols, cfg.grid_rows)
    except imaging.RegionError as e:
        return _err(e)
    try:
        f = await _frame()
    except CaptureError as e:
        return _err(e)
    if f.signal != "OK":
        return _blank(f)
    size, dnote = _detail(detail)
    part = imaging.crop(f.image, r)
    # Upscaling small crops makes text span more vision-encoder patches; cap at 3x to avoid mush.
    img = imaging.fit_long_side(part, size, max_upscale=3.0)
    content, save = _image(img, "zoom")
    ph, pw = part.shape[:2]
    h, w = img.shape[:2]
    cells = imaging.region_to_cells(r, cfg.grid_cols, cfg.grid_rows)
    meta = (f"zoomed region {'full screen' if r.is_full else cells} = {r.norm1000()} (0-1000), "
            f"{pw}x{ph} screen px shown at {w}x{h}{dnote}, captured {_when(f)}. "
            "No grid on zoomed images; regions always refer to the full-screen grid.")
    return [content, _text(meta + save)]


@tool(description="Wait until the screen (or a region) changes, e.g. a page finishes loading (等待画面变化). Use this instead of repeated screenshots.")
async def wait_for_change(
    timeout_s: Annotated[float, Field(description="Max seconds to wait, 1-60.")] = 10,
    region: RegionArg = "",
    threshold: Annotated[float, Field(description="Fraction of pixels that must change, default 0.02 (2%).")] = 0.02,
    return_image: Annotated[bool, Field(description="Also return a screenshot when it changes.")] = False,
) -> Content:
    try:
        r = imaging.parse_region(region, cfg.grid_cols, cfg.grid_rows)
    except imaging.RegionError as e:
        return _err(e)
    timeout_s = max(1.0, min(float(timeout_s or 10), 60.0))
    threshold = float(threshold) if threshold else 0.02
    if threshold > 1:  # model sent a percentage
        threshold /= 100
    threshold = max(0.0005, min(threshold, 0.9))
    where = "screen" if r.is_full else f"region {imaging.region_to_cells(r, cfg.grid_cols, cfg.grid_rows)}"
    try:
        base = await _frame()
        start = time.monotonic()
        while True:
            await anyio.sleep(0.3)
            cur = await _frame()
            elapsed = time.monotonic() - start
            if cur.signal != base.signal:
                msg = (f"after {elapsed:.1f}s the screen became blank ({cur.signal_desc})" if cur.signal != "OK"
                       else f"after {elapsed:.1f}s the screen came back from blank")
                break
            if cur.signal == "OK":
                ratio, bbox = imaging.diff_ratio(base.image, cur.image, r)
                if ratio >= threshold:
                    area = imaging.region_to_cells(bbox, cfg.grid_cols, cfg.grid_rows) if bbox else "?"
                    msg = f"CHANGED: {where} changed after {elapsed:.1f}s ({ratio:.1%} of pixels), changed area ≈ {area}"
                    break
            if elapsed >= timeout_s:
                return [_text(f"NO CHANGE: {where} did not change within {timeout_s:.0f}s.")]
    except CaptureError as e:
        return _err(e)
    if not return_image:
        return [_text(msg + ". Call screenshot or zoom to see it.")]
    if cur.signal != "OK":
        return [_text(msg)]
    img = imaging.draw_grid(imaging.fit_long_side(cur.image, cfg.max_side), cfg.grid_cols, cfg.grid_rows)
    content, save = _image(img, "change")
    return [content, _text(f"{msg}. Captured {_when(cur)}; grid {imaging.grid_spec_text(cfg.grid_cols, cfg.grid_rows)}.{save}")]


def _same(a: devices.DeviceInfo | None, b: devices.DeviceInfo) -> bool:
    if a is None:
        return False
    if a.path and b.path:
        return a.path == b.path
    return a.index == b.index and a.name == b.name


def _fmt(actual: dict) -> str:
    if not actual:
        return ""
    return f"{actual.get('width')}x{actual.get('height')} {actual.get('fourcc')} {actual.get('fps')}fps"


@tool(description="List video devices and check which one shows the other computer (查找采集卡). preview=true adds one thumbnail image per device (this also opens webcams).")
async def list_devices(
    preview: Annotated[bool, Field(description="Return a thumbnail sheet of all devices, including webcams.")] = False,
) -> Content:
    def work():
        devs = devices.list_devices(grabber.api)
        running = grabber.status == "open" and grabber.device is not None
        lines, tiles = [], []
        for d in devs:
            current = running and _same(grabber.device, d)
            img = None
            if current:
                try:
                    f = grabber.get_frame()
                    img, state, desc, actual = f.image, f.signal, f.signal_desc, grabber.actual
                except CaptureError as e:
                    state, desc, actual = "ERROR", str(e), {}
            elif d.kind() == "webcam (likely)" and not preview:
                state, desc, actual = "SKIPPED", "not opened (looks like a webcam; preview=true checks it anyway)", {}
            else:
                pr = probe(d, cfg)
                img, state, desc, actual = pr.image, pr.signal, pr.desc, pr.actual
            if state == "OK":
                desc = "picture OK"
            elif state == "BLANK":
                desc = "black/flat picture: no HDMI signal or source asleep"
            mark = "  <== CURRENT" if current else ""
            lines.append(f"{d.label} | {d.kind()} | {desc} {_fmt(actual)}".rstrip() + mark)
            if preview:
                tiles.append((img, d.label))
        if not devs:
            lines.append("No video devices found. Is the capture card plugged in?")
        if running:
            cur = f"current: {grabber.device.label}, chosen by: {grabber.source}"
        else:
            cur = f"current: not open yet; next capture will use {grabber.planned()}"
        saved = settings.load().get("device")
        lines.append(cur)
        if saved:
            lines.append(f'remembered choice: "{saved.get("name")}" (index {saved.get("index")})')
        lines.append('To use a device: select_device(device="<index>"). The capture card is the one whose '
                     "picture is the other computer's screen (with preview=true you can see it).")
        return lines, tiles

    try:
        lines, tiles = await anyio.to_thread.run_sync(work)
    except Exception as e:  # enumeration/driver errors should still reach the model as text
        return _err(e)
    out: Content = []
    if preview and tiles:
        sheet = imaging.contact_sheet(tiles)
        content, save = _image(sheet, "devices")
        out.append(content)
        lines.append("Image: one tile per device, labelled with its index." + save)
    out.append(_text("\n".join(lines)))
    return out


@tool(description='Use a video device: index ("1") or part of its name ("USB Video"); "auto" = automatic. Remembered for next start. Call list_devices first.')
async def select_device(
    device: Annotated[Any, WithJsonSchema({"type": "string", "description": 'Index like "1", a name part, or "auto".'})],
    remember: Annotated[bool, Field(description="Save this choice for future starts.")] = True,
) -> Content:
    spec = str(device).strip().strip("\"'")
    if spec.lower() in ("auto", "automatic", "default", ""):
        def reset():
            grabber.close()
            grabber.selected = None
            if remember:
                data = settings.load()
                data.pop("device", None)
                settings.save(data)
            grabber.open()
        try:
            await anyio.to_thread.run_sync(reset)
        except CaptureError as e:
            return _err(e)
        return [_text(f"switched to automatic selection: {grabber.device.label} ({grabber.source})")]

    def switch():
        devs = devices.list_devices(grabber.api)
        dev = devices.pick(spec, devs, grabber.api)
        grabber.select(dev)
        return dev, grabber.get_frame()

    try:
        dev, f = await anyio.to_thread.run_sync(switch)
    except (CaptureError, LookupError) as e:
        back = f" Still using {grabber.device.label}." if grabber.status == "open" and grabber.device else ""
        return _err(f"{e}{back}")
    notes = []
    if f.signal != "OK":
        notes.append("WARNING: picture is blank. Either this is not the right device, or the other computer "
                     "is off/asleep/not sending HDMI.")
    if dev.kind() == "webcam (likely)":
        notes.append("WARNING: this looks like a webcam, not a capture card.")
    if remember:
        path = settings.save({**settings.load(), "device": dev.to_saved()})
        notes.append(f"Remembered in {path}.")
        if cfg.device:
            notes.append(f'Note: CAPTURE_DEVICE="{cfg.device}" is set in the client config and wins at the next start; '
                         "remove it to use the remembered choice.")
    return [_text(f"now using {dev.label} | {_fmt(grabber.actual)} | signal: {f.signal_desc}\n" + " ".join(notes)
                  + "\nCall screenshot to check it shows the other computer.")]


# ---------------------------------------------------------------- optional OCR

def _load_ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        return RapidOCR, "legacy"
    except ImportError:
        pass
    try:
        from rapidocr import RapidOCR  # type: ignore

        return RapidOCR, "v2"
    except ImportError:
        return None, None


_OCR_CLS, _OCR_KIND = _load_ocr()
_ocr_engine = None


def _ocr_ready() -> None:
    # Loading the models takes seconds on first use; do it before grabbing the frame.
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = _OCR_CLS()


def _run_ocr(img) -> list[tuple[list, str, float]]:
    _ocr_ready()
    res = _ocr_engine(img)
    if _OCR_KIND == "legacy":
        result = res[0] or []
        return [(box, txt, float(score)) for box, txt, score in result]
    if res is None or getattr(res, "boxes", None) is None:
        return []
    return [(b.tolist() if hasattr(b, "tolist") else b, t, float(s)) for b, t, s in zip(res.boxes, res.txts, res.scores)]


if _OCR_CLS is not None:

    @tool(description='Read text on screen with OCR (识别文字). Returns text lines with their grid cell. region optional, e.g. "B2:D4".')
    async def read_text(region: RegionArg = "") -> Content:
        try:
            r = imaging.parse_region(region, cfg.grid_cols, cfg.grid_rows)
        except imaging.RegionError as e:
            return _err(e)
        await anyio.to_thread.run_sync(_ocr_ready)
        try:
            f = await _frame()
        except CaptureError as e:
            return _err(e)
        if f.signal != "OK":
            return _blank(f)
        part = imaging.crop(f.image, r)
        scale = 2.0 if max(part.shape[:2]) < 800 else 1.0  # small regions OCR better when enlarged
        img = imaging.fit_long_side(part, int(max(part.shape[:2]) * scale), max_upscale=2.0)
        t_ocr = time.monotonic()
        items = await anyio.to_thread.run_sync(_run_ocr, img)
        t_ocr = time.monotonic() - t_ocr
        ih, iw = img.shape[:2]
        rows = []
        for box, txt, score in items:
            if score < 0.5 or not str(txt).strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            # map back to full-screen fractions
            fx1 = r.x1 + min(xs) / iw * (r.x2 - r.x1)
            fx2 = r.x1 + max(xs) / iw * (r.x2 - r.x1)
            fy1 = r.y1 + min(ys) / ih * (r.y2 - r.y1)
            fy2 = r.y1 + max(ys) / ih * (r.y2 - r.y1)
            box_r = imaging.Region(fx1, fy1, max(fx2, fx1 + 1e-3), max(fy2, fy1 + 1e-3))
            center = imaging.Region((fx1 + fx2) / 2, (fy1 + fy2) / 2, (fx1 + fx2) / 2 + 1e-4, (fy1 + fy2) / 2 + 1e-4)
            cell = imaging.region_to_cells(center, cfg.grid_cols, cfg.grid_rows)
            rows.append((round((fy1 + fy2) / 2 * 60), fx1, f"[{cell}] {txt}  ({box_r.norm1000()})"))
        rows.sort(key=lambda t: (t[0], t[1]))
        where = "full screen" if r.is_full else imaging.region_to_cells(r, cfg.grid_cols, cfg.grid_rows)
        if not rows:
            return [_text(f"No text found in {where}.")]
        lines = imaging.iter_lines((x[2] for x in rows), 150)
        return [_text(f"OCR of {where}, screen captured {_when(f)}, OCR took {t_ocr:.1f}s. "
                      "OCR can misread characters; zoom to double-check important values. Format: [cell] text (x1,y1,x2,y2 in 0-1000)\n" + "\n".join(lines))]
