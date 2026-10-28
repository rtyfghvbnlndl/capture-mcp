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
import sys
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

from . import devices, imaging, layout, ocr, settings
from .camera import CaptureError, Frame, Grabber, probe
from .config import Config

log = logging.getLogger(__name__)

cfg = Config.from_env()
grabber = Grabber(cfg)
reader = ocr.ScreenReader(cfg)

INSTRUCTIONS = """\
Tools to look at another computer's screen through an HDMI capture card (view only, no mouse/keyboard).
How to use them well:
1. Call screenshot first. The image has a grid: columns A,B,C... and rows 1,2,3...; each cell is labelled in its top-left corner.
2. If text is small or you are not sure what it says, call zoom with the cell(s), e.g. zoom(region="C3") or zoom(region="B2:D3"). Do not guess unreadable text.
3. To read a lot of text quickly, call read_screen (if available): it returns the screen as text with its layout, no image.
4. To wait for something to load or change, call wait_for_change instead of taking screenshots repeatedly.
5. If a tool says the screen is blank / no signal, tell the user; do not describe an imaginary screen.
6. If a tool reports a device error, or the picture is a camera view instead of a computer screen, call list_devices
   and then select_device with the capture card's index. The choice is remembered for next time.
"""

TEXT_INSTRUCTIONS = """\
Tools to read another computer's screen as text (OCR through an HDMI capture card; view only, no mouse/keyboard).
1. Call read_screen. It returns the text on the screen with its layout: separate areas ("── pane B1:H5 ──"), tables
   (cells joined by " | "), selected rows (▶ ...[selected]), buttons ([button]), colored or gray text ([red] [dim] ...).
2. Positions use a grid over the screen: columns A-H left to right, rows 1-5 top to bottom ("B2:D4" = an area).
   read_screen(region="B2:D4") reads only that area (faster, more precise).
3. To follow something that is changing, call wait_for_change, or read_screen(changes_only=true) to get only
   the lines that appeared (+) or disappeared (-) since your last read.
4. OCR can misread characters, and it cannot see pictures: "Non-text areas" lists where images or graphics are.
   Never invent text that is not in the result. If a tool says the screen is blank / no signal, tell the user.
5. On a device error call list_devices, then select_device with the capture card's index.
"""

mcp = MCPServer("capture-card", instructions=TEXT_INSTRUCTIONS if cfg.text_only else INSTRUCTIONS)

Content = list[ImageContent | TextContent]


def tool(description: str):
    # unstructured: the result is the image + text blocks themselves, no JSON output schema
    return mcp.tool(description=description, structured_output=False)


def image_tool(description: str):
    """Like tool(), but not registered in text-only mode (CAPTURE_TEXT_ONLY=1): a model without vision
    should not be able to ask for images at all."""
    if cfg.text_only:
        return lambda fn: fn
    return tool(description)

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

@image_tool(description="Take a screenshot of the other computer's screen (截图). Returns one image with a labelled grid (A1, B2, ...).")
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


@image_tool(description='Zoom into part of the screen to read small text (放大). region: grid cell "C3", range "B2:D4", or "x1,y1,x2,y2" (0-1000).')
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


@tool(description=(
    "Wait until the screen (or a region) changes, e.g. a page finishes loading (等待画面变化). "
    + ("Returns the text lines that changed." if cfg.text_only else "Use this instead of repeated screenshots.")
))
async def wait_for_change(
    timeout_s: Annotated[float, Field(description="Max seconds to wait, 1-60.")] = 10,
    region: RegionArg = "",
    threshold: Annotated[float, Field(description="Fraction of pixels that must change; 0 = automatic (2%, or a text change in text-only mode).")] = 0,
    return_image: Annotated[bool, Field(description="Also return a screenshot when it changes (ignored in text-only mode).")] = False,
) -> Content:
    try:
        r = imaging.parse_region(region, cfg.grid_cols, cfg.grid_rows)
    except imaging.RegionError as e:
        return _err(e)
    timeout_s = max(1.0, min(float(timeout_s or 10), 60.0))
    # Image mode waits for this fraction of changed pixels. Text-only mode instead reacts to any visible
    # change and lets OCR decide whether text changed (a new line can be ~1% of the screen, one digit far less).
    threshold = float(threshold) if threshold else 0.02
    if threshold > 1:  # model sent a percentage
        threshold /= 100
    threshold = max(0.0005, min(threshold, 0.9))
    where = "screen" if r.is_full else f"region {imaging.region_to_cells(r, cfg.grid_cols, cfg.grid_rows)}"
    base_lines = None
    try:
        base = await _frame()
        if cfg.text_only and base.signal == "OK":
            try:
                base_lines = (await anyio.to_thread.run_sync(reader.read, base.image, r)).lines
                reader.remember(r.norm1000(), base_lines)  # final answer = changes relative to this
            except ocr.OcrUnavailable:
                base_lines = None
        start = time.monotonic()
        while True:
            await anyio.sleep(0.3)
            cur = await _frame()
            elapsed = time.monotonic() - start
            if cur.signal != base.signal:
                msg = (f"after {elapsed:.1f}s the screen became blank ({cur.signal_desc})" if cur.signal != "OK"
                       else f"after {elapsed:.1f}s the screen came back from blank")
                break
            if cur.signal == "OK" and base_lines is not None:
                # text mode: any visible change at half resolution (one changed digit counts) -> let OCR
                # decide whether text changed; a blinking cursor or an animation does not count
                boxes = imaging.changed_boxes(imaging.diff_prep(imaging.crop(base.image, r), reader.PREP_SIZE),
                                              imaging.diff_prep(imaging.crop(cur.image, r), reader.PREP_SIZE))
                if boxes != []:
                    lines = (await anyio.to_thread.run_sync(reader.read, cur.image, r)).lines
                    if layout.diff_lines(base_lines, lines):
                        msg = f"CHANGED: text in {where} changed after {elapsed:.1f}s"
                        break
                    base = cur
            elif cur.signal == "OK":
                ratio, bbox = imaging.diff_ratio(base.image, cur.image, r)
                if ratio >= threshold:
                    area = imaging.region_to_cells(bbox, cfg.grid_cols, cfg.grid_rows) if bbox else "?"
                    msg = f"CHANGED: {where} changed after {elapsed:.1f}s ({ratio:.1%} of pixels), changed area ≈ {area}"
                    break
            if elapsed >= timeout_s:
                return [_text(f"NO CHANGE: {'text in ' if base_lines is not None else ''}{where} "
                              f"did not change within {timeout_s:.0f}s.")]
    except CaptureError as e:
        return _err(e)
    if cfg.text_only:
        if cur.signal != "OK":
            return [_text(msg)]
        try:
            text = await _read_text(cur, r, changes_only=True)
        except ocr.OcrUnavailable as e:
            return [_text(f"{msg}. (OCR unavailable: {e})")]
        return [_text(f"{msg}.\n{text}")]
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
    if preview and tiles and not cfg.text_only:
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


# ---------------------------------------------------------------- text reading (OCR)

LEGEND = [  # (marker that triggers it, explanation)
    ("── pane", "── pane X ── = a separate column/window area"),
    ("── table", "── table ── rows with cells joined by ' | '"),
    ("[selected]", "▶ ...[selected] = selected row"),
    ("[button]", "[button] = text on a button/box"),
    (("[red]", "[green]", "[blue]"), "[red]/[green]/[blue] = text color"),
    ("[dim]", "[dim] = gray text (disabled/placeholder)"),
    ("(?)", "(?) = uncertain OCR"),
]


def _legend(lines: list[str]) -> str:
    text = "\n".join(lines)
    parts = [desc for marker, desc in LEGEND
             if any(m in text for m in (marker if isinstance(marker, tuple) else (marker,)))]
    return ("Legend: " + "; ".join(parts) + ".") if parts else ""


async def _read_text(f: Frame, r: imaging.Region, changes_only: bool) -> str:
    """OCR one frame and format it for the model. Raises ocr.OcrUnavailable."""
    await anyio.to_thread.run_sync(reader.engine)
    reading = await anyio.to_thread.run_sync(reader.read, f.image, r)
    where = "full screen" if r.is_full else imaging.region_to_cells(r, cfg.grid_cols, cfg.grid_rows)
    previous = reader.remember(r.norm1000(), reading.lines)
    h, w = f.image.shape[:2]
    mode = {"unchanged": "screen unchanged since last read, cached result",
            "incremental": "only the changed area was re-read"}.get(reading.mode, "")
    head = (f"SCREEN TEXT (OCR, no image) of {where} | screen {w}x{h} | captured {_when(f)} | "
            f"OCR {reading.seconds:.2f}s ({reading.engine}{', ' + mode if mode else ''})")
    body = reading.lines
    note = ""
    if changes_only:
        if previous is None:
            note = "No earlier read_screen of this area: full text below."
        else:
            body = layout.diff_lines(previous, reading.lines)
            note = ("Changes since your last read_screen of this area (+ appeared, - disappeared):"
                    if body else "No text changed since your last read_screen of this area.")
    out = [head]
    if note:
        out.append(note)
    if not reading.lines:
        out.append(f"No text found in {where}.")
    else:
        legend = _legend(body)
        if legend:
            out.append(legend)
        kept, cut = layout.truncate(body, cfg.text_max_chars)
        out.append("")
        out.extend(kept)
        if cut:
            out.append(f'... (truncated; read a smaller area, e.g. read_screen(region="B2:D4"))')
    if reading.graphics:
        out.append("")
        out.append("Non-text areas (images/graphics, OCR cannot read them): " + ", ".join(reading.graphics))
    return "\n".join(out)


def _ocr_installed() -> bool:
    import importlib.util

    if cfg.ocr_engine == "none":
        return False
    names = ["winrt.windows.media.ocr", "rapidocr"] if sys.platform == "win32" else ["rapidocr"]
    for n in names:
        try:
            if importlib.util.find_spec(n) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


if cfg.text_only or _ocr_installed():

    @tool(description=(
        "Read the screen as text with its layout (OCR, no image; 读屏/识别文字). Faster than looking at a "
        'screenshot. region optional, e.g. "B2:D4". changes_only=true returns only lines that changed since the last call.'
    ))
    async def read_screen(
        region: RegionArg = "",
        changes_only: Annotated[bool, Field(description="Only lines that appeared/disappeared since the last read_screen of this area.")] = False,
    ) -> Content:
        try:
            r = imaging.parse_region(region, cfg.grid_cols, cfg.grid_rows)
        except imaging.RegionError as e:
            return _err(e)
        try:
            await anyio.to_thread.run_sync(reader.engine)  # load the engine before grabbing the frame
            f = await _frame()
        except ocr.OcrUnavailable as e:
            return _err(f"OCR is not available: {e}")
        except CaptureError as e:
            return _err(e)
        if f.signal != "OK":
            return _blank(f)
        try:
            return [_text(await _read_text(f, r, changes_only))]
        except ocr.OcrUnavailable as e:
            return _err(f"OCR is not available: {e}")

    if cfg.ocr_engine != "none":
        reader.warm_up_in_background()  # model loading must not delay the first read_screen
