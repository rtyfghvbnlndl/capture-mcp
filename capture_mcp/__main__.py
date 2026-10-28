"""Entry point.

  python -m capture_mcp            run the MCP server on stdio
  python -m capture_mcp --list     list video devices
  python -m capture_mcp --probe    grab one frame, save probe.jpg (+ grid version), print diagnostics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Must be set before cv2 is imported: MSMF hardware transforms make some capture cards take
# 10-30 s to open; quieter OpenCV logs keep stderr readable.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def main() -> None:
    ap = argparse.ArgumentParser(prog="capture_mcp")
    ap.add_argument("--list", action="store_true", help="list video devices and exit")
    ap.add_argument("--probe", action="store_true", help="grab one frame to a file and print diagnostics")
    ap.add_argument("--out", default="probe.jpg", help="output file for --probe")
    ap.add_argument("--ocr", action="store_true", help="with --probe: also print the frame as read_screen text")
    ap.add_argument("--ocr-test", action="store_true", help="check the OCR engine on a generated image (no device)")
    args = ap.parse_args()

    # stdout belongs to the MCP protocol; all logs go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.list or args.probe or args.ocr_test:
        try:  # diagnostics print Chinese text and box characters; never crash on a legacy console codepage
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    if args.ocr_test:
        _ocr_test()
        return
    if args.list or args.probe:
        _diagnose(args)
        return

    from .server import mcp

    mcp.run()


def _diagnose(args) -> None:
    import time

    import cv2

    from . import devices, imaging
    from .camera import CaptureError, Grabber
    from .config import Config

    from . import settings

    cfg = Config.from_env()
    g = Grabber(cfg)
    devs = devices.list_devices(g.api)
    print(f"backend: {devices.backend_name(g.api)}")
    print("devices:" if devs else "devices: none found")
    for d in devs:
        print(f"  {d.label}  [{d.kind()}]")
    saved = settings.load().get("device")
    if saved:
        print(f'remembered choice: "{saved.get("name")}" (index {saved.get("index")}) in {settings.settings_path()}')
    try:
        chosen, why = g._resolve()
        print(f"would use: {chosen.label} ({why})")
    except LookupError as e:
        print(f"selection failed: {e}")
        sys.exit(1)
    if not args.probe:
        return

    try:
        t0 = time.monotonic()
        f = g.get_frame(timeout=8)
        print(f"opened in {time.monotonic() - t0:.2f}s: {g.describe()}")
        # measure real delivered fps over ~1s
        t1, n = time.monotonic(), 0
        while time.monotonic() - t1 < 1.0:
            g.get_frame()
            n += 1
        print(f"decoded frames/s on demand: {n / (time.monotonic() - t1):.1f}")
    except CaptureError as e:
        print(f"capture failed: {e}")
        sys.exit(1)
    finally:
        g.close()
    print(f"raw {f.raw_size[0]}x{f.raw_size[1]}, crop_box={f.crop_box}, signal: {f.signal_desc}")
    cv2.imwrite(args.out, f.image)
    root, ext = os.path.splitext(args.out)
    grid_path = f"{root}_grid{ext or '.jpg'}"
    cv2.imwrite(grid_path, imaging.draw_grid(imaging.fit_long_side(f.image, cfg.max_side), cfg.grid_cols, cfg.grid_rows))
    print(f"saved {args.out} and {grid_path}")
    if args.ocr and f.signal == "OK":
        from . import ocr

        reader = ocr.ScreenReader(cfg)
        try:
            t0 = time.monotonic()
            reader.engine()
            t1 = time.monotonic()
            rd = reader.read(f.image)
        except ocr.OcrUnavailable as e:
            print(f"OCR unavailable: {e}")
            return
        print(f"\nread_screen text ({rd.engine}: engine load {t1 - t0:.2f}s, OCR {rd.seconds:.2f}s, "
              f"{len(rd.words)} words):")
        print("\n".join(rd.lines) or "(no text found)")
        if rd.graphics:
            print("Non-text areas:", ", ".join(rd.graphics))


def _ocr_test() -> None:
    """Render a screen-like image with known text, read it back, report engine, speed and accuracy."""
    import time

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from . import ocr
    from .config import Config

    cfg = Config.from_env()
    lines = ["OCR self-test 文字识别测试", "Invoice total: 1,234.56 USD", "会议时间 2026-10-28 14:30", "保存   取消"]
    font = None
    for name in ("msyh.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc", "wqy-microhei.ttc", "PingFang.ttc"):
        for d in (os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"), "/usr/share/fonts", "/System/Library/Fonts"):
            for root, _, files in os.walk(d) if os.path.isdir(d) else []:
                if name in files:
                    font = ImageFont.truetype(os.path.join(root, name), 20)
                    break
            if font:
                break
        if font:
            break
    if font is None:
        print("note: no CJK font found, Chinese lines will not render properly")
        font = ImageFont.load_default(size=20)
    img = Image.new("RGB", (1920, 1080), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((200, 200 + i * 40), t, font=font, fill=(20, 20, 20))
    frame = np.asarray(img)[:, :, ::-1].copy()
    reader = ocr.ScreenReader(cfg)
    t0 = time.monotonic()
    try:
        eng = reader.engine()
    except ocr.OcrUnavailable as e:
        print(f"OCR unavailable: {e}")
        sys.exit(1)
    t1 = time.monotonic()
    rd = reader.read(frame)
    t2 = time.monotonic()
    reader.read(frame)
    t3 = time.monotonic()
    got = "".join("".join(rd.lines).split())
    ok = sum(1 for t in lines if "".join(t.split()) in got)
    print(f"engine: {eng.name}  (load {t1 - t0:.2f}s, full-screen OCR {t2 - t1:.2f}s, unchanged re-read {(t3 - t2) * 1000:.0f} ms)")
    print("\n".join(rd.lines))
    print(f"lines read exactly: {ok}/{len(lines)}")


if __name__ == "__main__":
    main()
