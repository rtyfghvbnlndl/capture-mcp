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
    args = ap.parse_args()

    # stdout belongs to the MCP protocol; all logs go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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


if __name__ == "__main__":
    main()
