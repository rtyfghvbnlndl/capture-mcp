"""Entry point.

  python -m capture_mcp            run the MCP server on stdio
  python -m capture_mcp --list     list video devices and their formats
  python -m capture_mcp --probe    open the device, save probe.jpg (+ grid version), print diagnostics
                                   including the CPU cost of keeping the card streaming
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


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

    from PIL import Image

    from . import imaging, settings, source
    from .camera import CaptureError, Grabber
    from .config import Config

    cfg = Config.from_env()
    g = Grabber(cfg)
    devs = g.list_devices()
    print(f"backend: {g.backend}")
    print("devices:" if devs else "devices: none found")
    for d in devs:
        print(f"  {d.label}  [{d.kind()}]")
        fmts = source.list_formats(d)
        kinds = sorted({f.kind for f in fmts})
        if fmts:
            best = source.choose_format(fmts, cfg)
            print(f"      formats: {', '.join(kinds)}; would use {best.kind} {best.width}x{best.height}"
                  + ("" if best.compressed else "  (no MJPG: raw frames are copied, costs more CPU)"))
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
        t1, n = time.monotonic(), 0
        while time.monotonic() - t1 < 1.0:
            g.get_frame()
            n += 1
        print(f"screenshots/s (decode on demand): {n / (time.monotonic() - t1):.1f}")
        c0, w0 = time.process_time(), time.monotonic()
        time.sleep(5)
        cpu = 100 * (time.process_time() - c0) / (time.monotonic() - w0)
        print(f"CPU while the card stays active and no one asks for frames: {cpu:.1f}% of one core")
    except CaptureError as e:
        print(f"capture failed: {e}")
        sys.exit(1)
    finally:
        g.close()
    print(f"raw {f.raw_size[0]}x{f.raw_size[1]}, crop_box={f.crop_box}, signal: {f.signal_desc}")
    Image.fromarray(f.image).save(args.out, quality=90)
    root, ext = os.path.splitext(args.out)
    grid_path = f"{root}_grid{ext or '.jpg'}"
    grid = imaging.draw_grid(imaging.fit_long_side(f.image, cfg.max_side), cfg.grid_cols, cfg.grid_rows)
    Image.fromarray(grid).save(grid_path, quality=90)
    print(f"saved {args.out} and {grid_path}")


if __name__ == "__main__":
    main()
