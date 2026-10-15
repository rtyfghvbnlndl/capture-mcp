"""Parsing FFmpeg's device/format listings and choosing the capture format."""

from capture_mcp import source
from capture_mcp.config import Config

DSHOW_DEVICES = '''"Integrated Camera" (video)
  Alternative name "@device_pnp_\\\\?\\usb#vid_04f2&pid_b806&mi_00#6&97d2ab&0&0000#{65e8773d}\\global"
"USB Video" (video)
  Alternative name "@device_pnp_\\\\?\\usb#vid_534d&pid_2109&mi_00#7&1a2b3c&0&0000#{65e8773d}\\global"
"USB Video" (video)
  Alternative name "@device_pnp_\\\\?\\usb#vid_534d&pid_2109&mi_00#7&ffff&0&0000#{65e8773d}\\global"
"Microphone (USB Audio)" (audio)
  Alternative name "@device_cm_{33D9A762}\\wave_{A537}"
'''

DSHOW_OPTIONS = '''DirectShow video device options (from video devices)
 Pin "Capture" (alternative pin name "0")
  vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30
  vcodec=mjpeg  min s=1280x720 fps=5 max s=1280x720 fps=60
  pixel_format=yuyv422  min s=1920x1080 fps=5 max s=1920x1080 fps=5
  pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
'''

V4L2_FORMATS = '''[video4linux2,v4l2 @ 0x55] Compressed:       mjpeg :          Motion-JPEG : 1920x1080 1280x720 640x480
[video4linux2,v4l2 @ 0x55] Raw       :     yuyv422 :           YUYV 4:2:2 : 1920x1080 640x480
'''


def test_parse_dshow_devices_keeps_video_only_with_unique_paths():
    devs = source.parse_dshow_devices(DSHOW_DEVICES)
    assert [d.name for d in devs] == ["Integrated Camera", "USB Video", "USB Video"]
    assert [d.index for d in devs] == [0, 1, 2]
    assert devs[1].path != devs[2].path and devs[1].path.startswith("@device_pnp_")
    assert (devs[1].vid, devs[1].pid) == (0x534D, 0x2109)
    assert source.dshow_url(devs[2]) == "video=" + devs[2].path


def test_parse_dshow_formats_and_prefer_mjpeg_at_target_size():
    fmts = source.parse_dshow_formats(DSHOW_OPTIONS)
    assert source.Format("mjpeg", 1920, 1080, 30.0) in fmts
    cfg = Config()
    best = source.choose_format(fmts, cfg)
    assert (best.kind, best.width, best.height) == ("mjpeg", 1920, 1080)


def test_choose_format_respects_forced_format_and_size_fallback():
    fmts = source.parse_dshow_formats(DSHOW_OPTIONS)
    cfg = Config()
    cfg.format = "yuyv422"
    best = source.choose_format(fmts, cfg)
    assert (best.kind, best.width) == ("yuyv422", 1920)
    cfg = Config()
    cfg.width, cfg.height = 1600, 900  # not offered: largest mjpeg size not above it
    assert (source.choose_format(fmts, cfg).width) == 1280


def test_choose_format_raw_only_device_prefers_nv12():
    fmts = [source.Format("yuyv422", 1920, 1080, 30), source.Format("nv12", 1920, 1080, 30)]
    assert source.choose_format(fmts, Config()).kind == "nv12"


def test_parse_v4l2_formats():
    fmts = source.parse_v4l2_formats(V4L2_FORMATS)
    assert source.Format("mjpeg", 1920, 1080) in fmts and source.Format("yuyv422", 640, 480) in fmts


def test_v4l2_sysfs_listing(tmp_path):
    for n, name, idx in ((0, "USB Video", "0"), (1, "USB Video", "1"), (2, "Integrated Camera", "0")):
        d = tmp_path / f"video{n}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(idx + "\n")
    devs = source._v4l2_devices(str(tmp_path))
    assert [(d.name, d.path) for d in devs] == [("USB Video", "/dev/video0"), ("Integrated Camera", "/dev/video2")]


def test_fourcc_env_still_works(monkeypatch):
    monkeypatch.delenv("CAPTURE_FORMAT", raising=False)
    monkeypatch.setenv("CAPTURE_FOURCC", "MJPG")
    assert Config.from_env().format == "mjpeg"
    monkeypatch.setenv("CAPTURE_FORMAT", "nv12")
    assert Config.from_env().format == "nv12"
