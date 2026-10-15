import numpy as np
import pytest

from capture_mcp import imaging
from capture_mcp.imaging import FULL, Region, RegionError, parse_region

COLS, ROWS = 8, 5


def approx(r: Region, *vals):
    assert (r.x1, r.y1, r.x2, r.y2) == pytest.approx(vals, abs=1e-6)


# ---------------------------------------------------------------- region parsing

@pytest.mark.parametrize("v", ["C3", "c3", " C 3 ", '"C3"', "[C3]"])
def test_cell_variants(v):
    approx(parse_region(v, COLS, ROWS), 2 / 8, 2 / 5, 3 / 8, 3 / 5)


@pytest.mark.parametrize("v", ["B2:D4", "b2-d4", "B2 to D4", "D4:B2", "B2, D4", "B2 D4"])
def test_range_variants(v):
    approx(parse_region(v, COLS, ROWS), 1 / 8, 1 / 5, 4 / 8, 4 / 5)


@pytest.mark.parametrize("v", ["100,200,500,600", "[100, 200, 500, 600]", "(100 200 500 600)",
                               [100, 200, 500, 600], {"x1": 100, "y1": 200, "x2": 500, "y2": 600},
                               "500,600,100,200", [0.1, 0.2, 0.5, 0.6]])
def test_numeric_variants(v):
    approx(parse_region(v, COLS, ROWS), 0.1, 0.2, 0.5, 0.6)


@pytest.mark.parametrize("v", ["", None, "full", "ALL", [], "全屏"])
def test_full(v):
    assert parse_region(v, COLS, ROWS) == FULL


@pytest.mark.parametrize("v,needle", [
    ("Z9", "outside the grid"),
    ("C9", "outside the grid"),
    ("1,2,3", "4 numbers"),
    ("100,200,1500,600", "not pixels"),
    ("100,200,100,600", "too small"),
    ("the button", "Cannot parse"),
])
def test_errors_include_example(v, needle):
    with pytest.raises(RegionError) as ei:
        parse_region(v, COLS, ROWS)
    msg = str(ei.value)
    assert needle in msg
    assert '"C3"' in msg  # every error teaches the right syntax


def test_region_to_cells_roundtrip():
    for s in ["A1", "H5", "C3", "B2:D4", "A1:H5"]:
        assert imaging.region_to_cells(parse_region(s, COLS, ROWS), COLS, ROWS) == s


def test_region_to_cells_partial():
    # 0.30..0.40 x spans columns C (0.25-0.375) and D (0.375-0.5)
    assert imaging.region_to_cells(Region(0.30, 0.05, 0.40, 0.10), COLS, ROWS) == "C1:D1"


# ---------------------------------------------------------------- geometry

def test_fit_long_side_and_upscale_cap():
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert imaging.fit_long_side(img, 1280).shape[:2] == (720, 1280)
    small = np.zeros((100, 200, 3), np.uint8)
    assert imaging.fit_long_side(small, 1920, max_upscale=3).shape[:2] == (300, 600)
    assert imaging.fit_long_side(small, 1920).shape[:2] == (100, 200)  # default: never upscale


def test_draw_grid_keeps_shape_and_marks():
    img = np.full((720, 1280, 3), 128, np.uint8)
    out = imaging.draw_grid(img, COLS, ROWS)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert np.array_equal(img, np.full((720, 1280, 3), 128, np.uint8))  # input untouched


def test_crop_pixels():
    img = np.arange(100 * 200).reshape(100, 200).astype(np.uint8)
    part = imaging.crop(img, Region(0.5, 0.5, 1.0, 1.0))
    assert part.shape == (50, 100)


# ---------------------------------------------------------------- analysis

def _desktop(h=1080, w=1920):
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (h, w, 3), dtype=np.uint8)


def test_letterbox_symmetric_pillarbox():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, 240:1680] = _desktop(1080, 1440)  # 4:3 source in 16:9 frame
    assert imaging.letterbox_box(frame) == (240, 0, 1680, 1080)


def test_letterbox_ignores_one_sided_dark_area():
    frame = _desktop()
    frame[:, :300] = 0  # dark sidebar on the left only
    assert imaging.letterbox_box(frame) is None


def test_letterbox_none_for_normal_frame():
    assert imaging.letterbox_box(_desktop()) is None


def test_signal_state():
    assert imaging.signal_state(_desktop())[0] == "OK"
    state, desc = imaging.signal_state(np.zeros((1080, 1920, 3), np.uint8))
    assert state == "BLANK" and "black" in desc
    blue = np.zeros((1080, 1920, 3), np.uint8)
    blue[..., 2] = 200  # RGB: capture-card "no signal" blue screen
    state, desc = imaging.signal_state(blue)
    assert state == "BLANK" and "RGB 0,0,200" in desc


def test_mostly_white_page_is_not_blank():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1920, 1080), (255, 255, 255))  # near-empty editor: one short line of text
    ImageDraw.Draw(img).text((40, 70), "hello world", fill=(0, 0, 0), font=ImageFont.load_default(size=16))
    assert imaging.signal_state(np.asarray(img))[0] == "OK"


def test_black_with_noise_is_blank():
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 6, (1080, 1920, 3), dtype=np.uint8)  # MJPG noise on a black frame
    assert imaging.signal_state(frame)[0] == "BLANK"


def test_diff_ratio_detects_and_localizes():
    a = np.full((1080, 1920, 3), 100, np.uint8)
    b = a.copy()
    b[540:1080, 960:1920] = 250  # bottom-right quarter changes
    ratio, bbox = imaging.diff_ratio(a, b)
    assert ratio == pytest.approx(0.25, abs=0.02)
    assert imaging.region_to_cells(bbox, COLS, ROWS) == "E3:H5"
    # restricted to the untouched top-left region: no change
    ratio, bbox = imaging.diff_ratio(a, b, parse_region("A1:B2", COLS, ROWS))
    assert ratio == 0 and bbox is None


def test_diff_ratio_sees_light_dialog_on_white():
    a = np.full((1080, 1920, 3), 245, np.uint8)
    b = a.copy()
    b[500:700, 800:1200] = 230
    ratio, _ = imaging.diff_ratio(a, b)
    assert ratio > 0.03


def test_diff_ratio_ignores_noise():
    a = _desktop()
    rng = np.random.default_rng(1)
    noise = rng.integers(-6, 7, a.shape)
    b = np.clip(a.astype(int) + noise, 0, 255).astype(np.uint8)
    ratio, _ = imaging.diff_ratio(a, b)
    assert ratio < 0.005


def test_diff_ratio_shape_change():
    a = np.zeros((1080, 1920, 3), np.uint8)
    b = np.zeros((1000, 1920, 3), np.uint8)
    assert imaging.diff_ratio(a, b)[0] == 1.0


# ---------------------------------------------------------------- device selection

from capture_mcp import devices  # noqa: E402


def _dev(i, name, path="", vid=0, pid=0):
    return devices.DeviceInfo(i, name, "dshow", path=path, vid=vid, pid=pid)


def test_kind_guess():
    assert _dev(0, "Integrated Camera").kind() == "webcam (likely)"
    assert _dev(1, "USB Video").kind() == "capture card (likely)"
    assert _dev(2, "Something").kind() == "unknown"


def test_auto_pick_prefers_capture_card_and_refuses_webcam_only():
    devs = [_dev(0, "Integrated Camera"), _dev(1, "USB Video")]
    assert devices.pick("", devs).index == 1
    with pytest.raises(LookupError, match="list_devices"):
        devices.pick("", [_dev(0, "Integrated Camera")])


def test_match_saved_survives_index_shift_and_duplicate_names():
    saved = _dev(1, "USB Video", path="p-card", vid=0x534D, pid=0x2109).to_saved()
    # after replugging, the same card is now index 2 and a second identical card took index 1
    now = [_dev(0, "Integrated Camera"), _dev(1, "USB Video", path="p-other", vid=0x534D, pid=0x2109),
           _dev(2, "USB Video", path="p-card", vid=0x534D, pid=0x2109)]
    assert devices.match_saved(saved, now).index == 2
    # without paths, duplicate names fall back to the old index
    saved2 = _dev(1, "USB Video").to_saved()
    assert devices.match_saved(saved2, [_dev(0, "USB Video"), _dev(1, "USB Video")]).index == 1
    assert devices.match_saved(saved2, [_dev(0, "Integrated Camera")]) is None


def test_settings_roundtrip(tmp_path, monkeypatch):
    from capture_mcp import settings

    monkeypatch.setenv("CAPTURE_SETTINGS_FILE", str(tmp_path / "s.json"))
    assert settings.load() == {}
    settings.save({"device": {"name": "USB Video", "index": 1}})
    assert settings.load()["device"]["name"] == "USB Video"


def test_contact_sheet():
    sheet = imaging.contact_sheet([(np.zeros((1080, 1920, 3), np.uint8), '0: "A"'), (None, '1: "B"'),
                                   (np.zeros((480, 640, 3), np.uint8), '2: "C"')])
    assert sheet.shape == (2 * (270 + 34), 960, 3)
