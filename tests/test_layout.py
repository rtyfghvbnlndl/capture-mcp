"""Layout reconstruction from OCR words (no OCR engine needed)."""

import numpy as np

from capture_mcp import layout
from capture_mcp.layout import Word

W, H = 1920, 1080


def line(text: str, x: float, y: float, h: float = 16, cw: float = 9, conf=None) -> list[Word]:
    """Words of one text line, laid out like an OCR engine would report them."""
    out, cx = [], x
    for tok in text.split(" "):
        w = len(tok) * cw
        out.append(Word(tok, cx, y, cx + w, y + h, conf))
        cx += w + cw  # one space
    return out


def cjk_chars(text: str, x: float, y: float, h: float = 16) -> list[Word]:
    """Windows OCR style: one 'word' per Chinese character."""
    return [Word(ch, x + i * h, y, x + (i + 1) * h - 1, y + h) for i, ch in enumerate(text)]


def render(words, **kw):
    return layout.render(layout.xy_cut(words), W, H, 8, 5, **kw)


def test_cjk_joined_without_spaces_latin_with_spaces():
    words = cjk_chars("发票总额", 100, 100) + line("1,234.56 USD", 100 + 4 * 16 + 9, 100)
    assert layout.join_words(words) == "发票总额 1,234.56 USD"


def test_rows_and_wide_gaps():
    words = line("Name", 100, 100) + line("Size", 600, 100) + line("report.pdf", 100, 130) + line("2 MB", 600, 130)
    out = render(words)
    # aligned columns with >=2 rows are a table
    assert out[0].startswith("── table")
    assert out[1:] == ["Name | Size", "report.pdf | 2 MB"]


def test_menu_bar_is_one_row_not_columns():
    words = line("File", 10, 5) + line("Edit", 80, 5) + line("View", 150, 5) + line("Untitled - Notepad", 800, 5)
    assert render(words) == ["File   Edit   View   Untitled - Notepad"]


def test_sidebar_and_content_are_separate_panes_in_reading_order():
    side = line("Inbox", 20, 100) + line("Sent", 20, 140) + line("Drafts", 20, 180) + line("Trash", 20, 220)
    body = []
    for i, t in enumerate(["Dear John,", "the invoice is attached.", "Total due: 1,234.56 USD",
                           "Payment within 30 days.", "Regards", "Mary"]):
        body += line(t, 400, 110 + i * 26)  # different line spacing than the sidebar
    out = render(side + body)
    headers = [l for l in out if l.startswith("──")]
    assert len(headers) == 2 and all("pane" in h for h in headers)
    assert out.index("Inbox") < out.index("Dear John,")  # left pane first
    assert "Total due: 1,234.56 USD" in out
    assert not any("Inbox" in l and "Dear" in l for l in out)  # panes never mixed on one line


def test_bands_separated_by_blank_line_and_paragraph_gap():
    words = line("Settings", 100, 50) + line("Wi-Fi", 100, 300) + line("Bluetooth", 100, 326) + line("Display", 100, 400)
    out = render(words)
    assert out[0] == "Settings" and out[1] == ""
    assert "Wi-Fi" in out and "Display" in out


def test_low_confidence_marked():
    out = render(line("Total", 100, 100, conf=0.95) + line("1,2B4", 100 + 5 * 9 + 9, 100, conf=0.3))
    assert out == ["Total 1,2B4(?)"]


def test_truncate():
    lines, cut = layout.truncate(["a" * 10] * 5, 25)
    assert cut and lines == ["a" * 10, "a" * 10]


def test_diff_lines():
    old = ["── pane A1 ──", "Status: running", "Progress 40%"]
    new = ["── pane A1 ──", "Status: running", "Progress 80%", "Done"]
    d = layout.diff_lines(old, new)
    assert d == ["── pane A1 ──", "+ Progress 80%", "+ Done", "- Progress 40%"]


def _canvas(bg=(255, 255, 255)):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = bg
    return img


def test_annotations_colors_dim_selected():
    img = _canvas()
    rows = []
    texts = ["Apple", "Banana", "Cherry", "Grape"]
    for i, t in enumerate(texts):
        ws = line(t, 100, 100 + i * 30)
        rows.append(ws)
        x1, y1, x2, y2 = ws[0].x1, ws[0].y1, ws[0].x2, ws[0].y2
        color = (30, 30, 30)
        if t == "Banana":
            img[y1 - 6:y2 + 6, 60:400] = (0, 120, 215)  # selected row: blue background (RGB)
            color = (255, 255, 255)
        if t == "Cherry":
            color = (220, 20, 20)    # red text
        if t == "Grape":
            color = (190, 190, 190)  # gray, low contrast on white
        img[int(y1) + 3:int(y2) - 3, int(x1):int(x2)] = color
    node = layout.xy_cut([w for r in rows for w in r])
    layout.annotate(node, img, rgb=True)
    out = layout.render(node, W, H, 8, 5)
    assert "▶ Banana[selected]" in out
    assert "Cherry[red]" in out
    assert "Grape[dim]" in out
    assert "Apple" in out


def test_button_vs_selected_row():
    img = _canvas()
    words = []
    for i, t in enumerate(["Save changes?", "Your edits will be lost.", "Autosave is off."]):
        ws = line(t, 700, 400 + i * 32)
        words += ws
        img[int(ws[0].y1) + 3:int(ws[0].y2) - 3, int(ws[0].x1):int(ws[-1].x2)] = (30, 30, 30)
    for x, t in [(800, "Save"), (950, "Cancel")]:
        ws = line(t, x, 520)
        words += ws
        img[505:552, x - 20:int(ws[-1].x2) + 20] = (225, 225, 225)  # button box
        img[int(ws[0].y1) + 3:int(ws[0].y2) - 3, int(ws[0].x1):int(ws[-1].x2)] = (0, 0, 0)
    node = layout.xy_cut(words)
    layout.annotate(node, img, rgb=True)
    out = "\n".join(layout.render(node, W, H, 8, 5))
    assert "Save[button]" in out and "Cancel[button]" in out
    assert "selected" not in out


def test_normalize_fixes_typical_chinese_recogniser_slips_but_leaves_chinese_alone():
    n = layout.normalize
    assert n("14：32") == "14:32" and n("2026一10一28") == "2026-10-28"
    assert n("1乙800.00 元") == "12800.00 元" and n("1 乙304 KB") == "12304 KB"
    assert n("deploy tO the") == "deploy to the" and n("ExceI 工作表") == "Excel 工作表"
    assert n("会议纪要．txt") == "会议纪要.txt"
    for keep in ["OCR", "macOS", "iOS", "第一章", "张伟一李", "Untitled - Notepad", "负责人：张伟"]:
        assert n(keep) == keep


def test_join_uses_gaps_not_engine_spacing():
    def w(t, x1, x2, h=16):
        return Word(t, x1, 100, x2, 100 + h)
    # Windows OCR style: every CJK char and every token is a separate word
    assert layout.join_words([w("项", 0, 16), w("目", 18, 34), w("2026", 40, 80), w("-", 82, 88, 2),
                              w("10", 90, 108)]) == "项目 2026-10"
    assert layout.join_words([w("Frid", 0, 30), w("ay", 31, 45), w("now", 52, 75)]) == "Friday now"
    assert layout.join_words([w("1", 0, 4), w("1", 10, 14), w(":05", 15, 35)]) == "11:05"
