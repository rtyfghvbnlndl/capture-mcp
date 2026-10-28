"""Synthetic 1080p test screens with known text, for OCR integration tests and benchmarks.

Each builder returns (bgr_image, expected_texts) where expected_texts are strings that must appear in the
OCR output (layout-independent checks), so tests do not depend on exact spacing.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT_DIRS = [os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"), "/usr/share/fonts", "/usr/share/fonts/truetype",
             "/System/Library/Fonts"]
CJK_FONTS = ["msyh.ttc", "msyh.ttf", "simhei.ttf", "NotoSansCJK-Regular.ttc", "wqy-microhei.ttc", "PingFang.ttc"]
LATIN_FONTS = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"]


def _find(names: list[str]) -> str | None:
    for d in FONT_DIRS:
        for root, _, files in os.walk(d) if os.path.isdir(d) else []:
            for n in names:
                if n in files:
                    return os.path.join(root, n)
    return None


CJK_PATH = _find(CJK_FONTS)
LATIN_PATH = _find(LATIN_FONTS) or CJK_PATH


def have_fonts() -> bool:
    return CJK_PATH is not None


def font(size: int, cjk: bool = True):
    path = CJK_PATH if cjk else LATIN_PATH
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


DRAWN: list[str] = []


class _Draw(ImageDraw.ImageDraw):
    def text(self, xy, text, *a, **k):  # remember the ground truth
        DRAWN.append(text)
        return super().text(xy, text, *a, **k)


def _draw(img: Image.Image) -> "_Draw":
    DRAWN.clear()
    return _Draw(img)


def _bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def notepad() -> tuple[np.ndarray, list[str]]:
    img = Image.new("RGB", (W, H), (32, 64, 110))
    d = _draw(img)
    d.rectangle([200, 100, 1500, 900], fill=(255, 255, 255))
    d.rectangle([200, 100, 1500, 140], fill=(240, 240, 240))
    d.text((216, 108), "会议纪要.txt - 记事本", font=font(18), fill=(0, 0, 0))
    d.text((216, 150), "文件   编辑   格式   查看   帮助", font=font(16), fill=(0, 0, 0))
    lines = ["项目周会 2026-10-28", "1. 完成采集卡驱动测试，延迟约 120 ms。",
             "2. OCR 文字识别：Windows 引擎平均 0.3 秒。", "3. Next step: deploy to the edge device by Friday.",
             "负责人：张伟、Li Na"]
    for i, t in enumerate(lines):
        d.text((220, 200 + i * 32), t, font=font(18), fill=(20, 20, 20))
    d.rectangle([0, 1040, W, H], fill=(20, 20, 20))
    d.text((1780, 1050), "14:32", font=font(16), fill=(255, 255, 255))
    return _bgr(img), ["会议纪要", "记事本", "项目周会", "延迟约", "deploy to the edge device", "14:32"]


def mail() -> tuple[np.ndarray, list[str]]:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = _draw(img)
    d.rectangle([0, 0, 320, H], fill=(243, 243, 243))
    folders = ["收件箱 (12)", "已发送", "草稿", "垃圾邮件", "已删除"]
    for i, t in enumerate(folders):
        y = 120 + i * 48
        if i == 0:
            d.rectangle([8, y - 10, 312, y + 30], fill=(0, 120, 215))
            d.text((28, y), t, font=font(18), fill=(255, 255, 255))
        else:
            d.text((28, y), t, font=font(18), fill=(30, 30, 30))
    d.text((400, 60), "发票确认 Invoice #2041", font=font(26), fill=(0, 0, 0))
    body = ["王经理您好：", "附件是十月份的发票，金额为 12,800.00 元。", "Please confirm by replying to this email.",
            "如有疑问请联系财务部。", "", "此致", "李明"]
    for i, t in enumerate(body):
        d.text((400, 140 + i * 30), t, font=font(18), fill=(30, 30, 30))
    d.text((400, 400), "付款已逾期 3 天", font=font(18), fill=(200, 20, 20))
    return _bgr(img), ["收件箱", "已发送", "发票确认", "12,800.00", "Please confirm", "付款已逾期"]


def table() -> tuple[np.ndarray, list[str]]:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = _draw(img)
    cols = [120, 620, 1000, 1300]
    head = ["名称", "修改日期", "类型", "大小"]
    rows = [["report_2026.xlsx", "2026/10/27 09:12", "Excel 工作表", "48 KB"],
            ["会议录音.mp3", "2026/10/26 15:40", "MP3 文件", "12,304 KB"],
            ["setup.exe", "2026/10/20 11:05", "应用程序", "86,112 KB"],
            ["readme.txt", "2026/10/01 08:00", "文本文档", "2 KB"]]
    for c, t in zip(cols, head):
        d.text((c, 100), t, font=font(18), fill=(90, 90, 90))
    for r, row in enumerate(rows):
        for c, t in zip(cols, row):
            d.text((c, 150 + r * 36), t, font=font(18), fill=(20, 20, 20))
    return _bgr(img), ["report_2026.xlsx", "修改日期", "12,304 KB", "setup.exe", "文本文档"]


def dialog() -> tuple[np.ndarray, list[str]]:
    img = Image.new("RGB", (W, H), (60, 90, 120))
    d = _draw(img)
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (300, 420, 3), dtype=np.uint8)).resize((420, 300))
    img.paste(photo, (1350, 120))  # a picture with no text
    d.rectangle([660, 360, 1260, 680], fill=(250, 250, 250))
    d.text((690, 380), "保存更改", font=font(22), fill=(0, 0, 0))
    d.text((690, 440), "是否将更改保存到 预算表.xlsx？", font=font(18), fill=(30, 30, 30))
    d.text((690, 480), "Unsaved changes will be lost.", font=font(18), fill=(30, 30, 30))
    d.text((690, 520), "(自动保存已关闭)", font=font(16), fill=(170, 170, 170))
    for x, t, primary in [(860, "保存", True), (990, "不保存", False), (1120, "取消", False)]:
        d.rectangle([x - 20, 600, x + 90, 650], fill=(0, 103, 192) if primary else (225, 225, 225))
        d.text((x, 612), t, font=font(18), fill=(255, 255, 255) if primary else (0, 0, 0))
    return _bgr(img), ["保存更改", "预算表", "Unsaved changes", "不保存", "取消"]


ALL = {"notepad": notepad, "mail": mail, "table": table, "dialog": dialog}
