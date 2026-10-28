"""Turn OCR words into layout-preserving plain text, for models that cannot (or should not) look at images.

Pipeline (all pure functions, no OCR engine needed, unit-testable):
  1. XY-cut: recursively split the words into bands (by horizontal whitespace) and columns/panes (by
     vertical whitespace). Columns whose rows line up are a *table* and are kept together as rows.
  2. Inside a leaf block: group words into visual rows, rows into segments (runs of words with small gaps).
  3. Render: panes get a header with their grid-cell range, bands are separated by a blank line, table
     cells are joined with " | ", wide gaps inside a row become "   ".
  4. Optional annotations from the pixels under each segment: ▶ ...[selected] (a row on a background that
     spans the pane, e.g. a selected list item), [button] (text on its own small box), [red]/[green]/[blue]
     text, [dim] (low contrast: disabled / placeholder), "(?)" for low OCR confidence.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .imaging import Region, region_to_cells


@dataclass
class Word:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None   # 0..1, None when the engine does not report it

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def _box(words: list[Word]) -> tuple[float, float, float, float]:
    return min(w.x1 for w in words), min(w.y1 for w in words), max(w.x2 for w in words), max(w.y2 for w in words)


def _median_h(words: list[Word]) -> float:
    return statistics.median(w.h for w in words) if words else 1.0


# ---------------------------------------------------------------- text joining

def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3000 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)


_DASHES = set("一—–－_")
_FULL_TO_ASCII = {"：": ":", "．": ".", "，": ",", "／": "/", "（": "(", "）": ")", "；": ";", "％": "%", "＋": "+",
                  "＝": "=", "＃": "#", "＠": "@", "＆": "&"}


def join_words(words: list[Word], line_h: float | None = None) -> str:
    """Join the words of one segment using the measured gaps: OCR engines split Chinese into single
    characters and sometimes split numbers/words ("1 20", "Frid ay"), so a space is inserted only where
    the gap is a real space (> 0.3 x text height)."""
    ws = [w for w in sorted(words, key=lambda w: w.x1) if w.text.strip()]
    if not ws:
        return ""
    h = line_h or statistics.median(w.h for w in ws)
    parts: list[tuple[str, Word]] = []
    for w in ws:
        t = w.text.strip()
        # a dash-like glyph that is much flatter than the text around it is a hyphen/minus, not "一"
        if t in _DASHES and w.h < 0.35 * h:
            t = "-"
        parts.append((t, w))
    out = parts[0][0]
    for (t, w), (_, pw) in zip(parts[1:], parts):
        gap = w.x1 - pw.x2
        # narrow digits ("1") have wide side bearings, so digit-to-digit gaps look bigger than they are
        limit = 0.55 * h if (out[-1].isdigit() and t[0].isdigit()) else 0.3 * h
        if gap > limit and not (_is_cjk(out[-1]) and _is_cjk(t[0]) and gap < 0.8 * h):
            out += " "
        out += t
    return normalize(out)


# characters a Chinese-tuned recogniser returns for digits it misreads, when they sit between digits
_DIGIT_LOOKALIKES = {"乙": "2", "己": "2", "已": "2", "〇": "0", "O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}
# a lone capital among lowercase letters is usually a misread lowercase letter ("tO", "ExceI")
_CAPITAL_LOOKALIKES = {"O": "o", "I": "l"}


def _fix_word(tok: str) -> str:
    letters = [c for c in tok if c.isascii() and c.isalpha()]
    if len(letters) < 2 or not tok.isascii():
        return tok
    caps = [i for i, c in enumerate(tok) if c.isupper() and i > 0]
    if len(caps) == 1 and tok[caps[0]] in _CAPITAL_LOOKALIKES and any(c.islower() for c in tok):
        others = [c for i, c in enumerate(tok) if i > 0 and i != caps[0] and c.isalpha()]
        if all(c.islower() for c in others):
            i = caps[0]
            return tok[:i] + _CAPITAL_LOOKALIKES[tok[i]] + tok[i + 1:]
    return tok


def normalize(s: str) -> str:
    """Full-width punctuation and dashes between ASCII letters/digits back to ASCII ("14：32" -> "14:32",
    "2026一10" -> "2026-10"), as recognisers tuned for Chinese tend to emit them. Chinese text is untouched."""
    # full-width period and slash are not used in running Chinese text (it uses 。): always ASCII
    chars = list(s.replace("．", ".").replace("／", "/"))
    for i, ch in enumerate(chars):
        if ch in _FULL_TO_ASCII or ch in _DASHES:
            left = next((c for c in reversed(chars[:i]) if c != " "), "")
            right = next((c for c in chars[i + 1:] if c != " "), "")
            ascii_ok = left.isascii() and left.isalnum() and right.isascii() and right.isalnum()
            if ch in _DASHES:  # "一" is also a common character: only between digits ("2026一10")
                ascii_ok = left.isdigit() and right.isdigit() and left.isascii() and right.isascii()
            if ascii_ok:
                chars[i] = _FULL_TO_ASCII.get(ch, "-")
                # drop the spaces the engine put around it ("14 : 32" -> "14:32")
                j = i - 1
                while j >= 0 and chars[j] == " ":
                    chars[j] = ""
                    j -= 1
                j = i + 1
                while j < len(chars) and chars[j] == " ":
                    chars[j] = ""
                    j += 1
    s = "".join(chars)
    # digit look-alikes between digits: "1乙800" -> "12800", "1 乙304" -> "12304", "1O5" -> "105"
    out = list(s)
    for i, ch in enumerate(out):
        if ch in _DIGIT_LOOKALIKES:
            j = i - 1
            while j >= 0 and out[j] == " ":
                j -= 1
            k = i + 1
            while k < len(out) and out[k] == " ":
                k += 1
            if j >= 0 and k < len(out) and out[j].isdigit() and out[k].isdigit() and out[j].isascii() and out[k].isascii():
                out[i] = _DIGIT_LOOKALIKES[ch]
                for m in range(j + 1, i):
                    out[m] = ""
                for m in range(i + 1, k):
                    out[m] = ""
    return " ".join(_fix_word(t) for t in "".join(out).split(" "))


# ---------------------------------------------------------------- rows / segments

@dataclass
class Segment:
    words: list[Word]
    text: str
    tags: list[str] = field(default_factory=list)

    @property
    def box(self):
        return _box(self.words)


@dataclass
class Row:
    words: list[Word]
    segments: list[Segment] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def box(self):
        return _box(self.words)

    @property
    def cy(self) -> float:
        b = self.box
        return (b[1] + b[3]) / 2


def group_rows(words: list[Word]) -> list[Row]:
    """Words whose vertical extents overlap by >= half the smaller height share a row."""
    rows: list[tuple[float, float, list[Word]]] = []  # (center, height, words)
    for w in sorted(words, key=lambda w: (w.cy, w.x1)):
        best, best_ov = None, 0.0
        for i, (c, h, _) in enumerate(rows):
            ov = min(w.y2, c + h / 2) - max(w.y1, c - h / 2)
            if ov >= 0.5 * min(w.h, h) and ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            rows.append((w.cy, w.h, [w]))
        else:
            rows[best][2].append(w)
    groups = [ws for _, _, ws in rows]
    # Punctuation and other tiny glyphs (",", "。", a hyphen) sit lower/higher than the line and can end
    # up as a row of their own: give them back to the nearest real row.
    h_all = _median_h(words)
    real = [g for g in groups if not _minor(g, h_all)]
    if real:
        for g in groups:
            if _minor(g, h_all):
                cy = statistics.mean(w.cy for w in g)
                target = min(real, key=lambda r: abs(statistics.mean(w.cy for w in r) - cy))
                if abs(statistics.mean(w.cy for w in target) - cy) <= 1.2 * h_all:
                    target.extend(g)
                else:
                    real.append(g)
        groups = real
    out = [Row(ws) for ws in groups]
    for r in out:
        r.segments = split_segments(r.words)
    out.sort(key=lambda r: r.cy)
    return out


def _minor(ws: list[Word], h: float) -> bool:
    return all(not any(c.isalnum() for c in w.text) or w.h < 0.5 * h for w in ws)


def split_segments(words: list[Word], gap_factor: float = 1.0) -> list[Segment]:
    ws = sorted(words, key=lambda w: w.x1)
    h = statistics.median(w.h for w in ws if any(c.isalnum() for c in w.text)) if any(
        any(c.isalnum() for c in w.text) for w in ws) else _median_h(ws)
    groups: list[list[Word]] = []
    for w in ws:
        if groups and w.x1 - max(x.x2 for x in groups[-1]) <= gap_factor * h:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [Segment(g, join_words(g, h)) for g in groups]


# ---------------------------------------------------------------- XY-cut

@dataclass
class Node:
    kind: str                       # "leaf" | "bands" (stacked vertically) | "columns" (side by side) | "table"
    words: list[Word]
    children: list["Node"] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)          # leaf
    table: list[list[str]] = field(default_factory=list)   # table: rows of cells
    cols: list[tuple[float, float]] = field(default_factory=list)  # table: x range of each column

    @property
    def box(self):
        return _box(self.words)


def _merge_table_headers(children: list[Node], h: float) -> list[Node]:
    """A one-row band right above a table whose pieces sit over the table's columns is its header row
    (headers are often set apart by a larger gap, so the band split separated them)."""
    out: list[Node] = []
    for c in children:
        prev = out[-1] if out else None
        if (c.kind == "table" and prev is not None and prev.kind == "leaf" and len(prev.rows) == 1
                and c.box[1] - prev.box[3] <= 4 * h and len(prev.rows[0].segments) >= 2):
            cells: list[list[str]] = [[] for _ in c.cols]
            ok = True
            for seg in prev.rows[0].segments:
                x1, _, x2, _ = seg.box
                hit = [i for i, (a, b) in enumerate(c.cols) if x1 < b + h and x2 > a - h]
                if not hit:
                    ok = False
                    break
                cells[hit[0]].append(seg.text)
            if ok:
                out[-1] = Node("table", prev.words + c.words, table=[["   ".join(x) for x in cells]] + c.table,
                               cols=c.cols)
                continue
        out.append(c)
    return out


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[float]:
    """Split positions where the union of intervals leaves a gap >= min_gap."""
    cuts = []
    ivs = sorted(intervals)
    end = ivs[0][1]
    for a, b in ivs[1:]:
        if a - end >= min_gap:
            cuts.append((end + a) / 2)
        end = max(end, b)
    return cuts


def _split(words: list[Word], cuts: list[float], key) -> list[list[Word]]:
    parts: list[list[Word]] = [[] for _ in range(len(cuts) + 1)]
    for w in words:
        i = sum(1 for c in cuts if key(w) > c)
        parts[i].append(w)
    return [p for p in parts if p]


def _aligned_fraction(parts: list[list[Word]], tol: float) -> float:
    """How many rows of each column have a row at the same height in another column (tables do)."""
    row_sets = [[r.cy for r in group_rows(p)] for p in parts]
    total = sum(len(rs) for rs in row_sets)
    hits = 0
    for i, rs in enumerate(row_sets):
        others = [c for j, o in enumerate(row_sets) if j != i for c in o]
        hits += sum(1 for c in rs if any(abs(c - o) <= tol for o in others))
    return hits / total if total else 0.0


def _table(parts: list[list[Word]], tol: float) -> list[list[str]]:
    col_rows = [group_rows(p) for p in parts]
    clusters: list[list[float]] = []
    for c in sorted(r.cy for rows in col_rows for r in rows):  # cluster row centers across columns
        if clusters and c - statistics.mean(clusters[-1]) <= tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    centers = [statistics.mean(c) for c in clusters]
    table = [[[] for _ in parts] for _ in centers]
    for ci, rows in enumerate(col_rows):
        for r in rows:  # every row lands in exactly one table line: the nearest
            li = min(range(len(centers)), key=lambda i: abs(centers[i] - r.cy))
            table[li][ci].extend(s.text for s in r.segments)
    return [["   ".join(cell) for cell in line] for line in table]


def xy_cut(words: list[Word], depth: int = 0) -> Node:
    h = _median_h(words)
    if len(words) > 1 and depth < 12:
        # 1) horizontal whitespace -> bands, top to bottom
        cuts = _gaps([(w.y1, w.y2) for w in words], 2.0 * h)
        if cuts:
            parts = _split(words, cuts, key=lambda w: w.cy)
            if len(parts) > 1:
                children = _merge_table_headers([xy_cut(p, depth + 1) for p in parts], h)
                return children[0] if len(children) == 1 else Node("bands", words, children)
        # 2) vertical whitespace -> columns, unless everything is one row (a menu bar is not columns)
        if len(group_rows(words)) > 1:
            cuts = _gaps([(w.x1, w.x2) for w in words], 2.0 * h)
            if cuts:
                parts = _split(words, cuts, key=lambda w: (w.x1 + w.x2) / 2)
                if len(parts) > 1:
                    if all(len(group_rows(p)) >= 2 for p in parts) and _aligned_fraction(parts, 0.5 * h) >= 0.7:
                        return Node("table", words, table=_table(parts, 0.5 * h),
                                    cols=[(_box(p)[0], _box(p)[2]) for p in parts])
                    return Node("columns", words, [xy_cut(p, depth + 1) for p in parts])
    return Node("leaf", words, rows=group_rows(words))


# ---------------------------------------------------------------- annotations (pixels)

def _luma(c: np.ndarray) -> float:
    return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


def _bg_fg(img: np.ndarray, box, rgb: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Background = median of a 2px ring around the box; foreground = median of clearly different pixels."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros(3), None
    pad = 2
    ox1, oy1, ox2, oy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    outer = img[oy1:oy2, ox1:ox2].reshape(-1, 3).astype(np.int16)
    inner = img[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
    ring = np.concatenate([img[oy1:y1, ox1:ox2].reshape(-1, 3), img[y2:oy2, ox1:ox2].reshape(-1, 3),
                           img[y1:y2, ox1:x1].reshape(-1, 3), img[y1:y2, x2:ox2].reshape(-1, 3)]).astype(np.int16)
    bg = np.median(ring if len(ring) else outer, axis=0)
    diff = np.abs(inner - bg).sum(axis=1)
    fg_px = inner[diff > max(60, np.percentile(diff, 80) * 0.6)]
    fg = np.median(fg_px, axis=0) if len(fg_px) >= 4 else None
    if not rgb:  # BGR frames (OpenCV)
        bg = bg[::-1]
        fg = fg[::-1] if fg is not None else None
    return bg, fg


def _color_tag(fg: np.ndarray) -> str | None:
    r, g, b = (float(v) for v in fg)
    if r > 150 and r > g + 70 and r > b + 70:
        return "red"
    if g > 110 and g > r + 50 and g > b + 30:
        return "green"
    if b > 140 and b > r + 60 and b > g + 20:
        return "blue"
    return None


def annotate(node: Node, img: np.ndarray, rgb: bool = False) -> None:
    """Add [selected]/[button]/[red]/[green]/[blue]/[dim] tags in place. img is the full screen (BGR by default).

    Highlight is judged per pane (a column/window area), not per leaf: a sidebar with a gap in it is split
    into several small leaves, which would leave too few rows to tell what the normal background is."""
    for pane in _panes(node):
        segs: list[tuple[Row, Segment, np.ndarray]] = []
        for leaf in pane:
            for row in leaf.rows:
                for seg in row.segments:
                    bg, fg = _bg_fg(img, seg.box, rgb)
                    segs.append((row, seg, bg))
                    if fg is not None:
                        tag = _color_tag(fg)
                        if tag:
                            seg.tags.append(tag)
                        elif abs(_luma(fg) - _luma(bg)) < 100:
                            seg.tags.append("dim")
        if len(segs) < 3:
            continue
        # Text on a background that differs from most of the pane: a selected row if that background
        # spans most of the pane's width, else a button / badge.
        base = np.median(np.array([bg for _, _, bg in segs]), axis=0)
        odd = [(row, seg, bg) for row, seg, bg in segs if np.abs(bg - base).sum() > 60]
        if not odd or 2 * len(odd) >= len(segs):  # must be the minority
            continue
        px1, _, px2, _ = _box([w for leaf in pane for w in leaf.words])
        pane_w = max(1.0, px2 - px1)
        for row, seg, bg in odd:
            if _bg_run(img, seg.box, bg, rgb) >= 0.6 * pane_w:
                if "selected" not in row.tags:
                    row.tags.append("selected")
            else:
                seg.tags.append("button")


def _bg_run(img: np.ndarray, box, bg: np.ndarray, rgb: bool) -> float:
    """Width of the run of background colour `bg` through the segment, measured just above the text."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    y = max(0, min(H - 1, y1 - 2))
    line = img[y, :, :].astype(np.int16)
    target = bg if rgb else bg[::-1]
    match = np.abs(line - target).sum(axis=1) <= 45
    left = max(0, min(W - 1, x1))
    right = max(0, min(W - 1, x2))
    while left > 0 and match[left - 1]:
        left -= 1
    while right < W - 1 and match[right + 1]:
        right += 1
    return float(right - left)


def _panes(node: Node) -> list[list[Node]]:
    """Leaves grouped by pane: each child of a 'columns' node is its own pane; the root is one otherwise."""
    panes: list[list[Node]] = []

    def walk(n: Node, current: list[Node]) -> None:
        if n.kind == "leaf":
            current.append(n)
        elif n.kind == "columns":
            for c in n.children:
                sub: list[Node] = []
                panes.append(sub)
                walk(c, sub)
        else:
            for c in n.children:
                walk(c, current)

    root: list[Node] = []
    panes.append(root)
    walk(node, root)
    return [p for p in panes if p]


def _leaves(node: Node):
    if node.kind == "leaf":
        yield node
    for c in node.children:
        yield from _leaves(c)


# ---------------------------------------------------------------- render

def _seg_text(seg: Segment, low_conf: float) -> str:
    t = seg.text
    if any(w.conf is not None and w.conf < low_conf for w in seg.words):
        t += "(?)"
    if seg.tags:
        t += "".join(f"[{x}]" for x in seg.tags)
    return t


def render(node: Node, screen_w: int, screen_h: int, cols: int, rows: int, low_conf: float = 0.6) -> list[str]:
    def cells(box) -> str:
        x1, y1, x2, y2 = box
        r = Region(max(0, x1 / screen_w), max(0, y1 / screen_h), min(1, x2 / screen_w), min(1, y2 / screen_h))
        return region_to_cells(r, cols, rows)

    out: list[str] = []

    def emit(n: Node, header: bool) -> None:
        if header:
            label = "table" if n.kind == "table" else "pane"
            out.append(f"── {label} {cells(n.box)} ──")
        if n.kind == "leaf":
            prev_y2, h = None, _median_h(n.words)
            for row in n.rows:
                b = row.box
                if prev_y2 is not None and b[1] - prev_y2 > 1.0 * h:
                    out.append("")  # paragraph gap
                line = "   ".join(_seg_text(s, low_conf) for s in row.segments)
                if row.tags:
                    line = "▶ " + line + "".join(f"[{t}]" for t in row.tags)
                out.append(line)
                prev_y2 = b[3]
        elif n.kind == "table":
            if not header:
                out.append(f"── table {cells(n.box)} ──")
            out.extend(" | ".join(c) for c in n.table)
        elif n.kind == "bands":
            for i, c in enumerate(n.children):
                if i and out and out[-1] != "":
                    out.append("")
                emit(c, False)
        else:  # columns: each pane gets a header with its grid range
            for c in n.children:
                if out and out[-1] != "":
                    out.append("")
                emit(c, c.kind != "table")
    emit(node, False)
    while out and out[-1] == "":
        out.pop()
    return out


def truncate(lines: list[str], max_chars: int) -> tuple[list[str], bool]:
    total, kept = 0, []
    for ln in lines:
        total += len(ln) + 1
        if total > max_chars:
            return kept, True
        kept.append(ln)
    return kept, False


def diff_lines(old: list[str], new: list[str]) -> list[str]:
    """Rows that appeared (+) or disappeared (-) since the previous reading; headers kept for context."""
    old_c, new_c = Counter(l for l in old if l and not l.startswith("──")), Counter(l for l in new if l and not l.startswith("──"))
    added, removed = new_c - old_c, old_c - new_c
    out, header = [], None
    for ln in new:
        if ln.startswith("──"):
            header = ln
        elif added.get(ln, 0) > 0:
            if header:
                out.append(header)
                header = None
            out.append("+ " + ln)
            added[ln] -= 1
    for ln in old:
        if removed.get(ln, 0) > 0:
            out.append("- " + ln)
            removed[ln] -= 1
    return out
