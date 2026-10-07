"""Extract one NCERT chapter PDF into a contract-v1 chapter record (Track B).

Policy notes (documented per data_contract.md §4.1):
- A chapter is one PDF file; page_start/page_end are 1-based PDF page indices.
- Running headers/footers/page numbers are excluded from `content_blocks` and
  from `text_cleaned`, but remain in the chapter-level `text_raw`.
- A paragraph that continues across pages becomes one block per page.
- Reading order: top-to-bottom, except on two-column pages, where each column
  is read top-to-bottom between full-width blocks.
- Stacked fractions in equations are rebuilt into one line, e.g. `∆L/L`;
  multi-term parts are parenthesised, e.g. `(F × L)/(A × ∆L)`.
- Tables are rebuilt from the table's filled background rectangles plus the text
  lines inside them (rows by baseline, cells by horizontal gaps) as one `table`
  block with rows on separate lines and cells joined by " | ". Tables that
  cannot be located this way stay as paragraph text and are counted in the
  chapter stats for review.
- Some PDFs (e.g. keph107) store text as private-use codes U+F0xx; these are
  decoded to the characters they stand for (cp1252 for text fonts, a Greek /
  operator table for Symbol-style fonts).
- Chapter openers are "CHAPTER <n>" or "UNIT <n>" (any weight); the title is the
  large heading(s) right after it, and stays null when it is part of a graphic.
- Images are recorded as `figure` blocks with null text; binary extraction and
  `asset_ids` linking belong to Track C (asset_ids stay empty here).
- Vector-only drawings are not detected as figures (Track C territory).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from src.data.ids import make_block_id, make_chapter_id
from src.data.ncert_clean import (
    boilerplate_key,
    clean_block_text,
    decode_private_use,
    private_use_ratio,
)

MIN_FIGURE_SIDE_PT = 40
MAX_FIGURE_PAGE_FRACTION = 0.8
HEADER_ZONE = 0.125
FOOTER_ZONE = 0.93
PUA_REVIEW_THRESHOLD = 0.005
FULL_WIDTH_FRACTION = 0.55
TABLE_SEARCH_BELOW_CAPTION_PT = 60
TABLE_MAX_PAGE_FRACTION = 0.7

_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE "
        "THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN TWENTY".split()
    )
}

_CHAPTER_RE = re.compile(r"^(?:CHAPTER|UNIT)\s+(\w+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\b")
_EXAMPLE_RE = re.compile(r"^(example\s+\d+(?:\.\d+)*)\b", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^((?:fig(?:ure)?\.?|table)\s*\d+(?:\.\d+)*)", re.IGNORECASE)
_EQ_NUMBER_RE = re.compile(r"(\(\d+(?:\.\d+)*[a-z]?\))\s*$")
_LIST_RE = re.compile(r"^\s*(?:\d+[.)]|\([a-z]{1,3}\)|\([ivx]+\)|[•▪●])\s+")
_LABEL_ONLY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|\([a-z]{1,3}\))$")
_PANEL_LABEL_RE = re.compile(r"^\([a-z]\)$", re.IGNORECASE)
_LONG_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_MATH_FONTS = ("symbol", "math", "cmsy", "cmmi")
_SYMBOL_FONTS = ("symbol", "mtextra", "euclid")
_BOLD_FONT_NAMES = ("bold", "demi", "black", "heavy")


@dataclass
class Line:
    """One text line with its geometry (PDF points, origin top-left)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    base: float


@dataclass
class RawBlock:
    page: int
    kind: str  # "text" | "image" | "table"
    bbox: tuple[float, float, float, float]
    text: str | None = None
    size: float = 0.0
    bold: bool = False
    math_ratio: float = 0.0
    page_height: float = 0.0
    lines: list[Line] = field(default_factory=list)


@dataclass
class ChapterExtraction:
    record: dict
    review_reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Raw block collection
# --------------------------------------------------------------------------- #


def _is_bold(span: dict) -> bool:
    font = span["font"].lower()
    return any(k in font for k in _BOLD_FONT_NAMES) or bool(span["flags"] & 16)


def _text_block(page_no: int, block: dict, page_height: float) -> RawBlock | None:
    lines: list[Line] = []
    spans: list[dict] = []
    for line in block["lines"]:
        # Wingdings spans are decorative bullet/marker glyphs ("u", "⊳"), not content.
        kept = [
            {**s, "text": decode_private_use(s["text"], any(k in s["font"].lower() for k in _SYMBOL_FONTS))}
            for s in line["spans"]
            if "wingdings" not in s["font"].lower()
        ]
        line_text = "".join(s["text"] for s in kept)
        if line_text.strip():
            x0, y0, x1, y1 = line["bbox"]
            lines.append(Line(line_text, x0, y0, x1, y1, kept[0]["origin"][1]))
            spans.extend(s for s in kept if s["text"].strip())
    if not lines:
        return None

    chars = sum(len(s["text"]) for s in spans) or 1
    math_chars = sum(
        len(s["text"]) for s in spans if any(m in s["font"].lower() for m in _MATH_FONTS)
    )
    return RawBlock(
        page=page_no,
        kind="text",
        bbox=tuple(block["bbox"]),
        text="\n".join(line.text for line in lines),
        size=max(s["size"] for s in spans),
        bold=all(_is_bold(s) for s in spans),
        math_ratio=math_chars / chars,
        page_height=page_height,
        lines=lines,
    )


def _union(boxes) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _merge_label_blocks(blocks: list[RawBlock]) -> list[RawBlock]:
    """Join a label-only block ("8.1", "1.") with the block on the same line."""
    merged: list[RawBlock] = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if (
            cur.kind == "text"
            and nxt is not None
            and nxt.kind == "text"
            and _LABEL_ONLY_RE.match((cur.text or "").strip())
            and abs(cur.bbox[1] - nxt.bbox[1]) < 4
        ):
            merged.append(
                RawBlock(
                    page=cur.page,
                    kind="text",
                    bbox=_union([cur.bbox, nxt.bbox]),
                    text=f"{cur.text.strip()} {nxt.text}",
                    size=max(cur.size, nxt.size),
                    bold=cur.bold and nxt.bold,
                    math_ratio=nxt.math_ratio,
                    page_height=cur.page_height,
                    lines=cur.lines + nxt.lines,
                )
            )
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged


# --------------------------------------------------------------------------- #
# Stacked fractions
# --------------------------------------------------------------------------- #


@dataclass
class _Tok:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    base: float


def _wrap(text: str) -> str:
    return f"({text})" if re.search(r"[\s+\-−×·=±]", text) else text


def _merge_adjacent(toks: list[_Tok]) -> list[_Tok]:
    """Join fragments sitting side by side on one baseline (e.g. "∆" + "L")."""
    out: list[_Tok] = []
    for t in sorted(toks, key=lambda t: (round(t.base), t.x0)):
        for o in out:
            if abs(o.base - t.base) < 1.0 and -1.0 <= t.x0 - o.x1 < 2.0:
                o.text += t.text
                o.x1, o.y0, o.y1 = max(o.x1, t.x1), min(o.y0, t.y0), max(o.y1, t.y1)
                break
        else:
            out.append(_Tok(t.text, t.x0, t.y0, t.x1, t.y1, t.base))
    return out


def _can_stack(a: _Tok, b: _Tok) -> bool:
    """True when `a` sits directly above `b` like a fraction's numerator/denominator."""
    if not (a.y1 <= b.y0 + 2 and b.y0 - a.y1 < 6):
        return False
    wa, wb = a.x1 - a.x0, b.x1 - b.x0
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    if overlap <= 0 or overlap < 0.4 * min(wa, wb) or wa > 110 or wb > 110:
        return False
    # A fraction's parts are centred on each other; consecutive equation rows are not.
    if abs((a.x0 + a.x1) / 2 - (b.x0 + b.x1) / 2) > 0.35 * max(wa, wb) + 2:
        return False
    for t in (a.text.strip(), b.text.strip()):
        if len(t) > 30 or _LONG_WORD_RE.search(t) or _PANEL_LABEL_RE.match(t):
            return False
        if "=" in t or not re.search(r"[A-Za-z0-9∆σεθλμπαβγδ]", t):
            return False
    return True


def _stack_fractions(toks: list[_Tok]) -> tuple[list[_Tok], int]:
    stacked = 0
    changed = True
    while changed:
        changed = False
        for a in toks:
            partner = next((b for b in toks if b is not a and _can_stack(a, b)), None)
            if partner is None:
                continue
            frac = _Tok(
                f"{_wrap(a.text.strip())}/{_wrap(partner.text.strip())}",
                min(a.x0, partner.x0),
                a.y0,
                max(a.x1, partner.x1),
                partner.y1,
                (a.base + partner.base) / 2,
            )
            toks = [t for t in toks if t is not a and t is not partner] + [frac]
            stacked += 1
            changed = True
            break
    return toks, stacked


def _flatten_rows(toks: list[_Tok]) -> str:
    """Group tokens into visual rows (by vertical overlap), left to right."""
    rows: list[list[_Tok]] = []
    for t in sorted(toks, key=lambda t: (t.y0, t.x0)):
        for row in rows:
            top, bottom = min(r.y0 for r in row), max(r.y1 for r in row)
            overlap = min(bottom, t.y1) - max(top, t.y0)
            if overlap >= 0.4 * max(t.y1 - t.y0, 1e-6):
                row.append(t)
                break
        else:
            rows.append([t])
    return "\n".join(
        " ".join(" ".join(t.text.split()) for t in sorted(row, key=lambda t: t.x0))
        for row in sorted(rows, key=lambda r: min(t.y0 for t in r))
    )


def _is_math_candidate(b: RawBlock) -> bool:
    if b.kind != "text" or not b.text or len(b.lines) > 8 or len(b.text) > 220:
        return False
    flat = " ".join(b.text.split())
    if _CAPTION_RE.match(flat) or b.bold:
        return False
    return "=" in flat or bool(_EQ_NUMBER_RE.search(flat)) or b.math_ratio >= 0.3


def _rebuild_math_blocks(blocks: list[RawBlock]) -> tuple[list[RawBlock], int]:
    """Replace groups of nearby math fragments by one block with fractions rebuilt."""
    ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    groups: list[list[RawBlock]] = []
    for b in ordered:
        if not _is_math_candidate(b):
            continue
        for g in groups:
            gx0, gy0, gx1, gy1 = _union([m.bbox for m in g])
            near_y = b.bbox[1] - gy1 < 6 and b.bbox[3] > gy0 - 6
            near_x = b.bbox[0] < gx1 + 40 and b.bbox[2] > gx0 - 40
            if near_y and near_x:
                g.append(b)
                break
        else:
            groups.append([b])

    replaced: dict[int, RawBlock | None] = {}
    total = 0
    for g in groups:
        toks = [
            _Tok(l.text.strip(), l.x0, l.y0, l.x1, l.y1, l.base)
            for b in g
            for l in b.lines
        ]
        toks, n = _stack_fractions(_merge_adjacent(toks))
        if n == 0:
            continue
        total += n
        new = RawBlock(
            page=g[0].page,
            kind="text",
            bbox=_union([m.bbox for m in g]),
            text=_flatten_rows(toks),
            size=max(m.size for m in g),
            bold=False,
            math_ratio=max(m.math_ratio for m in g),
            page_height=g[0].page_height,
            lines=[l for m in g for l in m.lines],
        )
        replaced[id(g[0])] = new
        for m in g[1:]:
            replaced[id(m)] = None
    out: list[RawBlock] = []
    for b in blocks:
        if id(b) in replaced:
            if replaced[id(b)] is not None:
                out.append(replaced[id(b)])
        else:
            out.append(b)
    return out, total


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _caption_label(flat: str) -> str | None:
    """Return "Fig. 8.1"/"Table 8.2" for a real caption, None for in-text mentions."""
    m = _CAPTION_RE.match(flat)
    if not m:
        return None
    rest = flat[m.end() :].lstrip()
    if rest and rest[0].islower():  # "Table 8.4 shows ..." is a sentence, not a caption
        return None
    return m.group(1)


def _is_table_caption(b: RawBlock) -> bool:
    if b.kind != "text" or not b.text:
        return False
    label = _caption_label(" ".join(b.text.split()))
    return bool(label and label.lower().startswith("table"))


def _fill_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    width = page.rect.width
    rects = []
    for dr in page.get_drawings():
        fill, r = dr.get("fill"), dr["rect"]
        if fill is None or r.width < 20 or r.height < 6 or r.width > 0.95 * width:
            continue
        if sum(fill) < 0.15 or min(fill) > 0.97:  # black masks / white backgrounds
            continue
        rects.append(r)
    return rects


def _table_region(cap: RawBlock, fills: list[pymupdf.Rect], page_h: float) -> pymupdf.Rect | None:
    cx0, cy0, cx1, cy1 = cap.bbox
    seeds = [
        r
        for r in fills
        if r.y0 >= cy0 - 5
        and r.y0 - cy1 < TABLE_SEARCH_BELOW_CAPTION_PT
        and min(r.x1, cx1) - max(r.x0, cx0) > 0
    ]
    if not seeds:
        return None
    region = pymupdf.Rect(min(seeds, key=lambda r: r.y0))
    grew = True
    while grew:
        grew = False
        for r in fills:
            grown = pymupdf.Rect(region.x0 - 4, region.y0 - 4, region.x1 + 4, region.y1 + 4)
            if r.y0 >= region.y0 - 5 and r.intersects(grown) and not region.contains(r):
                region |= r
                grew = True
    if region.height > TABLE_MAX_PAGE_FRACTION * page_h or region.width < 60:
        return None
    return region


def _share_inside(box: tuple, region: pymupdf.Rect) -> float:
    inter = pymupdf.Rect(box) & region
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return 0.0 if inter.is_empty else inter.width * inter.height / area


def _rows_from_lines(lines: list[Line]) -> str:
    """Rows by baseline; cells by horizontal gaps; header continuations folded up."""
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.base, l.x0)):
        if rows and abs(rows[-1][0].base - ln.base) < 3.5:
            rows[-1].append(ln)
        else:
            rows.append([ln])

    built: list[list[dict]] = []
    for row in rows:
        cells: list[dict] = []
        for ln in sorted(row, key=lambda l: l.x0):
            if cells and ln.x0 - cells[-1]["x1"] < 8:
                cells[-1]["text"] += " " + ln.text.strip()
                cells[-1]["x1"] = max(cells[-1]["x1"], ln.x1)
            else:
                cells.append({"text": ln.text.strip(), "x0": ln.x0, "x1": ln.x1})
        if built and len(cells) == 1 and len(built[-1]) > 1:
            # Continuation of a wrapped header/cell: fold into the cell it sits under.
            target = max(
                built[-1],
                key=lambda c: min(c["x1"], cells[0]["x1"]) - max(c["x0"], cells[0]["x0"]),
            )
            if min(target["x1"], cells[0]["x1"]) - max(target["x0"], cells[0]["x0"]) > 0:
                target["text"] += " " + cells[0]["text"]
                continue
        built.append(cells)
    return "\n".join(" | ".join(c["text"] for c in cells) for cells in built)


def _detect_tables(
    page: pymupdf.Page, blocks: list[RawBlock]
) -> tuple[list[RawBlock], int, int]:
    """Return (blocks with table blocks substituted, tables found, captions missed)."""
    captions = [b for b in blocks if _is_table_caption(b)]
    if not captions:
        return blocks, 0, 0
    fills = _fill_rects(page)
    found = missed = 0
    for cap in captions:
        region = _table_region(cap, fills, page.rect.height)
        if region is None:
            missed += 1
            continue
        members = [
            b
            for b in blocks
            if b is not cap and b.kind == "text" and _share_inside(b.bbox, region) >= 0.6
        ]
        lines = [l for m in members for l in m.lines]
        if len(lines) < 2:
            missed += 1
            continue
        text = _rows_from_lines(lines)
        table = RawBlock(
            page=cap.page,
            kind="table",
            bbox=(region.x0, region.y0, region.x1, region.y1),
            text=text,
            page_height=page.rect.height,
            lines=lines,
        )
        member_ids = {id(m) for m in members}
        blocks = [b for b in blocks if id(b) not in member_ids] + [table]
        found += 1
    return blocks, found, missed


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #


def _in_margin(b: RawBlock) -> bool:
    return b.bbox[3] < HEADER_ZONE * b.page_height or b.bbox[1] > FOOTER_ZONE * b.page_height


def _reading_order(blocks: list[RawBlock], page_w: float) -> list[RawBlock]:
    """Top-to-bottom; on two-column pages read each column between full-width blocks."""
    top = [b for b in blocks if b.bbox[3] < HEADER_ZONE * b.page_height]
    bottom = [b for b in blocks if b.bbox[1] > FOOTER_ZONE * b.page_height]
    body = [b for b in blocks if not _in_margin(b)]
    by_y = lambda b: (round(b.bbox[1] / 3), b.bbox[0])  # noqa: E731
    body.sort(key=by_y)

    mid = page_w / 2
    width = lambda b: b.bbox[2] - b.bbox[0]  # noqa: E731
    centre = lambda b: (b.bbox[0] + b.bbox[2]) / 2  # noqa: E731
    wide = lambda b: width(b) >= FULL_WIDTH_FRACTION * page_w or abs(centre(b) - mid) < 30  # noqa: E731
    cols = [b for b in body if not wide(b) and width(b) >= 100]
    left = [b for b in cols if centre(b) < mid]
    right = [b for b in cols if centre(b) >= mid]
    if len(left) < 4 or len(right) < 4:
        return sorted(top, key=by_y) + body + sorted(bottom, key=by_y)

    ordered: list[RawBlock] = []
    band: list[RawBlock] = []

    def flush() -> None:
        ordered.extend(sorted((b for b in band if centre(b) < mid), key=by_y))
        ordered.extend(sorted((b for b in band if centre(b) >= mid), key=by_y))
        band.clear()

    for b in body:
        if wide(b):
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return sorted(top, key=by_y) + ordered + sorted(bottom, key=by_y)


def _page_blocks(page: pymupdf.Page, page_no: int) -> tuple[list[RawBlock], dict]:
    height, width = page.rect.height, page.rect.width
    blocks: list[RawBlock] = []

    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] == 0:
            raw = _text_block(page_no, block, height)
            if raw:
                blocks.append(raw)
        elif block["type"] == 1:
            x0, y0, x1, y1 = block["bbox"]
            w, h = x1 - x0, y1 - y0
            if (
                w >= MIN_FIGURE_SIDE_PT
                and h >= MIN_FIGURE_SIDE_PT
                and w * h < MAX_FIGURE_PAGE_FRACTION * width * height
            ):
                blocks.append(RawBlock(page_no, "image", (x0, y0, x1, y1), page_height=height))

    blocks.sort(key=lambda b: (round(b.bbox[1] / 3), b.bbox[0]))
    blocks = _merge_label_blocks(blocks)
    blocks, fractions = _rebuild_math_blocks(blocks)
    blocks, tables, tables_missed = _detect_tables(page, blocks)
    blocks = _reading_order(blocks, width)
    return blocks, {"fractions_rebuilt": fractions, "tables": tables, "tables_missed": tables_missed}


# --------------------------------------------------------------------------- #
# Chapter assembly
# --------------------------------------------------------------------------- #


def _boilerplate_keys(blocks: list[RawBlock], n_pages: int) -> set[str]:
    """Header/footer strings: page numbers, or text repeating across pages."""
    counts: Counter[str] = Counter()
    seen_per_page: set[tuple[int, str]] = set()
    for b in blocks:
        if b.kind != "text" or not _in_margin(b):
            continue
        key = boilerplate_key(b.text or "")
        if (b.page, key) not in seen_per_page:
            seen_per_page.add((b.page, key))
            counts[key] += 1
    threshold = max(2, 0.3 * n_pages)
    return {k for k, c in counts.items() if re.fullmatch(r"#+", k) or c >= threshold}


def _classify(b: RawBlock) -> tuple[str, str | None]:
    """Return (block_type, original_label) for a non-boilerplate block."""
    if b.kind == "image":
        return "figure", None
    if b.kind == "table":
        return "table", None

    text = (b.text or "").strip()
    flat = " ".join(text.split())
    n_lines = text.count("\n") + 1

    if label := _caption_label(flat):
        return "caption", label

    # Chapter/unit opener ("Chapter One", "UNIT 1"): large, but not always bold.
    first_line = text.split("\n")[0].strip()
    if (
        b.size >= 14
        and n_lines <= 6
        and len(flat) <= 160
        and len(first_line) <= 40
        and _CHAPTER_RE.match(first_line)
    ):
        return "heading", first_line

    if b.bold and n_lines <= 3 and len(flat) <= 120:
        if m := _SECTION_RE.match(flat):
            return "heading", m.group(1)
        if m := _EXAMPLE_RE.match(flat):
            return "heading", m.group(1)
        letters = [c for c in flat if c.isalpha()]
        if b.size >= 12 or (len(letters) >= 3 and flat.upper() == flat):
            return "heading", None

    if m := _EQ_NUMBER_RE.search(text):
        if len(flat) <= 200 and n_lines <= 4:
            return "equation", m.group(1)
    long_words = _LONG_WORD_RE.findall(flat)
    if len(flat) <= 160 and (b.math_ratio >= 0.3 or ("=" in flat and len(long_words) <= 2)):
        return "equation", None

    if _LIST_RE.match(text):
        return "list", None

    return "paragraph", None


def _chapter_header(
    blocks: list[tuple[RawBlock, str, str | None]],
) -> tuple[str | None, str | None]:
    """Chapter number/title from the opener ("CHAPTER EIGHT" + large title)."""
    number = title = None
    for idx, (b, kind, label) in enumerate(blocks):
        if kind == "heading" and label and (m := _CHAPTER_RE.match(label)):
            word = m.group(1).upper()
            number = _NUMBER_WORDS.get(word, word if word.isdigit() else None)
            # Title either follows the opener line inside the same block
            # ("Chapter One\nELECTRIC CHARGES\nAND FIELDS") or in the next large block(s).
            inline = " ".join(" ".join((b.text or "").split("\n")[1:]).split())
            parts = [inline] if inline else []
            if not parts:
                below = sorted(
                    (
                        nb
                        for nb, nkind, _ in blocks[max(idx - 4, 0) : idx + 8]
                        if nkind == "heading"
                        and nb is not b
                        and nb.size >= 18
                        and nb.page == b.page
                        and nb.bbox[1] > b.bbox[1]
                    ),
                    key=lambda nb: nb.bbox[1],
                )
                parts = [" ".join((nb.text or "").split()) for nb in below[:3]]
            title = " ".join(parts) or None
            break
    return number, title


def extract_chapter(
    pdf_path: str,
    *,
    source_id: str,
    subject: str,
    class_level: int | None,
    language: str,
    chapter_number_hint: str | None = None,
) -> ChapterExtraction:
    """Extract one chapter PDF; never raises for content problems (returns failed)."""
    doc = pymupdf.open(pdf_path)
    n_pages = doc.page_count
    reasons: list[str] = []

    # Raw text is decoded from private-use codes (cp1252); Symbol-font glyphs in it
    # may stay approximate here, while block text decodes them per font.
    page_texts = [decode_private_use(doc[i].get_text()) for i in range(n_pages)]
    text_raw = "\n\n".join(page_texts)

    all_blocks: list[RawBlock] = []
    totals: Counter[str] = Counter()
    for i in range(n_pages):
        page_blocks, page_stats = _page_blocks(doc[i], i + 1)
        all_blocks.extend(page_blocks)
        totals.update(page_stats)

    boiler = _boilerplate_keys(all_blocks, n_pages)
    content = [
        b
        for b in all_blocks
        if not (b.kind == "text" and _in_margin(b) and boilerplate_key(b.text or "") in boiler)
    ]
    classified = [(b, *_classify(b)) for b in content]

    chapter_number, chapter_title = _chapter_header(classified)
    chapter_number = chapter_number or chapter_number_hint
    if chapter_number is None:
        reasons.append("chapter_number_not_found")
    # Some books set the chapter title inside a graphic, so it is not extractable
    # text; a null title is valid per the contract and only flagged when the
    # chapter number is missing too.
    if chapter_title is None and chapter_number is None:
        reasons.append("chapter_title_not_found")
    if totals["tables_missed"]:
        reasons.append(f"table_not_reconstructed:{totals['tables_missed']}")

    chapter_key = chapter_number if chapter_number else "p0001"
    chapter_id = make_chapter_id(source_id, chapter_key)

    content_blocks = []
    cleaned_parts: list[str] = []
    for order, (b, kind, label) in enumerate(classified):
        cleaned = clean_block_text(b.text, kind)
        if cleaned:
            cleaned_parts.append(cleaned)
        content_blocks.append(
            {
                "block_id": make_block_id(chapter_id, order),
                "block_type": kind,
                "text_raw": b.text,
                "text_cleaned": cleaned,
                "page": b.page,
                "asset_ids": [],
                "order_index": order,
                "original_label": label,
            }
        )

    pages_with_blocks = {b.page for b in content}
    empty_pages = [p for p in range(1, n_pages + 1) if p not in pages_with_blocks]
    if empty_pages:
        reasons.append(f"pages_without_content:{empty_pages}")
    pua = private_use_ratio(text_raw)
    if pua > PUA_REVIEW_THRESHOLD:
        reasons.append(f"possible_garbled_text:private_use_ratio={pua:.4f}")

    if not content_blocks or not text_raw.strip():
        status = "failed"
        reasons.append("no_extractable_text")
        content_blocks = []
    else:
        status = "needs_review" if reasons else "success"

    record = {
        "chapter_id": chapter_id,
        "source_id": source_id,
        "subject": subject,
        "class_level": class_level,
        "language": language,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "page_start": 1,
        "page_end": max(n_pages, 1),
        "text_raw": text_raw,
        "text_cleaned": "\n\n".join(cleaned_parts),
        "content_blocks": content_blocks,
        "asset_ids": [],
        "extraction_status": status,
    }
    kinds = Counter(b["block_type"] for b in content_blocks)
    fragmented = sum(
        1
        for b in content_blocks
        if b["block_type"] == "equation"
        and sum(1 for ln in (b["text_cleaned"] or "").split("\n") if len(ln) <= 2) >= 3
    )
    return ChapterExtraction(
        record=record,
        review_reasons=reasons,
        stats={
            "pages": n_pages,
            "blocks": len(content_blocks),
            "block_types": dict(kinds),
            "fractions_rebuilt": totals["fractions_rebuilt"],
            "tables_reconstructed": totals["tables"],
            "tables_not_reconstructed": totals["tables_missed"],
            "fragmented_equation_blocks": fragmented,
            "chapter_title_missing": chapter_title is None,
        },
    )
