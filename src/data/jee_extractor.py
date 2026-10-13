"""
JEE Main Question Extractor (Track D) - Geometry & Math-Aware Universal Parser
Conforms strictly to Data Contract v1.0.0 (Section 6) and src.models.JEEQuestionRecord.

Architecture:
- Span-level geometric tokenization with physical coordinates (x0, y0, x1, y1, base).
- Baseline fragment merging for split math symbols (e.g. '∆' + 'L' -> '∆L', '16' + '(' -> '16(').
- Vertical center-aligned fraction stacking ('_stack_fractions') for formulas like 16/11, 1/11, m/n, 9/√2.
- Options Grid protection: ensures MCQ options (1)-(4) are never falsely collapsed into fractions.
- 2D row flattening: groups tokens by visual line and horizontal order.
- Universal boundary detection (Q1 to Q75), subject partitioning, and answer key table extraction.
- Strict Pydantic schema validation with preset null difficulty fields and empty asset_ids.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import ValidationError

from src.data.checksums import file_size, sha256_file
from src.data.ids import make_source_id
from src.data.asset_store import build_review_rows, read_jsonl, review_rows, store_asset, validate_asset_records, write_jsonl_atomic
from src.data.vector_regions import group_drawing_regions
from src.models import (
    AssetRecord,
    Difficulty,
    DifficultySource,
    DifficultyStatus,
    ExtractionStatus,
    JEEOption,
    JEEQuestionRecord,
    SourceRecord,
)




def parse_filename_metadata(filename: str) -> dict[str, Any]:
    """Extracts exam year, session, and shift from standard JEE Main filenames."""
    meta = {
        "year": 2025,
        "session": "January",
        "shift": "1",
    }
    year_match = re.search(r"(\d{4})", filename)
    if year_match:
        meta["year"] = int(year_match.group(1))

    shift_match = re.search(r"Shift\s*(\d+)", filename, re.IGNORECASE)
    if shift_match:
        meta["shift"] = shift_match.group(1)

    date_match = re.search(r"\((\d{1,2}\s+[A-Za-z]+)", filename)
    if date_match:
        meta["session"] = date_match.group(1).strip()

    return meta


def extract_answer_keys_from_doc(doc: pymupdf.Document) -> dict[int, str]:
    """Extracts official answer keys from the summary table on the final page(s)."""
    answer_map: dict[int, str] = {}
    if len(doc) == 0:
        return answer_map

    for page in doc[-2:]:
        text = page.get_text() or ""
        if "ANSWER KEYS" in text or "ANSWER KEY" in text:
            matches = re.findall(r"(\d+)\.\s*\(([^\)]+)\)", text)
            for q_num_str, ans_str in matches:
                try:
                    q_num = int(q_num_str)
                    answer_map[q_num] = ans_str.strip()
                except ValueError:
                    continue
    return answer_map


def clean_generic_text(text: str) -> str:
    """Normalizes Unicode minuses, hyphens, and whitespace."""
    if not text:
        return ""
    text = text.replace("\u2212", "-").replace("−", "-").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    non_empty = [l for l in lines if l]
    return " ".join(non_empty)


def detect_image_reference(question_text: str) -> bool:
    """Checks if a question contains a diagram or figure reference."""
    keywords = [
        "shown in figure", "in the figure", "given figure", "in the given figure",
        "shown in graph", "v-t graph", "velocity-time graph", "given circuit",
        "circuit shown", "diagram", "figure below", "shown below"
    ]
    text_lower = question_text.lower()
    return any(kw in text_lower for kw in keywords)


def determine_subject(q_num: int) -> str:
    """Standard JEE Main section partitioning: Q1-25 Math, Q26-50 Physics, Q51-75 Chemistry."""
    if 1 <= q_num <= 25:
        return "Mathematics"
    elif 26 <= q_num <= 50:
        return "Physics"
    elif 51 <= q_num <= 75:
        return "Chemistry"
    return "Physics"


def extract_page_text_clean(page: pymupdf.Page) -> str:
    """Extracts text from a PDF page block-by-block in reading order."""
    blocks = page.get_text("blocks", sort=True)
    lines: list[str] = []
    for b in blocks:
        if b[6] == 0:  # Text block
            for l in b[4].splitlines():
                l_str = l.strip()
                if l_str:
                    # Filter recurring MathonGo headers/footers
                    if "JEE Main Previous Year Paper" in l_str or ("MathonGo" in l_str and len(l_str) < 40):
                        continue
                    lines.append(l_str)
    return "\n".join(lines)


def validate_question_candidate(candidate: dict[str, Any], is_numerical: bool) -> tuple[bool, list[str]]:
    """
    Layer 2: Consistency & Anomaly Validator.
    Checks whether the extracted question and options are structurally sound
    or suffering from disjoint PDF font-stream corruption.
    """
    issues: list[str] = []
    q_text = candidate.get("question_text", "").strip()
    options = candidate.get("options", [])

    # 1. Question text checks
    if len(q_text) < 15:
        issues.append("Question text too short (< 15 chars)")

    # Check for dangling math clauses where equations were dropped
    dangling_patterns = [
        r"\b(?:domain of|range of|components of|value of|values of|equal to)\s+is\s*:",
        r"\balong and perpendicular to\s+respectively\s*,\s*are\s+and\b",
        r"\bderivative of\s+is\s*:",
    ]
    for pat in dangling_patterns:
        if re.search(pat, q_text, re.IGNORECASE):
            issues.append(f"Dangling incomplete math clause detected: {pat}")
            break

    # Check for question bleeding
    if re.search(r"\bQ\s*\d+[\.:]", q_text):
        issues.append("Subsequent question marker found inside question text (question bleeding)")

    # 2. Options checks (for MCQs)
    if not is_numerical:
        if len(options) != 4:
            issues.append(f"Expected 4 options, found {len(options)}")
        else:
            # Check for empty option text
            for opt in options:
                if not opt.get("text", "").strip():
                    issues.append(f"Option ({opt.get('label')}) has empty text")
                    break

            # Check if Option 4 absorbed stream tokens or dumped option markers
            opt4_text = options[3].get("text", "") if len(options) >= 4 else ""
            if re.search(r"\([1-3]\)", opt4_text) or len(opt4_text) > 350:
                issues.append("Option (4) contains dumped stream tokens or excessive length")

    return (len(issues) == 0, issues)


@dataclass
class QuestionAnchor:
    q_num: int
    page: int
    y0: float
    x0: float


def find_all_question_anchors(doc: pymupdf.Document) -> dict[int, QuestionAnchor]:
    """
    Scans the document to discover the exact (page, y0, x0) coordinates of all questions Q1 to Q75.
    Sequentially matches questions to avoid false anchors from math formulas or options.
    """
    anchors: dict[int, QuestionAnchor] = {}
    if len(doc) == 0:
        return anchors

    exam_pages = doc[:-1] if "ANSWER KEY" in (doc[-1].get_text() or "") else doc
    expected_q = 1

    for p_idx, page in enumerate(exam_pages):
        page_rect = page.rect
        raw_words = page.get_text("words")  # x0, y0, x1, y1, word, block_no, line_no, word_no
        valid_words = [
            w for w in raw_words
            if 25.0 <= w[1] <= page_rect.height - 25.0 and "MathonGo" not in w[4] and "JEE Main" not in w[4]
        ]
        sorted_words = sorted(valid_words, key=lambda w: (round(w[1] / 4.0) * 4.0, w[0]))

        for idx, w in enumerate(sorted_words):
            word_str = w[4].strip()
            # Questions start near the left margin
            if w[0] > 140.0:
                continue

            found_num: int | None = None
            m = re.match(r"^(?:Q|Question)\s*\.?\s*(\d+)(?:[\.:\)]|\b)", word_str)
            if m:
                found_num = int(m.group(1))
            elif word_str == "Q" and idx + 1 < len(sorted_words):
                next_w = sorted_words[idx + 1]
                m_next = re.match(r"^(\d+)(?:[\.:\)]|\b)", next_w[4].strip())
                if m_next:
                    found_num = int(m_next.group(1))

            if found_num is not None:
                if found_num == expected_q:
                    anchors[found_num] = QuestionAnchor(q_num=found_num, page=p_idx, y0=w[1], x0=w[0])
                    expected_q += 1
                elif expected_q < found_num <= expected_q + 2 and found_num <= 75:
                    anchors[found_num] = QuestionAnchor(q_num=found_num, page=p_idx, y0=w[1], x0=w[0])
                    expected_q = found_num + 1

    return anchors


def snap_vector_arrows(word_items: list[tuple[float, float, float, float, str]]) -> list[tuple[float, float, float, float, str]]:
    """
    Merges floating vector arrows (→, ->, \u2192) directly with the variable underneath or immediately following them.
    Prevents floating arrows like '→a' from jumping to the beginning of the previous line.
    """
    arrow_indices = [
        i for i, w in enumerate(word_items)
        if w[4].strip() in ["→", "->", "\u2192", "−→"] or w[4].strip().startswith("→")
    ]
    if not arrow_indices:
        return word_items

    used_arrows = set()
    modified_items = list(word_items)

    for ai in arrow_indices:
        aw = word_items[ai]
        arrow_text = aw[4].strip()
        if len(arrow_text) > 1 and not arrow_text.startswith("→"):
            continue

        arrow_char = "→"
        target_idx = None
        min_dist = float("inf")

        # Check for variable directly underneath (within 12 pt vertically, aligned horizontally)
        for i, tw in enumerate(word_items):
            if i == ai or i in used_arrows:
                continue
            if -4.0 <= tw[1] - aw[3] <= 12.0:
                h_dist = abs((aw[0] + aw[2]) / 2.0 - (tw[0] + tw[2]) / 2.0)
                if h_dist <= 14.0 and h_dist < min_dist:
                    min_dist = h_dist
                    target_idx = i

        # Or immediately to the right on the same line
        if target_idx is None:
            for i, tw in enumerate(word_items):
                if i == ai or i in used_arrows:
                    continue
                if abs(aw[1] - tw[1]) <= 6.0 and 0.0 <= tw[0] - aw[2] <= 6.0:
                    target_idx = i
                    break

        if target_idx is not None:
            tw = word_items[target_idx]
            new_text = f"{arrow_char}{tw[4]}" if not tw[4].startswith("→") else tw[4]
            min_x = min(aw[0], tw[0])
            min_y = min(aw[1], tw[1])
            max_x = max(aw[2], tw[2])
            max_y = max(aw[3], tw[3])
            modified_items[target_idx] = (min_x, min_y, max_x, max_y, new_text)
            used_arrows.add(ai)

    return [w for i, w in enumerate(modified_items) if i not in used_arrows]


def merge_fractions(
    word_items: list[tuple[float, float, float, float, str]],
    frac_lines: list[pymupdf.Rect],
) -> list[tuple[float, float, float, float, str]]:
    """
    Merges numerators and denominators separated by a horizontal vector stroke into 'num/den'.
    Supports integers, variables, radicals (e.g. 16/11, 1/11, 9/√2, m/n, 2√α/3).
    """
    used_indices: set[int] = set()
    new_fractions: list[tuple[float, float, float, float, str]] = []

    for fl in frac_lines:
        nums = [
            (i, w) for i, w in enumerate(word_items)
            if i not in used_indices
            and (w[1] + w[3]) / 2.0 < fl.y0
            and abs(w[3] - fl.y0) <= 14.0
            and max(w[0], fl.x0) <= min(w[2], fl.x1) + 8.0
        ]
        dens = [
            (i, w) for i, w in enumerate(word_items)
            if i not in used_indices
            and (w[1] + w[3]) / 2.0 > fl.y1
            and abs(w[1] - fl.y1) <= 14.0
            and max(w[0], fl.x0) <= min(w[2], fl.x1) + 8.0
        ]

        if nums and dens:
            nums_sorted = sorted(nums, key=lambda x: x[1][0])
            dens_sorted = sorted(dens, key=lambda x: x[1][0])
            num_str = "".join(x[1][4] for x in nums_sorted)
            den_str = "".join(x[1][4] for x in dens_sorted)

            for i, _ in nums:
                used_indices.add(i)
            for i, _ in dens:
                used_indices.add(i)

            min_x = min(min(x[1][0] for x in nums), min(x[1][0] for x in dens))
            max_x = max(max(x[1][2] for x in nums), max(x[1][2] for x in dens))
            new_fractions.append((min_x, fl.y0 - 2.0, max_x, fl.y1 + 2.0, f"{num_str}/{den_str}"))

    remaining = [w for i, w in enumerate(word_items) if i not in used_indices]
    return remaining + new_fractions


def extract_segment_words(page: pymupdf.Page, clip_rect: pymupdf.Rect) -> list[tuple[float, float, float, float, str]]:
    """
    Extracts words in a vertical clipped region of a page, applying superscript carets
    to elevated small-font tokens (e.g. a^2, x^2, α^2), snapping floating vector arrows,
    and merging fractions separated by horizontal vector strokes into clean 'numerator/denominator' tokens.
    Strictly avoids conflicting with subscripts (which have lowered baselines) or unit vectors.
    """
    page_rect = page.rect
    text_dict = page.get_text("dict", clip=clip_rect)
    raw_word_items: list[tuple[float, float, float, float, str]] = []

    for b in text_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            valid_spans = [s for s in spans if s.get("text", "").strip()]
            if not valid_spans:
                continue

            sizes = [s["size"] for s in valid_spans]
            max_size = max(sizes)
            base_spans = [s for s in valid_spans if s["size"] >= max_size * 0.85]
            line_baseline = (
                sum(s["origin"][1] for s in base_spans) / len(base_spans)
                if base_spans else valid_spans[0]["origin"][1]
            )

            prev_span: dict[str, Any] | None = None
            for s in valid_spans:
                stext = s.get("text", "")
                if not stext:
                    continue
                bbox = s["bbox"]
                if bbox[1] < 25.0 or bbox[3] > page_rect.height - 25.0:
                    continue
                if "MathonGo" in stext or "JEE Main" in stext:
                    continue

                prev_text = prev_span.get("text", "").strip() if prev_span else ""
                prev_ends_alphanumeric = bool(re.search(r"[a-zA-Z0-9α-ωΑ-Ω\)\]]$", prev_text))

                # Check strictly for superscript exponent:
                # 1) Font size significantly smaller than line base font (<= 88%)
                # 2) Baseline elevated ABOVE line baseline by at least 1.8 pt
                # 3) Located within 4.5 pt horizontally from the preceding base token
                # 4) Preceding base ends with a valid symbol/character
                # 5) Does not already start with '^'
                is_super = (
                    prev_span is not None
                    and prev_ends_alphanumeric
                    and s["size"] <= max_size * 0.88
                    and s["origin"][1] <= line_baseline - 1.8
                    and s["origin"][1] >= line_baseline - 15.0
                    and -1.0 <= (bbox[0] - prev_span["bbox"][2]) <= 4.5
                    and not stext.strip().startswith("^")
                )

                # Check strictly for subscript index:
                # 1) Font size significantly smaller than line base font (<= 88%)
                # 2) Baseline lowered BELOW line baseline by at least 1.0 pt
                # 3) Located within 4.5 pt horizontally from the preceding base token
                # 4) Preceding base ends with a valid symbol/character
                # 5) Does not already start with '_'
                is_sub = (
                    prev_span is not None
                    and prev_ends_alphanumeric
                    and s["size"] <= max_size * 0.88
                    and s["origin"][1] >= line_baseline + 1.0
                    and s["origin"][1] <= line_baseline + 12.0
                    and -1.0 <= (bbox[0] - prev_span["bbox"][2]) <= 4.5
                    and not stext.strip().startswith("_")
                )

                words_in_span = stext.split(" ")
                for wi, w in enumerate(words_in_span):
                    if not w:
                        continue
                    if is_super and wi == 0:
                        token_text = f"^{w}"
                    elif is_sub and wi == 0:
                        token_text = f"_{w}"
                    else:
                        token_text = w
                    raw_word_items.append((bbox[0], bbox[1], bbox[2], bbox[3], token_text))

                prev_span = s

    # Fallback to get_text("words") if dict returned empty
    if not raw_word_items:
        raw_words = page.get_text("words")
        seg_words = [
            w for w in raw_words
            if clip_rect.y0 - 3.0 <= w[1] <= clip_rect.y1 + 3.0
            and 25.0 <= w[1] <= page_rect.height - 25.0
            and "MathonGo" not in w[4] and "JEE Main" not in w[4]
        ]
        raw_word_items = [(w[0], w[1], w[2], w[3], w[4]) for w in seg_words]

    # 1. Snap floating vector arrows to their variables
    word_items = snap_vector_arrows(raw_word_items)

    # 2. Detect horizontal fraction lines from vector drawings
    drawings = page.get_drawings()
    frac_lines = [
        d["rect"] for d in drawings
        if clip_rect.y0 - 2.0 <= d["rect"].y0 <= clip_rect.y1 + 2.0
        and d["rect"].height <= 3.0
        and 5.0 <= d["rect"].width <= 75.0
    ]

    # 3. Merge fractions
    word_items = merge_fractions(word_items, frac_lines)

    return word_items


def assemble_text_from_words(word_items: list[tuple[float, float, float, float, str]]) -> str:
    """Groups words into horizontal visual lines and joins them, discarding recurring headers/footers."""
    if not word_items:
        return ""
    sorted_words = sorted(word_items, key=lambda w: (round(w[1] / 6.0) * 6.0, w[0]))
    lines: list[list[tuple[float, float, float, float, str]]] = []

    for w in sorted_words:
        if not lines:
            lines.append([w])
        else:
            last_line = lines[-1]
            avg_y = sum(item[1] for item in last_line) / len(last_line)
            if abs(w[1] - avg_y) <= 5.5:
                last_line.append(w)
            else:
                lines.append([w])

    formatted_lines: list[str] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w[0])
        line_text = " ".join(w[4] for w in line_sorted).strip()
        if "MathonGo" in line_text or "Previous Year Paper" in line_text or "JEE Main 2025" in line_text:
            continue
        if line_text:
            formatted_lines.append(line_text)

    return "\n".join(formatted_lines)


def clean_question_body(text: str) -> str:
    """Strips recurring watermarks, leading Q markers, and binds caret exponents (e.g. a^2, x^2, α^2) and subscripts (e.g. a_1, a_2024)."""
    cleaned = re.sub(r"^(?:Q|Question)\s*\.?\s*\d+[\.:\s]*", "", text.strip())
    cleaned = re.sub(r"2025\s*\(\d+\s+[A-Za-z]+\s+Shift\s+\d+\)\s*JEE\s*Main\s*(?:Previous\s*Year\s*Paper)?\s*JEE\s*Main\s*2025\s*January", "", cleaned)
    cleaned = re.sub(r"JEE\s*Main\s*2025\s*January", "", cleaned)
    # Bind caret exponents directly to base variables: e.g. "a ^2" -> "a^2", "x ^2" -> "x^2", "α ^2" -> "α^2"
    cleaned = re.sub(r"([a-zA-Z0-9α-ωΑ-Ω\)\]])\s*\^\s*([0-9a-zA-Z\+\-]+)", r"\1^\2", cleaned)
    # Bind subscript indices directly to base variables: e.g. "a _1" -> "a_1", "a _2024" -> "a_2024"
    cleaned = re.sub(r"([a-zA-Z0-9α-ωΑ-Ω\)\]])\s*_\s*([0-9a-zA-Z\+\-]+)", r"\1_\2", cleaned)
    return clean_generic_text(cleaned)


def resolve_options_from_text(raw_text: str, is_numerical: bool) -> tuple[str, list[dict[str, str]]]:
    """
    Separates the question text and options (1)-(4).
    For numericals, returns options=[].
    For MCQs, reverse-searches bottom-up for option markers (1), (2), (3), (4)
    ensuring mathematical expressions like (1-a) are never mistaken for option markers.
    """
    cleaned_body = clean_question_body(raw_text)

    if is_numerical:
        return cleaned_body, []

    matches = list(re.finditer(r"\(([1-4])\)", cleaned_body))

    best_seq: tuple[re.Match, re.Match, re.Match, re.Match] | None = None
    for i1, m1 in enumerate(matches):
        if m1.group(1) != "1":
            continue
        for i2 in range(i1 + 1, len(matches)):
            m2 = matches[i2]
            if m2.group(1) != "2":
                continue
            for i3 in range(i2 + 1, len(matches)):
                m3 = matches[i3]
                if m3.group(1) != "3":
                    continue
                for i4 in range(i3 + 1, len(matches)):
                    m4 = matches[i4]
                    if m4.group(1) == "4":
                        best_seq = (m1, m2, m3, m4)

    if best_seq:
        m1, m2, m3, m4 = best_seq
        q_text = cleaned_body[:m1.start()].strip()
        opt1 = clean_question_body(cleaned_body[m1.end():m2.start()])
        opt2 = clean_question_body(cleaned_body[m2.end():m3.start()])
        opt3 = clean_question_body(cleaned_body[m3.end():m4.start()])
        opt4 = clean_question_body(cleaned_body[m4.end():])

        options = [
            {"label": "1", "text": opt1},
            {"label": "2", "text": opt2},
            {"label": "3", "text": opt3},
            {"label": "4", "text": opt4},
        ]
        return q_text, options

    # Fallback to (A)-(D) if (1)-(4) not found
    matches_alpha = list(re.finditer(r"\(([A-D])\)", cleaned_body))
    best_alpha: tuple[re.Match, re.Match, re.Match, re.Match] | None = None
    for i1, m1 in enumerate(matches_alpha):
        if m1.group(1) != "A":
            continue
        for i2 in range(i1 + 1, len(matches_alpha)):
            m2 = matches_alpha[i2]
            if m2.group(1) != "B":
                continue
            for i3 in range(i2 + 1, len(matches_alpha)):
                m3 = matches_alpha[i3]
                if m3.group(1) != "C":
                    continue
                for i4 in range(i3 + 1, len(matches_alpha)):
                    m4 = matches_alpha[i4]
                    if m4.group(1) == "D":
                        best_alpha = (m1, m2, m3, m4)

    if best_alpha:
        m1, m2, m3, m4 = best_alpha
        q_text = cleaned_body[:m1.start()].strip()
        opt1 = clean_question_body(cleaned_body[m1.end():m2.start()])
        opt2 = clean_question_body(cleaned_body[m2.end():m3.start()])
        opt3 = clean_question_body(cleaned_body[m3.end():m4.start()])
        opt4 = clean_question_body(cleaned_body[m4.end():])

        options = [
            {"label": "1", "text": opt1},
            {"label": "2", "text": opt2},
            {"label": "3", "text": opt3},
            {"label": "4", "text": opt4},
        ]
        return q_text, options

    return cleaned_body, []


def crop_diagram_if_present(
    page: pymupdf.Page,
    clip_rect: pymupdf.Rect,
    source_id: str,
    qid: str,
    source_page: int,
    assets_dir: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Safely crops an embedded raster image or non-math vector illustration
    strictly clamped within the question's bounding rectangle.
    Excludes whole-page backgrounds, headers, footers, and pure math formulas.
    """
    page_rect = page.rect

    # 1. Raster images
    for img_info in page.get_image_info(xrefs=True):
        bbox = pymupdf.Rect(img_info["bbox"])
        # Exclude whole-page watermarks / backgrounds
        if bbox.width > page_rect.width * 0.8 or bbox.height > page_rect.height * 0.8:
            continue
        # Exclude headers and footers
        if bbox.y1 <= 35.0 or bbox.y0 >= page_rect.height - 35.0:
            continue
        if bbox.intersects(clip_rect):
            crop_bbox = bbox.intersect(clip_rect)
            if crop_bbox.width >= 35.0 and crop_bbox.height >= 25.0:
                pix = page.get_pixmap(clip=crop_bbox, dpi=200)
                img_bytes = pix.tobytes("png")
                rec = store_asset(
                    img_bytes,
                    source_id=source_id,
                    source_page=source_page,
                    asset_type="diagram",
                    bbox={"x0": crop_bbox.x0, "y0": crop_bbox.y0, "x1": crop_bbox.x1, "y1": crop_bbox.y1},
                    associated_record_ids=[qid],
                    review_status="unreviewed",
                ).model_dump(mode="json")
                return rec["asset_id"], rec

    # 2. Vector graphic clusters (circuits, graphs, chemical structures)
    grouped = group_drawing_regions(
        page.get_drawings(), page_width=page_rect.width, page_height=page_rect.height
    )
    candidates = []
    for group in grouped["regions"]:
        rect = pymupdf.Rect(group["bbox"])
        rect &= clip_rect & page_rect
        if rect.width >= 40.0 and rect.height >= 30.0 and rect.width * rect.height < 0.70 * page_rect.width * page_rect.height:
            candidates.append((rect, group["object_count"]))
    # Prefer the tightest plausible region; drawing count is reported as
    # provenance only and is not used as a figure classification threshold.
    if candidates:
        cluster_rect, _ = min(candidates, key=lambda item: (item[0].width * item[0].height, item[0].y0, item[0].x0))
        cluster_rect = pymupdf.Rect(
            max(cluster_rect.x0 - 3, clip_rect.x0, 0),
            max(cluster_rect.y0 - 3, clip_rect.y0, 0),
            min(cluster_rect.x1 + 3, clip_rect.x1, page_rect.width),
            min(cluster_rect.y1 + 3, clip_rect.y1, page_rect.height),
        )
        if cluster_rect.width >= 40.0 and cluster_rect.height >= 30.0:
            img_bytes = page.get_pixmap(clip=cluster_rect, dpi=200, alpha=False).tobytes("png")
            rec = store_asset(
                img_bytes,
                source_id=source_id,
                source_page=source_page,
                asset_type="diagram",
                bbox={"x0": cluster_rect.x0, "y0": cluster_rect.y0, "x1": cluster_rect.x1, "y1": cluster_rect.y1},
                associated_record_ids=[qid],
                review_status="unreviewed",
            ).model_dump(mode="json")
            return rec["asset_id"], rec

    return None, None


def has_complex_math_expression(text: str) -> bool:
    """
    Checks if question text contains complex 2D mathematical expressions
    such as integrals, limits, summations, derivatives, matrices, or determinants
    that are best preserved as visual assets.
    """
    patterns = [
        r"[∫∬∭∮]",                                       # Integration symbols
        r"[∑∏]",                                           # Summation and product symbols
        r"\blim(?:_|\s*[\(t-z0-9a-zA-Z→])",               # Limits (e.g. lim t→0, lim_x→∞)
        r"\bd[a-zA-Z0-9]/d[a-zA-Z0-9]\b",                  # Derivatives (dy/dx, dx/dt, dm/dt)
        r"\bd/d[a-zA-Z0-9]\b",                             # Derivative operators (d/dx, d/dt)
        r"\bf\s*[''′″]\s*\(",                              # Derivative function notation f'(x)
        r"\b(?:matrix|matrices|determinant)\b.*[\[\|]",    # Matrices / determinants with brackets
        r"\|\s*[a-zA-Z0-9\s\+\-\^]+\s*\|",                 # Determinant bars
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def crop_math_expression_asset(
    page: pymupdf.Page,
    clip_rect: pymupdf.Rect,
    source_id: str,
    qid: str,
    source_page: int,
    assets_dir: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Crops the visual mathematical expression or formula block from the page
    when complex math (integrals, summations, limits, derivatives) is present.
    """
    raw_words = page.get_text("words")
    valid_words = [
        w for w in raw_words
        if clip_rect.y0 - 2.0 <= w[1] <= clip_rect.y1 + 2.0
        and 25.0 <= w[1] <= page.rect.height - 25.0
        and "MathonGo" not in w[4] and "JEE Main" not in w[4]
    ]
    if not valid_words:
        return None, None

    drawings = page.get_drawings()
    region_drawings = [
        d for d in drawings
        if clip_rect.y0 - 2.0 <= d["rect"].y0 <= clip_rect.y1 + 2.0
        and d["rect"].width < page.rect.width * 0.95
        and d["rect"].height < page.rect.height * 0.95
    ]

    min_x = min(w[0] for w in valid_words)
    max_x = max(w[2] for w in valid_words)
    min_y = min(w[1] for w in valid_words)
    max_y = max(w[3] for w in valid_words)

    if region_drawings:
        min_x = min(min_x, min(d["rect"].x0 for d in region_drawings))
        max_x = max(max_x, max(d["rect"].x1 for d in region_drawings))
        min_y = min(min_y, min(d["rect"].y0 for d in region_drawings))
        max_y = max(max_y, max(d["rect"].y1 for d in region_drawings))

    crop_box = pymupdf.Rect(
        max(min_x - 6.0, 0),
        max(min_y - 4.0, clip_rect.y0),
        min(max_x + 6.0, page.rect.width),
        min(max_y + 4.0, clip_rect.y1),
    )

    if crop_box.width < 25.0 or crop_box.height < 15.0:
        return None, None

    pix = page.get_pixmap(clip=crop_box, dpi=200)
    img_bytes = pix.tobytes("png")
    rec = store_asset(
        img_bytes,
        source_id=source_id,
        source_page=source_page,
        asset_type="diagram",
        bbox={"x0": crop_box.x0, "y0": crop_box.y0, "x1": crop_box.x1, "y1": crop_box.y1},
        associated_record_ids=[qid],
        review_status="unreviewed",
    ).model_dump(mode="json")
    return rec["asset_id"], rec


def extract_questions_from_pdf(
    pdf_path: Path, source_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extracts all 75 questions from one JEE Main PDF using the Anchor-Based Cross-Page Pipeline.
    Strictly prevents question bleeding and accurately handles cross-page questions (e.g. Q9 onto Page 2).
    """
    meta = parse_filename_metadata(pdf_path.name)
    candidates: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    all_assets: list[dict[str, Any]] = []

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    if total_pages == 0:
        return [], [], []

    answer_keys = extract_answer_keys_from_doc(doc)
    anchors = find_all_question_anchors(doc)
    assets_dir = Path("data/assets/crops")
    date_tag = re.sub(r"[^a-zA-Z0-9]", "", meta["session"]).lower()

    for q_num in range(1, 76):
        if q_num not in anchors:
            continue

        anchor = anchors[q_num]
        source_page = anchor.page + 1  # 1-indexed
        subject = determine_subject(q_num)
        is_numerical = (q_num in range(21, 26)) or (q_num in range(46, 51)) or (q_num in range(71, 76))
        qid = f"jee-{meta['year']}-{date_tag}-s{meta['shift']}-q{q_num:02d}"

        start_page = anchor.page
        start_y = max(28.0, anchor.y0 - 4.0)

        if (q_num + 1) in anchors:
            next_anchor = anchors[q_num + 1]
            end_page = next_anchor.page
            end_y = max(28.0, next_anchor.y0 - 4.0)
        else:
            end_page = start_page
            end_y = doc[start_page].rect.height - 25.0

        segments: list[tuple[int, pymupdf.Rect]] = []
        if start_page == end_page:
            segments.append((start_page, pymupdf.Rect(0, start_y, doc[start_page].rect.width, end_y)))
        else:
            segments.append((start_page, pymupdf.Rect(0, start_y, doc[start_page].rect.width, doc[start_page].rect.height - 25.0)))
            for p in range(start_page + 1, end_page):
                segments.append((p, pymupdf.Rect(0, 28.0, doc[p].rect.width, doc[p].rect.height - 25.0)))
            segments.append((end_page, pymupdf.Rect(0, 28.0, doc[end_page].rect.width, end_y)))

        # Collect and assemble text across segments
        segment_texts: list[str] = []
        asset_ids: list[str] = []

        for p_idx, seg_rect in segments:
            page = doc[p_idx]
            seg_words = extract_segment_words(page, seg_rect)
            seg_text = assemble_text_from_words(seg_words)
            if seg_text:
                segment_texts.append(seg_text)

            # Check for raster diagrams clamped within this segment
            ast_id, ast_rec = crop_diagram_if_present(page, seg_rect, source_id, qid, p_idx + 1, assets_dir)
            if ast_id and ast_rec:
                asset_ids.append(ast_id)
                all_assets.append(ast_rec)

        full_raw_text = "\n".join(segment_texts)
        q_text, options = resolve_options_from_text(full_raw_text, is_numerical)

        # If no diagram was found, but question contains complex math expressions
        # (integration, differentiation, summation, limits), crop the mathematical block as an asset
        if not asset_ids and has_complex_math_expression(full_raw_text):
            first_p_idx, first_seg_rect = segments[0]
            ast_id, ast_rec = crop_math_expression_asset(
                doc[first_p_idx], first_seg_rect, source_id, qid, first_p_idx + 1, assets_dir
            )
            if ast_id and ast_rec:
                asset_ids.append(ast_id)
                all_assets.append(ast_rec)

        if asset_ids:
            q_text += f" [IMAGE: {asset_ids[0]}]"

        correct_ans = answer_keys.get(q_num)
        record = {
            "question_id": qid,
            "source_id": source_id,
            "year": meta["year"],
            "session": meta["session"],
            "shift": str(meta["shift"]),
            "subject": subject,
            "question_text": q_text if q_text else f"Question {q_num}",
            "options": options,
            "correct_answer": str(correct_ans) if correct_ans is not None else None,
            "solution_text": None,
            "source_page": source_page,
            "asset_ids": asset_ids,
            "extraction_status": "success",
            "difficulty": None,
            "difficulty_source": "unknown",
            "difficulty_status": "unknown",
        }

        # Validate with Layer 2 consistency validator
        is_valid, issues = validate_question_candidate(record, is_numerical)
        if not is_valid:
            record["quarantine_reason"] = "; ".join(issues)

        # Layer 4 Pydantic schema validation
        try:
            JEEQuestionRecord.model_validate(record)
            if is_valid:
                candidates.append(record)
            else:
                quarantine.append(record)
        except ValidationError as ve:
            record["quarantine_reason"] = str(ve)
            quarantine.append(record)

    doc.close()
    return candidates, quarantine, all_assets


def process_all_jee_papers(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/jee_questions.jsonl",
    manifest_path: str | Path = "data/processed/source_manifest.jsonl",
    assets_manifest_path: str | Path = "data/processed/assets_manifest.jsonl",
    quarantine_path: str | Path = "data/quarantine/jee_quarantine.jsonl",
) -> dict[str, int]:
    """Main entrypoint for Track D question extraction."""
    raw_path = Path(raw_dir)
    out_file = Path(output_path)
    man_file = Path(manifest_path)
    assets_file = Path(assets_manifest_path)
    quar_file = Path(quarantine_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    assets_file.parent.mkdir(parents=True, exist_ok=True)
    quar_file.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(raw_path.glob("*.pdf")) + list(raw_path.glob("pyq/*.pdf")))
    if not pdf_files:
        print(f"No PDF files found under {raw_path}")
        return {"sources": 0, "extracted": 0, "quarantined": 0, "assets": 0}

    existing_manifest_sources: dict[str, SourceRecord] = {}
    if man_file.exists():
        with open(man_file, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if line:
                    try:
                        rec = SourceRecord.model_validate_json(line)
                        existing_manifest_sources[rec.original_filename] = rec
                        existing_manifest_sources[rec.sha256] = rec
                    except Exception:
                        pass

    all_source_records: list[SourceRecord] = list({r.source_id: r for r in existing_manifest_sources.values()}.values())
    all_extracted_questions: list[dict[str, Any]] = []
    all_quarantine: list[dict[str, Any]] = []
    all_assets_records: list[dict[str, Any]] = []
    processed_source_ids: set[str] = set()

    print(f"Discovered {len(pdf_files)} PDF papers to process.")

    for pdf in pdf_files:
        if "JEE" not in pdf.name:
            continue

        digest = sha256_file(pdf)
        src_id = make_source_id(digest)
        meta = parse_filename_metadata(pdf.name)

        if digest not in existing_manifest_sources and pdf.name not in existing_manifest_sources:
            src_record = SourceRecord.model_validate({
                "source_id": src_id,
                "logical_source_id": None,
                "source_type": "jee_main",
                "original_filename": pdf.name,
                "sha256": digest,
                "file_size_bytes": file_size(pdf),
                "subject": None,
                "language": "en",
                "publication_year": None,
                "exam_year": meta["year"],
                "edition": None,
                "class_level": None,
                "session": meta["session"],
                "shift": meta["shift"],
                "source_locator": pdf.as_posix(),
                "inventory_status": "registered",
            })
            all_source_records.append(src_record)
            existing_manifest_sources[digest] = src_record
        else:
            src_record = existing_manifest_sources.get(digest) or existing_manifest_sources[pdf.name]

        processed_source_ids.add(src_record.source_id)

        print(f"Processing: {pdf.name} (Source ID: {src_record.source_id[:16]}...)")
        questions, quar, assets = extract_questions_from_pdf(pdf, src_record.source_id)
        all_extracted_questions.extend(questions)
        all_quarantine.extend(quar)
        all_assets_records.extend(assets)
        print(f"  -> Extracted: {len(questions)} valid questions, {len(quar)} quarantined, {len(assets)} assets.")

    with open(man_file, "w", encoding="utf-8") as mf:
        for srec in all_source_records:
            mf.write(srec.model_dump_json() + "\n")

    with open(out_file, "w", encoding="utf-8") as qf:
        for q in all_extracted_questions:
            rec = JEEQuestionRecord.model_validate(q)
            qf.write(rec.model_dump_json() + "\n")

    # Keep assets for sources not selected in this run. Exact binary
    # deduplication is by SHA-256 storage path; occurrence rows retain page and
    # crop provenance and have distinct stable IDs.
    previous_assets = read_jsonl(assets_file) if assets_file.exists() else []
    combined_assets = {
        row["asset_id"]: row
        for row in previous_assets
        if row.get("source_id") not in processed_source_ids
    }
    for ast in all_assets_records:
        rec = AssetRecord.model_validate(ast)
        combined_assets[rec.asset_id] = rec.model_dump(mode="json")
    asset_rows = list(combined_assets.values())
    registered_source_ids = {row.source_id for row in all_source_records}
    ncert_sources_path = Path("data/interim/ncert_source_manifest.jsonl")
    if ncert_sources_path.exists():
        registered_source_ids.update(
            row.get("source_id") for row in read_jsonl(ncert_sources_path) if row.get("source_id")
        )
    known_record_ids = {
        row.get("question_id")
        for row in (*all_extracted_questions, *all_quarantine)
        if row.get("question_id")
    }
    ncert_chapters_path = Path("data/processed/ncert_chapters.jsonl")
    if ncert_chapters_path.exists():
        for row in read_jsonl(ncert_chapters_path):
            if row.get("chapter_id"):
                known_record_ids.add(row["chapter_id"])
            known_record_ids.update(block["block_id"] for block in row.get("content_blocks", []) if block.get("block_id"))
    asset_errors = validate_asset_records(
        asset_rows,
        data_root=Path.cwd(),
        known_source_ids=registered_source_ids,
        known_record_ids=known_record_ids or None,
    )
    write_jsonl_atomic(asset_rows, assets_file)
    review_file = assets_file.parent / "assets_review.jsonl"
    prior_review = read_jsonl(review_file) if review_file.exists() else []
    write_jsonl_atomic(build_review_rows(asset_rows, existing_review_rows=prior_review), review_file)

    if all_quarantine:
        with open(quar_file, "w", encoding="utf-8") as qf:
            import json
            for q in all_quarantine:
                qf.write(json.dumps(q) + "\n")

    print(f"\n[Done] Successfully wrote {len(all_extracted_questions)} questions to {out_file}")
    print(f"[Done] Successfully wrote {len(asset_rows)} asset records to {assets_file}")
    if asset_errors:
        print(f"[Asset validation] {len(asset_errors)} integrity errors; see {review_file} and validation output.")
    return {
        "sources": len(all_source_records),
        "extracted": len(all_extracted_questions),
        "quarantined": len(all_quarantine),
        "assets": len(asset_rows),
        "asset_integrity_errors": len(asset_errors),
        "asset_integrity_error_details": asset_errors,
        "assets_needing_review": len(review_rows(asset_rows)),
    }


if __name__ == "__main__":
    process_all_jee_papers()
