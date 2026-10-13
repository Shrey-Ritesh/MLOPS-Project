"""Deterministic, fine-grained NCERT learning-unit generation (policy v2).

Run from the repository root:
    python -m src.data.learning_unit_generation --input data/processed/ncert_chapters.jsonl

Requires transformers so unit sizes are measured with the embedding model's tokenizer.
Learning units are final retrieval documents; no downstream chunking is expected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

CONTRACT_VERSION = "2.0.0"
POLICY_VERSION = "learning-unit-v2"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_TOKENS = 180
MAX_TOKENS = 220
MAX_BLOCKS = 5
SECTION_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s*[\.\s:—-]*\s*(.*)$")
TOC_MARKERS = ("contents", "table of contents", "points to ponder", "exercises", "summary", "appendix", "index")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\]])")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def block_text(block: dict[str, Any]) -> str:
    value = block.get("text_cleaned")
    if value is None:
        value = block.get("text_raw")
    return value.strip() if isinstance(value, str) else ""


def section_heading(block: dict[str, Any]) -> tuple[str, str] | None:
    if block.get("block_type") != "heading":
        return None
    text = block_text(block)
    match = SECTION_PREFIX_RE.match(text)
    if match:
        return match.group(1), (match.group(2).strip() or text)
    label = block.get("original_label")
    if isinstance(label, str) and re.fullmatch(r"\s*\d+(?:\.\d+)+\s*", label) and text:
        return label.strip(), text
    return None


def looks_like_toc(block: dict[str, Any]) -> bool:
    text = re.sub(r"\s+", " ", block_text(block)).strip().lower()
    if not text:
        return False
    count = len(re.findall(r"\b\d+(?:\.\d+)+\b", text))
    markers = sum(marker in text for marker in TOC_MARKERS)
    return count >= 3 or (count >= 2 and markers >= 1)


def stable_unit_id(chapter_id: str, first_block_id: str, segment_index: int) -> str:
    raw = f"{chapter_id}|{first_block_id}|{segment_index}|{POLICY_VERSION}"
    return "lu-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def embedding_text(chapter: dict[str, Any], title: str, text: str) -> str:
    # Keep in sync with configs/retrieval.yaml's document.text_template.
    return (
        f"Subject: {chapter.get('subject', '')}\n"
        f"Class: {chapter.get('class_level') or ''}\n"
        f"Chapter: {chapter.get('chapter_title') or ''}\n"
        f"Topic: {title}\n\n{text}"
    )


def token_count(tokenizer: Any, chapter: dict[str, Any], title: str, text: str) -> int:
    return len(tokenizer.encode(embedding_text(chapter, title, text), add_special_tokens=True, truncation=False))


def split_long_text(text: str, tokenizer: Any, chapter: dict[str, Any], title: str) -> list[str]:
    """Split oversized block text at sentence boundaries, then word boundaries as fallback."""
    text = text.strip()
    if not text:
        return [""]
    if token_count(tokenizer, chapter, title, text) <= MAX_TOKENS:
        return [text]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        # Conservative fallback for long equations/tables or OCR paragraphs.
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and token_count(tokenizer, chapter, title, candidate) > TARGET_TOKENS:
                parts.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            parts.append(" ".join(current))
        return parts

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and token_count(tokenizer, chapter, title, candidate) > TARGET_TOKENS:
            parts.append(current)
            current = sentence
        else:
            current = candidate
        # A single sentence can itself exceed the target; split it safely.
        if token_count(tokenizer, chapter, title, current) > MAX_TOKENS:
            parts.extend(split_long_text(current, tokenizer, chapter, title))
            current = ""
    if current:
        parts.append(current)
    return parts


def make_unit(chapter: dict[str, Any], entries: list[dict[str, Any]], order: int,
              title: str, unit_type: str, tokenizer: Any,
              flags: list[str] | None = None) -> dict[str, Any]:
    if not entries:
        raise ValueError("Cannot create a learning unit without source blocks")
    blocks = [entry["block"] for entry in entries]
    block_ids = list(dict.fromkeys(block["block_id"] for block in blocks))
    pages = [b["page"] for b in blocks if isinstance(b.get("page"), int)]
    texts = [entry["text"] for entry in entries if entry["text"].strip()]
    text = "\n\n".join(texts).strip()
    asset_ids: list[str] = []
    for block in blocks:
        for asset_id in block.get("asset_ids", []) or []:
            if asset_id not in asset_ids:
                asset_ids.append(asset_id)

    review_flags = list(flags or [])
    if chapter.get("extraction_status") == "needs_review":
        review_flags.append("source_chapter_needs_review")
    count = token_count(tokenizer, chapter, title, text)
    if count > MAX_TOKENS:
        review_flags.append("token_limit_exceeded")
    if len(block_ids) > MAX_BLOCKS:
        review_flags.append("block_count_limit_exceeded")
    if not text:
        review_flags.append("unit_has_no_textual_content")

    first_entry = entries[0]
    unit_id = stable_unit_id(chapter["chapter_id"], first_entry["block"]["block_id"], order)
    return {
        "unit_id": unit_id,
        "chapter_id": chapter["chapter_id"],
        "source_id": chapter["source_id"],
        "subject": chapter["subject"],
        "class_level": chapter.get("class_level"),
        "language": chapter["language"],
        "chapter_number": chapter.get("chapter_number"),
        "chapter_title": chapter.get("chapter_title"),
        "unit_order": order,
        "unit_title": title.strip() or f"Unit {order + 1}",
        "unit_type": unit_type,
        "text": text,
        "block_ids": block_ids,
        "asset_ids": asset_ids,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "source_extraction_status": chapter.get("extraction_status"),
        "generation_status": "needs_review" if review_flags else "success",
        "review_flags": list(dict.fromkeys(review_flags)),
        "token_count": count,
    }


def generate_chapter_units(chapter: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[str]]:
    chapter_id = chapter.get("chapter_id", "<missing-chapter-id>")
    required = ("chapter_id", "source_id", "subject", "language", "content_blocks")
    missing = [key for key in required if key not in chapter]
    if missing:
        return [], [f"{chapter_id}: missing required fields {missing}"]
    if chapter.get("extraction_status") == "failed":
        return [], []
    blocks = chapter["content_blocks"]
    if not isinstance(blocks, list) or not blocks:
        return [], [f"{chapter_id}: content_blocks must be a non-empty list"]

    errors: list[str] = []
    seen: set[str] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            errors.append(f"{chapter_id}: block at index {index} is not an object")
            continue
        bid = block.get("block_id")
        if not isinstance(bid, str) or not bid:
            errors.append(f"{chapter_id}: block at index {index} has no block_id")
        elif bid in seen:
            errors.append(f"{chapter_id}: duplicate block_id {bid}")
        seen.add(bid)
        if block.get("order_index") != index:
            errors.append(f"{chapter_id}: block {bid!r} order_index is not {index}")
        if not isinstance(block.get("page"), int) or block["page"] < 1:
            errors.append(f"{chapter_id}: block {bid!r} has invalid page")
    if errors:
        return [], errors

    # Split into semantic sections first. Each source block remains traceable even
    # when its text must be split across multiple units.
    sections: list[tuple[str, str, list[dict[str, Any]]]] = []
    prefix_end = next((i for i, b in enumerate(blocks) if section_heading(b)), len(blocks))
    if prefix_end:
        prefix = blocks[:prefix_end]
        run_type: str | None = None
        run: list[dict[str, Any]] = []
        for block in prefix:
            kind = "contents_overview" if block.get("block_type") == "paragraph" and looks_like_toc(block) else "chapter_intro"
            if run and kind != run_type:
                title = "Section overview (extracted contents)" if run_type == "contents_overview" else (chapter.get("chapter_title") or "Chapter introduction")
                sections.append((title, run_type or "chapter_intro", run))
                run = []
            run_type = kind
            run.append(block)
        if run:
            title = "Section overview (extracted contents)" if run_type == "contents_overview" else (chapter.get("chapter_title") or "Chapter introduction")
            sections.append((title, run_type or "chapter_intro", run))

    boundaries = [(i, *section_heading(b)) for i, b in enumerate(blocks) if section_heading(b)]
    for idx, (start, number, title) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(blocks)
        sections.append((title or f"Section {number}", "section", blocks[start:end]))
    if not sections:
        sections = [(chapter.get("chapter_title") or "Chapter content", "chapter_content", blocks)]

    units: list[dict[str, Any]] = []
    for section_title, section_type, section_blocks in sections:
        entries: list[dict[str, Any]] = []
        # Long individual blocks are split into text fragments; each fragment
        # retains the original block reference and therefore its provenance.
        for block in section_blocks:
            text = block_text(block)
            fragments = split_long_text(text, tokenizer, chapter, section_title) if text else [""]
            for fragment in fragments:
                entries.append({"block": block, "text": fragment})

        current: list[dict[str, Any]] = []
        fragment_counts: Counter[str] = Counter()
        for entry in entries:
            candidate = current + [entry]
            unique_blocks = list(dict.fromkeys(e["block"]["block_id"] for e in candidate))
            candidate_text = "\n\n".join(e["text"] for e in candidate if e["text"].strip()).strip()
            count = token_count(tokenizer, chapter, section_title, candidate_text)
            # If this fragment would exceed token or source-block limits, flush
            # current first. A single oversized fragment is retained and flagged.
            if current and (count > TARGET_TOKENS or len(unique_blocks) > MAX_BLOCKS):
                unit_title = section_title if not fragment_counts["unit"] else f"{section_title} (continued {fragment_counts['unit'] + 1})"
                units.append(make_unit(chapter, current, len(units), unit_title, section_type, tokenizer))
                fragment_counts["unit"] += 1
                current = [entry]
            else:
                current = candidate

            current_ids = list(dict.fromkeys(e["block"]["block_id"] for e in current))
            current_text = "\n\n".join(e["text"] for e in current if e["text"].strip()).strip()
            # If a single entry still exceeds the hard maximum, mark it during
            # make_unit; split_long_text should normally prevent this.
            if len(current_ids) > MAX_BLOCKS:
                raise AssertionError("Internal error: unit exceeded MAX_BLOCKS")
        if current:
            unit_title = section_title if not fragment_counts["unit"] else f"{section_title} (continued {fragment_counts['unit'] + 1})"
            units.append(make_unit(chapter, current, len(units), unit_title, section_type, tokenizer))

    # A source block may occur in multiple units only when its text was split.
    # Validate that the first appearances of block IDs preserve source order and
    # that every original block appears at least once.
    first_appearances: list[str] = []
    emitted_set: set[str] = set()
    for unit in units:
        for bid in unit["block_ids"]:
            if bid not in emitted_set:
                emitted_set.add(bid)
                first_appearances.append(bid)
    expected_ids = [block["block_id"] for block in blocks]
    if first_appearances != expected_ids:
        return [], [f"{chapter_id}: block coverage/order mismatch after fine-grained splitting"]
    return units, []


def run(input_path: Path, output_path: Path, limit: int | None = None,
        model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    chapters = read_jsonl(input_path)
    if limit is not None:
        chapters = chapters[:limit]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    units: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_chapters: set[str] = set()
    seen_units: set[str] = set()
    status_counts: Counter[str] = Counter()
    skipped_failed_chapters = 0

    for chapter in chapters:
        chapter_id = chapter.get("chapter_id", "<missing-chapter-id>")
        if chapter_id in seen_chapters:
            errors.append(f"duplicate chapter_id in input: {chapter_id}")
            continue
        seen_chapters.add(chapter_id)
        status = str(chapter.get("extraction_status", "unknown"))
        status_counts[status] += 1
        if status == "failed":
            skipped_failed_chapters += 1
            continue
        chapter_units, chapter_errors = generate_chapter_units(chapter, tokenizer)
        errors.extend(chapter_errors)
        for unit in chapter_units:
            if unit["unit_id"] in seen_units:
                errors.append(f"duplicate unit_id generated: {unit['unit_id']}")
                continue
            seen_units.add(unit["unit_id"])
            units.append(unit)

    token_counts = [u["token_count"] for u in units]
    report = {
        "contract_version": CONTRACT_VERSION,
        "pipeline": "ncert_learning_unit_generation",
        "boundary_policy": POLICY_VERSION,
        "tokenizer_model": model_name,
        "target_tokens": TARGET_TOKENS,
        "max_tokens": MAX_TOKENS,
        "max_source_blocks_per_unit": MAX_BLOCKS,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "chapters_seen": len(chapters),
        "chapters_by_extraction_status": dict(status_counts),
        "failed_chapters_skipped": skipped_failed_chapters,
        "learning_units_generated": len(units),
        "units_by_type": dict(Counter(u["unit_type"] for u in units)),
        "units_needing_review": sum(u["generation_status"] == "needs_review" for u in units),
        "units_over_token_limit": sum(n > MAX_TOKENS for n in token_counts),
        "units_over_block_limit": sum(len(u["block_ids"]) > MAX_BLOCKS for u in units),
        "token_count_summary": {
            "min": min(token_counts) if token_counts else 0,
            "median": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        },
        "errors": errors,
        "validation_outcome": "failed" if errors else "passed",
    }
    report_path = Path("reports/data_quality/learning_unit_generation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"Generation failed with {len(errors)} error(s); output not promoted. See {report_path}")
    write_jsonl_atomic(units, output_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/ncert_chapters.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ncert_learning_units.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face tokenizer/model name")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    print(json.dumps(run(args.input, args.output, args.limit, args.model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
