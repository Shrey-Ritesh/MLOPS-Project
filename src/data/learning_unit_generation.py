"""
Deterministic NCERT learning-unit generation.

Run from the repository root:
    python -m src.data.learning_unit_generation
    python -m src.data.learning_unit_generation --limit 2 --output data/interim/learning_units_sample.jsonl

Learning units are final retrieval documents; this script does not create a
second chunking stage. It groups ordered canonical NCERT content blocks using
numbered subsection headings and preserves source-block provenance.

Boundary policy v1:
- A heading such as "14.2 TRANSVERSE AND LONGITUDINAL WAVES" starts a unit.
- Content before the first subsection is retained in a chapter-opening unit.
- Likely table-of-contents paragraphs are isolated and flagged for review.
- Figures, captions, tables and equations stay with their surrounding section.
- Asset IDs are carried through; empty lists are valid before NCERT asset extraction.
- Failed source chapters are skipped and reported; needs_review chapters are retained.
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

CONTRACT_VERSION = "1.0.0"
SECTION_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s*[\.\s:—-]*\s*(.*)$")
TOC_MARKERS = ("contents", "table of contents", "points to ponder", "exercises", "summary", "appendix", "index")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    rows = []
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


def stable_unit_id(chapter_id: str, first_block_id: str, unit_type: str) -> str:
    raw = f"{chapter_id}|{first_block_id}|{unit_type}|learning-unit-v1"
    return "lu-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_unit(chapter: dict[str, Any], blocks: list[dict[str, Any]], order: int,
              title: str, unit_type: str, flags: list[str] | None = None) -> dict[str, Any]:
    if not blocks:
        raise ValueError("Cannot create a learning unit without source blocks")
    block_ids = [b["block_id"] for b in blocks]
    pages = [b["page"] for b in blocks if isinstance(b.get("page"), int)]
    texts = [block_text(b) for b in blocks if block_text(b)]
    asset_ids: list[str] = []
    for block in blocks:
        for asset_id in block.get("asset_ids", []) or []:
            if asset_id not in asset_ids:
                asset_ids.append(asset_id)
    # Attach assets only when a constituent block explicitly references them.
    # Chapter-level asset_ids are too coarse to assign to every unit safely.

    review_flags = list(flags or [])
    if chapter.get("extraction_status") == "needs_review":
        review_flags.append("source_chapter_needs_review")
    return {
        "unit_id": stable_unit_id(chapter["chapter_id"], block_ids[0], unit_type),
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
        "text": "\n\n".join(texts).strip(),
        "block_ids": block_ids,
        "asset_ids": asset_ids,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "source_extraction_status": chapter.get("extraction_status"),
        "generation_status": "needs_review" if review_flags else "success",
        "review_flags": review_flags,
    }


def generate_chapter_units(chapter: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
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

    boundaries = []
    for index, block in enumerate(blocks):
        parsed = section_heading(block)
        if parsed:
            boundaries.append((index, parsed[0], parsed[1]))

    if not boundaries:
        flags = ["no_numbered_subsection_boundaries"]
        unit = make_unit(chapter, blocks, 0, chapter.get("chapter_title") or "Chapter content",
                         "chapter_content", flags)
        return [unit], []

    units: list[dict[str, Any]] = []
    first_start = boundaries[0][0]
    if first_start:
        prefix = blocks[:first_start]

        # Preserve the source order: split the prefix into contiguous runs of
        # intro content and likely table-of-contents paragraphs. Filtering all
        # TOC blocks out and appending them afterward would reorder block IDs.
        runs: list[tuple[str, list[dict[str, Any]]]] = []
        for block in prefix:
            is_toc = (
                block.get("block_type") == "paragraph"
                and looks_like_toc(block)
            )
            run_type = "contents_overview" if is_toc else "chapter_intro"
            if runs and runs[-1][0] == run_type:
                runs[-1][1].append(block)
            else:
                runs.append((run_type, [block]))

        for run_type, run_blocks in runs:
            if run_type == "contents_overview":
                units.append(make_unit(
                    chapter, run_blocks, len(units),
                    "Section overview (extracted contents)",
                    "contents_overview", ["possible_table_of_contents"]
                ))
            else:
                units.append(make_unit(
                    chapter, run_blocks, len(units),
                    chapter.get("chapter_title") or "Chapter introduction",
                    "chapter_intro", ["chapter_opening_content"]
                ))

    for idx, (start, number, title) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(blocks)
        group = blocks[start:end]
        flags = []
        if not any(block_text(b) for b in group):
            flags.append("unit_has_no_textual_content")
        units.append(make_unit(chapter, group, len(units), title or f"Section {number}", "section", flags))

    emitted_ids = [bid for unit in units for bid in unit["block_ids"]]
    expected_ids = [block["block_id"] for block in blocks]
    if emitted_ids != expected_ids:
        return [], [f"{chapter_id}: block coverage/order mismatch"]
    return units, []


def run(input_path: Path, output_path: Path, limit: int | None = None) -> dict[str, Any]:
    chapters = read_jsonl(input_path)
    if limit is not None:
        chapters = chapters[:limit]
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
        chapter_units, chapter_errors = generate_chapter_units(chapter)
        errors.extend(chapter_errors)
        for unit in chapter_units:
            if unit["unit_id"] in seen_units:
                errors.append(f"duplicate unit_id generated: {unit['unit_id']}")
                continue
            seen_units.add(unit["unit_id"])
            units.append(unit)

    report = {
        "contract_version": CONTRACT_VERSION,
        "pipeline": "ncert_learning_unit_generation",
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "chapters_seen": len(chapters),
        "chapters_by_extraction_status": dict(status_counts),
        "failed_chapters_skipped": skipped_failed_chapters,
        "learning_units_generated": len(units),
        "units_by_type": dict(Counter(u["unit_type"] for u in units)),
        "units_needing_review": sum(u["generation_status"] == "needs_review" for u in units),
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
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    print(json.dumps(run(args.input, args.output, args.limit), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
