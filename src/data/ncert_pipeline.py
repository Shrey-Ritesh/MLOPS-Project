"""NCERT PDFs -> validated chapter records, linked raster assets, and quality report.

Run from the repository root (source locators are repo-relative):
    python -m src.data.ncert_pipeline [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml
import pymupdf
from pydantic import ValidationError

from src.data.inventory import build_source_record
from src.data.ncert_extract import FigureAsset, extract_chapter
from src.data.asset_store import build_review_rows, read_jsonl, store_asset, validate_asset_records
from src.models import AssetType, NCERTChapterRecord
from src.models.assets import ReviewStatus as AssetReviewStatus
from src.data.vector_regions import group_drawing_regions
from src.data.extract_pdf_assets import raster_placement_candidates

CONTRACT_VERSION = "1.0.0"


def _write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    """Write to a temp file in the same directory, then promote with os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def validate_dataset(chapters: list[dict], source_ids: dict[str, str]) -> list[str]:
    """Dataset-level integrity checks; returns fatal error messages."""
    errors: list[str] = []
    ids = [c["chapter_id"] for c in chapters]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate chapter_id: {dup}")
    for c in chapters:
        if source_ids.get(c["source_id"]) != "ncert":
            errors.append(f"{c['chapter_id']}: source_id missing or not an NCERT source")
        blocks = c["content_blocks"]
        if [b["order_index"] for b in blocks] != list(range(len(blocks))):
            errors.append(f"{c['chapter_id']}: order_index not contiguous from 0")
    return errors


def _register_figure_assets(
    figure_assets: list[FigureAsset],
    record: dict,
    source_id: str,
    assets_dir: Path,
) -> list[dict]:
    """Persist embedded image bytes and link each asset to its figure block."""
    block_by_id = {block["block_id"]: block for block in record["content_blocks"]}
    manifest_rows: list[dict] = []

    for figure in figure_assets:
        block = block_by_id.get(figure.block_id)
        if block is None or block["block_type"] != "figure":
            raise ValueError(f"Figure asset refers to a non-figure block: {figure.block_id}")

        asset = store_asset(
            figure.image_bytes,
            source_id=source_id,
            source_page=figure.page,
            data_root=Path.cwd(),
            storage_dir=assets_dir,
            asset_type=AssetType.IMAGE,
            bbox={
                "x0": figure.bbox[0],
                "y0": figure.bbox[1],
                "x1": figure.bbox[2],
                "y1": figure.bbox[3],
            },
            associated_record_ids=[figure.block_id],
            review_status=AssetReviewStatus.UNREVIEWED,
        )
        if asset.asset_id not in block["asset_ids"]:
            block["asset_ids"].append(asset.asset_id)
        if asset.asset_id not in record["asset_ids"]:
            record["asset_ids"].append(asset.asset_id)
        manifest_rows.append(json.loads(asset.model_dump_json()))

    NCERTChapterRecord.model_validate(record)
    return manifest_rows


def _vector_review_candidates(pdf_path: Path, source_id: str) -> list[dict]:
    """Return geometric vector candidates without assigning figure semantics."""
    document = pymupdf.open(pdf_path)
    candidates = []
    try:
        for page_index in range(document.page_count):
            page = document[page_index]
            _, excluded_rasters = raster_placement_candidates(page)
            for raster in excluded_rasters:
                candidates.append({
                    "source_id": source_id,
                    "source_page": page_index + 1,
                    "bbox": {
                        "x0": raster["bbox"].x0,
                        "y0": raster["bbox"].y0,
                        "x1": raster["bbox"].x1,
                        "y1": raster["bbox"].y1,
                    },
                    "xref": raster["xref"],
                    "status": "raster_candidate_requires_review",
                    "reason": raster["reason"],
                })
            grouped = group_drawing_regions(
                page.get_drawings(), page_width=page.rect.width, page_height=page.rect.height
            )
            for region in grouped["regions"]:
                bbox = region["bbox"]
                width, height = bbox["x1"] - bbox["x0"], bbox["y1"] - bbox["y0"]
                if width < 30 or height < 20 or width * height >= 0.75 * page.rect.width * page.rect.height:
                    continue
                candidates.append({
                    "source_id": source_id,
                    "source_page": page_index + 1,
                    "bbox": bbox,
                    "drawing_indices": region["drawing_indices"],
                    "object_count": region["object_count"],
                    "status": "candidate_not_classified",
                    "reason": "Vector geometry grouping is not semantic figure detection; select for rendering after review.",
                })
    finally:
        document.close()
    return candidates


def run(
    config_path: Path,
    out_dir: Path,
    limit: int | None = None,
    assets_dir: Path = Path("data/assets/sha256"),
) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(cfg["raw_dir"])
    defaults = cfg["defaults"]

    sources, chapters, rejected, per_file, assets, vector_candidates = [], [], [], [], [], []
    processed_chapter_source_ids: set[str] = set()
    chapter_entries = 0

    for entry in cfg["files"]:
        meta = {**defaults, **entry}
        path = raw_dir / meta["file"]
        if not path.exists():
            per_file.append({"file": meta["file"], "outcome": "missing"})
            continue

        source = build_source_record(
            path,
            source_type=meta["source_type"],
            subject=meta["subject"],
            language=meta["language"],
            class_level=meta["class_level"],
        )
        sources.append(source)

        if meta["kind"] != "chapter":
            per_file.append({"file": meta["file"], "outcome": "skipped_non_chapter"})
            continue
        if limit is not None and chapter_entries >= limit:
            per_file.append({"file": meta["file"], "outcome": "skipped_limit"})
            continue
        chapter_entries += 1
        processed_chapter_source_ids.add(source.source_id)

        result = extract_chapter(
            str(path),
            source_id=source.source_id,
            subject=meta["subject"],
            class_level=meta["class_level"],
            language=meta["language"],
            chapter_number_hint=meta.get("chapter_number"),
        )
        vector_candidates.extend(_vector_review_candidates(path, source.source_id))
        try:
            record = NCERTChapterRecord.model_validate(result.record)
        except ValidationError as exc:
            rejected.append({"file": meta["file"], "errors": json.loads(exc.json())})
            per_file.append({"file": meta["file"], "outcome": "rejected"})
            continue

        record_data = record.model_dump(mode="json")
        chapter_assets = _register_figure_assets(
            result.figure_assets,
            record_data,
            source.source_id,
            assets_dir,
        )
        assets.extend(chapter_assets)
        chapters.append(record_data)
        per_file.append(
            {
                "file": meta["file"],
                "outcome": record.extraction_status.value,
                "chapter_id": record.chapter_id,
                "assets_extracted": len(chapter_assets),
                "review_reasons": result.review_reasons,
                **result.stats,
            }
        )

    source_types = {s.source_id: s.source_type.value for s in sources}
    integrity_errors = validate_dataset(chapters, source_types)
    if not chapters:
        integrity_errors.append(
            "no NCERT chapter records were produced; check raw_dir and configured PDF files"
        )

    status_counts = Counter(c["extraction_status"] for c in chapters)
    report = {
        "contract_version": CONTRACT_VERSION,
        "track": "B_ncert",
        "sources_registered": [
            {"source_id": s.source_id, "file": s.original_filename, "sha256": s.sha256}
            for s in sources
        ],
        "chapters_produced": len(chapters),
        "assets_extracted": len(assets),
        "by_status": dict(status_counts),
        "rejected_records": len(rejected),
        "integrity_errors": integrity_errors,
        "validation_outcome": "failed" if integrity_errors or rejected else "passed",
        "files": per_file,
    }

    interim = Path("data/interim")
    _write_jsonl_atomic([json.loads(s.model_dump_json()) for s in sources], interim / "ncert_source_manifest.jsonl")
    _write_jsonl_atomic(rejected, Path("data/quarantine") / "ncert_rejected.jsonl")
    assets_manifest_path = out_dir / "assets_manifest.jsonl"
    existing_assets = []
    if assets_manifest_path.exists():
        existing_assets = [
            json.loads(line)
            for line in assets_manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    preserved_assets = [
        asset
        for asset in existing_assets
        if asset.get("source_id") not in processed_chapter_source_ids
    ]
    combined_assets = preserved_assets + assets
    known_sources = {source.source_id for source in sources}
    source_manifest_paths = [out_dir / "source_manifest.jsonl", Path("data/interim/ncert_source_manifest.jsonl")]
    for source_path in source_manifest_paths:
        if source_path.exists():
            known_sources.update(row.get("source_id") for row in read_jsonl(source_path) if row.get("source_id"))
    known_records = set()
    for chapter in chapters:
        known_records.add(chapter["chapter_id"])
        known_records.update(block["block_id"] for block in chapter.get("content_blocks", []) if block.get("block_id"))
    for record_path, key in ((out_dir / "jee_questions.jsonl", "question_id"), (out_dir / "ncert_chapters.jsonl", "chapter_id")):
        if record_path.exists():
            for row in read_jsonl(record_path):
                if row.get(key):
                    known_records.add(row[key])
                known_records.update(block["block_id"] for block in row.get("content_blocks", []) if block.get("block_id"))
    asset_integrity_errors = validate_asset_records(
        combined_assets,
        data_root=Path.cwd().resolve(),
        known_source_ids=known_sources or None,
        known_record_ids=known_records or None,
    )
    integrity_errors.extend(asset_integrity_errors)
    review_path = out_dir / "assets_review.jsonl"
    existing_review = read_jsonl(review_path) if review_path.exists() else []
    review_list = build_review_rows(
        combined_assets,
        existing_review_rows=existing_review,
        new_candidates=vector_candidates,
    )
    report["assets_needing_review"] = len(review_list)
    report["vector_candidates_for_review"] = len(vector_candidates)
    report["asset_integrity_errors"] = asset_integrity_errors
    report["validation_outcome"] = "failed" if integrity_errors or rejected else "passed"
    Path("reports/data_quality").mkdir(parents=True, exist_ok=True)
    (Path("reports/data_quality") / "ncert_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Promote canonical output only after integrity checks pass.
    if not integrity_errors:
        _write_jsonl_atomic(chapters, out_dir / "ncert_chapters.jsonl")
        _write_jsonl_atomic(combined_assets, assets_manifest_path)
        _write_jsonl_atomic(review_list, review_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ncert_sources.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("data/assets/sha256"),
        help="directory for extracted NCERT image files",
    )
    parser.add_argument("--limit", type=int, default=None, help="extract at most N chapters")
    args = parser.parse_args()
    report = run(args.config, args.out_dir, args.limit, args.assets_dir)
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "sources_registered")}, indent=2))
    raise SystemExit(0 if report["validation_outcome"] == "passed" else 1)


if __name__ == "__main__":
    main()
