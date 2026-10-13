"""Validate asset manifest schema, source links, stored files and checksums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.asset_store import (
    build_review_rows,
    clear_unresolved_associations,
    read_jsonl,
    validate_asset_records,
    validate_record_asset_references,
    write_jsonl_atomic,
)
from src.models.source import SourceRecord


def validate_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path = ".",
    source_manifests: list[str | Path] | None = None,
    record_manifests: list[str | Path] | None = None,
    clear_unresolved: bool = False,
) -> dict:
    manifest = Path(manifest_path)
    rows = read_jsonl(manifest)
    source_ids = None
    if source_manifests:
        source_ids = set()
        for path in source_manifests:
            source_ids.update(SourceRecord.model_validate(row).source_id for row in read_jsonl(path))
    record_ids = None
    if record_manifests:
        record_ids = set()
        for path in record_manifests:
            for row in read_jsonl(path):
                record_id = row.get("chapter_id") or row.get("question_id")
                if record_id:
                    record_ids.add(record_id)
                for block in row.get("content_blocks", []):
                    if block.get("block_id"):
                        record_ids.add(block["block_id"])
    cleared_links = 0
    if clear_unresolved:
        if not record_ids:
            raise ValueError("--clear-unresolved-associations requires complete --records inputs")
        rows, cleared_links = clear_unresolved_associations(rows, record_ids)
        write_jsonl_atomic(rows, manifest)
    errors = validate_asset_records(
        rows,
        data_root=data_root,
        known_source_ids=source_ids,
        known_record_ids=record_ids,
    )
    if record_manifests:
        errors.extend(validate_record_asset_references(rows, record_manifests))
    review_path = manifest.parent / "assets_review.jsonl"
    prior_review = read_jsonl(review_path) if review_path.exists() else []
    review = build_review_rows(rows, existing_review_rows=prior_review)
    write_jsonl_atomic(review, review_path)
    return {
        "manifest": str(manifest),
        "records": len(rows),
        "unresolved_associations_cleared": cleared_links,
        "errors": errors,
        "review_items": len(review),
        "review_manifest": str(review_path),
        "outcome": "failed" if errors else "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("data/processed/assets_manifest.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--records", type=Path, action="append", default=[], help="chapter/question JSONL; repeat as needed")
    parser.add_argument("--report", type=Path, help="optional JSON validation report path")
    parser.add_argument(
        "--clear-unresolved-associations",
        action="store_true",
        help="remove links absent from the supplied complete --records manifests and mark affected assets needs_review",
    )
    args = parser.parse_args()
    report = validate_manifest(
        args.manifest,
        data_root=args.data_root,
        source_manifests=args.source_manifest,
        record_manifests=args.records,
        clear_unresolved=args.clear_unresolved_associations,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
