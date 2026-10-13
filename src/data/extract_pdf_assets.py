"""Incrementally extract embedded raster instances and explicitly selected crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pymupdf

from src.data.asset_store import build_review_rows, extract_xobject_png, read_jsonl, store_asset, write_jsonl_atomic
from src.data.vector_regions import group_drawing_regions
from src.models.assets import AssetRecord, AssetType, ReviewStatus


def _clip_bbox(raw: tuple, page: pymupdf.Page) -> pymupdf.Rect:
    return pymupdf.Rect(raw) & page.rect


def _bbox_dict(rect: pymupdf.Rect) -> dict[str, float]:
    return {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1}


def raster_placement_candidates(page: pymupdf.Page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs = {int(row[0]): int(row[1]) if len(row) > 1 else 0 for row in page.get_images(full=True)}
    candidates = []
    excluded = []
    page_area = max(page.rect.width * page.rect.height, 1.0)
    seen: set[tuple[int, tuple[float, ...]]] = set()
    for info in page.get_image_info(xrefs=True):
        xref = int(info.get("xref", 0))
        rect = _clip_bbox(info["bbox"], page)
        if rect.is_empty:
            continue
        key = (xref, tuple(round(v, 3) for v in rect))
        if key in seen:
            continue
        seen.add(key)
        placement = {"xref": xref, "soft_mask_xref": refs.get(xref, 0), "bbox": rect}
        # Page-sized placements may be backgrounds, scans, or full-page
        # figures. Keep these visible for review instead of silently dropping.
        if (
            rect.width < 40
            or rect.height < 40
            or rect.width >= 0.85 * page.rect.width
            or rect.height >= 0.85 * page.rect.height
            or rect.width * rect.height >= 0.80 * page_area
        ):
            placement["reason"] = "small raster placement" if rect.width < 40 or rect.height < 40 else "large raster placement may be background or full-page figure"
            excluded.append(placement)
        else:
            candidates.append(placement)
    return candidates, excluded


def _parse_region(value: str) -> tuple[int, pymupdf.Rect]:
    try:
        page_raw, *coords_raw = value.split(":")
        if len(coords_raw) != 1:
            raise ValueError
        coords = [float(part) for part in coords_raw[0].split(",")]
        if len(coords) != 4:
            raise ValueError
        return int(page_raw), pymupdf.Rect(coords)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("region must use PAGE:x0,y0,x1,y1 (one-based page, PDF points)") from exc


def extract_pdf_assets(
    pdf_path: str | Path,
    *,
    source_id: str,
    manifest_path: str | Path = "data/processed/assets_manifest.jsonl",
    data_root: str | Path = ".",
    regions: list[tuple[int, pymupdf.Rect]] | None = None,
    associated_record_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Extract placed raster images and render user-selected page regions.

    Vector drawing groups are emitted to a review queue. They are rendered only
    when explicitly selected with ``regions``; region-to-record links are empty
    unless a reviewer supplies an association for these selected crops.
    """
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Cannot open source PDF {pdf_path}: {exc}") from exc
    records: list[AssetRecord] = []
    candidates_for_review: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    root = Path(data_root)
    page_count = document.page_count
    try:
        for index in range(document.page_count):
            page = document[index]
            page_number = index + 1
            raster_candidates, excluded_rasters = raster_placement_candidates(page)
            for candidate in excluded_rasters:
                candidates_for_review.append({
                    "source_id": source_id,
                    "source_page": page_number,
                    "bbox": _bbox_dict(candidate["bbox"]),
                    "xref": candidate["xref"],
                    "status": "raster_candidate_requires_review",
                    "reason": candidate["reason"],
                })
            for candidate in raster_candidates:
                try:
                    fallback = candidate["xref"] <= 0
                    if not fallback:
                        try:
                            image_bytes = extract_xobject_png(document, candidate["xref"], candidate["soft_mask_xref"])
                        except ValueError:
                            fallback = True
                    if fallback:
                        image_bytes = page.get_pixmap(clip=candidate["bbox"], dpi=200, alpha=False).tobytes("png")
                    records.append(store_asset(
                        image_bytes,
                        source_id=source_id,
                        source_page=page_number,
                        data_root=root,
                        asset_type=AssetType.PAGE_CROP if fallback else AssetType.IMAGE,
                        bbox=_bbox_dict(candidate["bbox"]),
                        associated_record_ids=(),
                        review_status=ReviewStatus.UNREVIEWED,
                    ))
                except Exception as exc:
                    errors.append({"page": page_number, "xref": candidate["xref"], "error": str(exc)})

            region_groups = group_drawing_regions(
                page.get_drawings(), page_width=page.rect.width, page_height=page.rect.height
            )
            for region in region_groups["regions"]:
                bounds = region["bbox"]
                region_width = bounds["x1"] - bounds["x0"]
                region_height = bounds["y1"] - bounds["y0"]
                if (
                    region_width < 30
                    or region_height < 20
                    or region_width * region_height >= 0.75 * page.rect.width * page.rect.height
                ):
                    continue
                candidates_for_review.append({
                    "source_id": source_id,
                    "source_page": page_number,
                    "bbox": region["bbox"],
                    "drawing_indices": region["drawing_indices"],
                    "object_count": region["object_count"],
                    "status": "candidate_not_classified",
                    "reason": "Vector geometry grouping is not semantic figure detection; select for rendering after review.",
                })

        for source_page, requested in regions or []:
            if source_page < 1 or source_page > page_count:
                errors.append({"page": source_page, "error": f"selected region page outside PDF range 1..{page_count}"})
                continue
            page = document[source_page - 1]
            crop = requested & page.rect
            if crop.is_empty or crop.width <= 0 or crop.height <= 0:
                errors.append({"page": source_page, "error": "selected region has no visible area inside page bounds"})
                continue
            try:
                payload = page.get_pixmap(clip=crop, dpi=200, alpha=False).tobytes("png")
                records.append(store_asset(
                    payload,
                    source_id=source_id,
                    source_page=source_page,
                    data_root=root,
                    asset_type=AssetType.DIAGRAM,
                    bbox=_bbox_dict(crop),
                    associated_record_ids=associated_record_ids or (),
                    review_status=ReviewStatus.UNREVIEWED,
                ))
            except Exception as exc:
                errors.append({"page": source_page, "bbox": _bbox_dict(crop), "error": str(exc)})
    finally:
        document.close()

    manifest = Path(manifest_path)
    existing = read_jsonl(manifest) if manifest.exists() else []
    # Preserve unrelated sources. Reprocessed identical occurrences replace
    # their prior rows by stable asset_id, while byte-identical assets share a
    # single checksum path on disk.
    merged = {row.get("asset_id"): row for row in existing}
    merged.update({row.asset_id: row.model_dump(mode="json") for row in records})
    merged_rows = list(merged.values())
    write_jsonl_atomic(merged_rows, manifest)
    review_path = manifest.parent / "assets_review.jsonl"
    prior_review = read_jsonl(review_path) if review_path.exists() else []
    review_items = build_review_rows(
        merged_rows,
        existing_review_rows=[row for row in prior_review if row.get("source_id") != source_id],
        new_candidates=candidates_for_review,
    )
    write_jsonl_atomic(review_items, review_path)
    return {
        "source_id": source_id,
        "pages": page_count,
        "assets_written": len(records),
        "unique_asset_files_written_or_reused": len({r.sha256 for r in records}),
        "vector_candidates_for_review": len(candidates_for_review),
        "errors": errors,
        "manifest_path": str(manifest),
        "review_path": str(review_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--source-id", required=True, help="registered source_id from the source manifest")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/assets_manifest.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--region", action="append", type=_parse_region, default=[], help="selected crop: PAGE:x0,y0,x1,y1")
    parser.add_argument("--associate-record-id", action="append", default=[], help="review-confirmed association for selected regions")
    args = parser.parse_args()
    result = extract_pdf_assets(
        args.pdf,
        source_id=args.source_id,
        manifest_path=args.manifest,
        data_root=args.data_root,
        regions=args.region,
        associated_record_ids=args.associate_record_id,
    )
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
