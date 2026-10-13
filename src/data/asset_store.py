"""Shared content-addressed image storage and manifest integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pymupdf
from pydantic import ValidationError

from src.models.assets import AssetRecord, AssetType, ReviewStatus


def _image_is_openable(data: bytes) -> bool:
    try:
        pixmap = pymupdf.Pixmap(data)
        return pixmap.width > 0 and pixmap.height > 0
    except Exception:
        try:
            image = pymupdf.open(stream=data, filetype="png")
            try:
                return image.page_count > 0 and image[0].get_pixmap(alpha=False).width > 0
            finally:
                image.close()
        except Exception:
            return False


def _image_extension(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "tif"
    if data[4:8] == b"ftyp":
        return "heic"
    return "img"


def extract_xobject_png(document: pymupdf.Document, xref: int, soft_mask_xref: int = 0) -> bytes:
    """Decode an image XObject, applying its soft mask when present."""
    try:
        base = pymupdf.Pixmap(document, int(xref))
        if base.colorspace is not None and base.colorspace.name not in ("DeviceGray", "DeviceRGB"):
            base = pymupdf.Pixmap(pymupdf.csRGB, base)
        if soft_mask_xref:
            mask = pymupdf.Pixmap(document, int(soft_mask_xref))
            combined = pymupdf.Pixmap(base, mask)
        else:
            combined = base
        if combined.colorspace is None or combined.colorspace.name not in ("DeviceGray", "DeviceRGB"):
            combined = pymupdf.Pixmap(pymupdf.csRGB, combined)
        payload = combined.tobytes("png")
    except Exception as exc:
        raise ValueError(f"Cannot decode image xref={xref} (soft mask {soft_mask_xref}): {exc}") from exc
    if not _image_is_openable(payload):
        raise ValueError(f"Decoded image xref={xref} is not a readable image")
    return payload


def _stable_asset_id(source_id: str, sha256: str, source_page: int, bbox: dict | None) -> str:
    canonical_box = "none" if bbox is None else ",".join(f"{bbox[k]:.3f}" for k in ("x0", "y0", "x1", "y1"))
    key = f"{source_id}|{sha256}|{source_page}|{canonical_box}"
    return "ast-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def store_asset(
    image_bytes: bytes,
    *,
    source_id: str,
    source_page: int,
    data_root: str | Path = ".",
    storage_dir: str | Path = "data/assets/sha256",
    asset_type: AssetType | str = AssetType.IMAGE,
    bbox: dict[str, float] | None = None,
    associated_record_ids: Iterable[str] = (),
    review_status: ReviewStatus | str = ReviewStatus.UNREVIEWED,
) -> AssetRecord:
    """Persist one occurrence; identical PNG bytes share a single binary path.

    Manifest rows remain occurrence-specific so each source page and crop box
    stays traceable. Association IDs are opt-in and stay empty when uncertain.
    """
    if not image_bytes or not _image_is_openable(image_bytes):
        raise ValueError("Asset bytes are empty or cannot be opened as an image")
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    root = Path(data_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.absolute()
    extension = _image_extension(image_bytes)
    storage = Path(storage_dir)
    if storage.is_absolute() or ".." in storage.parts:
        raise ValueError("storage_dir must be a relative path inside data_root")
    target = root / storage / sha256[:2] / f"{sha256}.{extension}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256 or not _image_is_openable(target.read_bytes()):
            raise ValueError(f"Content-addressed asset path contains invalid bytes: {target}")
    else:
        fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(image_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
    relative = target.relative_to(root).as_posix()
    normalized_bbox = None if bbox is None else {key: float(bbox[key]) for key in ("x0", "y0", "x1", "y1")}
    record = AssetRecord(
        asset_id=_stable_asset_id(source_id, sha256, source_page, normalized_bbox),
        source_id=source_id,
        sha256=sha256,
        asset_type=asset_type,
        file_path=relative,
        source_page=source_page,
        bbox=normalized_bbox,
        associated_record_ids=list(dict.fromkeys(associated_record_ids)),
        review_status=review_status,
    )
    return record


def validate_asset_records(
    records: Iterable[dict[str, Any] | AssetRecord],
    *,
    data_root: str | Path = ".",
    known_source_ids: set[str] | None = None,
    known_record_ids: set[str] | None = None,
) -> list[str]:
    """Validate schema, IDs, binary presence/readability, checksums and links."""
    root = Path(data_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.absolute()
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(records, start=1):
        try:
            record = raw if isinstance(raw, AssetRecord) else AssetRecord.model_validate(raw)
        except ValidationError as exc:
            errors.append(f"row {line_no}: invalid AssetRecord: {exc}")
            continue
        if record.asset_id in seen:
            errors.append(f"row {line_no}: duplicate asset_id {record.asset_id}")
        seen.add(record.asset_id)
        if known_source_ids is not None and record.source_id not in known_source_ids:
            errors.append(f"{record.asset_id}: source_id is not registered: {record.source_id}")
        if known_record_ids is not None:
            for linked_id in record.associated_record_ids:
                if linked_id not in known_record_ids:
                    errors.append(f"{record.asset_id}: associated record does not exist: {linked_id}")
        target = root / Path(record.file_path)
        resolved_root = root.resolve()
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(resolved_root):
            errors.append(f"{record.asset_id}: file_path resolves outside data root")
            continue
        if not target.is_file():
            errors.append(f"{record.asset_id}: asset file missing: {record.file_path}")
            continue
        try:
            payload = target.read_bytes()
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != record.sha256:
                errors.append(f"{record.asset_id}: checksum mismatch (manifest {record.sha256}, file {actual_hash})")
            if not _image_is_openable(payload):
                errors.append(f"{record.asset_id}: asset cannot be opened as an image")
        except OSError as exc:
            errors.append(f"{record.asset_id}: cannot read {record.file_path}: {exc}")
    return errors


def validate_record_asset_references(
    asset_records: Iterable[dict[str, Any] | AssetRecord],
    record_manifests: Iterable[str | Path],
) -> list[str]:
    """Check chapter/question `asset_ids` links resolve to manifest entries."""
    asset_ids = {
        row.asset_id if isinstance(row, AssetRecord) else row.get("asset_id")
        for row in asset_records
    }
    errors = []
    for manifest_path in record_manifests:
        for line_no, record in enumerate(read_jsonl(manifest_path), start=1):
            parent = record.get("chapter_id") or record.get("question_id") or f"row {line_no}"
            for asset_id in record.get("asset_ids", []) or []:
                if asset_id not in asset_ids:
                    errors.append(f"{manifest_path}: {parent}: missing asset_id {asset_id}")
            for block in record.get("content_blocks", []) or []:
                block_id = block.get("block_id", "unknown-block")
                for asset_id in block.get("asset_ids", []) or []:
                    if asset_id not in asset_ids:
                        errors.append(f"{manifest_path}: {parent}/{block_id}: missing asset_id {asset_id}")
    return errors


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl_atomic(rows: Iterable[dict[str, Any] | AssetRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                value = row.model_dump(mode="json") if isinstance(row, AssetRecord) else row
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def review_rows(records: Iterable[dict[str, Any] | AssetRecord]) -> list[dict[str, Any]]:
    review = []
    for raw in records:
        row = raw.model_dump(mode="json") if isinstance(raw, AssetRecord) else raw
        if row.get("review_status") != ReviewStatus.VERIFIED.value or not row.get("associated_record_ids"):
            status = row.get("review_status")
            reason = (
                "association or extraction needs review"
                if status == ReviewStatus.NEEDS_REVIEW.value
                else "unreviewed asset"
                if status != ReviewStatus.VERIFIED.value
                else "association is unknown"
            )
            review.append({
                "asset_id": row.get("asset_id"),
                "source_id": row.get("source_id"),
                "source_page": row.get("source_page"),
                "file_path": row.get("file_path"),
                "associated_record_ids": row.get("associated_record_ids", []),
                "review_status": row.get("review_status"),
                "reason": reason,
            })
    return review


def build_review_rows(
    records: Iterable[dict[str, Any] | AssetRecord],
    *,
    existing_review_rows: Iterable[dict[str, Any]] = (),
    new_candidates: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Refresh asset review items while retaining vector candidates by source."""
    combined = review_rows(records)
    candidates = [
        row for row in existing_review_rows
        if row.get("status") in {"candidate_not_classified", "raster_candidate_requires_review"}
    ]
    candidates.extend(new_candidates)
    seen = set()
    for candidate in candidates:
        key = (
            candidate.get("source_id"), candidate.get("source_page"),
            json.dumps(candidate.get("bbox"), sort_keys=True),
        )
        if key not in seen:
            combined.append(candidate)
            seen.add(key)
    return combined


def clear_unresolved_associations(
    records: Iterable[dict[str, Any]], known_record_ids: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Remove only links absent from a complete supplied record-ID set.

    Affected assets are marked ``needs_review`` and retain source/page/bbox
    provenance. Callers must provide all relevant record manifests.
    """
    repaired = []
    cleared = 0
    for raw in records:
        row = dict(raw)
        links = row.get("associated_record_ids", [])
        valid_links = [link for link in links if link in known_record_ids]
        if len(valid_links) != len(links):
            cleared += len(links) - len(valid_links)
            row["associated_record_ids"] = valid_links
            row["review_status"] = ReviewStatus.NEEDS_REVIEW.value
        repaired.append(row)
    return repaired, cleared
