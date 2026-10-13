import hashlib
import json
from pathlib import Path

import pymupdf

from src.data.asset_store import (
    clear_unresolved_associations,
    store_asset,
    validate_asset_records,
    validate_record_asset_references,
    write_jsonl_atomic,
)
from src.data.extract_pdf_assets import extract_pdf_assets
from src.data.vector_regions import group_drawing_regions


def _png(color=(220, 30, 20)):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 24, 24), 0)
    pix.clear_with((color[0] << 16) | (color[1] << 8) | color[2])
    return pix.tobytes("png")


def test_store_asset_deduplicates_binary_but_preserves_occurrences(tmp_path):
    payload = _png()
    first = store_asset(payload, source_id="src-a", source_page=1, data_root=tmp_path, bbox={"x0": 1, "y0": 2, "x1": 20, "y1": 21})
    second = store_asset(payload, source_id="src-a", source_page=2, data_root=tmp_path, bbox={"x0": 2, "y0": 3, "x1": 21, "y1": 22})

    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.file_path == second.file_path
    assert first.asset_id != second.asset_id
    path = tmp_path / first.file_path
    assert path.is_file()
    assert pymupdf.Pixmap(path.as_posix()).width == 24
    assert validate_asset_records([first, second], data_root=tmp_path) == []


def test_asset_integrity_validator_reports_missing_and_corrupt_assets(tmp_path):
    record = store_asset(_png(), source_id="src-a", source_page=1, data_root=tmp_path)
    path = tmp_path / record.file_path
    path.write_bytes(b"not an image")
    errors = validate_asset_records([record.model_dump(mode="json")], data_root=tmp_path)
    assert any("checksum mismatch" in error for error in errors)
    assert any("cannot be opened" in error for error in errors)
    path.unlink()
    errors = validate_asset_records([record.model_dump(mode="json")], data_root=tmp_path)
    assert any("asset file missing" in error for error in errors)


def test_unresolved_links_are_cleared_and_flagged_for_review():
    rows, count = clear_unresolved_associations(
        [{"asset_id": "a", "associated_record_ids": ["known", "missing"], "review_status": "unreviewed"}],
        {"known"},
    )
    assert count == 1
    assert rows[0]["associated_record_ids"] == ["known"]
    assert rows[0]["review_status"] == "needs_review"


def test_record_asset_links_must_resolve_to_manifest(tmp_path):
    records = tmp_path / "questions.jsonl"
    write_jsonl_atomic([{"question_id": "q1", "asset_ids": ["absent"]}], records)
    errors = validate_record_asset_references([], [records])
    assert errors == [f"{records}: q1: missing asset_id absent"]


def test_vector_groups_are_stable_clipped_and_report_invalid_objects():
    drawings = [
        {"rect": pymupdf.Rect(10, 10, 20, 20)},
        {"rect": pymupdf.Rect(20.5, 10, 30, 20)},
        {"rect": pymupdf.Rect(-10, 40, 5, 50)},
        {"rect": pymupdf.Rect(5, 5, 5, 8)},
    ]
    first = group_drawing_regions(drawings, page_width=100, page_height=100, distance_fraction=0.01)
    second = group_drawing_regions(drawings, page_width=100, page_height=100, distance_fraction=0.01)
    assert first == second
    assert first["regions"][0]["drawing_indices"] == [0, 1]
    assert first["regions"][1]["bbox"]["x0"] == 0
    assert first["rejected_drawings"][0]["drawing_index"] == 3


def test_pdf_extraction_reuses_manifest_and_keeps_vector_links_unassigned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = _png()
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=500)
    page.insert_image(pymupdf.Rect(80, 80, 150, 150), stream=image)
    page.draw_rect(pymupdf.Rect(200, 200, 230, 230), color=(0, 0, 0))
    pdf = tmp_path / "source.pdf"
    doc.save(pdf)
    doc.close()

    manifest = Path("data/processed/assets_manifest.jsonl")
    result = extract_pdf_assets(
        pdf,
        source_id="src-synthetic",
        manifest_path=manifest,
        data_root=tmp_path,
        regions=[(1, pymupdf.Rect(195, 195, 235, 235))],
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert result["assets_written"] == 2
    assert len(rows) == 2
    assert all(Path(tmp_path / row["file_path"]).is_file() for row in rows)
    crop = next(row for row in rows if row["asset_type"] == "diagram")
    assert crop["associated_record_ids"] == []
    review = Path(result["review_path"]).read_text(encoding="utf-8")
    assert '"status": "candidate_not_classified"' in review
