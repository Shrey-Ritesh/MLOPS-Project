"""Register NCERT source PDFs referenced by canonical chapter/asset records.

Run from the repository root:
    python -m src.data.source_extraction

The script:
- preserves existing source_manifest.jsonl records;
- finds source PDFs by matching their SHA-256-derived source_id to ncert_chapters.jsonl;
- reuses subject/class/language metadata from chapter records;
- adds only missing NCERT source records;
- refuses to overwrite the manifest if a referenced NCERT source cannot be matched.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.data.checksums import sha256_file
from src.data.ids import make_source_id
from src.data.inventory import build_source_record
from src.models import SourceRecord


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL, with a useful line-number error for malformed records."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}") from exc
    return rows


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    """Write to a temporary file in the destination directory, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def build_chapter_metadata(chapters: list[dict]) -> dict[str, dict]:
    """Build one metadata entry per source_id from canonical NCERT chapter records."""
    metadata: dict[str, dict] = {}
    for chapter in chapters:
        source_id = chapter.get("source_id")
        if not source_id:
            raise ValueError(
                f"Chapter {chapter.get('chapter_id', '<unknown>')} has no source_id"
            )

        candidate = {
            "subject": chapter.get("subject"),
            "language": chapter.get("language"),
            "class_level": chapter.get("class_level"),
        }

        if source_id in metadata:
            # Multiple chapters can refer to one source PDF, but their source-level
            # metadata should agree.
            previous = metadata[source_id]
            for key in ("subject", "language", "class_level"):
                a, b = previous.get(key), candidate.get(key)
                if a is not None and b is not None and a != b:
                    raise ValueError(
                        f"Conflicting {key} metadata for source_id {source_id}: "
                        f"{a!r} vs {b!r}"
                    )
                if previous.get(key) is None and b is not None:
                    previous[key] = b
        else:
            metadata[source_id] = candidate

    return metadata


def run(
    *,
    raw_dir: Path,
    chapters_path: Path,
    assets_path: Path,
    manifest_path: Path,
) -> dict:
    chapters = read_jsonl(chapters_path)
    assets = read_jsonl(assets_path) if assets_path.exists() else []
    existing_rows = read_jsonl(manifest_path) if manifest_path.exists() else []

    # Validate existing manifest rows and preserve their serialized values.
    existing_by_id: dict[str, dict] = {}
    for row in existing_rows:
        record = SourceRecord.model_validate(row)
        source_id = record.source_id
        if source_id in existing_by_id:
            if existing_by_id[source_id] != row:
                raise ValueError(
                    f"Conflicting duplicate source_id in existing manifest: {source_id}"
                )
            continue
        existing_by_id[source_id] = row

    chapter_metadata = build_chapter_metadata(chapters)
    required_ncert_ids = set(chapter_metadata)
    asset_source_ids = {
        row["source_id"] for row in assets if row.get("source_id")
    }

    missing_ids = required_ncert_ids - set(existing_by_id)
    if not missing_ids:
        # Still validate that chapter sources are registered as NCERT.
        wrong_type = [
            source_id
            for source_id in required_ncert_ids
            if existing_by_id[source_id].get("source_type") != "ncert"
        ]
        if wrong_type:
            raise ValueError(
                "NCERT chapter records point to source IDs registered with a "
                f"non-NCERT type: {wrong_type}"
            )
        return {
            "manifest_path": str(manifest_path),
            "existing_sources": len(existing_by_id),
            "ncert_sources_required": len(required_ncert_ids),
            "new_sources_added": 0,
            "asset_source_ids_not_registered": sorted(
                asset_source_ids - set(existing_by_id)
            ),
            "status": "already_up_to_date",
        }

    if not raw_dir.exists():
        raise FileNotFoundError(f"NCERT raw PDF directory not found: {raw_dir}")

    # Hash candidate PDFs once to identify which files correspond to missing IDs.
    # Only matching files are passed to build_source_record (which validates the
    # full source record and computes the digest/size used by the contract).
    candidates: dict[str, Path] = {}
    for pdf_path in sorted(raw_dir.rglob("*.pdf")):
        digest = sha256_file(pdf_path)
        candidate_id = make_source_id(digest)
        if candidate_id in missing_ids:
            if candidate_id in candidates:
                raise ValueError(
                    "Two PDFs have identical bytes for source_id "
                    f"{candidate_id}: {candidates[candidate_id]} and {pdf_path}. "
                    "Resolve the duplicate explicitly before registering."
                )
            candidates[candidate_id] = pdf_path

    unresolved = sorted(missing_ids - set(candidates))
    if unresolved:
        raise FileNotFoundError(
            "Could not match these NCERT source IDs to PDFs under "
            f"{raw_dir}:\n  " + "\n  ".join(unresolved)
            + "\nNo manifest changes were written. Check that the original PDFs "
              "are present and byte-identical to those used during extraction."
        )

    additions: list[dict] = []
    for source_id in sorted(missing_ids):
        pdf_path = candidates[source_id]
        meta = chapter_metadata[source_id]

        # Use a repository-relative locator where possible, avoiding machine-specific
        # absolute paths in the canonical manifest.
        try:
            locator = pdf_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            locator = pdf_path.as_posix()

        record = build_source_record(
            locator,
            source_type="ncert",
            subject=meta["subject"],
            language=meta["language"],
            class_level=meta["class_level"],
        )

        # Guard against any identity mismatch before adding it.
        if record.source_id != source_id:
            raise ValueError(
                f"Source ID mismatch for {pdf_path}: expected {source_id}, "
                f"built {record.source_id}"
            )

        additions.append(json.loads(record.model_dump_json()))

    # Ensure asset references are not silently left dangling when they refer to a
    # source expected to be an NCERT source. Existing JEE sources are preserved.
    final_ids = set(existing_by_id) | {row["source_id"] for row in additions}
    unregistered_asset_ids = sorted(asset_source_ids - final_ids)

    # Ensure all NCERT source references are present and correctly typed.
    for row in additions:
        SourceRecord.model_validate(row)
    final_by_id = {**existing_by_id, **{r["source_id"]: r for r in additions}}
    wrong_type = [
        source_id
        for source_id in required_ncert_ids
        if final_by_id[source_id].get("source_type") != "ncert"
    ]
    if wrong_type:
        raise ValueError(f"NCERT source IDs have incorrect source_type: {wrong_type}")

    combined = list(existing_by_id.values()) + additions
    write_jsonl_atomic(combined, manifest_path)

    return {
        "manifest_path": str(manifest_path),
        "existing_sources": len(existing_by_id),
        "ncert_sources_required": len(required_ncert_ids),
        "new_sources_added": len(additions),
        "asset_source_ids_not_registered": unregistered_asset_ids,
        "status": "updated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ncert"))
    parser.add_argument(
        "--chapters",
        type=Path,
        default=Path("data/processed/ncert_chapters.jsonl"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("data/processed/assets.jsonl"),
        help="Asset manifest; if your file is named assets_manifest.jsonl, pass it here.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/source_manifest.jsonl"),
    )
    args = parser.parse_args()

    result = run(
        raw_dir=args.raw_dir,
        chapters_path=args.chapters,
        assets_path=args.assets,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
