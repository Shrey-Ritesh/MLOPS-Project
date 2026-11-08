"""
Remove Physics records from NCERT chapters, JEE questions, and NCERT learning
units, then prune manifest entries that are no longer referenced.

Assumptions:
- JSONL contains one JSON object per line.
- Each record has a top-level "subject" field.
- Source references use a top-level "source_id" field.
- Assets are intentionally left untouched.

Run `python remove_physics_records_with_learning_units.py --dry-run` first.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

NCERT_CHAPTERS = Path("data/processed/ncert_chapters.jsonl")
JEE_QUESTIONS = Path("data/processed/jee_questions.jsonl")
LEARNING_UNITS = Path("data/processed/ncert_learning_units.jsonl")
SOURCE_MANIFEST = Path("data/processed/source_manifest.jsonl")

SUBJECT_TO_REMOVE = "physics"
SUBJECT_FIELD = "subject"
SOURCE_ID_FIELD = "source_id"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )
            records.append(record)
    return records


def is_physics(record: dict[str, Any]) -> bool:
    subject = record.get(SUBJECT_FIELD)
    return isinstance(subject, str) and subject.strip().casefold() == SUBJECT_TO_REMOVE


def source_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        value.strip()
        for record in records
        if isinstance((value := record.get(SOURCE_ID_FIELD)), str) and value.strip()
    }


def write_jsonl_atomically(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove Physics records and prune unreferenced manifest entries."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned changes without writing files.")
    args = parser.parse_args()

    # Parse all inputs before modifying any file.
    ncert = read_jsonl(NCERT_CHAPTERS)
    jee = read_jsonl(JEE_QUESTIONS)
    learning_units = read_jsonl(LEARNING_UNITS)
    manifest = read_jsonl(SOURCE_MANIFEST)

    kept_ncert = [r for r in ncert if not is_physics(r)]
    kept_jee = [r for r in jee if not is_physics(r)]
    kept_units = [r for r in learning_units if not is_physics(r)]

    removed_ncert = [r for r in ncert if is_physics(r)]
    removed_jee = [r for r in jee if is_physics(r)]
    removed_units = [r for r in learning_units if is_physics(r)]

    removed_ids = source_ids(removed_ncert + removed_jee + removed_units)
    remaining_ids = source_ids(kept_ncert + kept_jee + kept_units)
    safe_to_prune = removed_ids - remaining_ids

    kept_manifest = []
    removed_manifest = []
    for entry in manifest:
        value = entry.get(SOURCE_ID_FIELD)
        if isinstance(value, str) and value.strip() in safe_to_prune:
            removed_manifest.append(entry)
        else:
            kept_manifest.append(entry)

    print("Planned cleanup:")
    for label, all_records, removed, kept in (
        ("NCERT chapters", ncert, removed_ncert, kept_ncert),
        ("JEE questions", jee, removed_jee, kept_jee),
        ("Learning units", learning_units, removed_units, kept_units),
        ("Source manifest", manifest, removed_manifest, kept_manifest),
    ):
        print(f"  {label}: {len(all_records)} total; "
              f"{len(removed)} to remove; {len(kept)} to keep")
    print(f"  Source IDs on removed records: {len(removed_ids)}")
    print(f"  Source IDs still referenced: {len(remaining_ids)}")

    if args.dry_run:
        print("\nDry run only; no files changed.")
        return

    write_jsonl_atomically(NCERT_CHAPTERS, kept_ncert)
    write_jsonl_atomically(JEE_QUESTIONS, kept_jee)
    write_jsonl_atomically(LEARNING_UNITS, kept_units)
    write_jsonl_atomically(SOURCE_MANIFEST, kept_manifest)
    print("\nCleanup complete. Assets were not modified.")


if __name__ == "__main__":
    main()
