"""Track B pipeline: NCERT PDFs -> validated ncert_chapters.jsonl + quality report.

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
from pydantic import ValidationError

from src.data.inventory import build_source_record
from src.data.ncert_extract import extract_chapter
from src.models import NCERTChapterRecord

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


def run(config_path: Path, out_dir: Path, limit: int | None = None) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(cfg["raw_dir"])
    defaults = cfg["defaults"]

    sources, chapters, rejected, per_file = [], [], [], []
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

        result = extract_chapter(
            str(path),
            source_id=source.source_id,
            subject=meta["subject"],
            class_level=meta["class_level"],
            language=meta["language"],
            chapter_number_hint=meta.get("chapter_number"),
        )
        try:
            record = NCERTChapterRecord.model_validate(result.record)
        except ValidationError as exc:
            rejected.append({"file": meta["file"], "errors": json.loads(exc.json())})
            per_file.append({"file": meta["file"], "outcome": "rejected"})
            continue

        chapters.append(record.model_dump(mode="json"))
        per_file.append(
            {
                "file": meta["file"],
                "outcome": record.extraction_status.value,
                "chapter_id": record.chapter_id,
                "review_reasons": result.review_reasons,
                **result.stats,
            }
        )

    source_types = {s.source_id: s.source_type.value for s in sources}
    integrity_errors = validate_dataset(chapters, source_types)

    status_counts = Counter(c["extraction_status"] for c in chapters)
    report = {
        "contract_version": CONTRACT_VERSION,
        "track": "B_ncert",
        "sources_registered": [
            {"source_id": s.source_id, "file": s.original_filename, "sha256": s.sha256}
            for s in sources
        ],
        "chapters_produced": len(chapters),
        "by_status": dict(status_counts),
        "rejected_records": len(rejected),
        "integrity_errors": integrity_errors,
        "validation_outcome": "failed" if integrity_errors or rejected else "passed",
        "files": per_file,
    }

    interim = Path("data/interim")
    _write_jsonl_atomic([json.loads(s.model_dump_json()) for s in sources], interim / "ncert_source_manifest.jsonl")
    _write_jsonl_atomic(rejected, Path("data/quarantine") / "ncert_rejected.jsonl")
    Path("reports/data_quality").mkdir(parents=True, exist_ok=True)
    (Path("reports/data_quality") / "ncert_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Promote canonical output only after integrity checks pass.
    if not integrity_errors:
        _write_jsonl_atomic(chapters, out_dir / "ncert_chapters.jsonl")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ncert_sources.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None, help="extract at most N chapters")
    args = parser.parse_args()
    report = run(args.config, args.out_dir, args.limit)
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "sources_registered")}, indent=2))
    raise SystemExit(0 if report["validation_outcome"] == "passed" else 1)


if __name__ == "__main__":
    main()
