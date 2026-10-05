from pathlib import Path
import json

from src.data.checksums import file_size, sha256_file
from src.data.ids import make_source_id, make_logical_source_id
from src.models import SourceRecord



def build_source_record(
    path: str | Path,
    *,
    source_type: str,
    subject: str | None = None,
    language: str | None = None,
    publication_year: int | None = None,
    exam_year: int | None = None,
    edition: str | None = None,
    class_level: int | None = None,
    session: str | None = None,
    shift: str | None = None,
    logical_source_id: str | None = None,
    inventory_status: str = "registered",
) -> SourceRecord:
    """Create a validated SourceRecord from one local source file."""
    path = Path(path)
    digest = sha256_file(path)

    return SourceRecord.model_validate({
        "source_id": make_source_id(digest),
        "logical_source_id": (
            make_logical_source_id(logical_source_id)
            if logical_source_id is not None else None
        ),
        "source_type": source_type,
        "original_filename": path.name,
        "sha256": digest,
        "file_size_bytes": file_size(path),
        "subject": subject,
        "language": language,
        "publication_year": publication_year,
        "exam_year": exam_year,
        "edition": edition,
        "class_level": class_level,
        "session": session,
        "shift": shift,
        "source_locator": path.as_posix(),
        "inventory_status": inventory_status,
    })

def write_jsonl(recs: list[SourceRecord], path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs: 
            f.write(rec.model_dump_json() + "\n") 

def append_jsonl(rec: SourceRecord, path):
    with open(path, "a", encoding='utf-8') as f:
        f.write(rec.model_dump_json() + "\n")  
