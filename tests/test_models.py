
import pytest
from pydantic import ValidationError

from src.models import SourceRecord, NCERTChapterRecord, JEEQuestionRecord


def test_valid_source_record():
    data = {
        "source_id": "src-001",
        "logical_source_id": None,
        "source_type": "ncert",
        "original_filename": "physics.pdf",
        "sha256": "a" * 64,
        "file_size_bytes": 1024,
        "subject": "Physics",
        "language": "en",
        "publication_year": 2024,
        "exam_year": None,
        "edition": "2024",
        "class_level": 11,
        "session": None,
        "shift": None,
        "source_locator": "data/raw/physics.pdf",
        "inventory_status": "registered",
    }
    assert SourceRecord.model_validate(data)


def test_source_rejects_invalid_sha256():
    with pytest.raises(ValidationError):
        SourceRecord.model_validate({
            "source_id": "src-001",
            "logical_source_id": None,
            "source_type": "ncert",
            "original_filename": "physics.pdf",
            "sha256": "not-a-hash",
            "file_size_bytes": 1024,
            "subject": "Physics",
            "language": "en",
            "publication_year": 2024,
            "exam_year": None,
            "edition": "2024",
            "class_level": 11,
            "session": None,
            "shift": None,
            "source_locator": "data/raw/physics.pdf",
            "inventory_status": "registered",
        })


def test_source_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        SourceRecord.model_validate({"source_id": "src-001"})


def test_ncert_rejects_invalid_page_range():
    data = {
        "chapter_id": "ch-001",
        "source_id": "src-001",
        "subject": "Physics",
        "class_level": 11,
        "language": "en",
        "chapter_number": 1,
        "chapter_title": "Physical World",
        "page_start": 10,
        "page_end": 5,
        "text_raw": "",
        "text_cleaned": "",
        "content_blocks": [],
        "asset_ids": [],
        "extraction_status": "success",
    }
    with pytest.raises(ValidationError):
        NCERTChapterRecord.model_validate(data)

def test_valid_jee_question_record():
    data = {
        "question_id": "jee-2025-001",
        "source_id": "src-jee-2025",
        "year": 2025,
        "session": "January",
        "shift": "1",
        "subject": "Physics",
        "question_text": "A particle moves with constant velocity...",
        "options": [
            {"label": "A", "text": "10 m/s"},
            {"label": "B", "text": "20 m/s"},
            {"label": "C", "text": "30 m/s"},
            {"label": "D", "text": "40 m/s"},
        ],
        "correct_answer": "B",
        "solution_text": None,
        "source_page": 12,
        "asset_ids": [],
        "extraction_status": "success",
        "difficulty": "medium",
        "difficulty_source": "expert_review",
        "difficulty_status": "reviewed",
    }

    assert JEEQuestionRecord.model_validate(data)


def test_jee_rejects_inconsistent_difficulty_metadata():
    data = {
        "question_id": "jee-2025-002",
        "source_id": "src-jee-2025",
        "year": 2025,
        "session": "January",
        "shift": "1",
        "subject": "Physics",
        "question_text": "A particle moves with constant velocity...",
        "options": [],
        "correct_answer": None,
        "solution_text": None,
        "source_page": 12,
        "asset_ids": [],
        "extraction_status": "success",
        "difficulty": "medium",
        "difficulty_source": "unknown",
        "difficulty_status": "unknown",
    }

    with pytest.raises(ValidationError):
        JEEQuestionRecord.model_validate(data)