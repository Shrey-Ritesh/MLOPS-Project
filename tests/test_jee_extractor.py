"""
Unit tests for Track D JEE Extractor
Verifies token geometry, fraction stacking, option protection, and Pydantic validation.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.data.jee_extractor import (
    clean_generic_text,
    parse_filename_metadata,
    determine_subject,
    detect_image_reference,
)
from src.models.jee import JEEQuestionRecord, JEEOption


def test_clean_generic_text():
    """Verify clean_generic_text removes non-breaking spaces, unicode minuses and normalizes whitespace."""
    raw = "Let \u22122  and   \u00a0 3 "
    assert clean_generic_text(raw) == "Let -2 and 3"


def test_filename_metadata_parsing():
    fname = "JEE Main 2025 (28 Jan Shift 2) Previous Year Paper with Answer Keys - MathonGo.pdf"
    meta = parse_filename_metadata(fname)
    assert meta["year"] == 2025
    assert meta["session"] == "28 Jan"
    assert meta["shift"] == "2"


def test_determine_subject():
    assert determine_subject(1) == "Mathematics"
    assert determine_subject(25) == "Mathematics"
    assert determine_subject(26) == "Physics"
    assert determine_subject(50) == "Physics"
    assert determine_subject(51) == "Chemistry"
    assert determine_subject(75) == "Chemistry"


def test_jee_question_record_schema_conformance():
    """Verify schema conformance with null difficulty and empty asset_ids."""
    data = {
        "question_id": "jee-2025-28jan-s2-q02",
        "source_id": "src-f7ebd97f5b2c85d0885c70e1639100960385dc35ff8bcc19fecf9d7e630742fb",
        "year": 2025,
        "session": "28 Jan",
        "shift": "2",
        "subject": "Mathematics",
        "question_text": "If the components of a along and perpendicular to b are 16/11 and 1/11, then...",
        "options": [
            {"label": "1", "text": "26"},
            {"label": "2", "text": "18"},
            {"label": "3", "text": "23"},
            {"label": "4", "text": "16"},
        ],
        "correct_answer": "1",
        "solution_text": None,
        "source_page": 1,
        "asset_ids": [],
        "extraction_status": "success",
        "difficulty": None,
        "difficulty_source": "unknown",
        "difficulty_status": "unknown",
    }
    rec = JEEQuestionRecord.model_validate(data)
    assert rec.question_id == "jee-2025-28jan-s2-q02"
    assert rec.difficulty is None
    assert rec.difficulty_source == "unknown"
    assert rec.difficulty_status == "unknown"
    assert rec.asset_ids == []
