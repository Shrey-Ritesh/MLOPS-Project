"""Comprehensive failure-path resilience and API error semantics tests (Day 5).

Verifies:
- Vector store outages, socket timeouts, empty hits, and malformed hit formats
- Disputed, dropped, cancelled, or unparseable answer keys (MCQ & Numerical)
- Attempt store disk write failures fail soft (evaluations are never blocked)
- Question repository skips corrupted lines and tracks corrupted record counts
- Missing question repository raises RepositoryUnavailableError
- FastAPI HTTP error semantics (404, 422, 500, 503) with structured error envelopes
- Sanitized 500 error envelopes that never leak stack traces or system paths
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.evaluation.attempts import AttemptStore
from src.evaluation.repository import QuestionRepository, RepositoryUnavailableError
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    EvaluationOutcome,
    EvidenceStatus,
    InputMode,
)
from src.evaluation.service import EvaluationService, QuestionNotFoundError
from src.models.jee import ExtractionStatus, JEEOption, JEEQuestionRecord


# --- Fixtures ---

@pytest.fixture
def mock_mcq_with_key():
    def _create(correct_answer: str | None = "2"):
        return JEEQuestionRecord(
            question_id="fail-mcq-01",
            source_id="src-fail-01",
            year=2025,
            session="jan",
            shift="s1",
            subject="Chemistry",
            question_text="Which gas is evolved when zinc reacts with dilute hydrochloric acid?",
            options=[
                JEEOption(label="1", text="Oxygen"),
                JEEOption(label="2", text="Hydrogen"),
                JEEOption(label="3", text="Chlorine"),
                JEEOption(label="4", text="Nitrogen"),
            ],
            correct_answer=correct_answer,
            solution_text=None,
            source_page=1,
            asset_ids=[],
            extraction_status=ExtractionStatus.SUCCESS,
            difficulty=None,
        )
    return _create


@pytest.fixture
def mock_num_with_key():
    def _create(correct_answer: str | None = "275"):
        return JEEQuestionRecord(
            question_id="fail-num-01",
            source_id="src-fail-02",
            year=2025,
            session="jan",
            shift="s1",
            subject="Chemistry",
            question_text="Calculate the mass of precipitate formed in grams.",
            options=[],
            correct_answer=correct_answer,
            solution_text=None,
            source_page=2,
            asset_ids=[],
            extraction_status=ExtractionStatus.SUCCESS,
            difficulty=None,
        )
    return _create


class SimpleRepo(QuestionRepository):
    def __init__(self, questions: list[JEEQuestionRecord]):
        super().__init__(data_path="dummy.jsonl")
        self._questions_by_id = {q.question_id: q for q in questions}
        self._loaded = True


# ==============================================================================
# 1. RETRIEVER ADAPTER FAILURE TESTS
# ==============================================================================

def test_retriever_offline_socket_returns_retrieval_unavailable():
    """Verify retriever fast-probe returns RETRIEVAL_UNAVAILABLE when host is unreachable."""
    adapter = NCERTRetrieverAdapter(chroma_host="192.0.2.1", chroma_port=65432)  # Non-routable TEST-NET
    items, status = adapter.fetch_evidence(query="Equilibrium constant Kc")
    assert items == []
    assert status == EvidenceStatus.RETRIEVAL_UNAVAILABLE


def test_retriever_exception_in_retrieve_fn():
    """Verify unhandled exceptions in retrieve_fn gracefully return RETRIEVAL_UNAVAILABLE."""
    def broken_retrieve(query, **kw):
        raise ConnectionResetError("Remote server closed connection unexpectedly")

    adapter = NCERTRetrieverAdapter(retrieve_fn=broken_retrieve)
    items, status = adapter.fetch_evidence(query="Thermodynamics")
    assert items == []
    assert status == EvidenceStatus.RETRIEVAL_UNAVAILABLE


def test_retriever_empty_hits_returns_insufficient_evidence():
    """Verify empty hits return INSUFFICIENT_EVIDENCE rather than RETRIEVAL_UNAVAILABLE."""
    adapter = NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: {"status": "ok", "ncert": []})
    items, status = adapter.fetch_evidence(query="Unknown topic with no match")
    assert items == []
    assert status == EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_retriever_non_dict_and_non_list_payloads():
    """Verify retriever handles non-standard return formats without crashing."""
    # Returns a string instead of dict
    adapter_str = NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: "unexpected error string")
    items, status = adapter_str.fetch_evidence(query="Test")
    assert items == []
    assert status == EvidenceStatus.INSUFFICIENT_EVIDENCE

    # Returns ncert as string instead of list
    adapter_bad_list = NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: {"status": "ok", "ncert": "not a list"})
    items, status = adapter_bad_list.fetch_evidence(query="Test")
    assert items == []
    assert status == EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_retriever_corrupted_hit_metadata():
    """Verify retriever safely skips non-dict hits and handles malformed metadata."""
    malformed_hits = [
        "not a dict hit",
        {"id": "hit-1"},  # Missing document
        {
            "id": "hit-2",
            "document": "Valid NCERT excerpt regarding Le Chatelier's principle.",
            "metadata": {"chapter_title": "Equilibrium", "page": "invalid_page_str"},
            "score": "not_a_float",
        },
    ]
    adapter = NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: {"status": "ok", "ncert": malformed_hits})
    items, status = adapter.fetch_evidence(query="Equilibrium")
    assert len(items) == 1
    assert items[0].unit_id == "hit-2"
    assert items[0].page is None  # Safely fell back to None
    assert items[0].relevance_score is None  # Safely fell back to None
    assert status == EvidenceStatus.GROUNDED


# ==============================================================================
# 2. DISPUTED, DROPPED, AND UNPARSEABLE ANSWER KEYS
# ==============================================================================

def test_mcq_dropped_answer_key_safeguard(mock_mcq_with_key):
    """Verify question with dropped/cancelled key (e.g. 'DROP') returns KEY_UNAVAILABLE."""
    q_drop = mock_mcq_with_key(correct_answer="DROP")
    repo = SimpleRepo([q_drop])
    service = EvaluationService(repository=repo)

    sub = AnswerSubmissionRequest(
        question_id=q_drop.question_id,
        answer_text="2",
        session_id="test-session",
    )
    resp = service.evaluate_answer(sub)
    assert resp.outcome == EvaluationOutcome.KEY_UNAVAILABLE
    assert resp.is_correct is None
    assert resp.score == 0.0
    assert "disputed" in resp.feedback_text.lower() or "dropped" in resp.feedback_text.lower()


def test_mcq_out_of_range_answer_key(mock_mcq_with_key):
    """Verify question with key outside options (e.g. '9' for 4 options) returns KEY_UNAVAILABLE."""
    q_invalid = mock_mcq_with_key(correct_answer="9")
    repo = SimpleRepo([q_invalid])
    service = EvaluationService(repository=repo)

    sub = AnswerSubmissionRequest(
        question_id=q_invalid.question_id,
        answer_text="Option 2",
        session_id="test-session",
    )
    resp = service.evaluate_answer(sub)
    assert resp.outcome == EvaluationOutcome.KEY_UNAVAILABLE
    assert resp.is_correct is None


def test_numerical_cancelled_or_unparseable_answer_key(mock_num_with_key):
    """Verify numerical question with non-numeric key ('Cancelled') returns KEY_UNAVAILABLE."""
    q_cancelled = mock_num_with_key(correct_answer="Cancelled")
    repo = SimpleRepo([q_cancelled])
    service = EvaluationService(repository=repo)

    sub = AnswerSubmissionRequest(
        question_id=q_cancelled.question_id,
        answer_text="275",
        session_id="test-session",
    )
    resp = service.evaluate_answer(sub)
    assert resp.outcome == EvaluationOutcome.KEY_UNAVAILABLE
    assert resp.is_correct is None
    assert resp.score == 0.0
    assert "could not be parsed" in resp.feedback_text.lower()


def test_absent_empty_answer_key(mock_mcq_with_key):
    """Verify missing/whitespace answer key cleanly returns KEY_UNAVAILABLE."""
    q_empty = mock_mcq_with_key(correct_answer="   ")
    repo = SimpleRepo([q_empty])
    service = EvaluationService(repository=repo)

    sub = AnswerSubmissionRequest(
        question_id=q_empty.question_id,
        answer_text="2",
        session_id="test-session",
    )
    resp = service.evaluate_answer(sub)
    assert resp.outcome == EvaluationOutcome.KEY_UNAVAILABLE
    assert resp.is_correct is None


# ==============================================================================
# 3. ATTEMPT STORAGE FAIL-SOFT ISOLATION
# ==============================================================================

def test_attempt_store_disk_write_error_fails_soft(mock_mcq_with_key):
    """Verify evaluation succeeds even if the underlying disk persistence raises an OSError."""
    q = mock_mcq_with_key(correct_answer="2")
    repo = SimpleRepo([q])
    store = AttemptStore(data_path="test_readonly_attempts.jsonl")

    # Simulate disk failure or read-only volume on write
    with patch.object(store, "record_attempt", side_effect=OSError("Read-only file system")):
        service = EvaluationService(repository=repo, attempt_store=store)
        sub = AnswerSubmissionRequest(
            question_id=q.question_id,
            answer_text="2",
            session_id="session-readonly",
        )
        resp = service.evaluate_answer(sub)
        # Evaluation response MUST still succeed and be returned to the student
        assert resp.outcome == EvaluationOutcome.CORRECT
        assert resp.is_correct is True
        assert resp.score == 1.0


def test_attempt_store_skips_corrupted_lines(tmp_path):
    """Verify AttemptStore handles corrupted lines in existing attempts.jsonl without crashing."""
    attempts_file = tmp_path / "attempts.jsonl"
    attempts_file.write_text(
        '{"attempt_id": "att-01", "session_id": "s1", "question_id": "q1", "subject": "Chemistry", "concept_tags": [], "input_mode": "typed", "outcome": "correct", "score": 1.0, "timestamp": "2026-10-29T12:00:00Z"}\n'
        'CORRUPTED_NON_JSON_LINE\n'
        '{"incomplete_record": true}\n'
        '{"attempt_id": "att-02", "session_id": "s1", "question_id": "q2", "subject": "Chemistry", "concept_tags": [], "input_mode": "typed", "outcome": "incorrect", "score": 0.0, "timestamp": "2026-10-29T12:01:00Z"}\n',
        encoding="utf-8",
    )

    store = AttemptStore(data_path=attempts_file)
    history = store.get_attempts_for_session("s1")
    assert len(history) == 2
    assert history[0].attempt_id == "att-01"
    assert history[1].attempt_id == "att-02"


# ==============================================================================
# 4. QUESTION REPOSITORY CORRUPTION & READINESS
# ==============================================================================

def test_repository_missing_file_raises_unavailable(tmp_path):
    """Verify missing questions dataset file raises RepositoryUnavailableError."""
    missing_path = tmp_path / "non_existent_dataset.jsonl"
    repo = QuestionRepository(data_path=missing_path)
    with pytest.raises(RepositoryUnavailableError):
        repo.load()
    assert repo.is_ready is False


def test_repository_tracks_corrupted_lines(tmp_path):
    """Verify QuestionRepository skips corrupted lines and accurately tracks corrupted_records_count."""
    dataset_file = tmp_path / "test_questions.jsonl"
    dataset_file.write_text(
        '{"question_id": "q1", "source_id": "s1", "year": 2025, "session": "jan", "shift": "1", "subject": "Chemistry", "question_text": "What is water?", "options": [], "correct_answer": "H2O", "source_page": 1, "asset_ids": [], "extraction_status": "success", "difficulty": null, "solution_text": null}\n'
        'INVALID_RAW_SYNTAX_JSON\n'
        '{"incomplete": "record_missing_fields"}\n',
        encoding="utf-8",
    )

    repo = QuestionRepository(data_path=dataset_file)
    repo.load()
    assert repo.count() == 1
    assert repo.corrupted_records_count == 2
    assert repo.is_ready is True


# ==============================================================================
# 5. FASTAPI HTTP ERROR SEMANTICS & ERROR ENVELOPES
# ==============================================================================

def test_http_404_question_not_found():
    """Verify HTTP 404 response structure and error envelope for non-existent questions."""
    import asyncio
    from fastapi import HTTPException
    from starlette.requests import Request
    from src.api.routes import get_question, evaluate_answer

    # 1. Question details endpoint raises HTTP 404
    with pytest.raises(HTTPException) as exc_info:
        get_question("jee-ghost-question-999")
    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail).lower()

    # 2. Evaluation endpoint raises HTTP 404 for non-existent question
    sub = AnswerSubmissionRequest(
        question_id="jee-ghost-question-999",
        answer_text="2",
        input_mode=InputMode.TYPED,
        session_id="test-sess",
    )
    with pytest.raises(HTTPException) as exc_info_eval:
        evaluate_answer(sub)
    assert exc_info_eval.value.status_code == 404

    # 3. Verify exception handler produces structured error envelope
    app = create_app()
    req = Request({"type": "http", "method": "GET", "path": "/api/questions/ghost"})
    handler = app.exception_handlers[HTTPException]
    resp = handler(req, exc_info.value)
    if hasattr(resp, "__await__"):
        resp = asyncio.run(resp)
    assert resp.status_code == 404
    data = json.loads(resp.body.decode("utf-8"))
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
    assert data["error"]["status_code"] == 404
    assert "not found" in data["error"]["message"].lower()


def test_http_422_validation_error_envelope():
    """Verify HTTP 422 schema validation returns a structured, sanitized error envelope."""
    import asyncio
    from pydantic import ValidationError
    from fastapi.exceptions import RequestValidationError
    from starlette.requests import Request

    # Sending payload with extra forbidden field (ContractModel extra="forbid")
    payload = {
        "question_id": "jee-2025-22jan-s1-q52",
        "answer_text": "2",
        "session_id": "test-sess",
        "malicious_extra_field": "injected",
    }
    with pytest.raises(ValidationError) as exc_info:
        AnswerSubmissionRequest.model_validate(payload)

    app = create_app()
    req = Request({"type": "http", "method": "POST", "path": "/api/evaluate"})
    handler = app.exception_handlers[RequestValidationError]
    resp = handler(req, RequestValidationError(exc_info.value.errors()))
    if hasattr(resp, "__await__"):
        resp = asyncio.run(resp)
    assert resp.status_code == 422
    data = json.loads(resp.body.decode("utf-8"))
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["status_code"] == 422
    assert "extra_field" in data["error"]["message"].lower() or "extra fields not permitted" in str(data["detail"]).lower()


def test_http_503_health_check_when_repository_empty():
    """Verify /health returns HTTP 503 and unhealthy status when repository is empty."""
    from starlette.responses import Response
    from src.api.routes import get_repository, health_check
    repo = get_repository()

    # Temporarily simulate empty repository
    original_questions = repo._questions_by_id.copy()
    try:
        repo._questions_by_id = {}
        res = Response()
        data = health_check(response=res)
        assert res.status_code == 503
        assert data["status"] == "unhealthy"
        assert "unavailable" in data["reason"].lower() or "no questions" in data["reason"].lower()
    finally:
        repo._questions_by_id = original_questions


def test_http_500_sanitized_error_envelope():
    """Verify unhandled exceptions return a sanitized HTTP 500 without leaking stack traces."""
    import asyncio
    from starlette.requests import Request
    app = create_app()

    req = Request({"type": "http", "method": "GET", "path": "/api/test-crash"})
    fake_exc = RuntimeError("Database connection string leaked: postgres://user:secret@internal:5432/db")

    handler = app.exception_handlers[Exception]
    resp = handler(req, fake_exc)
    if hasattr(resp, "__await__"):
        resp = asyncio.run(resp)
    assert resp.status_code == 500
    data = json.loads(resp.body.decode("utf-8"))

    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["status_code"] == 500
    # Must NOT leak secret credentials or system stack traces
    assert "secret" not in data["error"]["message"]
    assert "Traceback" not in data["error"]["message"]
    assert "postgres://" not in data["error"]["message"]

