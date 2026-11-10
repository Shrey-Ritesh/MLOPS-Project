"""Integration tests for the complete typed-answer vertical slice (Day 3).

Verifies:
- Question -> Typed Answer -> Grading -> Grounded NCERT Feedback -> Persisted Attempt
- Traceable NCERT citations with chapter and page provenance
- Graceful degradation when Chroma vector database is offline
- Learner attempt persistence and retrieval via API (for Person 4)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.api.routes import evaluate_answer, get_learner_attempts
from src.evaluation.attempts import AttemptStore
from src.evaluation.repository import QuestionRepository
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    AttemptRecord,
    EvaluationOutcome,
    EvidenceItem,
    EvidenceStatus,
    InputMode,
    QuestionType,
)
from src.evaluation.service import EvaluationService
from src.models.jee import ExtractionStatus, JEEOption, JEEQuestionRecord


# --- Fixtures ---

@pytest.fixture
def sample_question():
    return JEEQuestionRecord(
        question_id="vert-chem-q01",
        source_id="src-vert-01",
        year=2025,
        session="22 Jan",
        shift="1",
        subject="Chemistry",
        question_text="The incorrect statements regarding geometrical isomerism are : Propene shows geometrical isomerism.",
        options=[
            JEEOption(label="1", text="Propene shows geometrical isomerism"),
            JEEOption(label="2", text="Propene does not show geometrical isomerism"),
        ],
        correct_answer="2",
        solution_text=None,
        source_page=9,
        asset_ids=[],
        extraction_status=ExtractionStatus.SUCCESS,
        difficulty=None,
    )


def mock_successful_retrieve(query: str, level: str = "medium", subject: str = "Chemistry") -> dict[str, Any]:
    return {
        "status": "ok",
        "query": query,
        "ncert": [
            {
                "id": "ncert-hydrocarbons-unit-42",
                "document": "Propene cannot show geometrical isomerism because one of the doubly bonded carbon atoms carries two identical hydrogen atoms.",
                "metadata": {
                    "unit_id": "ncert-hydrocarbons-unit-42",
                    "chapter_title": "Hydrocarbons - Alkenes",
                    "page": 378,
                    "subject": "Chemistry",
                },
                "score": 0.88,
            }
        ],
        "jee_questions": [],
    }


def mock_failing_retrieve(query: str, level: str = "medium", subject: str = "Chemistry") -> dict[str, Any]:
    raise ConnectionError("ChromaDB vector store connection refused on localhost:8000")


# --- 1. Retriever Adapter Tests ---

def test_retriever_adapter_success():
    adapter = NCERTRetrieverAdapter(retrieve_fn=mock_successful_retrieve)
    evidence, status = adapter.fetch_evidence("geometrical isomerism propene", subject="Chemistry")

    assert status == EvidenceStatus.GROUNDED
    assert len(evidence) == 1
    assert evidence[0].unit_id == "ncert-hydrocarbons-unit-42"
    assert evidence[0].chapter_title == "Hydrocarbons - Alkenes"
    assert evidence[0].page == 378
    assert "Propene cannot show geometrical isomerism" in evidence[0].excerpt
    assert evidence[0].relevance_score == 0.88


def test_retriever_adapter_handles_offline_chroma_safely():
    adapter = NCERTRetrieverAdapter(retrieve_fn=mock_failing_retrieve)
    evidence, status = adapter.fetch_evidence("geometrical isomerism", subject="Chemistry")

    assert status == EvidenceStatus.RETRIEVAL_UNAVAILABLE
    assert evidence == []


def test_retriever_adapter_empty_query():
    adapter = NCERTRetrieverAdapter(retrieve_fn=mock_successful_retrieve)
    evidence, status = adapter.fetch_evidence("   ")

    assert status == EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert evidence == []


# --- 2. Attempt Store Tests ---

def test_attempt_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "attempts.jsonl"
        store = AttemptStore(data_path=store_path)

        assert store.count() == 0

        attempt = AttemptRecord(
            attempt_id="att-test-01",
            session_id="learner-session-A",
            question_id="vert-chem-q01",
            subject="Chemistry",
            concept_tags=["chemistry"],
            input_mode=InputMode.TYPED,
            outcome=EvaluationOutcome.CORRECT,
            score=1.0,
        )
        store.record_attempt(attempt)

        assert store.count() == 1
        records_a = store.get_attempts_for_session("learner-session-A")
        assert len(records_a) == 1
        assert records_a[0].attempt_id == "att-test-01"

        records_b = store.get_attempts_for_session("learner-session-B")
        assert len(records_b) == 0


# --- 3. End-to-End Vertical Slice Tests ---

def test_vertical_slice_correct_submission_with_grounded_evidence(sample_question):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = QuestionRepository(data_path="dummy.jsonl")
        repo._questions_by_id = {sample_question.question_id: sample_question}
        repo._loaded = True

        retriever = NCERTRetrieverAdapter(retrieve_fn=mock_successful_retrieve)
        attempt_store = AttemptStore(data_path=Path(tmpdir) / "attempts.jsonl")

        service = EvaluationService(
            repository=repo,
            retriever_adapter=retriever,
            attempt_store=attempt_store,
        )

        sub = AnswerSubmissionRequest(
            question_id="vert-chem-q01",
            answer_text="2",
            input_mode=InputMode.TYPED,
            session_id="learner-42",
        )
        response = service.evaluate_answer(sub)

        # 1. Verification of Grading
        assert response.outcome == EvaluationOutcome.CORRECT
        assert response.is_correct is True
        assert response.score == 1.0

        # 2. Verification of Grounded Evidence & Provenance
        assert response.evidence_status == EvidenceStatus.GROUNDED
        assert len(response.evidence) == 1
        assert response.evidence[0].chapter_title == "Hydrocarbons - Alkenes"
        assert response.evidence[0].page == 378
        assert "Reference: NCERT Hydrocarbons - Alkenes (p. 378)" in response.feedback_text

        # 3. Verification of Attempt Persistence
        attempts = attempt_store.get_attempts_for_session("learner-42")
        assert len(attempts) == 1
        assert attempts[0].outcome == EvaluationOutcome.CORRECT
        assert attempts[0].score == 1.0


def test_vertical_slice_incorrect_submission_with_misconception_explanation(sample_question):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = QuestionRepository(data_path="dummy.jsonl")
        repo._questions_by_id = {sample_question.question_id: sample_question}
        repo._loaded = True

        retriever = NCERTRetrieverAdapter(retrieve_fn=mock_successful_retrieve)
        attempt_store = AttemptStore(data_path=Path(tmpdir) / "attempts.jsonl")

        service = EvaluationService(
            repository=repo,
            retriever_adapter=retriever,
            attempt_store=attempt_store,
        )

        sub = AnswerSubmissionRequest(
            question_id="vert-chem-q01",
            answer_text="1",  # Incorrect choice
            input_mode=InputMode.TYPED,
            session_id="learner-42",
        )
        response = service.evaluate_answer(sub)

        # 1. Verification of Grading
        assert response.outcome == EvaluationOutcome.INCORRECT
        assert response.is_correct is False
        assert response.score == 0.0

        # 2. Verification of Misconception Feedback with Citation
        assert "Incorrect" in response.feedback_text
        assert "recorded correct answer is option 2" in response.feedback_text
        assert "NCERT Hydrocarbons - Alkenes (p. 378)" in response.feedback_text

        # 3. Verification of Attempt Persistence
        attempts = attempt_store.get_attempts_for_session("learner-42")
        assert len(attempts) == 1
        assert attempts[0].outcome == EvaluationOutcome.INCORRECT
        assert attempts[0].score == 0.0


def test_vertical_slice_fallback_when_chroma_offline(sample_question):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = QuestionRepository(data_path="dummy.jsonl")
        repo._questions_by_id = {sample_question.question_id: sample_question}
        repo._loaded = True

        # Vector store throws connection error
        retriever = NCERTRetrieverAdapter(retrieve_fn=mock_failing_retrieve)
        attempt_store = AttemptStore(data_path=Path(tmpdir) / "attempts.jsonl")

        service = EvaluationService(
            repository=repo,
            retriever_adapter=retriever,
            attempt_store=attempt_store,
        )

        sub = AnswerSubmissionRequest(
            question_id="vert-chem-q01",
            answer_text="2",
            input_mode=InputMode.TYPED,
            session_id="learner-43",
        )
        response = service.evaluate_answer(sub)

        # Grading succeeds with zero crashes
        assert response.outcome == EvaluationOutcome.CORRECT
        assert response.is_correct is True
        assert response.evidence_status == EvidenceStatus.RETRIEVAL_UNAVAILABLE
        assert response.evidence == []

        # Attempt is still safely logged
        attempts = attempt_store.get_attempts_for_session("learner-43")
        assert len(attempts) == 1


def test_api_attempt_history_retrieval():
    sub = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q52",
        answer_text="2",
        input_mode=InputMode.TYPED,
        session_id="history-test-session-99",
    )
    evaluate_answer(sub)

    attempts = get_learner_attempts("history-test-session-99")
    assert len(attempts) >= 1
    assert attempts[-1].session_id == "history-test-session-99"
    assert attempts[-1].outcome == EvaluationOutcome.CORRECT

