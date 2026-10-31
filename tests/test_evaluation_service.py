"""Unit tests for QuestionRepository and EvaluationService (Day 2).

Verifies:
- QuestionRepository indexed loading and subject filtering
- EvaluationService deterministic grading of MCQs (correct, incorrect, invalid option)
- EvaluationService numerical grading with tolerance (exact, tolerance pass, out-of-tolerance, unit stripping)
- Resilient error handling (missing question raises QuestionNotFoundError, missing answer key)
"""
from __future__ import annotations

import pytest

from src.evaluation.attempts import AttemptStore
from src.evaluation.repository import QuestionRepository
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    EvaluationOutcome,
    InputMode,
    QuestionType,
)
from src.evaluation.service import EvaluationService, QuestionNotFoundError
from src.models.jee import ExtractionStatus, JEEOption, JEEQuestionRecord


# --- Test Fixtures ---

@pytest.fixture
def mock_mcq_question():
    return JEEQuestionRecord(
        question_id="test-chem-mcq-01",
        source_id="src-test-01",
        year=2025,
        session="22 Jan",
        shift="1",
        subject="Chemistry",
        question_text="Which of the following compounds exhibits geometrical isomerism?",
        options=[
            JEEOption(label="1", text="Propene"),
            JEEOption(label="2", text="But-2-ene"),
            JEEOption(label="3", text="Ethene"),
            JEEOption(label="4", text="Methane"),
        ],
        correct_answer="2",
        solution_text=None,
        source_page=1,
        asset_ids=[],
        extraction_status=ExtractionStatus.SUCCESS,
        difficulty=None,
    )


@pytest.fixture
def mock_numerical_question():
    return JEEQuestionRecord(
        question_id="test-chem-num-01",
        source_id="src-test-02",
        year=2025,
        session="22 Jan",
        shift="1",
        subject="Chemistry",
        question_text="The percentage of sulphur in the compound is ______.",
        options=[],
        correct_answer="275",
        solution_text=None,
        source_page=2,
        asset_ids=[],
        extraction_status=ExtractionStatus.SUCCESS,
        difficulty=None,
    )


@pytest.fixture
def mock_no_key_question():
    return JEEQuestionRecord(
        question_id="test-chem-no-key-01",
        source_id="src-test-03",
        year=2025,
        session="22 Jan",
        shift="1",
        subject="Chemistry",
        question_text="A question with an unavailable answer key.",
        options=[],
        correct_answer=None,
        solution_text=None,
        source_page=3,
        asset_ids=[],
        extraction_status=ExtractionStatus.SUCCESS,
        difficulty=None,
    )


class MockQuestionRepository(QuestionRepository):
    """Test repository loaded with in-memory fixtures."""

    def __init__(self, questions: list[JEEQuestionRecord]) -> None:
        super().__init__(data_path="dummy_path.jsonl")
        self._questions_by_id = {q.question_id: q for q in questions}
        self._loaded = True


@pytest.fixture
def eval_service(mock_mcq_question, mock_numerical_question, mock_no_key_question, tmp_path):
    repo = MockQuestionRepository([
        mock_mcq_question,
        mock_numerical_question,
        mock_no_key_question,
    ])
    retriever = NCERTRetrieverAdapter(retrieve_fn=lambda *args, **kwargs: {"status": "ok", "ncert": []})
    attempt_store = AttemptStore(data_path=tmp_path / "attempts.jsonl")
    return EvaluationService(
        repository=repo,
        retriever_adapter=retriever,
        attempt_store=attempt_store,
    )


# --- Service Unit Tests ---

def test_evaluate_mcq_correct(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-mcq-01",
        answer_text="2",
        session_id="sess-01",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0
    assert resp.student_answer_normalized == "2"
    assert resp.question_type == QuestionType.MCQ
    assert "Option 2 is the right answer" in resp.feedback_text


def test_evaluate_mcq_correct_via_letter_alias(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-mcq-01",
        answer_text="Option B",  # Key is "2"
        session_id="sess-01",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.student_answer_normalized == "2"


def test_evaluate_mcq_incorrect(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-mcq-01",
        answer_text="1",  # Wrong option
        session_id="sess-01",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.INCORRECT
    assert resp.is_correct is False
    assert resp.score == 0.0
    assert resp.student_answer_normalized == "1"


def test_evaluate_mcq_invalid_option(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-mcq-01",
        answer_text="Option 9",  # Out of range
        session_id="sess-01",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.INVALID_INPUT
    assert resp.is_correct is None
    assert resp.score == 0.0
    assert "Invalid answer" in resp.feedback_text


def test_evaluate_numerical_correct(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-num-01",
        answer_text="275",
        session_id="sess-02",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0
    assert resp.student_answer_normalized == "275.0"
    assert resp.question_type == QuestionType.NUMERICAL


def test_evaluate_numerical_with_unit_and_scientific(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-num-01",
        answer_text="2.75 x 10^2 %",
        session_id="sess-02",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0


def test_evaluate_numerical_tolerance_pass(eval_service):
    # 274.5 is within 1% of 275 (0.18% difference)
    sub = AnswerSubmissionRequest(
        question_id="test-chem-num-01",
        answer_text="274.5",
        session_id="sess-02",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True


def test_evaluate_numerical_incorrect(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-num-01",
        answer_text="300",  # Out of tolerance
        session_id="sess-02",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.INCORRECT
    assert resp.is_correct is False
    assert resp.score == 0.0


def test_evaluate_numerical_invalid_text(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-num-01",
        answer_text="two hundred",
        session_id="sess-02",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.INVALID_INPUT
    assert resp.is_correct is None
    assert resp.score == 0.0


def test_evaluate_question_not_found(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="non-existent-question",
        answer_text="2",
        session_id="sess-03",
    )
    with pytest.raises(QuestionNotFoundError):
        eval_service.evaluate_answer(sub)


def test_evaluate_question_missing_answer_key(eval_service):
    sub = AnswerSubmissionRequest(
        question_id="test-chem-no-key-01",
        answer_text="2",
        session_id="sess-04",
    )
    resp = eval_service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.KEY_UNAVAILABLE
    assert resp.is_correct is None
    assert resp.score == 0.0
    assert "Answer key is unavailable" in resp.feedback_text


# --- Real Dataset Repository Test ---

def test_real_question_repository_loading():
    repo = QuestionRepository()
    repo.load()
    chem_questions = repo.list_questions(subject="Chemistry")
    assert len(chem_questions) == 215

    q = repo.get_question("jee-2025-22jan-s1-q52")
    assert q is not None
    assert q.subject == "Chemistry"
    assert q.correct_answer == "2"
