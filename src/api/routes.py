"""FastAPI API routes for answer evaluation, questions, health checks, and learner attempts."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status

from src.evaluation.attempts import AttemptStore
from src.evaluation.repository import QuestionRepository
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    AttemptRecord,
    EvaluationResponse,
)
from src.evaluation.service import EvaluationService, QuestionNotFoundError

router = APIRouter()

# Default shared service singletons
_repository = QuestionRepository()
_attempt_store = AttemptStore()
_retriever_adapter = NCERTRetrieverAdapter()
_service = EvaluationService(
    repository=_repository,
    retriever_adapter=_retriever_adapter,
    attempt_store=_attempt_store,
)


def get_repository() -> QuestionRepository:
    return _repository


def get_attempt_store() -> AttemptStore:
    return _attempt_store


def get_service() -> EvaluationService:
    return _service


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(response: Response = None) -> dict[str, Any]:
    """Health and readiness check endpoint for Person 1 / Docker."""
    try:
        count = _repository.count(subject="Chemistry")
        is_ready = _repository.is_ready
    except Exception:
        count = 0
        is_ready = False

    if not is_ready or count == 0:
        if response is not None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "service": "jee-evaluation-api",
            "subject": "Chemistry",
            "chemistry_questions_loaded": count,
            "corrupted_records_count": getattr(_repository, "corrupted_records_count", 0),
            "attempts_recorded": _attempt_store.count(),
            "reason": "Question repository has no questions loaded or is unavailable.",
        }
    return {
        "status": "healthy",
        "service": "jee-evaluation-api",
        "subject": "Chemistry",
        "chemistry_questions_loaded": count,
        "corrupted_records_count": _repository.corrupted_records_count,
        "attempts_recorded": _attempt_store.count(),
    }


@router.post(
    "/api/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
)
def evaluate_answer(submission: AnswerSubmissionRequest) -> EvaluationResponse:
    """Evaluate a student's answer submission (typed or OCR reviewed)."""
    try:
        return _service.evaluate_answer(submission)
    except QuestionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/api/attempts/{session_id}",
    response_model=list[AttemptRecord],
    status_code=status.HTTP_200_OK,
)
def get_learner_attempts(session_id: str) -> list[AttemptRecord]:
    """Retrieve attempt history for Person 4's personalization and recommendations."""
    return _attempt_store.get_attempts_for_session(session_id)


@router.get("/api/questions/{question_id}", status_code=status.HTTP_200_OK)
def get_question(question_id: str) -> dict[str, Any]:
    """Fetch public question details for learner rendering."""
    question = _repository.get_question(question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question with ID '{question_id}' not found.",
        )
    return {
        "question_id": question.question_id,
        "subject": question.subject,
        "year": question.year,
        "session": question.session,
        "shift": question.shift,
        "question_text": question.question_text,
        "options": [
            {"label": opt.label, "text": opt.text} for opt in question.options
        ],
        "difficulty": question.difficulty,
        "is_mcq": len(question.options) > 0,
    }


@router.get("/api/questions", status_code=status.HTTP_200_OK)
def list_questions(
    subject: str = "Chemistry",
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """List available questions for tutoring or practice."""
    questions = _repository.list_questions(subject=subject)[:limit]
    return {
        "subject": subject,
        "count": len(questions),
        "questions": [
            {
                "question_id": q.question_id,
                "question_text_preview": q.question_text[:100] + "...",
                "is_mcq": len(q.options) > 0,
            }
            for q in questions
        ],
    }
