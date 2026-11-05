"""API endpoint tests for the JEE Tutor FastAPI routes (Day 2).

Verifies:
- FastAPI application initialization and OpenAPI route registration
- Health check endpoint contract for Person 1 / Docker
- Question retrieval and 404 handling
- Answer evaluation endpoint behavior for MCQ and Numerical questions
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.app import create_app
from src.api.routes import (
    evaluate_answer,
    get_question,
    health_check,
    list_questions,
)
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    EvaluationOutcome,
    InputMode,
)


def test_fastapi_app_initialization():
    app = create_app()
    assert app.title == "JEE Chemistry Tutor Evaluation API"
    route_paths = set(app.openapi().get("paths", {}).keys())
    for route in app.routes:
        if hasattr(route, "path"):
            route_paths.add(route.path)
        elif hasattr(route, "routes"):
            route_paths.update(r.path for r in route.routes if hasattr(r, "path"))
        elif hasattr(route, "router"):
            route_paths.update(r.path for r in getattr(route.router, "routes", []) if hasattr(r, "path"))
    assert "/health" in route_paths
    assert "/api/evaluate" in route_paths
    assert "/api/questions/{question_id}" in route_paths
    assert "/api/questions" in route_paths


def test_health_check_endpoint():
    result = health_check()
    assert result["status"] == "healthy"
    assert result["service"] == "jee-evaluation-api"
    assert result["chemistry_questions_loaded"] > 0


def test_get_question_success():
    result = get_question("jee-2025-22jan-s1-q52")
    assert result["question_id"] == "jee-2025-22jan-s1-q52"
    assert result["subject"] == "Chemistry"
    assert result["is_mcq"] is True
    assert len(result["options"]) == 4


def test_get_question_not_found():
    with pytest.raises(HTTPException) as exc_info:
        get_question("jee-non-existent-id")
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_list_questions_endpoint():
    result = list_questions(subject="Chemistry", limit=5)
    assert result["subject"] == "Chemistry"
    assert result["count"] == 5
    assert len(result["questions"]) == 5


def test_evaluate_mcq_correct():
    submission = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q52",
        answer_text="2",
        input_mode=InputMode.TYPED,
        session_id="test-session-01",
    )
    resp = evaluate_answer(submission)
    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0
    assert resp.student_answer_normalized == "2"


def test_evaluate_mcq_incorrect():
    submission = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q52",
        answer_text="Option 1",
        input_mode=InputMode.TYPED,
        session_id="test-session-01",
    )
    resp = evaluate_answer(submission)
    assert resp.outcome == EvaluationOutcome.INCORRECT
    assert resp.is_correct is False
    assert resp.score == 0.0


def test_evaluate_mcq_invalid_option():
    submission = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q52",
        answer_text="Option 9",
        input_mode=InputMode.TYPED,
        session_id="test-session-01",
    )
    resp = evaluate_answer(submission)
    assert resp.outcome == EvaluationOutcome.INVALID_INPUT
    assert resp.is_correct is None
    assert resp.score == 0.0


def test_evaluate_numerical_correct():
    submission = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q71",
        answer_text="6.0",
        input_mode=InputMode.TYPED,
        session_id="test-session-02",
    )
    resp = evaluate_answer(submission)
    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0


def test_evaluate_question_not_found():
    submission = AnswerSubmissionRequest(
        question_id="jee-does-not-exist",
        answer_text="2",
        input_mode=InputMode.TYPED,
        session_id="test-session-03",
    )
    with pytest.raises(HTTPException) as exc_info:
        evaluate_answer(submission)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()
