"""Pydantic contract schemas for the JEE Tutor evaluation service.

These schemas define the interface for:
- Answer submission (typed and handwritten/OCR)
- Evaluation results and feedback
- NCERT evidence provenance
- Learner attempt persistence
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field

from src.models.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class InputMode(str, Enum):
    """Input channel used by the student."""
    TYPED = "typed"
    HANDWRITTEN_OCR = "handwritten_ocr"


class QuestionType(str, Enum):
    """Class of question being evaluated."""
    MCQ = "mcq"
    NUMERICAL = "numerical"
    CHEMICAL_FORMULA = "chemical_formula"
    SHORT_TEXT = "short_text"
    UNKNOWN = "unknown"


class EvaluationOutcome(str, Enum):
    """High-level outcome of an answer evaluation."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_TYPE = "unsupported_type"
    UNCERTAIN = "uncertain"
    KEY_UNAVAILABLE = "key_unavailable"


class EvidenceStatus(str, Enum):
    """Status of NCERT retrieval for feedback grounding."""
    GROUNDED = "grounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"


class EvidenceItem(ContractModel):
    """Grounded excerpt from NCERT content with traceable provenance."""
    unit_id: NonEmptyString
    chapter_title: str | None = None
    page: int | None = Field(default=None, ge=1)
    excerpt: NonEmptyString
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AnswerSubmissionRequest(ContractModel):
    """Request payload sent to POST /api/evaluate."""
    question_id: NonEmptyString
    answer_text: str
    input_mode: InputMode = InputMode.TYPED
    session_id: NonEmptyString
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_user_corrected: bool = False


class EvaluationResponse(ContractModel):
    """Response payload returned by the evaluation service."""
    submission_id: NonEmptyString
    question_id: NonEmptyString
    question_type: QuestionType
    outcome: EvaluationOutcome
    is_correct: bool | None = None
    score: float = Field(ge=0.0, le=1.0)
    student_answer_normalized: str | None = None
    feedback_text: str
    evidence_status: EvidenceStatus = EvidenceStatus.GROUNDED
    evidence: list[EvidenceItem] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AttemptRecord(ContractModel):
    """Persistent record of a student attempt, handed off for personalization."""
    attempt_id: NonEmptyString
    session_id: NonEmptyString
    question_id: NonEmptyString
    subject: str = "Chemistry"
    concept_tags: list[str] = Field(default_factory=list)
    input_mode: InputMode
    outcome: EvaluationOutcome
    score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APIErrorDetail(ContractModel):
    """Structured error details for client consumption."""
    code: str
    message: str
    status_code: int
    details: Any | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APIErrorResponse(ContractModel):
    """Top-level error response envelope ensuring consistent API errors."""
    error: APIErrorDetail
    detail: str


