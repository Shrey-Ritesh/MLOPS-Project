"""Evaluation package for the JEE Chemistry AI Tutor.

Provides deterministic answer evaluation, schema contracts, and feedback integration
for typed answers and OCR-transcribed submissions.
"""
from __future__ import annotations

from .schemas import (
    AnswerSubmissionRequest,
    AttemptRecord,
    EvaluationOutcome,
    EvaluationResponse,
    EvidenceItem,
    InputMode,
    QuestionType,
)

__all__ = [
    "AnswerSubmissionRequest",
    "AttemptRecord",
    "EvaluationOutcome",
    "EvaluationResponse",
    "EvidenceItem",
    "InputMode",
    "QuestionType",
]

