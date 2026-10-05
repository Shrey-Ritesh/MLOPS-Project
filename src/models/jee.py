from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.base import ContractModel


NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultySource(str, Enum):
    EXPERT_REVIEW = "expert_review"
    EMPIRICAL = "empirical"
    MODEL_ESTIMATE = "model_estimate"
    UNKNOWN = "unknown"


class DifficultyStatus(str, Enum):
    PROVISIONAL = "provisional"
    REVIEWED = "reviewed"
    VALIDATED = "validated"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

class JEEOption(ContractModel):
    """One answer option, preserving the source's label and order."""

    label: NonEmptyString
    text: str


class JEEQuestionRecord(ContractModel):
    """One JEE Main question; serialized to jee_questions.jsonl."""

    question_id: NonEmptyString
    source_id: NonEmptyString
    year: PositiveInt | None
    session: str | None
    shift: str | None
    subject: NonEmptyString
    question_text: str
    options: list[JEEOption]
    correct_answer: str | None
    solution_text: str | None
    source_page: PositiveInt
    asset_ids: list[NonEmptyString]
    extraction_status: ExtractionStatus
    difficulty: Difficulty | None
    difficulty_source: DifficultySource = DifficultySource.UNKNOWN
    difficulty_status: DifficultyStatus = DifficultyStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_question_record(self) -> "JEEQuestionRecord":
        if not self.question_text.strip() and self.extraction_status != ExtractionStatus.FAILED:
            raise ValueError(
                "question_text may be empty only when extraction_status='failed'"
            )

        if self.difficulty is None:
            if (
                self.difficulty_source != DifficultySource.UNKNOWN
                or self.difficulty_status != DifficultyStatus.UNKNOWN
            ):
                raise ValueError(
                    "When difficulty is null, difficulty_source and difficulty_status "
                    "must both be 'unknown'"
                )
        elif (
            self.difficulty_source == DifficultySource.UNKNOWN
            or self.difficulty_status == DifficultyStatus.UNKNOWN
        ):
            raise ValueError(
                "A non-null difficulty requires a known difficulty_source and "
                "a non-'unknown' difficulty_status"
            )

        if self.difficulty is not None and (
            self.difficulty_source == DifficultySource.UNKNOWN
            or self.difficulty_status == DifficultyStatus.UNKNOWN
        ):
            raise ValueError(
                "A not-null diffiulty level cannot exist without appropriate origin"
            )


        return self
