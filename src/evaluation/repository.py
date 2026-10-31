"""Repository for loading, indexing, and accessing JEE questions.

Provides in-memory indexed access to JEEQuestionRecord objects loaded
from the canonical data/processed/jee_questions.jsonl dataset.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.models.jee import JEEQuestionRecord

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = Path("data/processed/jee_questions.jsonl")


class RepositoryUnavailableError(FileNotFoundError):
    """Raised when the question repository dataset file is missing or inaccessible."""
    pass


class QuestionRepository:
    """In-memory indexed store for JEE questions."""

    def __init__(self, data_path: str | Path = DEFAULT_QUESTIONS_PATH) -> None:
        self.data_path = Path(data_path)
        self._questions_by_id: dict[str, JEEQuestionRecord] = {}
        self.corrupted_records_count: int = 0
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        """Return True if questions have been successfully loaded and count > 0."""
        try:
            self._ensure_loaded()
            return len(self._questions_by_id) > 0
        except Exception:
            return False

    def load(self) -> None:
        """Load and index questions from the JSONL file."""
        if not self.data_path.is_file():
            raise RepositoryUnavailableError(f"Questions data file not found: {self.data_path.resolve()}")

        self._questions_by_id.clear()
        self.corrupted_records_count = 0
        with self.data_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    record_dict = json.loads(raw_line)
                    question = JEEQuestionRecord.model_validate(record_dict)
                    self._questions_by_id[question.question_id] = question
                except Exception as exc:
                    self.corrupted_records_count += 1
                    logger.warning(
                        "Skipping invalid question record at %s:%d: %s",
                        self.data_path,
                        line_no,
                        exc,
                    )

        self._loaded = True
        logger.info("Loaded %d questions from %s", len(self._questions_by_id), self.data_path)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_question(self, question_id: str) -> JEEQuestionRecord | None:
        """Fetch a single question by question_id."""
        self._ensure_loaded()
        return self._questions_by_id.get(question_id.strip())

    def list_questions(self, subject: str | None = None) -> list[JEEQuestionRecord]:
        """List questions, optionally filtered by subject (e.g. 'Chemistry')."""
        self._ensure_loaded()
        if subject is None:
            return list(self._questions_by_id.values())
        target_subject = subject.casefold().strip()
        return [
            q for q in self._questions_by_id.values()
            if q.subject.casefold().strip() == target_subject
        ]

    def count(self, subject: str | None = None) -> int:
        """Return the number of loaded questions."""
        return len(self.list_questions(subject=subject))

