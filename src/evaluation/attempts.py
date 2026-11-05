"""Persistence layer for learner attempts.

Stores student evaluation attempts to disk in JSONL format and maintains an in-memory
index for quick retrieval by Person 4's personalization and recommendation modules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .schemas import AttemptRecord

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPTS_PATH = Path("data/attempts/attempts.jsonl")


class AttemptStore:
    """Thread-safe, append-only persistence store for student attempts."""

    def __init__(self, data_path: str | Path = DEFAULT_ATTEMPTS_PATH) -> None:
        self.data_path = Path(data_path)
        self._in_memory_records: list[AttemptRecord] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._in_memory_records.clear()
        if self.data_path.is_file():
            try:
                with self.data_path.open("r", encoding="utf-8") as stream:
                    for line_no, line in enumerate(stream, start=1):
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            data = json.loads(raw)
                            record = AttemptRecord.model_validate(data)
                            self._in_memory_records.append(record)
                        except Exception as exc:
                            logger.warning(
                                "Skipping invalid attempt record at %s:%d: %s",
                                self.data_path,
                                line_no,
                                exc,
                            )
            except Exception as exc:
                logger.warning("Could not read attempt store file %s: %s", self.data_path, exc)
        self._loaded = True

    def record_attempt(self, attempt: AttemptRecord) -> None:
        """Persist a single attempt record to memory and append to disk."""
        self._ensure_loaded()
        self._in_memory_records.append(attempt)

        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with self.data_path.open("a", encoding="utf-8") as stream:
                stream.write(attempt.model_dump_json() + "\n")
        except Exception as exc:
            logger.error("Failed to write attempt %s to %s: %s", attempt.attempt_id, self.data_path, exc)

    def get_attempts_for_session(self, session_id: str) -> list[AttemptRecord]:
        """Retrieve all attempts made by a specific session/student."""
        self._ensure_loaded()
        target = session_id.strip()
        return [att for att in self._in_memory_records if att.session_id == target]

    def get_attempts_for_question(self, question_id: str) -> list[AttemptRecord]:
        """Retrieve all attempts submitted for a specific question."""
        self._ensure_loaded()
        target = question_id.strip()
        return [att for att in self._in_memory_records if att.question_id == target]

    def count(self) -> int:
        """Return the total number of attempts recorded."""
        self._ensure_loaded()
        return len(self._in_memory_records)

    def clear(self) -> None:
        """Clear attempts (primarily used for test cleanup)."""
        self._in_memory_records.clear()
        self._loaded = True
        if self.data_path.is_file():
            try:
                self.data_path.unlink()
            except Exception as exc:
                logger.warning("Could not delete attempt file %s: %s", self.data_path, exc)

