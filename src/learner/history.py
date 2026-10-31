"""Minimal learner attempt persistence and deterministic practice selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal, Sequence
from uuid import uuid4


AnswerMode = Literal["typed", "handwritten"]
Outcome = Literal["correct", "incorrect", "partial", "uncertain"]
_SCORES: dict[str, float] = {"correct": 1.0, "incorrect": 0.0, "partial": 0.5}


@dataclass(frozen=True)
class AttemptRecord:
    """Small persisted attempt record; answer text and personal details are omitted."""

    learner_id: str
    question_id: str
    concept_ids: tuple[str, ...]
    outcome: Outcome
    answer_mode: AnswerMode
    attempted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.learner_id.strip():
            raise ValueError("learner_id must be a non-empty pseudonymous identifier")
        if not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.concept_ids or any(not concept.strip() for concept in self.concept_ids):
            raise ValueError("at least one non-empty concept_id is required")
        if self.outcome not in {"correct", "incorrect", "partial", "uncertain"}:
            raise ValueError("outcome must be correct, incorrect, partial, or uncertain")
        if self.answer_mode not in {"typed", "handwritten"}:
            raise ValueError("answer_mode must be typed or handwritten")
        if self.attempted_at.tzinfo is None:
            raise ValueError("attempted_at must include a timezone")


@dataclass(frozen=True)
class PracticeCandidate:
    question_id: str
    concept_ids: tuple[str, ...]
    title: str = "Practice question"


@dataclass(frozen=True)
class Recommendation:
    question_id: str
    title: str
    concept_id: str
    reason: str
    evidence_attempts: int
    accuracy: float | None


class SQLiteAttemptStore:
    """SQLite history store with learner-scoped reads."""

    def __init__(self, database_path: str | Path = "data/learner_history.sqlite3") -> None:
        self.database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(":memory:")
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    concept_ids TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('correct', 'incorrect', 'partial', 'uncertain')),
                    answer_mode TEXT NOT NULL CHECK (answer_mode IN ('typed', 'handwritten')),
                    attempted_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_learner_time "
                "ON attempts (learner_id, attempted_at DESC)"
            )

    def save(self, attempt: AttemptRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id, learner_id, question_id, concept_ids,
                    outcome, answer_mode, attempted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    attempt.learner_id,
                    attempt.question_id,
                    json.dumps(attempt.concept_ids),
                    attempt.outcome,
                    attempt.answer_mode,
                    attempt.attempted_at.astimezone(timezone.utc).isoformat(),
                ),
            )

    def list_for_learner(self, learner_id: str, limit: int = 500) -> list[AttemptRecord]:
        if not learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempt_id, learner_id, question_id, concept_ids,
                       outcome, answer_mode, attempted_at
                FROM attempts
                WHERE learner_id = ?
                ORDER BY attempted_at DESC, attempt_id DESC
                LIMIT ?
                """,
                (learner_id, limit),
            ).fetchall()
        return [
            AttemptRecord(
                attempt_id=row[0],
                learner_id=row[1],
                question_id=row[2],
                concept_ids=tuple(json.loads(row[3])),
                outcome=row[4],
                answer_mode=row[5],
                attempted_at=datetime.fromisoformat(row[6]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        return sqlite3.connect(self.database_path)


def recommend_practice(
    attempts: Sequence[AttemptRecord],
    candidates: Sequence[PracticeCandidate],
    *,
    minimum_evidence: int = 3,
    mastery_threshold: float = 0.7,
    window_days: int = 30,
    now: datetime | None = None,
) -> Recommendation | None:
    """Choose a practice question from recent concept evidence with an explanation."""
    if minimum_evidence < 1:
        raise ValueError("minimum_evidence must be at least 1")
    if not 0.0 <= mastery_threshold <= 1.0:
        raise ValueError("mastery_threshold must be between 0 and 1")
    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    if not candidates:
        return None

    now = now or datetime.now(timezone.utc)
    cutoff = now.astimezone(timezone.utc) - timedelta(days=window_days)
    recent = [
        attempt
        for attempt in attempts
        if attempt.attempted_at.astimezone(timezone.utc) >= cutoff
    ]
    concept_scores: dict[str, list[float]] = {}
    question_counts: dict[str, int] = {}
    for attempt in recent:
        question_counts[attempt.question_id] = question_counts.get(attempt.question_id, 0) + 1
        score = _SCORES.get(attempt.outcome)
        if score is None:
            continue
        for concept_id in set(attempt.concept_ids):
            concept_scores.setdefault(concept_id, []).append(score)

    candidate_concepts = sorted({concept for item in candidates for concept in item.concept_ids})
    if not candidate_concepts:
        return None

    stats = {
        concept: (len(concept_scores.get(concept, [])), _accuracy(concept_scores.get(concept, [])))
        for concept in candidate_concepts
    }
    under_threshold = [
        concept
        for concept, (count, accuracy) in stats.items()
        if count >= minimum_evidence and accuracy is not None and accuracy < mastery_threshold
    ]
    if under_threshold:
        concept = min(under_threshold, key=lambda item: (stats[item][1], item))
        count, accuracy = stats[concept]
        reason = (
            f"Recent accuracy is {accuracy:.0%} across {count} evaluated attempts, "
            f"below the {mastery_threshold:.0%} practice threshold."
        )
    else:
        under_sampled = [
            concept for concept in candidate_concepts if stats[concept][0] < minimum_evidence
        ]
        if under_sampled:
            concept = min(under_sampled, key=lambda item: (stats[item][0], item))
            count, _ = stats[concept]
            reason = (
                f"Build evidence for this concept: {count} evaluated attempt(s) "
                f"so far; at least {minimum_evidence} are needed before weak-area recommendations."
            )
        else:
            concept = min(candidate_concepts, key=lambda item: (stats[item][1], item))
            count, accuracy = stats[concept]
            reason = (
                f"Reinforce this concept: recent accuracy is {accuracy:.0%} "
                f"across {count} evaluated attempts."
            )

    matching = [item for item in candidates if concept in item.concept_ids]
    selected = min(
        matching,
        key=lambda item: (question_counts.get(item.question_id, 0), item.question_id),
    )
    count, accuracy = stats[concept]
    return Recommendation(
        question_id=selected.question_id,
        title=selected.title,
        concept_id=concept,
        reason=reason,
        evidence_attempts=count,
        accuracy=accuracy,
    )


def _accuracy(scores: Sequence[float]) -> float | None:
    return sum(scores) / len(scores) if scores else None
