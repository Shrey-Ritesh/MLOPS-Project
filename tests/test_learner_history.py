"""Unit tests for Person 4 learner history and practice recommendation (src/learner/history.py).

Verifies:
- AttemptRecord validation (learner_id, question_id, concept_ids, outcomes, timezone awareness)
- SQLiteAttemptStore persistence and scoped retrieval
- recommend_practice algorithm (under-sampled concepts, weak-area threshold, question selection)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.learner.history import (
    AttemptRecord,
    PracticeCandidate,
    Recommendation,
    SQLiteAttemptStore,
    recommend_practice,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --- 1. AttemptRecord Validation Tests ---

def test_attempt_record_valid():
    now = _utc_now()
    record = AttemptRecord(
        learner_id="learner-01",
        question_id="q-101",
        concept_ids=("hydrocarbons", "isomerism"),
        outcome="correct",
        answer_mode="handwritten",
        attempted_at=now,
    )
    assert record.learner_id == "learner-01"
    assert record.outcome == "correct"
    assert record.answer_mode == "handwritten"
    assert record.concept_ids == ("hydrocarbons", "isomerism")


def test_attempt_record_rejects_empty_ids():
    now = _utc_now()
    with pytest.raises(ValueError):
        AttemptRecord(learner_id="  ", question_id="q-1", concept_ids=("chem",), outcome="correct", answer_mode="typed", attempted_at=now)
    with pytest.raises(ValueError):
        AttemptRecord(learner_id="l-1", question_id="   ", concept_ids=("chem",), outcome="correct", answer_mode="typed", attempted_at=now)
    with pytest.raises(ValueError):
        AttemptRecord(learner_id="l-1", question_id="q-1", concept_ids=(), outcome="correct", answer_mode="typed", attempted_at=now)


def test_attempt_record_rejects_naive_datetime():
    naive = datetime.now()  # no tzinfo
    with pytest.raises(ValueError) as exc:
        AttemptRecord(learner_id="l-1", question_id="q-1", concept_ids=("chem",), outcome="correct", answer_mode="typed", attempted_at=naive)
    assert "timezone" in str(exc.value).lower()


# --- 2. SQLiteAttemptStore Tests ---

def test_sqlite_attempt_store_save_and_list():
    store = SQLiteAttemptStore(database_path=":memory:")
    now = _utc_now()

    att1 = AttemptRecord(
        learner_id="learner-A",
        question_id="q-1",
        concept_ids=("thermodynamics",),
        outcome="correct",
        answer_mode="typed",
        attempted_at=now - timedelta(minutes=5),
    )
    att2 = AttemptRecord(
        learner_id="learner-A",
        question_id="q-2",
        concept_ids=("equilibrium",),
        outcome="incorrect",
        answer_mode="handwritten",
        attempted_at=now,
    )
    att3 = AttemptRecord(
        learner_id="learner-B",  # different learner
        question_id="q-3",
        concept_ids=("equilibrium",),
        outcome="correct",
        answer_mode="typed",
        attempted_at=now,
    )

    store.save(att1)
    store.save(att2)
    store.save(att3)

    records_a = store.list_for_learner("learner-A")
    assert len(records_a) == 2
    # Ordered by attempted_at DESC
    assert records_a[0].question_id == "q-2"
    assert records_a[1].question_id == "q-1"

    records_b = store.list_for_learner("learner-B")
    assert len(records_b) == 1
    assert records_b[0].question_id == "q-3"


# --- 3. recommend_practice Algorithm Tests ---

def test_recommend_practice_under_sampled_concept():
    """When a concept has fewer attempts than minimum_evidence, recommend it to build evidence."""
    now = _utc_now()
    candidates = [
        PracticeCandidate(question_id="q-chem-01", concept_ids=("thermodynamics",), title="Entropy Calculation"),
        PracticeCandidate(question_id="q-chem-02", concept_ids=("equilibrium",), title="Le Chatelier's Principle"),
    ]
    # Learner has only 1 attempt on thermodynamics
    attempts = [
        AttemptRecord(
            learner_id="l-1",
            question_id="q-chem-01",
            concept_ids=("thermodynamics",),
            outcome="correct",
            answer_mode="typed",
            attempted_at=now - timedelta(days=1),
        )
    ]

    rec = recommend_practice(attempts, candidates, minimum_evidence=3, now=now)
    assert rec is not None
    assert rec.concept_id in {"equilibrium", "thermodynamics"}
    assert "build evidence" in rec.reason.lower()


def test_recommend_practice_weak_area():
    """When a concept has >= minimum_evidence and accuracy < 70%, prioritize it as weak."""
    now = _utc_now()
    candidates = [
        PracticeCandidate(question_id="q-thermo-01", concept_ids=("thermodynamics",), title="Heat capacity"),
        PracticeCandidate(question_id="q-thermo-02", concept_ids=("thermodynamics",), title="Gibbs Energy"),
        PracticeCandidate(question_id="q-bonding-01", concept_ids=("bonding",), title="Hybridization"),
    ]

    # Learner has 3 attempts on thermodynamics: 1 correct, 2 incorrect (33% accuracy)
    # Learner has 3 attempts on bonding: 3 correct (100% accuracy)
    attempts = [
        AttemptRecord(learner_id="l-1", question_id="q-thermo-01", concept_ids=("thermodynamics",), outcome="incorrect", answer_mode="typed", attempted_at=now - timedelta(days=3)),
        AttemptRecord(learner_id="l-1", question_id="q-thermo-01", concept_ids=("thermodynamics",), outcome="incorrect", answer_mode="typed", attempted_at=now - timedelta(days=2)),
        AttemptRecord(learner_id="l-1", question_id="q-thermo-01", concept_ids=("thermodynamics",), outcome="correct", answer_mode="typed", attempted_at=now - timedelta(days=1)),
        AttemptRecord(learner_id="l-1", question_id="q-bonding-01", concept_ids=("bonding",), outcome="correct", answer_mode="typed", attempted_at=now - timedelta(days=2)),
        AttemptRecord(learner_id="l-1", question_id="q-bonding-01", concept_ids=("bonding",), outcome="correct", answer_mode="typed", attempted_at=now - timedelta(days=1)),
        AttemptRecord(learner_id="l-1", question_id="q-bonding-01", concept_ids=("bonding",), outcome="correct", answer_mode="typed", attempted_at=now - timedelta(hours=5)),
    ]

    rec = recommend_practice(attempts, candidates, minimum_evidence=3, mastery_threshold=0.7, now=now)
    assert rec is not None
    assert rec.concept_id == "thermodynamics"
    assert rec.question_id == "q-thermo-02"  # Least recently attempted question for thermodynamics!
    assert "below the 70% practice threshold" in rec.reason.lower()

