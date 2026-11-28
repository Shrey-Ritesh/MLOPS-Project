"""Model contracts: feature lists and label rules are configuration, not code."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from src.monitoring.contract import ContractError, load_contract
from src.monitoring.label_joiner import join_labels
from src.monitoring.prediction_log import OutcomeRecord, PredictionLogger, PredictionRecord
from src.monitoring.run_window import evaluate_window, load_config

KEY = "test-pseudonym-key-0123456789"
T0 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)
TRACK_B = "configs/monitoring/contracts/track_b_difficulty_v1.yaml"
MASTERY = "configs/monitoring/contracts/chapter_mastery_draft.yaml"


def _iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _mastery_features(topic="Chemical Kinetics", **overrides):
    base = {"topic": topic, "recent_accuracy": 0.6, "mistake_count": 1, "hint_count": 0,
            "response_time_seconds": 40.0, "prior_interaction_count": 5, "topic_attempts": 2,
            "topic_accuracy": 0.5}
    base.update(overrides)
    return base


def _predict(logger, seq, at, topic="Chemical Kinetics", learner="lrn-a", source="live", **feature_overrides):
    return logger.log(PredictionRecord(
        request_id=uuid.uuid4().hex, source=source, learner_id=learner, learner_seq=seq, question_id=f"q{seq}",
        features=_mastery_features(topic, prior_interaction_count=seq, **feature_overrides),
        model_name="learner-chapter-mastery", model_version="1", latency_ms=3.0, status="ok",
        score=0.7, label=1, threshold=0.5, occurred_at=_iso(at)))


def _outcome(logger, at, outcome, topic="Chemical Kinetics", learner="lrn-a", source="live"):
    return logger.log_outcome(OutcomeRecord(source=source, learner_id=learner, topic=topic,
                                            question_id="q", outcome=outcome, occurred_at=_iso(at)))


def _labels(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return [r for r in conn.execute(
            "SELECT learner_seq, topic, label_observed, label_status FROM predictions ORDER BY learner_seq")]
    finally:
        conn.close()


@pytest.fixture()
def mastery():
    return load_contract(MASTERY)


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "monitoring.sqlite3"


# ---- loading and validation ------------------------------------------------------------------

def test_default_contract_is_track_b(monkeypatch):
    monkeypatch.delenv("MONITORING_CONTRACT", raising=False)
    c = load_contract()
    assert c.contract_id == "track_b_difficulty_v1" and c.status == "active"
    assert c.label_rule == "next_interaction_lower_difficulty"


def test_env_selects_the_contract(monkeypatch):
    monkeypatch.setenv("MONITORING_CONTRACT", MASTERY)
    c = load_contract()
    assert c.contract_id == "chapter_mastery_draft" and c.status == "draft"
    assert "selected_difficulty" not in c.feature_names and "topic_accuracy" in c.feature_names


def _write_contract(tmp_path, mutate):
    raw = yaml.safe_load(open(MASTERY))
    mutate(raw)
    path = tmp_path / f"c-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.mark.parametrize("mutate", [
    lambda r: r["label"].update(rule="guess"),                                     # unknown rule
    lambda r: r["label"]["params"].update(excluded=["partial"]),                    # 'uncertain' not mapped
    lambda r: r["features"].pop("topic"),                                           # outcome rule needs topic
    lambda r: r["features"]["hint_count"].update(type="float"),                     # unknown type
    lambda r: r.pop("label"),                                                       # missing section
])
def test_invalid_contracts_are_refused(tmp_path, mutate):
    with pytest.raises(ContractError):
        load_contract(_write_contract(tmp_path, mutate))


def test_feature_checks():
    c = load_contract(TRACK_B)
    assert c.feature("recent_accuracy").check(1.2) == "out_of_range"
    assert c.feature("mistake_count").check(1.5) == "out_of_range"    # integer feature
    assert c.feature("mistake_count").check(True) == "out_of_range"   # bool is not a count
    assert c.feature("selected_difficulty").check("Expert") == "out_of_range"
    assert c.feature("topic").check("Organic Name Reactions") == "unseen"
    assert c.feature("topic").check("Chemical Kinetics") is None


# ---- logging under the mastery contract -------------------------------------------------------

def test_logger_uses_the_active_contract(db_path, mastery):
    logger = PredictionLogger(db_path, KEY, contract=mastery)
    assert _predict(logger, 0, T0) is not None
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT contract_id FROM predictions").fetchone() == ("chapter_mastery_draft",)
    conn.close()
    # a Track B-shaped record is rejected under the mastery contract
    bad = PredictionRecord(request_id=uuid.uuid4().hex, source="live", learner_id="lrn-a", learner_seq=1,
                           question_id="q", features={**_mastery_features(), "selected_difficulty": "Easy"},
                           model_name="m", model_version="1", latency_ms=1.0, status="ok", score=0.5, label=1)
    assert logger.log(bad) is None


def test_outcome_logging_is_pseudonymous_and_validated(db_path, mastery):
    logger = PredictionLogger(db_path, KEY, contract=mastery)
    assert _outcome(logger, T0, "correct", learner="lrn-alice-1a2b3c4d") is not None
    assert b"alice" not in db_path.read_bytes()
    assert _outcome(logger, T0, "maybe") is None                       # not a known outcome
    assert logger.log_outcome(OutcomeRecord(source="live", learner_id="lrn-a", topic="Chemical Kinetics",
                                            question_id="q-lrn-a", outcome="correct")) is None  # id leak


# ---- next_outcome_same_topic label rule -------------------------------------------------------

def test_outcome_rule_joins_the_next_answer_in_the_same_chapter(db_path, mastery):
    logger = PredictionLogger(db_path, KEY, contract=mastery)
    _predict(logger, 0, T0, topic="Chemical Kinetics")
    _predict(logger, 1, T0 + timedelta(minutes=1), topic="Electrochemistry")
    _predict(logger, 2, T0 + timedelta(minutes=2), topic="Biomolecules")
    _predict(logger, 3, T0 + timedelta(minutes=3), topic="Atomic Structure")
    _outcome(logger, T0 - timedelta(minutes=5), "correct", topic="Chemical Kinetics")   # before: ignored
    _outcome(logger, T0 + timedelta(seconds=30), "incorrect", topic="Electrochemistry")  # other chapter first
    _outcome(logger, T0 + timedelta(minutes=4), "correct", topic="Chemical Kinetics")    # -> seq 0 = 1
    _outcome(logger, T0 + timedelta(minutes=5), "incorrect", topic="Electrochemistry")   # -> seq 1 = 0
    _outcome(logger, T0 + timedelta(minutes=6), "partial", topic="Biomolecules")         # -> seq 2 excluded
    result = join_labels(db_path, now=T0 + timedelta(minutes=10), contract=mastery)
    assert _labels(db_path) == [
        (0, "Chemical Kinetics", 1, "joined"),
        (1, "Electrochemistry", 0, "joined"),
        (2, "Biomolecules", None, "excluded"),
        (3, "Atomic Structure", None, "pending"),
    ]
    assert (result.joined, result.excluded, result.still_pending) == (2, 1, 1)


def test_outcome_rule_never_crosses_learners_or_sources(db_path, mastery):
    logger = PredictionLogger(db_path, KEY, contract=mastery)
    _predict(logger, 0, T0)
    _outcome(logger, T0 + timedelta(minutes=1), "correct", learner="lrn-b")
    _outcome(logger, T0 + timedelta(minutes=1), "correct", source="simulated")
    join_labels(db_path, now=T0 + timedelta(minutes=5), contract=mastery)
    assert _labels(db_path)[0][3] == "pending"


def test_outcome_rule_expires_unanswered_predictions(db_path, mastery):
    logger = PredictionLogger(db_path, KEY, contract=mastery)
    _predict(logger, 0, T0)
    assert join_labels(db_path, now=T0 + timedelta(hours=73), contract=mastery).expired == 1


# ---- model switch -----------------------------------------------------------------------------

def test_rows_from_another_contract_are_flagged_not_mixed(db_path, mastery):
    old = PredictionLogger(db_path, KEY, contract=load_contract(TRACK_B))
    for i in range(5):
        old.log(PredictionRecord(
            request_id=uuid.uuid4().hex, source="live", learner_id="lrn-old", learner_seq=i, question_id="q",
            features={"topic": "Chemical Kinetics", "selected_difficulty": "Medium", "recent_accuracy": 0.5,
                      "mistake_count": 1, "hint_count": 0, "response_time_seconds": 40.0,
                      "prior_interaction_count": i, "previous_difficulty": "Medium"},
            model_name="learner-difficulty", model_version="2", latency_ms=3.0, status="ok", score=0.4, label=0,
            occurred_at=_iso(T0 + timedelta(seconds=i))))
    new = PredictionLogger(db_path, KEY, contract=mastery)
    for i in range(10):
        _predict(new, i, T0 + timedelta(minutes=1, seconds=i), learner="lrn-new")
    report = evaluate_window(db_path, T0, T0 + timedelta(minutes=15), config=load_config(), evaluator=None,
                             contract=mastery)
    assert report["contract_id"] == "chapter_mastery_draft"
    assert report["validity"]["contract_mismatch"] == 5 and report["validity"]["schema_failures"] == 5
    assert report["validity"]["out_of_range"] == 0
    # labels for the old contract's rows are not produced by the new rule
    join_labels(db_path, now=T0 + timedelta(minutes=20), contract=mastery)
    conn = sqlite3.connect(db_path)
    statuses = {s for (s,) in conn.execute(
        "SELECT label_status FROM predictions WHERE contract_id = 'track_b_difficulty_v1'")}
    conn.close()
    assert statuses == {"pending"}
