"""Tests for Person 3 live inference: topic tagging, ProgressModel wrapper, and prediction logging hook."""
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.learner.history import AttemptRecord
from src.learner.inference_hook import InferenceService, build_learner_features
from src.learner.progress_model import (
    FeatureContractMismatchError,
    ProgressModel,
    SYNTHETIC_NOTICE,
)
from src.learner.topic_tagger import (
    QuestionTopicTagger,
    build_coverage_report,
    load_chemistry_topics,
    tag_text,
)
from src.monitoring.label_joiner import join_labels
from src.monitoring.prediction_log import PredictionLogger

TEST_KEY = "test-pseudonym-key-0123456789"


def _sample_features(**overrides):
    base = {
        "topic": "Chemical Kinetics",
        "selected_difficulty": "Hard",
        "recent_accuracy": 0.35,
        "mistake_count": 3,
        "hint_count": 2,
        "response_time_seconds": 95.0,
        "prior_interaction_count": 4,
        "previous_difficulty": "Hard",
    }
    base.update(overrides)
    return base


# ---- Topic tagging ---------------------------------------------------------------------------

def test_chemistry_topic_vocabulary_matches_track_b_priors():
    topics = load_chemistry_topics()
    assert len(topics) == 22
    assert "Chemical Kinetics" in topics
    assert "Coordination Compounds" in topics
    assert "Biomolecules" in topics


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Calculate the rate constant and half-life for a first-order reaction using Arrhenius equation.", "Chemical Kinetics"),
        ("Which crystal field splitting energy CFSE is highest for an octahedral complex ligand?", "Coordination Compounds"),
        ("Which of the following acids is a vitamin? Ascorbic acid is a biomolecule.", "Biomolecules"),
        ("A solution of aluminium chloride is electrolysed using Nernst equation and Faraday's law.", "Electrochemistry"),
        ("Lanthanoid ions with 4f7 configuration such as Eu2+ and Gd3+.", "d- and f-Block Elements"),
    ],
)
def test_tag_text_identifies_expected_chemistry_chapters(text, expected):
    res = tag_text(text)
    assert res.topic == expected
    assert not res.is_unknown
    assert res.score > 0


def test_coverage_report_on_fixture_questions():
    tag_map, report = build_coverage_report("tests/fixtures/jee_questions_sample.jsonl")
    assert report["total_chemistry_questions"] > 0
    assert report["tagged_count"] <= report["total_chemistry_questions"]
    assert set(tag_map.values()) <= (set(load_chemistry_topics()) | {"unknown"})


def test_topic_tagger_falls_back_cleanly_for_empty_question():
    tagger = QuestionTopicTagger()
    topic = tagger.get_topic({"question_id": "q-empty", "question_text": ""})
    assert topic in set(load_chemistry_topics())


# ---- ProgressModel wrapper -------------------------------------------------------------------

def test_progress_model_loads_local_champion_and_enforces_boundary_guard():
    pm = ProgressModel(tracking_uri="", local_model_dir="models/track_b/candidate")
    meta = pm.current_metadata
    assert meta["source_backend"] == "local_joblib"
    assert meta["model_id"] == "candidate_hgb_shifted_v2"
    assert meta["threshold"] == pytest.approx(0.7)

    res = pm.predict(_sample_features(selected_difficulty="Hard"))
    assert res.status == "ok"
    assert not res.fallback
    assert 0.0 <= res.score <= 1.0
    assert res.label in (0, 1)
    assert res.latency_ms >= 0.0
    assert len(res.feature_hash) == 64
    assert "SYNTHETIC" in res.synthetic_notice.upper() or res.synthetic_notice == SYNTHETIC_NOTICE

    # Boundary guard: Easy difficulty cannot switch to a lower difficulty.
    easy_res = pm.predict(_sample_features(selected_difficulty="Easy"))
    assert easy_res.status == "ok"
    assert easy_res.score == 0.0
    assert easy_res.label == 0


def test_progress_model_works_with_model_missing():
    pm = ProgressModel(tracking_uri="", local_model_dir="models/does_not_exist")
    res = pm.predict(_sample_features(selected_difficulty="Hard", recent_accuracy=0.2, mistake_count=4, hint_count=2))
    assert res.status == "fallback"
    assert res.fallback is True
    assert res.model_name == "rule_fallback"
    assert res.model_version == "fallback-v1"
    assert 0.0 <= res.score <= 1.0
    assert res.label == 1


def test_progress_model_fails_loudly_on_feature_order_mismatch(tmp_path):
    # Train a dummy model with reversed column order and save it to tmp_path/model.joblib.
    wrong_cols = [
        "previous_difficulty",
        "prior_interaction_count",
        "response_time_seconds",
        "hint_count",
        "mistake_count",
        "recent_accuracy",
        "selected_difficulty",
        "topic",
    ]
    X = pd.DataFrame([["Easy", 0, 10.0, 0, 0, 0.5, "Easy", "Solutions"]], columns=wrong_cols)
    y = np.array([0])
    dummy = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    joblib.dump(dummy, tmp_path / "model.joblib")

    with pytest.raises(FeatureContractMismatchError, match="feature order mismatch"):
        ProgressModel(tracking_uri="", local_model_dir=tmp_path)


def test_progress_model_timeout_falls_back_safely():
    pm = ProgressModel(tracking_uri="", local_model_dir="models/track_b/candidate", timeout_ms=30.0)

    class _SlowModel:
        feature_names_in_ = np.array(pm.expected_features)

        def predict_proba(self, X):
            time.sleep(0.2)
            return np.array([[0.4, 0.6]])

    pm._model = _SlowModel()
    res = pm.predict(_sample_features())
    assert res.status == "timeout"
    assert res.fallback is True
    assert res.error_type == "TimeoutError"
    assert res.model_name == "rule_fallback"


def test_progress_model_loads_from_mlflow_alias_and_reloads_on_rollback(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from src.track_b.register_mlflow import TRUSTED_TYPES

    db_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment("test_live_reload")
    client = MlflowClient(tracking_uri=db_uri)

    base_model = joblib.load("models/track_b/baseline/model.joblib")
    cand_model = joblib.load("models/track_b/candidate/model.joblib")

    with mlflow.start_run(run_name="baseline_v1"):
        mlflow.log_params({"model_id": "baseline_logreg_v1", "decision_threshold": 0.5, "apply_boundary_guard": False})
        v1 = mlflow.sklearn.log_model(
            base_model, name="model", registered_model_name="learner-difficulty", skops_trusted_types=TRUSTED_TYPES
        ).registered_model_version

    with mlflow.start_run(run_name="candidate_v2"):
        mlflow.log_params({"model_id": "candidate_hgb_shifted_v2", "decision_threshold": 0.7, "apply_boundary_guard": True})
        v2 = mlflow.sklearn.log_model(
            cand_model, name="model", registered_model_name="learner-difficulty", skops_trusted_types=TRUSTED_TYPES
        ).registered_model_version

    client.set_registered_model_alias("learner-difficulty", "production", str(v2))

    pm = ProgressModel(
        tracking_uri=db_uri,
        model_name="learner-difficulty",
        registry_alias="production",
        allow_local_fallback=False,
        refresh_seconds=0.0,
    )
    r1 = pm.predict(_sample_features())
    assert r1.status == "ok"
    assert r1.model_version == str(v2)
    assert r1.model_id == "candidate_hgb_shifted_v2"
    assert r1.source_backend == "mlflow"

    # Rollback alias from v2 -> v1; ProgressModel picks it up without rebuilding or recreating.
    client.set_registered_model_alias("learner-difficulty", "production", str(v1))
    r2 = pm.predict(_sample_features())
    assert r2.status == "ok"
    assert r2.model_version == str(v1)
    assert r2.model_id == "baseline_logreg_v1"


# ---- InferenceService + PredictionLogger integration -----------------------------------------

def test_inference_service_logs_versioned_predictions_and_joins_labels(tmp_path):
    db_path = tmp_path / "monitoring.sqlite3"
    pm = ProgressModel(tracking_uri="", local_model_dir="models/track_b/candidate")
    pred_logger = PredictionLogger(db_path, TEST_KEY)
    svc = InferenceService(model=pm, prediction_logger=pred_logger)

    f0 = build_learner_features(
        topic="Chemical Kinetics",
        selected_difficulty="Hard",
        prior_attempts=[],
        hint_count=2,
        response_time_seconds=80.0,
        previous_difficulty="Hard",
    )
    e0 = svc.predict_and_log(learner_id="lrn-alice-1234", learner_seq=0, question_id="q-001", features=f0)
    assert e0.prediction_id is not None

    prior = [
        AttemptRecord(
            learner_id="lrn-alice-1234",
            question_id="q-001",
            concept_ids=("Chemical Kinetics",),
            outcome="incorrect",
            answer_mode="typed",
            attempted_at=datetime.now(timezone.utc),
        )
    ]
    f1 = build_learner_features(
        topic="Chemical Kinetics",
        selected_difficulty="Medium",
        prior_attempts=prior,
        hint_count=0,
        response_time_seconds=45.0,
        previous_difficulty="Hard",
    )
    assert f1["prior_interaction_count"] == 1
    assert f1["mistake_count"] == 1
    assert f1["recent_accuracy"] == 0.0

    e1 = svc.predict_and_log(learner_id="lrn-alice-1234", learner_seq=1, question_id="q-002", features=f1)
    assert e1.prediction_id is not None
    assert svc.log_outcome(learner_id="lrn-alice-1234", topic="Chemical Kinetics", question_id="q-002", outcome="correct")

    # Privacy check: raw learner username must never appear in the DB bytes.
    assert b"alice" not in db_path.read_bytes()

    # Label joiner should join seq 0 -> label_observed = 1 (Hard -> Medium).
    joined = join_labels(db_path)
    assert joined.joined == 1
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT model_version, label_observed, label_status FROM predictions WHERE learner_seq = 0"
    ).fetchone()
    conn.close()
    assert row == ("local-joblib", 1, "joined")
