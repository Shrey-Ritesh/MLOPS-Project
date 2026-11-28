"""End to end: the app's inference path -> prediction log -> label join -> window report.

This drives Person 3's ``InferenceService`` exactly the way ``app.py`` does on a practice
submit (features from the attempts saved so far, predict and log, save the attempt, log the
outcome), then runs Person 1's label joiner and window evaluation on the result.

It skips until the live inference code (PR #26) is on the branch, then runs in CI.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

inference_hook = pytest.importorskip("src.learner.inference_hook")
progress_model = pytest.importorskip("src.learner.progress_model")

from src.learner.history import AttemptRecord, SQLiteAttemptStore  # noqa: E402
from src.monitoring.contract import load_contract  # noqa: E402
from src.monitoring.label_joiner import join_labels  # noqa: E402
from src.monitoring.prediction_log import PredictionLogger  # noqa: E402
from src.monitoring.run_window import evaluate_window, load_config  # noqa: E402

KEY = "test-pseudonym-key-0123456789"
LEARNER = "lrn-priya-7f3a9c21"
TRACK_B = "configs/monitoring/contracts/track_b_difficulty_v1.yaml"


@pytest.fixture()
def contract():
    return load_contract(TRACK_B)


@pytest.fixture()
def model_dir(tmp_path, contract):
    """A tiny real sklearn model saved the way the committed candidate is (joblib + card)."""
    joblib = pytest.importorskip("joblib")
    pd = pytest.importorskip("pandas")
    from sklearn.dummy import DummyClassifier

    names = list(contract.feature_names)
    rows = [{"topic": "Chemical Kinetics", "selected_difficulty": d, "recent_accuracy": 0.5,
             "mistake_count": 1, "hint_count": 0, "response_time_seconds": 30.0,
             "prior_interaction_count": i, "previous_difficulty": d}
            for i, d in enumerate(["Easy", "Medium", "Hard", "Medium", "Hard"])]
    model = DummyClassifier(strategy="prior").fit(pd.DataFrame(rows, columns=names), [1, 0, 1, 1, 0])
    out = tmp_path / "model"
    out.mkdir()
    joblib.dump(model, out / "model.joblib")
    (out / "model_card.json").write_text(json.dumps(
        {"model_id": "it_dummy", "model_version": "7", "decision_threshold": 0.5, "apply_boundary_guard": False}))
    return out


def _service(db_path, contract, model_dir):
    model = progress_model.ProgressModel(tracking_uri="", local_model_dir=model_dir, contract=contract,
                                         refresh_seconds=3600)
    return inference_hook.InferenceService(model=model,
                                           prediction_logger=PredictionLogger(db_path, KEY, contract=contract))


def _submit(service, store, difficulty, outcome, previous, topic="Chemical Kinetics"):
    """What app.py's practice tab does on 'Submit', in the same order."""
    prior = store.list_for_learner(LEARNER)
    features = inference_hook.build_learner_features(
        topic=topic, selected_difficulty=difficulty, prior_attempts=prior, hint_count=1,
        response_time_seconds=42.0, previous_difficulty=previous)
    event = service.predict_and_log(learner_id=LEARNER, learner_seq=len(prior),
                                    question_id=f"q-{len(prior)}", features=features, source="live")
    store.save(AttemptRecord(learner_id=LEARNER, question_id=f"q-{len(prior)}", concept_ids=(topic,),
                             outcome=outcome, answer_mode="typed"))
    service.log_outcome(learner_id=LEARNER, topic=topic, question_id=f"q-{len(prior)}", outcome=outcome,
                        source="live", request_id=event.request_id)
    return event


def _rows(db_path, sql):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _window(db_path, contract):
    now = datetime.now(timezone.utc)
    return evaluate_window(db_path, now - timedelta(hours=1), now + timedelta(minutes=1), config=load_config(),
                           evaluator=None, contract=contract)


def test_practice_session_reaches_the_window_report(tmp_path, contract, model_dir, capsys):
    db_path = tmp_path / "monitoring.sqlite3"
    store = SQLiteAttemptStore(tmp_path / "learner.sqlite3")
    service = _service(db_path, contract, model_dir)

    plan = [("Hard", "incorrect"), ("Medium", "correct"), ("Medium", "partial"), ("Easy", "correct")]
    previous = None
    events = []
    for difficulty, outcome in plan:
        events.append(_submit(service, store, difficulty, outcome, previous))
        previous = difficulty

    # every submit was scored by the real model and logged once
    assert all(e.prediction_id for e in events)
    assert all(e.result.status == "ok" and not e.result.fallback for e in events)
    assert _rows(db_path, "SELECT learner_seq, model_name, model_version, contract_id, fallback FROM predictions "
                          "ORDER BY learner_seq") == [
        (i, "learner-difficulty", "7", "track_b_difficulty_v1", 0) for i in range(4)]
    assert _rows(db_path, "SELECT COUNT(*) FROM attempt_outcomes") == [(4,)]
    # features come only from attempts saved before the submit
    assert [json.loads(f)["prior_interaction_count"] for (f,) in
            _rows(db_path, "SELECT features FROM predictions ORDER BY learner_seq")] == [0, 1, 2, 3]

    # Track B label: did the learner step down to an easier level on the next attempt?
    join_labels(db_path, contract=contract)
    assert _rows(db_path, "SELECT learner_seq, label_observed, label_status FROM predictions ORDER BY learner_seq") == [
        (0, 1, "joined"),     # Hard -> Medium
        (1, 0, "joined"),     # Medium -> Medium
        (2, 1, "joined"),     # Medium -> Easy
        (3, None, "pending"),  # no next attempt yet
    ]

    report = _window(db_path, contract)
    assert (report["n_predictions"], report["n_labelled"], report["n_label_pending"]) == (4, 3, 1)
    assert report["health"]["by_source"] == {"live": 4}
    assert report["health"]["by_model_version"] == {"7": 4}
    assert report["validity"]["schema_failures"] == 0 and report["validity"]["contract_mismatch"] == 0
    assert report["status"] == "insufficient_data"  # 4 < 10: never reported as healthy

    # the learner id is in neither the database nor the logs
    for path in tmp_path.glob("monitoring.sqlite3*"):
        assert b"priya" not in path.read_bytes()
    assert "priya" not in capsys.readouterr().out


def test_missing_model_is_logged_as_fallback_and_alerts(tmp_path, contract):
    db_path = tmp_path / "monitoring.sqlite3"
    store = SQLiteAttemptStore(tmp_path / "learner.sqlite3")
    model = progress_model.ProgressModel(tracking_uri="", local_model_dir=tmp_path / "nothing-here",
                                         contract=contract, refresh_seconds=3600)
    service = inference_hook.InferenceService(model=model,
                                              prediction_logger=PredictionLogger(db_path, KEY, contract=contract))
    previous = None
    for i in range(12):
        difficulty = ("Hard", "Medium", "Easy")[i % 3]
        event = _submit(service, store, difficulty, "correct", previous)
        previous = difficulty
        assert event.prediction_id and event.result.fallback

    assert _rows(db_path, "SELECT DISTINCT model_name, status, fallback FROM predictions") == [
        ("rule_fallback", "fallback", 1)]
    report = _window(db_path, contract)
    assert report["health"]["fallback_rate"] == 1.0
    assert "fallback_rate" in {a["metric"] for a in report["alerts"]}


def test_app_still_predicts_without_the_pseudonym_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LOG_PSEUDONYM_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("PROGRESS_MODEL_LOCAL_DIR", "")
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "monitoring.sqlite3"))
    service = inference_hook.InferenceService.from_env()
    features = inference_hook.build_learner_features(
        topic="Chemical Kinetics", selected_difficulty="Hard", prior_attempts=[], hint_count=0,
        response_time_seconds=30.0)
    event = service.predict_and_log(learner_id=LEARNER, learner_seq=0, question_id="q-0", features=features)
    assert event.prediction_id is None and event.result.score is not None
    assert not (tmp_path / "monitoring.sqlite3").exists()
