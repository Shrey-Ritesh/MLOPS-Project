"""Phase 3 Day 2: prediction logger, label joiner, JSON event logs, and backups."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import app_logging, schema
from src.monitoring.backup import BackupError, backup, restore
from src.monitoring.label_joiner import join_labels
from src.monitoring.prediction_log import (
    PredictionLogConfigError,
    PredictionLogger,
    PredictionRecord,
    feature_hash,
    pseudonymize,
)

KEY = "test-pseudonym-key-0123456789"
T0 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _features(difficulty: str = "Medium", **overrides) -> dict:
    base = {
        "topic": "Chemical Kinetics",
        "selected_difficulty": difficulty,
        "recent_accuracy": 0.6,
        "mistake_count": 1,
        "hint_count": 0,
        "response_time_seconds": 42.0,
        "prior_interaction_count": 3,
        "previous_difficulty": "Medium",
    }
    base.update(overrides)
    return base


def _record(seq: int, difficulty: str = "Medium", *, learner: str = "lrn-alice-1a2b3c4d",
            source: str = "live", at: datetime = T0, **overrides) -> PredictionRecord:
    fields = dict(
        request_id=f"req-{seq}", source=source, learner_id=learner, learner_seq=seq,
        question_id=f"q-{seq}", features=_features(difficulty, prior_interaction_count=seq),
        model_name="learner-difficulty", model_version="2", model_id="candidate_hgb_shifted_v2",
        registry_alias="production", latency_ms=8.5, status="ok", score=0.8, label=1,
        threshold=0.7, occurred_at=_iso(at),
    )
    fields.update(overrides)
    return PredictionRecord(**fields)


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "state" / "monitoring.sqlite3"


@pytest.fixture()
def logger(db_path):
    return PredictionLogger(db_path, KEY)


def _rows(db_path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM predictions ORDER BY learner_seq, source")]
    finally:
        conn.close()


# ---- contract ------------------------------------------------------------------------------

def test_feature_contract_matches_track_b():
    from src.track_b import schema as track_b

    assert list(schema.FEATURE_COLUMNS) == track_b.FEATURE_COLUMNS
    assert list(schema.DIFFICULTY_LEVELS) == track_b.DIFFICULTY_LEVELS


def test_ddl_matches_documented_schema():
    doc = open("docs/monitoring/prediction_log_schema.md", encoding="utf-8").read()
    for column in ("prediction_id", "learner_key", "learner_seq", "feature_hash", "label_status"):
        assert f"`{column}`" in doc


# ---- logger --------------------------------------------------------------------------------

def test_exactly_one_row_per_prediction(logger, db_path):
    pid = logger.log(_record(0))
    rows = _rows(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["prediction_id"] == pid
    assert row["schema_version"] == schema.SCHEMA_VERSION
    assert row["model_version"] == "2" and row["registry_alias"] == "production"
    assert row["label_status"] == "pending" and row["label_observed"] is None
    assert list(json.loads(row["features"])) == list(schema.FEATURE_COLUMNS)
    assert row["feature_hash"] == feature_hash(json.loads(row["features"]))


def test_no_raw_learner_id_or_username_is_stored(logger, db_path):
    logger.log(_record(0, learner="lrn-alice-1a2b3c4d"))
    raw = db_path.read_bytes()
    assert b"alice" not in raw and b"lrn-" not in raw
    assert _rows(db_path)[0]["learner_key"] == pseudonymize("lrn-alice-1a2b3c4d", KEY)


def test_pseudonym_is_stable_and_key_dependent():
    assert pseudonymize("lrn-a", KEY) == pseudonymize("lrn-a", KEY)
    assert pseudonymize("lrn-a", KEY) != pseudonymize("lrn-b", KEY)
    assert pseudonymize("lrn-a", KEY) != pseudonymize("lrn-a", KEY + "x")


@pytest.mark.parametrize("key", ["", "changeme", "short"])
def test_logger_refuses_missing_or_weak_key(db_path, key):
    with pytest.raises(PredictionLogConfigError):
        PredictionLogger(db_path, key)


def test_feature_order_is_normalised(logger, db_path):
    shuffled = dict(reversed(list(_features().items())))
    logger.log(_record(0, features=shuffled))
    assert list(json.loads(_rows(db_path)[0]["features"])) == list(schema.FEATURE_COLUMNS)


@pytest.mark.parametrize("bad", [
    {"features": {k: v for k, v in _features().items() if k != "hint_count"}},  # missing feature
    {"features": {**_features(), "username": "alice"}},                         # extra field
    {"status": "ok", "score": None},                                             # ok without score
    {"source": "prod"},
    {"score": 1.5},
    {"latency_ms": float("nan")},
])
def test_contract_violations_are_rejected_not_written(logger, db_path, bad, capsys):
    assert logger.log(_record(0, **bad)) is None
    assert _rows(db_path) == []
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(e["event"] == "prediction_log_write_failed" for e in lines)


def test_fallback_and_error_predictions_are_logged(logger, db_path):
    logger.log(_record(0, status="fallback", fallback=True, model_name="rule_fallback",
                       model_version="fallback-v1", score=None, label=0))
    logger.log(_record(1, status="error", score=None, label=None, error_type="TimeoutError"))
    rows = _rows(db_path)
    assert [r["status"] for r in rows] == ["fallback", "error"]
    assert rows[0]["fallback"] == 1


def test_duplicate_learner_seq_does_not_raise(logger, db_path):
    assert logger.log(_record(0)) is not None
    assert logger.log(_record(0, request_id="req-dup")) is None  # double submit: rejected, app unaffected
    assert len(_rows(db_path)) == 1


def test_write_failure_never_raises(logger, db_path, capsys):
    db_path.unlink()
    db_path.mkdir()  # the database path is now a directory: every write fails
    assert logger.log(_record(0)) is None
    assert "prediction_log_write_failed" in capsys.readouterr().out


def test_prediction_event_line_has_no_personal_fields(logger, capsys):
    logger.log(_record(0))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    event = next(e for e in events if e["event"] == "prediction")
    assert {"request_id", "prediction_id", "model_version", "latency_ms", "status"} <= set(event)
    assert not {"learner_id", "learner_key", "features", "topic"} & set(event)
    blob = json.dumps(event)
    assert "alice" not in blob and "Chemical Kinetics" not in blob


def test_event_service_name_is_per_call(capsys):
    app_logging.log_event("a", service="app")
    app_logging.log_event("b", service="monitoring")
    services = [json.loads(line)["service"] for line in capsys.readouterr().out.splitlines()]
    assert services == ["app", "monitoring"]


def test_log_event_drops_unknown_fields(capsys):
    logged = app_logging.log_event("x", request_id="r1", learner_id="lrn-alice", answer_text="secret")
    assert logged == {"request_id": "r1"}
    assert "secret" not in capsys.readouterr().out


# ---- label joiner --------------------------------------------------------------------------

def test_labels_join_to_the_right_prediction(logger, db_path):
    logger.log(_record(0, "Hard", at=T0))
    logger.log(_record(1, "Medium", at=T0 + timedelta(minutes=2)))   # Hard -> Medium: switched down
    logger.log(_record(2, "Medium", at=T0 + timedelta(minutes=4)))   # Medium -> Medium: no switch
    logger.log(_record(3, "Hard", at=T0 + timedelta(minutes=6)))     # Medium -> Hard: no switch

    result = join_labels(db_path, now=T0 + timedelta(minutes=10))
    rows = {r["learner_seq"]: r for r in _rows(db_path)}
    assert [rows[i]["label_observed"] for i in range(3)] == [1, 0, 0]
    assert all(rows[i]["label_status"] == "joined" for i in range(3))
    assert rows[3]["label_status"] == "pending"  # no next interaction yet
    assert (result.joined, result.expired, result.still_pending) == (3, 0, 1)


def test_labels_never_cross_learners_or_sources(logger, db_path):
    logger.log(_record(0, "Hard", learner="lrn-a"))
    logger.log(_record(1, "Easy", learner="lrn-b"))
    logger.log(_record(1, "Easy", learner="lrn-a", source="simulated"))
    join_labels(db_path, now=T0 + timedelta(minutes=1))
    assert all(r["label_status"] == "pending" for r in _rows(db_path))


def test_unanswered_predictions_expire(logger, db_path):
    logger.log(_record(0, at=T0))
    result = join_labels(db_path, now=T0 + timedelta(hours=73), expiry_hours=72)
    assert result.expired == 1
    assert _rows(db_path)[0]["label_status"] == "expired"
    assert _rows(db_path)[0]["label_observed"] is None


def test_join_is_idempotent_and_only_touches_label_columns(logger, db_path):
    logger.log(_record(0, "Hard"))
    logger.log(_record(1, "Easy"))
    before = _rows(db_path)
    join_labels(db_path, now=T0 + timedelta(minutes=5))
    first = _rows(db_path)
    again = join_labels(db_path, now=T0 + timedelta(minutes=50))
    assert again.joined == 0
    assert _rows(db_path) == first
    label_cols = {"label_observed", "label_status", "label_joined_at"}
    for b, a in zip(before, first):
        assert {k: v for k, v in b.items() if k not in label_cols} == {k: v for k, v in a.items() if k not in label_cols}


# ---- backup --------------------------------------------------------------------------------

def test_backup_restore_and_rotation(logger, db_path, tmp_path):
    logger.log(_record(0))
    dest = tmp_path / "backups"
    made = [backup(db_path, dest, keep=2, now=T0 + timedelta(days=d)) for d in range(3)]
    kept = sorted(p.name for p in dest.glob("monitoring-*.sqlite3"))
    assert kept == sorted(p.name for p in made[1:])

    restored = restore(made[-1], tmp_path / "restored.sqlite3")
    assert len(_rows(restored)) == 1
    with pytest.raises(BackupError):
        restore(made[-1], restored)  # never overwrites an existing database


def test_backup_of_missing_database_fails_loudly(tmp_path):
    with pytest.raises(BackupError):
        backup(tmp_path / "nope.sqlite3", tmp_path / "b")
