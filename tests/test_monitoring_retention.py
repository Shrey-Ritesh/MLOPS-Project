"""Data retention: raw rows and snapshots age out; aggregates, pending labels and backups-first are respected."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import run_window, scheduler
from src.monitoring.prediction_log import OutcomeRecord, PredictionLogger, PredictionRecord
from src.monitoring.retention import apply_retention, load_retention

KEY = "test-pseudonym-key-0123456789"
NOW = datetime(2026, 12, 30, 3, 0, tzinfo=timezone.utc)
CFG = {"raw_predictions_days": 30, "attempt_outcomes_days": 30, "window_snapshot_days": 30, "backups_keep": 7}


def _iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _predict(logger, seq, at, learner="lrn-a"):
    logger.log(PredictionRecord(
        request_id=uuid.uuid4().hex, source="live", learner_id=learner, learner_seq=seq, question_id="q",
        features={"topic": "Chemical Kinetics", "selected_difficulty": "Medium", "recent_accuracy": 0.5,
                  "mistake_count": 1, "hint_count": 0, "response_time_seconds": 40.0,
                  "prior_interaction_count": seq, "previous_difficulty": "Medium"},
        model_name="m", model_version="2", latency_ms=3.0, status="ok", score=0.4, label=0, occurred_at=_iso(at)))


def _count(db_path, table, where="1=1"):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "monitoring.sqlite3"


def test_committed_retention_config_is_valid():
    cfg = load_retention()
    assert cfg["raw_predictions_days"] == 30 and cfg["window_reports"] == "keep" and cfg["alerts"] == "keep"


def test_old_raw_rows_and_snapshots_go_aggregates_stay(db_path, tmp_path):
    logger = PredictionLogger(db_path, KEY)
    old, recent = NOW - timedelta(days=40), NOW - timedelta(days=2)
    _predict(logger, 0, old)
    _predict(logger, 1, old + timedelta(minutes=1))        # labels seq 0; itself stays pending
    _predict(logger, 0, recent, learner="lrn-b")
    logger.log_outcome(OutcomeRecord(source="live", learner_id="lrn-a", topic="Chemical Kinetics",
                                     question_id="q", outcome="correct", occurred_at=_iso(old)))
    from src.monitoring.label_joiner import join_labels
    join_labels(db_path, now=old + timedelta(minutes=5))
    out = tmp_path / "out"
    cfg_h = run_window.load_config()
    run_window.evaluate_window(db_path, old, old + timedelta(minutes=15), config=cfg_h, evaluator=None, output_dir=out)
    run_window.evaluate_window(db_path, recent, recent + timedelta(minutes=15), config=cfg_h, evaluator=None,
                               output_dir=out)
    (out / "windows" / "not-a-window").mkdir()             # unrelated folders are never touched

    result = apply_retention(db_path, out, CFG, now=NOW)
    assert result == {"predictions_deleted": 1, "outcomes_deleted": 1, "snapshots_deleted": 1, "old_pending_kept": 1}
    assert _count(db_path, "predictions") == 2               # old pending row + recent row
    assert _count(db_path, "monitoring_windows") == 2        # aggregates kept
    assert sorted(p.name for p in (out / "windows").iterdir()) == ["not-a-window", "w-20261228T0300Z-15m"]


def test_dry_run_deletes_nothing(db_path, tmp_path):
    logger = PredictionLogger(db_path, KEY)
    _predict(logger, 0, NOW - timedelta(days=40))
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE predictions SET label_status = 'expired'")
    conn.commit()
    conn.close()
    assert apply_retention(db_path, None, CFG, now=NOW, dry_run=True)["predictions_deleted"] == 1
    assert _count(db_path, "predictions") == 1


@pytest.mark.parametrize("bad", [{"raw_predictions_days": 0}, {"window_snapshot_days": "30"}])
def test_invalid_retention_config_is_refused(tmp_path, bad):
    import yaml
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump({**CFG, **bad}))
    with pytest.raises(ValueError):
        load_retention(path)


def test_scheduler_runs_retention_once_a_day_after_the_backup(db_path, tmp_path, monkeypatch):
    s = scheduler.Settings(db_path=str(db_path), output_dir=str(tmp_path / "out"), backup_dir=str(tmp_path / "b"),
                           window=timedelta(minutes=15), label_lag=timedelta(minutes=5), max_catch_up=96,
                           tick_seconds=60, backup_hour_utc=2, config=run_window.load_config(), retention=CFG)
    logger = PredictionLogger(db_path, KEY)
    _predict(logger, 0, NOW - timedelta(days=40))
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE predictions SET label_status = 'expired'")
    conn.commit()
    conn.close()

    monkeypatch.setattr(scheduler, "backup", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert scheduler.tick(s, now=NOW)["retention"] is None            # no backup today -> nothing deleted
    assert _count(db_path, "predictions") == 1

    from src.monitoring.backup import backup
    monkeypatch.setattr(scheduler, "backup", backup)
    first = scheduler.tick(s, now=NOW + timedelta(minutes=1))
    assert first["backup"] and first["retention"]["predictions_deleted"] == 1
    assert scheduler.tick(s, now=NOW + timedelta(minutes=2))["retention"] is None  # once per day
