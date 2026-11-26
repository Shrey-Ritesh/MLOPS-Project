"""Phase 3 Day 5 (done early): monitoring failure paths. No silent gaps, missed windows reported."""
from __future__ import annotations

import json
import uuid
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import run_window, scheduler
from src.monitoring.backup import BackupError, backup, restore
from src.monitoring.label_joiner import join_labels
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord

KEY = "test-pseudonym-key-0123456789"
W0 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)
WIN = timedelta(minutes=15)


def _iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _feat(i, difficulty="Medium"):
    return {"topic": "Chemical Kinetics", "selected_difficulty": difficulty, "recent_accuracy": 0.5,
            "mistake_count": 1, "hint_count": 0, "response_time_seconds": 40.0,
            "prior_interaction_count": i, "previous_difficulty": "Medium"}


def _log(logger, n, start, learner="lrn-a", seq0=0):
    for i in range(n):
        logger.log(PredictionRecord(
            request_id=uuid.uuid4().hex, source="live", learner_id=learner, learner_seq=seq0 + i,
            question_id="q", features=_feat(seq0 + i), model_name="m", model_version="2", latency_ms=3.0,
            status="ok", score=0.4, label=0, threshold=0.7, occurred_at=_iso(start + timedelta(seconds=40 * i))))


@pytest.fixture()
def settings(tmp_path):
    return scheduler.Settings(
        db_path=str(tmp_path / "state" / "monitoring.sqlite3"), output_dir=str(tmp_path / "out"),
        backup_dir=str(tmp_path / "backups"), window=WIN, label_lag=timedelta(minutes=5), max_catch_up=96,
        tick_seconds=60, backup_hour_utc=2, config=run_window.load_config(),
    )


@pytest.fixture()
def logger(settings):
    return PredictionLogger(settings.db_path, KEY)


def _windows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT window_id FROM monitoring_windows ORDER BY window_start")]
    finally:
        conn.close()


def test_fresh_volume_without_database_starts_cleanly(settings):
    """First start on an empty app_state volume: the scheduler creates the schema itself."""
    summary = scheduler.tick(settings, now=W0 + WIN + timedelta(minutes=5))
    assert summary["windows"] == [("w-20261126T0900Z-15m", "insufficient_data")]


def test_crash_mid_run_does_not_skip_windows(settings, logger, monkeypatch):
    scheduler.tick(settings, now=W0 + timedelta(minutes=5))           # evaluated up to 09:00
    _log(logger, 12, W0)
    _log(logger, 12, W0 + WIN, learner="lrn-b")

    real = run_window.evaluate_window
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(scheduler, "evaluate_window", flaky)
    with pytest.raises(OSError):
        scheduler.tick(settings, now=W0 + 2 * WIN + timedelta(minutes=5))  # 09:00 ok, 09:15 crashes
    assert _windows(settings.db_path)[-1] == "w-20261126T0900Z-15m"

    monkeypatch.setattr(scheduler, "evaluate_window", real)
    summary = scheduler.tick(settings, now=W0 + 3 * WIN + timedelta(minutes=5))  # the next tick catches up
    assert [w for w, _ in summary["windows"]] == ["w-20261126T0915Z-15m", "w-20261126T0930Z-15m"]
    ids = _windows(settings.db_path)
    assert ids == sorted(set(ids))  # every window exactly once, in order


def test_main_loop_survives_a_failed_tick(settings, monkeypatch, capsys):
    monkeypatch.setattr(scheduler.Settings, "from_env", classmethod(lambda cls, path=None: settings))
    monkeypatch.setattr(scheduler, "tick", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert scheduler.main(["--once"]) == 1
    assert "monitor_tick_failed" in capsys.readouterr().out


def test_window_with_no_labels_yet_is_reported_honestly(settings, logger):
    _log(logger, 12, W0, learner="lrn-a")  # each learner's last prediction can never be labelled yet
    for k in range(12):
        _log(logger, 1, W0 + timedelta(seconds=k), learner=f"solo-{k}")
    report = run_window.evaluate_window(settings.db_path, W0, W0 + WIN, config=settings.config, evaluator=None)
    assert report["n_labelled"] == 0 and report["n_label_pending"] == 24
    assert report["status"] == "ok"            # health is still judged...
    assert report["performance"] == "not_configured"  # ...but no accuracy claim is made


def test_expired_labels_are_counted_in_later_reports(settings, logger):
    _log(logger, 12, W0)
    join_labels(settings.db_path, now=W0 + timedelta(hours=80))
    report = run_window.evaluate_window(settings.db_path, W0, W0 + WIN, config=settings.config, evaluator=None)
    assert report["n_labelled"] == 11 and report["n_label_expired"] == 1


def test_corrupt_feature_row_is_a_schema_failure_not_a_crash(settings, logger):
    _log(logger, 12, W0)
    conn = sqlite3.connect(settings.db_path)
    conn.execute("UPDATE predictions SET features = '{not json' WHERE learner_seq = 0")
    conn.commit()
    conn.close()
    report = run_window.evaluate_window(settings.db_path, W0, W0 + WIN, config=settings.config, evaluator=None)
    assert report["validity"]["schema_failures"] == 1
    assert any(a["metric"] == "schema_failure_rate" for a in report["alerts"])


def test_backup_restores_into_a_working_monitoring_database(settings, logger, tmp_path):
    _log(logger, 12, W0)
    scheduler.tick(settings, now=W0 + WIN + timedelta(minutes=5))
    made = backup(settings.db_path, settings.backup_dir, now=W0 + timedelta(hours=3))

    restored = tmp_path / "restored" / "monitoring.sqlite3"
    restored.parent.mkdir()
    restore(made, restored)
    # The restored copy is a fully working monitoring DB: same rows, and the job runs on it.
    conn = sqlite3.connect(restored)
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 12
    stored = json.loads(conn.execute("SELECT report FROM monitoring_windows").fetchone()[0])
    conn.close()
    again = run_window.evaluate_window(restored, W0, W0 + WIN, config=settings.config, evaluator=None)
    assert again["health"] == stored["health"]


def test_failed_backup_is_logged_and_retried_next_tick(settings, logger, monkeypatch, capsys):
    _log(logger, 1, W0)
    night = datetime(2026, 11, 27, 3, 0, tzinfo=timezone.utc)

    def broken(*a, **k):
        raise BackupError("integrity check failed")

    monkeypatch.setattr(scheduler, "backup", broken)
    assert scheduler.tick(settings, now=night)["backup"] is None
    assert "backup_failed" in capsys.readouterr().out
    monkeypatch.setattr(scheduler, "backup", backup)
    assert scheduler.tick(settings, now=night + timedelta(minutes=1))["backup"] is not None  # retried same day
