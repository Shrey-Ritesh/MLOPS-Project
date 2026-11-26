"""Phase 3 Day 3: monitoring window evaluation, alerts, drift hook, and the scheduler."""
from __future__ import annotations

import json
import uuid
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import drift_hook, scheduler
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.monitoring.run_window import evaluate_window, load_config, window_id_for

KEY = "test-pseudonym-key-0123456789"
W0 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)
W1 = W0 + timedelta(minutes=15)


def _iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _features(**overrides):
    base = {"topic": "Chemical Kinetics", "selected_difficulty": "Medium", "recent_accuracy": 0.6,
            "mistake_count": 1, "hint_count": 0, "response_time_seconds": 42.0,
            "prior_interaction_count": 3, "previous_difficulty": "Medium"}
    base.update(overrides)
    return base


def _log_many(logger, n, *, start=W0, learner="lrn-a", seq0=0, step=timedelta(seconds=30), **overrides):
    for i in range(n):
        fields = dict(
            request_id=uuid.uuid4().hex, source="live", learner_id=learner, learner_seq=seq0 + i,
            question_id="q", features=_features(prior_interaction_count=seq0 + i), model_name="learner-difficulty",
            model_version="2", latency_ms=5.0, status="ok", score=0.75, label=1, threshold=0.7,
            occurred_at=_iso(start + i * step),
        )
        fields.update({k: (v(i) if callable(v) else v) for k, v in overrides.items()})
        logger.log(PredictionRecord(**fields))


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "monitoring.sqlite3"


@pytest.fixture()
def logger(db_path):
    return PredictionLogger(db_path, KEY)


def _alerts(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return [dict(zip(("window_id", "severity", "category", "metric", "value", "threshold"), r))
                for r in conn.execute("SELECT window_id, severity, category, metric, value, threshold FROM alerts")]
    finally:
        conn.close()


# ---- window evaluation -----------------------------------------------------------------------

def test_healthy_window_reports_ok_with_full_shape(logger, db_path, cfg, tmp_path):
    _log_many(logger, 20)
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None, output_dir=tmp_path / "out")
    assert report["status"] == "ok"
    assert report["window_id"] == window_id_for(W0, W1) == "w-20261126T0900Z-15m"
    for key in ("window_id", "start", "end", "n_predictions", "n_labelled", "health", "validity", "drift",
                "prediction_psi", "performance", "alerts", "reference_dataset_sha256", "champion_version"):
        assert key in report
    assert report["n_predictions"] == 20 and report["champion_version"] == "2"
    assert report["health"]["p95"] == 5.0 and report["health"]["error_rate"] == 0.0
    assert report["prediction_distribution"]["score_counts"][7] == 20  # all scores in [0.7, 0.8)
    assert report["drift"] == "not_configured"  # never claims "no drift" when it was not measured
    snap = tmp_path / "out" / "windows" / report["window_id"]
    assert json.loads((snap / "report.json").read_text())["status"] == "ok"
    assert len((snap / "predictions.csv").read_text().splitlines()) == 21
    assert "lrn-a" not in (snap / "predictions.csv").read_text()


def test_small_window_is_insufficient_data_not_healthy(logger, db_path, cfg):
    _log_many(logger, 3, status="error", score=None, label=None)
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    assert report["status"] == "insufficient_data"
    assert report["health"]["status"] == "insufficient_data"
    assert report["health"]["error_rate"] == 1.0  # raw numbers still reported
    assert _alerts(db_path) == []                 # but nobody is paged for 3 requests


def test_empty_window_is_insufficient_data(db_path, cfg, logger):
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    assert report["status"] == "insufficient_data" and report["n_predictions"] == 0
    assert report["health"]["p95"] is None


def test_breaches_raise_alerts_with_severity(logger, db_path, cfg):
    # 20 predictions: 6 fallbacks (30% -> critical), slow latency (p95 > 250 -> warning)
    _log_many(logger, 20,
              status=lambda i: "fallback" if i < 6 else "ok",
              fallback=lambda i: i < 6,
              latency_ms=lambda i: 400.0 if i % 2 else 5.0)
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    assert report["status"] == "alert"
    by_metric = {a["metric"]: a for a in _alerts(db_path)}
    assert by_metric["fallback_rate"]["severity"] == "critical"
    assert by_metric["fallback_rate"]["value"] == 0.3 and by_metric["fallback_rate"]["threshold"] == 0.3
    assert by_metric["latency_p95_ms"]["severity"] == "warning"
    assert all(a["window_id"] == report["window_id"] for a in by_metric.values())


def test_validity_counts_impossible_values_and_unseen_topics(logger, db_path, cfg):
    _log_many(logger, 20,
              features=lambda i: _features(
                  prior_interaction_count=i,
                  recent_accuracy=1.5 if i < 1 else 0.5,                      # impossible -> out of range (5%)
                  topic="Organic Name Reactions" if i < 10 else "Chemical Kinetics"))  # unseen
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    v = report["validity"]
    assert v["out_of_range"] == 1 and v["unseen_topics"] == 10
    assert v["unseen_topic_names"] == {"Organic Name Reactions": 10}
    metrics = {a["metric"]: a["severity"] for a in _alerts(db_path)}
    assert metrics["unseen_topic_rate"] == "critical" and metrics["out_of_range_rate"] == "warning"


def test_re_evaluating_a_window_replaces_its_alerts(logger, db_path, cfg):
    _log_many(logger, 20, status="error", score=None, label=None)
    evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    assert [a["metric"] for a in _alerts(db_path)].count("error_rate") == 1


def test_simulated_and_live_counts_are_reported_separately(logger, db_path, cfg):
    _log_many(logger, 10)
    _log_many(logger, 10, learner="lrn-sim", source="simulated")
    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=None)
    assert report["health"]["by_source"] == {"live": 10, "simulated": 10}


# ---- drift hook ------------------------------------------------------------------------------

def test_drift_evaluator_results_and_alerts_are_included(logger, db_path, cfg):
    _log_many(logger, 20)

    def fake(rows):
        assert len(rows) == 20 and isinstance(rows[0]["features"], dict)
        return {"drift": {"hint_count": {"psi": 0.31, "ks_stat": 0.2, "ks_p": 0.001, "status": "drift"}},
                "prediction_psi": 0.12, "performance": "insufficient_data",
                "reference_dataset_sha256": "abc",
                "alerts": [{"severity": "critical", "category": "drift", "metric": "psi.hint_count",
                            "value": 0.31, "threshold": 0.2, "message": "hint_count drifted"}]}

    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=fake)
    assert report["drift"]["hint_count"]["psi"] == 0.31 and report["reference_dataset_sha256"] == "abc"
    assert report["status"] == "alert"
    assert {"psi.hint_count"} == {a["metric"] for a in _alerts(db_path)}


def test_a_crashing_drift_evaluator_does_not_kill_monitoring(logger, db_path, cfg):
    _log_many(logger, 20)

    def broken(rows):
        raise RuntimeError("boom")

    report = evaluate_window(db_path, W0, W1, config=cfg, evaluator=broken)
    assert report["drift"] == "error" and report["drift_error_type"] == "RuntimeError"
    assert report["health"]["error_rate"] == 0.0


def test_missing_evaluator_module_means_not_configured():
    assert drift_hook.load_evaluator() is None  # until Person 2 adds src/monitoring/drift_windows.py


# ---- scheduler -------------------------------------------------------------------------------

@pytest.fixture()
def settings(db_path, tmp_path, cfg):
    return scheduler.Settings(
        db_path=str(db_path), output_dir=str(tmp_path / "out"), backup_dir=str(tmp_path / "backups"),
        window=timedelta(minutes=15), label_lag=timedelta(minutes=5), max_catch_up=4,
        tick_seconds=60, backup_hour_utc=2, config=cfg,
    )


def test_first_tick_evaluates_latest_complete_window_after_label_lag(settings, logger):
    _log_many(logger, 12)
    # 09:19: window 09:00-09:15 closed but label lag (5 min) not over -> evaluates 08:45-09:00
    s1 = scheduler.tick(settings, now=W1 + timedelta(minutes=4))
    assert [w for w, _ in s1["windows"]] == ["w-20261126T0845Z-15m"]
    s2 = scheduler.tick(settings, now=W1 + timedelta(minutes=5))
    assert s2["windows"] == [("w-20261126T0900Z-15m", "ok")]
    assert scheduler.tick(settings, now=W1 + timedelta(minutes=6))["windows"] == []  # nothing new


def test_catch_up_after_downtime_and_gap_alert(settings, logger, db_path):
    scheduler.tick(settings, now=W1 + timedelta(minutes=5))  # last evaluated end = 09:15
    later = W1 + timedelta(hours=2, minutes=5)                 # 8 windows due, limit 4
    summary = scheduler.tick(settings, now=later)
    assert len(summary["windows"]) == 4 and summary["skipped_windows"] == 4
    assert summary["windows"][-1][0] == "w-20261126T1100Z-15m"
    assert any(a["metric"] == "monitoring_gap" for a in _alerts(db_path))


def test_changing_window_length_realigns_without_overlap(settings, logger):
    scheduler.tick(settings, now=W1 + timedelta(minutes=5))           # evaluated up to 09:15
    short = replace(settings, window=timedelta(minutes=2), label_lag=timedelta(minutes=0))
    summary = scheduler.tick(short, now=W1 + timedelta(minutes=6))    # 09:21
    # 09:15 is not on a 2-minute boundary: a 1-minute bridge window keeps coverage gap-free
    assert [w for w, _ in summary["windows"]] == ["w-20261126T0915Z-1m", "w-20261126T0916Z-2m",
                                                  "w-20261126T0918Z-2m"]


def test_daily_backup_runs_once_after_backup_hour(settings, logger, tmp_path):
    _log_many(logger, 1)
    night = datetime(2026, 11, 27, 1, 30, tzinfo=timezone.utc)
    assert scheduler.tick(settings, now=night)["backup"] is None                      # before 02:00
    first = scheduler.tick(settings, now=night + timedelta(hours=1))["backup"]
    assert first and first.startswith("monitoring-20261127")
    assert scheduler.tick(settings, now=night + timedelta(hours=3))["backup"] is None  # already done today
    assert len(list((tmp_path / "backups").glob("monitoring-*.sqlite3"))) == 1


def test_tick_joins_labels(settings, logger, db_path):
    _log_many(logger, 2, features=lambda i: _features(prior_interaction_count=i,
                                                      selected_difficulty="Hard" if i == 0 else "Easy"))
    scheduler.tick(settings, now=W1 + timedelta(minutes=5))
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT label_observed FROM predictions WHERE learner_seq = 0").fetchone() == (1,)
    finally:
        conn.close()


def test_heartbeat_written_and_checked(settings, logger):
    scheduler.tick(settings, now=datetime.now(timezone.utc))
    assert scheduler.heartbeat_is_fresh(settings.output_dir, 120)
    assert not scheduler.heartbeat_is_fresh(settings.output_dir + "-missing", 120)
