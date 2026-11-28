"""Detection-delay report over monitoring windows (graded topic 4 evidence)."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone

from src.monitoring.detection_report import build_report, main, to_markdown

FIXTURE = "tests/fixtures/monitoring/monitoring_sample.sqlite3"
SHIFT = datetime(2026, 11, 20, 3, 0, tzinfo=timezone.utc)


def _copy(tmp_path):
    path = tmp_path / "m.sqlite3"
    shutil.copy(FIXTURE, path)
    return path


def test_fixture_experiment_numbers(tmp_path):
    rep = build_report(_copy(tmp_path), SHIFT)
    assert rep["windows"]["control"] == 12 and rep["windows"]["shifted"] == 12
    assert rep["control_phase"]["false_alarm_windows_drift_or_performance"] == [
        "w-20261120T0000Z-15m", "w-20261120T0100Z-15m", "w-20261120T0115Z-15m"]
    assert rep["control_phase"]["windows_alerting_by_category"]["health"] == 2  # incidents, not drift
    first = rep["data_drift_detection"]
    assert first["metric"] == "psi.response_time_seconds"
    assert first["first_window"] == "w-20261120T0300Z-15m" and first["delay_minutes"] == 20.0
    assert rep["detection_by_metric"]["psi.response_time_seconds"]["share_of_shifted_windows"] == 1.0
    assert rep["performance_detection"] is None                      # no performance alert in the fixture...
    trend = rep["performance_trend"]
    assert trend["after_shift"]["mean_f1"] < trend["before_shift"]["mean_f1"]  # ...but the trend shows the drop


def test_straddling_window_is_excluded(tmp_path):
    rep = build_report(_copy(tmp_path), datetime(2026, 11, 20, 3, 5, tzinfo=timezone.utc))
    assert rep["windows"]["straddling_excluded"] == ["w-20261120T0300Z-15m"]
    assert rep["windows"]["control"] + rep["windows"]["shifted"] == 23


def test_start_end_filters_and_empty_db(tmp_path):
    rep = build_report(_copy(tmp_path), SHIFT, start=datetime(2026, 11, 20, 2, 0, tzinfo=timezone.utc),
                       end=datetime(2026, 11, 20, 4, 0, tzinfo=timezone.utc))
    assert rep["windows"]["total"] == 8
    empty = build_report(tmp_path / "empty.sqlite3", SHIFT)
    assert empty["windows"]["total"] == 0 and empty["data_drift_detection"] is None
    assert "not detected" in to_markdown(empty)


def test_cli_json_and_markdown(tmp_path, capsys):
    path = str(_copy(tmp_path))
    assert main(["--db", path, "--shift-at", "2026-11-20T03:00:00Z"]) == 0
    assert json.loads(capsys.readouterr().out)["shift_at"] == "2026-11-20T03:00:00.000Z"
    assert main(["--db", path, "--shift-at", "2026-11-20T03:00:00Z", "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert "| psi.response_time_seconds | drift |" in out and "Data drift vs performance" in out


def test_report_reads_only(tmp_path):
    path = _copy(tmp_path)
    before = sqlite3.connect(path).execute("SELECT COUNT(*) FROM monitoring_windows").fetchone()
    build_report(path, SHIFT)
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM monitoring_windows").fetchone() == before
