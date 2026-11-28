"""The status command is read-only and summarises the fixture correctly."""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone

from src.monitoring import status

FIXTURE = "tests/fixtures/monitoring/monitoring_sample.sqlite3"


def test_status_summarises_without_writing(tmp_path):
    db_path = tmp_path / "m.sqlite3"
    shutil.copy(FIXTURE, db_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "monitor.heartbeat").write_text("2026-11-20T06:04:30.000Z")
    (tmp_path / "bk").mkdir()
    (tmp_path / "bk" / "monitoring-20261120T020000Z.sqlite3").write_bytes(b"x")

    s = status.collect(db_path, tmp_path / "out", tmp_path / "bk", n_windows=3, n_alerts=2,
                       now=datetime(2026, 11, 20, 6, 5, tzinfo=timezone.utc))

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob("m.sqlite3-*"))  # no WAL or journal created
    assert sum(r["n"] for r in s["predictions"]) == 2406
    assert len(s["windows"]) == 3 and len(s["alerts"]) == 2
    assert s["heartbeat_age_seconds"] == 30 and s["backups_total"] == 1
    md = status.to_markdown(s)
    assert "Latest alerts" in md and "30 s ago" in md


def test_status_without_a_database(tmp_path):
    s = status.collect(tmp_path / "missing.sqlite3")
    assert "error" in s and "not found" in status.to_markdown(s)
    assert not (tmp_path / "missing.sqlite3").exists()
