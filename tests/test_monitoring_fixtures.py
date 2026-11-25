"""The shared monitoring fixtures must keep matching the live schema and report shape."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.monitoring import db
from src.monitoring.run_window import evaluate_window, load_config, parse_iso
from src.monitoring.schema import FEATURE_COLUMNS, SCHEMA_VERSION

FIXTURES = Path("tests/fixtures/monitoring")
DB = FIXTURES / "monitoring_sample.sqlite3"
REPORT_KEYS = {"window_id", "start", "end", "n_predictions", "n_labelled", "health", "validity", "drift",
               "prediction_psi", "performance", "alerts", "reference_dataset_sha256", "champion_version",
               "status", "report_version"}


@pytest.fixture()
def conn():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    yield c
    c.close()


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_fixture_tables_match_current_schema(conn, tmp_path):
    fresh = sqlite3.connect(tmp_path / "fresh.sqlite3")
    db.ensure_schema(fresh)
    for table in ("predictions", "monitoring_windows", "alerts", "monitor_state"):
        assert _columns(conn, table) == _columns(fresh, table), f"{table} differs: regenerate fixtures"
    fresh.close()


def test_fixture_predictions_are_synthetic_and_pseudonymous(conn):
    sources = {s for (s,) in conn.execute("SELECT DISTINCT source FROM predictions")}
    assert sources == {"simulated"}
    assert conn.execute("SELECT COUNT(*) FROM predictions WHERE schema_version != ?", (SCHEMA_VERSION,)).fetchone()[0] == 0
    raw = DB.read_bytes()
    assert b"lrn-" not in raw and b"base-L0" not in raw  # only HMAC learner keys are stored
    features = json.loads(conn.execute("SELECT features FROM predictions LIMIT 1").fetchone()[0])
    assert list(features) == list(FEATURE_COLUMNS)


def test_fixture_reports_have_the_contract_shape(conn):
    reports = [json.loads(r) for (r,) in conn.execute("SELECT report FROM monitoring_windows ORDER BY window_start")]
    assert len(reports) == 24
    for report in reports:
        assert REPORT_KEYS <= set(report)
    for path in sorted((FIXTURES / "window_reports").glob("*.json")):
        assert REPORT_KEYS <= set(json.loads(path.read_text()))


def test_fixture_timeline_has_the_documented_events(conn):
    rows = {wid: json.loads(rep) for wid, rep in conn.execute("SELECT window_id, report FROM monitoring_windows")}
    metrics = lambda wid: {a["metric"] for a in rows[wid]["alerts"]}  # noqa: E731
    assert "fallback_rate" in metrics("w-20261120T0100Z-15m")
    assert "latency_p95_ms" in metrics("w-20261120T0145Z-15m")
    assert rows["w-20261120T0215Z-15m"]["status"] == "insufficient_data"
    assert "unseen_topic_rate" in metrics("w-20261120T0430Z-15m")
    shifted = [w for w in rows if w >= "w-20261120T0300Z"]
    assert all("psi.response_time_seconds" in metrics(w) for w in shifted)  # shift visible after 03:00


def test_monitoring_job_reproduces_a_fixture_window(tmp_path):
    """Re-running the real job over the fixture predictions gives the stored health numbers."""
    copy = tmp_path / "copy.sqlite3"
    shutil.copy(DB, copy)
    conn = sqlite3.connect(copy)
    stored = json.loads(conn.execute(
        "SELECT report FROM monitoring_windows WHERE window_id = 'w-20261120T0145Z-15m'").fetchone()[0])
    conn.close()
    again = evaluate_window(copy, parse_iso(stored["start"]), parse_iso(stored["end"]), config=load_config(),
                            evaluator=None, now=datetime(2026, 11, 21, tzinfo=timezone.utc))
    assert again["health"] == stored["health"]
    assert again["validity"] == stored["validity"]
    assert again["n_predictions"] == stored["n_predictions"]


def test_registry_snapshot_is_labelled_fixture():
    snap = json.loads((FIXTURES / "registry_snapshot.json").read_text())
    assert "SYNTHETIC FIXTURE" in snap["_fixture"]
    assert {v["version"] for v in snap["versions"]} == {"1", "2", "3"}
    assert all(h["approved_by"] in (None, "fixture-approver") for h in snap["alias_history"])
