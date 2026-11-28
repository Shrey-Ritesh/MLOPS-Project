"""Phase 3 Day 4: the monitor hands each evaluated window to the retraining trigger.

The trigger itself (conditions, cooldown, snapshot, pipeline) is Person 2's and is tested in
tests/test_trigger.py. These tests cover the wiring: it is off by default, it is called once per
new window, it never stops monitoring, and with the real trigger a shifted window fires on its own
while the next one is held by the cooldown.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring import scheduler, trigger
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.monitoring.run_window import load_config
from src.track_b import train_candidate
from src.track_b.schema import FEATURE_COLUMNS

KEY = "test-pseudonym-key-0123456789"
W0 = datetime(2026, 11, 28, 10, 0, tzinfo=timezone.utc)
BASELINE = Path("tests/fixtures/monitoring/sample_window_baseline.csv")
SHIFTED = Path("tests/fixtures/monitoring/sample_window_shifted.csv")


def _iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@pytest.fixture()
def settings(tmp_path):
    return scheduler.Settings(
        db_path=str(tmp_path / "monitoring.sqlite3"), output_dir=str(tmp_path / "out"),
        backup_dir=str(tmp_path / "backups"), window=timedelta(minutes=15), label_lag=timedelta(minutes=5),
        max_catch_up=4, tick_seconds=60, backup_hour_utc=2, config=load_config(),
        trigger_enabled=True, trigger_execute=False, trigger_simulated=True,
        trigger_state_dir=str(tmp_path / "retraining"),
    )


def _log_one(settings, at, seq=0):
    PredictionLogger(settings.db_path, KEY).log(PredictionRecord(
        request_id=uuid.uuid4().hex, source="simulated", learner_id="lrn-a", learner_seq=seq, question_id="q",
        features={"topic": "Chemical Kinetics", "selected_difficulty": "Medium", "recent_accuracy": 0.6,
                  "mistake_count": 1, "hint_count": 0, "response_time_seconds": 42.0,
                  "prior_interaction_count": seq, "previous_difficulty": "Medium"},
        model_name="learner-difficulty", model_version="2", latency_ms=5.0, status="ok",
        score=0.4, label=0, threshold=0.7, occurred_at=_iso(at),
    ))


def _log_shifted_window(settings, start, seq: dict[str, int]):
    """Replay Person 2's shifted sample (60 rows) with documented input shift, in learner order so labels join."""
    df = pd.read_csv(SHIFTED).copy()
    df["response_time_seconds"] = (df["response_time_seconds"] * 0.45).round(2)
    df["hint_count"] = df["hint_count"] + 2
    logger = PredictionLogger(settings.db_path, KEY)
    step = timedelta(minutes=14) / len(df)
    for i, row in df.iterrows():
        learner = str(row["learner_id"])
        n = seq.get(learner, 0)
        seq[learner] = n + 1
        feats = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FEATURE_COLUMNS}
        score = min(1.0, 0.2 + 0.1 * float(row["mistake_count"]) + 0.1 * float(row["hint_count"]))
        logger.log(PredictionRecord(
            request_id=uuid.uuid4().hex, source="simulated", learner_id=learner, learner_seq=n,
            question_id=f"q{i}", features=feats, model_name="learner-difficulty", model_version="2",
            latency_ms=5.0, status="ok", score=score, label=int(score >= 0.7), threshold=0.7,
            occurred_at=_iso(start + i * step),
        ))


def _log_baseline_window(settings, start, seq: dict[str, int]):
    """Replay the unshifted 60-row baseline fixture scored by the champion model, in learner order."""
    import joblib

    df = pd.read_csv(BASELINE).copy()
    model = joblib.load("models/track_b/candidate/model.joblib")
    scores = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    logger = PredictionLogger(settings.db_path, KEY)
    step = timedelta(minutes=14) / len(df)
    for i, row in df.iterrows():
        learner = str(row["learner_id"])
        n = seq.get(learner, 0)
        seq[learner] = n + 1
        feats = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FEATURE_COLUMNS}
        score = float(scores[i])
        logger.log(PredictionRecord(
            request_id=uuid.uuid4().hex, source="simulated", learner_id=learner, learner_seq=n,
            question_id=f"qb{i}", features=feats, model_name="learner-difficulty", model_version="2",
            latency_ms=5.0, status="ok", score=score, label=int(score >= 0.7), threshold=0.7,
            occurred_at=_iso(start + i * step),
        ))


def test_trigger_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MONITORING_TRIGGER_ENABLED", raising=False)
    s = scheduler.Settings.from_env()
    assert s.trigger_enabled is False
    assert s.trigger_execute is True  # once switched on, it retrains unless told to dry-run


def test_env_switches_the_trigger_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MONITORING_TRIGGER_ENABLED", "1")
    monkeypatch.setenv("MONITORING_TRIGGER_EXECUTE", "0")
    monkeypatch.setenv("MONITORING_TRIGGER_SIMULATED", "true")
    monkeypatch.setenv("TRIGGER_STATE_DIR", str(tmp_path / "r"))
    s = scheduler.Settings.from_env()
    assert (s.trigger_enabled, s.trigger_execute, s.trigger_simulated) == (True, False, True)
    assert s.trigger_state_dir == str(tmp_path / "r")


def test_disabled_trigger_is_never_called(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(trigger, "process_window_for_trigger", lambda *a, **k: calls.append(a))
    _log_one(settings, W0 + timedelta(minutes=1))
    off = scheduler.Settings(**{**settings.__dict__, "trigger_enabled": False})
    summary = scheduler.tick(off, now=W0 + timedelta(minutes=20))
    assert summary["windows"] and calls == []


def test_each_new_window_is_passed_once_with_state_on_the_volume(settings, monkeypatch):
    calls = []

    def fake(report, **kwargs):
        calls.append((report["window_id"], kwargs))
        return None

    monkeypatch.setattr(trigger, "process_window_for_trigger", fake)
    _log_one(settings, W0 + timedelta(minutes=1))
    scheduler.tick(settings, now=W0 + timedelta(minutes=20))
    scheduler.tick(settings, now=W0 + timedelta(minutes=21))  # same window: not passed again
    assert [c[0] for c in calls] == ["w-20261128T1000Z-15m"]
    kwargs = calls[0][1]
    assert kwargs["state_path"] == Path(settings.trigger_state_dir) / "trigger_state.json"
    assert kwargs["events_path"] == Path(settings.trigger_state_dir) / "trigger_events.jsonl"
    assert kwargs["db_path"] == settings.db_path
    assert (kwargs["simulated_mode"], kwargs["execute_pipeline"]) == (True, False)


def test_a_crashing_trigger_never_stops_monitoring(settings, monkeypatch, capsys):
    def boom(report, **kwargs):
        raise RuntimeError("secret detail that must not be logged")

    monkeypatch.setattr(trigger, "process_window_for_trigger", boom)
    _log_one(settings, W0 + timedelta(minutes=1))
    summary = scheduler.tick(settings, now=W0 + timedelta(minutes=20))
    assert summary["windows"] == [("w-20261128T1000Z-15m", "insufficient_data")]
    out = capsys.readouterr().out
    assert '"event":"trigger_failed"' in out and '"error_type":"RuntimeError"' in out
    assert "secret detail" not in out
    # The window is still recorded as evaluated, so it is not re-run (and re-triggered) next tick.
    assert scheduler.tick(settings, now=W0 + timedelta(minutes=21))["windows"] == []


def test_real_trigger_fires_on_a_shifted_window_then_cooldown_holds(settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trigger, "DEFAULT_SNAPSHOT_DIR", tmp_path / "snapshots")
    seq: dict[str, int] = {}  # learner_seq continues across windows, as in a real session
    _log_shifted_window(settings, W0, seq)
    _log_shifted_window(settings, W0 + timedelta(minutes=15), seq)

    first = scheduler.tick(settings, now=W0 + timedelta(minutes=20))
    second = scheduler.tick(settings, now=W0 + timedelta(minutes=35))

    fired = first["trigger_events"][0]
    assert fired["status"] == "success" and fired["condition_met"].startswith("feature_drift_exceeded_threshold")
    assert fired["evidence_window_ids"] == ["w-20261128T1000Z-15m"]
    assert fired["snapshot_source"] == "monitoring_db" and len(fired["data_snapshot_sha256"]) == 64
    assert fired["pipeline_run_id"] == "dry_run"
    assert list((tmp_path / "snapshots").glob("snapshot_*.csv"))

    assert second["trigger_events"][0]["status"] == "cooldown_blocked"

    events = [json.loads(line) for line in
              (Path(settings.trigger_state_dir) / "trigger_events.jsonl").read_text().splitlines()]
    assert [e["trigger_id"] for e in events] == [fired["trigger_id"]]  # the blocked window adds no event
    out = capsys.readouterr().out
    assert '"event":"trigger_fired"' in out and '"event":"trigger_blocked"' in out


def test_real_trigger_does_not_fire_on_baseline_fixture(settings):
    """Control phase: unshifted 60-row baseline replay must not false-alarm trigger."""
    seq: dict[str, int] = {}
    _log_baseline_window(settings, W0, seq)
    summary = scheduler.tick(settings, now=W0 + timedelta(minutes=20))
    assert summary["windows"]
    assert summary.get("trigger_events", []) == []
    events_file = Path(settings.trigger_state_dir) / "trigger_events.jsonl"
    assert not events_file.exists()


def test_git_commit_falls_back_to_the_image_build_arg(monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train_candidate.subprocess, "run", no_git)
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    assert train_candidate.git_commit() == "abc123"
    monkeypatch.delenv("GIT_COMMIT")
    assert train_candidate.git_commit() == "unknown"


def test_day4_five_baseline_then_shifted_300_row_windows(tmp_path, monkeypatch):
    """End-to-end Day 4 replay: 5x300 baseline windows stay quiet; shifted window 1 fires; window 2 blocked."""
    import joblib
    from scripts.simulate_traffic import iter_learner_ordered_rows, load_traffic_dataset
    from src.track_b.modeling import apply_difficulty_boundary_guard

    monkeypatch.setenv("MONITORING_WINDOW_MINUTES", "2")
    monkeypatch.setattr(trigger, "DEFAULT_SNAPSHOT_DIR", tmp_path / "snapshots")

    win_settings = scheduler.Settings(
        db_path=str(tmp_path / "monitoring.sqlite3"),
        output_dir=str(tmp_path / "out"),
        backup_dir=str(tmp_path / "backups"),
        window=timedelta(minutes=2),
        label_lag=timedelta(minutes=5),
        max_catch_up=10,
        tick_seconds=60,
        backup_hour_utc=2,
        config=load_config(),
        trigger_enabled=True,
        trigger_execute=False,
        trigger_simulated=True,
        trigger_state_dir=str(tmp_path / "retraining"),
    )

    model = joblib.load("models/track_b/candidate/model.joblib")
    pred_logger = PredictionLogger(win_settings.db_path, KEY)

    base_df = load_traffic_dataset("baseline", test_only=True)
    shift_df = load_traffic_dataset("shifted", test_only=True)
    base_rows = iter_learner_ordered_rows(base_df, limit=1500, seed=1)
    shift_rows = iter_learner_ordered_rows(shift_df, limit=600, seed=101)

    def _log_batch(batch, start_dt, prefix):
        bdf = pd.DataFrame([r for _, _, r in batch])
        scores = apply_difficulty_boundary_guard(model.predict_proba(bdf[FEATURE_COLUMNS])[:, 1], bdf["selected_difficulty"])
        step = timedelta(seconds=110) / len(batch)
        for idx, (lid, seq, row) in enumerate(batch):
            feats = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FEATURE_COLUMNS}
            score = float(scores[idx])
            pred_logger.log(PredictionRecord(
                request_id=f"req_{prefix}_{idx}",
                source="simulated",
                learner_id=f"{prefix}_{lid}",
                learner_seq=seq,
                question_id=f"q_{idx}",
                features=feats,
                model_name="learner-difficulty",
                model_version="2",
                latency_ms=5.0,
                status="ok",
                score=score,
                label=int(score >= 0.7),
                threshold=0.7,
                occurred_at=_iso(start_dt + idx * step),
            ))

    # 5 control windows (10:00..10:10Z), then 2 shifted windows (10:10..10:14Z)
    for w in range(5):
        _log_batch(base_rows[w * 300:(w + 1) * 300], W0 + timedelta(minutes=2 * w), "ctrl")
    for w in range(2):
        _log_batch(shift_rows[w * 300:(w + 1) * 300], W0 + timedelta(minutes=10 + 2 * w), "shift")

    # Evaluate control windows one by one (now = window_end + 5m label_lag)
    for w in range(5):
        summary = scheduler.tick(win_settings, now=W0 + timedelta(minutes=2 * (w + 1) + 5))
        assert summary.get("trigger_events", []) == []

    # First shifted window must fire on hint_count + response_time_seconds
    s1 = scheduler.tick(win_settings, now=W0 + timedelta(minutes=12 + 5))
    assert len(s1.get("trigger_events", [])) == 1
    fired = s1["trigger_events"][0]
    assert fired["status"] == "success"
    assert "hint_count" in fired["condition_met"]
    assert "response_time_seconds" in fired["condition_met"]

    # Second shifted window must be cooldown_blocked
    s2 = scheduler.tick(win_settings, now=W0 + timedelta(minutes=14 + 5))
    assert len(s2.get("trigger_events", [])) == 1
    assert s2["trigger_events"][0]["status"] == "cooldown_blocked"
