"""Unit tests for the automated retraining trigger engine."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest
import yaml

from src.monitoring import db
from src.monitoring.contract import load_contract
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.monitoring.trigger import (
    check_cooldown,
    evaluate_trigger_conditions,
    execute_retraining_pipeline,
    load_trigger_config,
    process_window_for_trigger,
    snapshot_from_monitoring,
)


@pytest.fixture
def trigger_config():
    return load_trigger_config(Path("configs/monitoring/retrain_trigger.yaml"))


def test_trigger_does_not_fire_on_clean_baseline(trigger_config):
    """Control phase validation: baseline report with 0 alerts must not fire."""
    clean_report = {
        "window_id": "win_clean_001",
        "n_predictions": 60,
        "n_labelled": 40,
        "alerts": [],
        "prediction_psi": 0.02,
        "performance": {"f1": 0.68, "roc_auc": 0.70},
    }
    should_fire, condition_met, _ = evaluate_trigger_conditions(clean_report, trigger_config)
    assert not should_fire
    assert condition_met is None


def test_trigger_fires_on_feature_drift(trigger_config):
    """Trigger must fire when >= 2 drift alerts are raised."""
    drifted_report = {
        "window_id": "win_drifted_001",
        "n_predictions": 60,
        "n_labelled": 40,
        "alerts": [
            {"severity": "critical", "category": "drift", "metric": "hint_count", "value": 0.25, "threshold": 0.20},
            {"severity": "critical", "category": "drift", "metric": "response_time_seconds", "value": 0.22, "threshold": 0.20},
        ],
        "prediction_psi": 0.05,
    }
    should_fire, condition_met, windows = evaluate_trigger_conditions(drifted_report, trigger_config)
    assert should_fire
    assert "feature_drift_exceeded_threshold" in condition_met
    assert "win_drifted_001" in windows


def test_trigger_fires_on_prediction_psi(trigger_config):
    """Trigger must fire when prediction PSI >= 0.20."""
    pred_shifted_report = {
        "window_id": "win_pred_shifted",
        "n_predictions": 60,
        "n_labelled": 40,
        "alerts": [],
        "prediction_psi": 0.24,
    }
    should_fire, condition_met, _ = evaluate_trigger_conditions(pred_shifted_report, trigger_config)
    assert should_fire
    assert "prediction_psi_exceeded_threshold" in condition_met


def test_cooldown_enforcement(tmp_path, trigger_config, monkeypatch):
    """Test that cooldown stops repeated trigger execution.

    snapshot_data and execute_retraining_pipeline are mocked so the test
    runs in CI without dvc, mlflow, or the full training dataset.
    """
    import src.monitoring.trigger as _trigger_mod

    state_file = tmp_path / "test_trigger_state.json"
    events_file = tmp_path / "test_trigger_events.jsonl"
    monkeypatch.setattr(_trigger_mod, "DEFAULT_EVENTS_LOG", events_file)

    # Mock snapshot_data to avoid reading the real training CSV
    monkeypatch.setattr(
        _trigger_mod,
        "snapshot_data",
        lambda source_csv, out_dir=None: (tmp_path / "snap.csv", "abcd1234" * 8),
    )

    # Mock execute_retraining_pipeline to avoid dvc/mlflow subprocess calls
    monkeypatch.setattr(
        _trigger_mod,
        "execute_retraining_pipeline",
        lambda **kw: {
            "pipeline_run_id": "run_mock_ci_test",
            "challenger_model_version": "candidate_mock",
        },
    )

    drifted_report = {
        "window_id": "win_test_cooldown",
        "n_predictions": 60,
        "n_labelled": 40,
        "alerts": [
            {"severity": "critical", "category": "drift", "metric": "hint_count", "value": 0.25, "threshold": 0.20},
            {"severity": "critical", "category": "drift", "metric": "response_time_seconds", "value": 0.22, "threshold": 0.20},
        ],
    }

    # First trigger execution: fires and sets cooldown
    first_event = process_window_for_trigger(
        drifted_report,
        state_path=state_file,
        simulated_mode=False,
    )
    assert first_event is not None
    assert first_event["status"] == "success"
    assert "data_snapshot_sha256" in first_event

    # Second immediate execution: must be blocked by cooldown
    second_event = process_window_for_trigger(
        drifted_report,
        state_path=state_file,
        simulated_mode=False,
    )
    assert second_event is not None
    assert second_event["status"] == "cooldown_blocked"
    assert "Cooldown active" in second_event["reason"]


def test_trigger_respects_insufficient_data_and_outage_guardrails(trigger_config):
    """Trigger must refuse insufficient_data windows and fallback/outage incidents."""
    insufficient = {
        "window_id": "win_small",
        "status": "insufficient_data",
        "n_predictions": 10,
        "prediction_psi": 0.90,
        "alerts": [],
    }
    fire_small, _, _ = evaluate_trigger_conditions(insufficient, trigger_config)
    assert not fire_small

    outage = {
        "window_id": "win_outage",
        "status": "alert",
        "n_predictions": 100,
        "n_labelled": 90,
        "prediction_psi": 0.92,
        "performance": {"f1": 0.1},
        "alerts": [
            {"severity": "critical", "category": "health", "metric": "fallback_rate", "value": 0.7, "threshold": 0.05},
        ],
    }
    fire_outage, _, _ = evaluate_trigger_conditions(outage, trigger_config)
    assert not fire_outage


def test_prediction_drift_not_counted_as_feature_drift(trigger_config):
    """One feature alert + one prediction-drift alert must NOT satisfy >= 2 feature alerts."""
    report = {
        "window_id": "win_mixed_001",
        "n_predictions": 80,
        "n_labelled": 40,
        "prediction_psi": 0.16,  # below Rule 2's 0.20 threshold
        "alerts": [
            {"severity": "critical", "category": "drift", "metric": "FEATURE_DRIFT:psi:hint_count", "value": 0.25, "threshold": 0.20},
            {"severity": "critical", "category": "drift", "metric": "PREDICTION_DRIFT:prediction_psi", "value": 0.16, "threshold": 0.15},
        ],
    }
    should_fire, cond, _ = evaluate_trigger_conditions(report, trigger_config)
    assert not should_fire
    assert cond is None


def test_trigger_reads_baseline_f1_and_pipeline_command_from_config(tmp_path, trigger_config, monkeypatch):
    """Rule 3 reads baseline_f1 from thresholds.yaml and pipeline executes cfg pipeline_command."""
    custom_thr = tmp_path / "thresholds.yaml"
    custom_thr.write_text(
        yaml.safe_dump({"performance_degradation": {"baseline_f1": 0.80}}),
        encoding="utf-8",
    )
    report = {
        "window_id": "win_perf_cfg",
        "n_predictions": 1000,
        "n_labelled": 1000,
        "alerts": [],
        "performance": {"f1": 0.74, "roc_auc": 0.85},
    }
    should_fire, cond, _ = evaluate_trigger_conditions(report, trigger_config, thresholds_path=custom_thr)
    assert should_fire
    assert "performance_f1_drop_exceeded_threshold" in cond

    seen_cmds = []

    def fake_run(cmd, **kw):
        seen_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    import src.monitoring.trigger as _trigger_mod
    monkeypatch.setattr(_trigger_mod, "_latest_version_number", lambda: "1")

    execute_retraining_pipeline(
        trigger_id="trig_cfg_test",
        evidence_windows=["win_perf_cfg"],
        snapshot_sha="deadbeef" * 8,
        cfg={"trigger_policy": {"pipeline_command": "dvc repro train_challenger"}},
        challenger_dir=tmp_path / "challenger",
    )
    assert seen_cmds[0] == ["dvc", "repro", "train_challenger"]


def test_simulated_cooldown_window_clock_and_clean_windows(tmp_path, monkeypatch):
    """Simulated cooldown uses window end time and config windows, and ignores clean windows."""
    import src.monitoring.trigger as _trigger_mod

    state_file = tmp_path / "state.json"
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("MONITORING_WINDOW_MINUTES", "15")
    monkeypatch.setattr(
        _trigger_mod,
        "snapshot_data",
        lambda source_csv, out_dir=None: (tmp_path / "snap.csv", "abcd1234" * 8),
    )

    drifted = {
        "window_id": "w1",
        "start": "2026-11-26T10:30:00.000Z",
        "end": "2026-11-26T10:45:00.000Z",
        "n_predictions": 100,
        "alerts": [
            {"severity": "critical", "category": "drift", "metric": "FEATURE_DRIFT:psi:hint_count", "value": 0.3, "threshold": 0.2},
            {"severity": "critical", "category": "drift", "metric": "FEATURE_DRIFT:psi:response_time_seconds", "value": 0.3, "threshold": 0.2},
        ],
    }
    e1 = process_window_for_trigger(
        drifted, state_path=state_file, events_path=events_file, simulated_mode=True, execute_pipeline=False
    )
    assert e1 is not None and e1["status"] == "success"
    assert e1["pipeline_run_id"] == "dry_run"
    assert e1["cooldown_until"] == "2026-11-26T12:00:00.000Z"  # 5 * 15m = 75m after 10:45Z

    # Clean window during cooldown -> None, NOT cooldown_blocked
    clean = {
        "window_id": "w_clean",
        "start": "2026-11-26T10:45:00.000Z",
        "end": "2026-11-26T11:00:00.000Z",
        "n_predictions": 100,
        "alerts": [],
    }
    assert process_window_for_trigger(
        clean, state_path=state_file, events_path=events_file, simulated_mode=True, execute_pipeline=False
    ) is None

    # Drifted window still inside 75-min cooldown -> cooldown_blocked
    drifted_w2 = {**drifted, "window_id": "w2", "end": "2026-11-26T11:15:00.000Z"}
    e2 = process_window_for_trigger(
        drifted_w2, state_path=state_file, events_path=events_file, simulated_mode=True, execute_pipeline=False
    )
    assert e2 is not None and e2["status"] == "cooldown_blocked"

    # Drifted window once window time passes 12:00Z -> fires again immediately
    drifted_w3 = {**drifted, "window_id": "w3", "end": "2026-11-26T12:15:00.000Z"}
    e3 = process_window_for_trigger(
        drifted_w3, state_path=state_file, events_path=events_file, simulated_mode=True, execute_pipeline=False
    )
    assert e3 is not None and e3["status"] == "success"
    assert len(events_file.read_text(encoding="utf-8").splitlines()) == 2


def test_mlflow_not_tagged_when_registration_fails_or_unchanged(tmp_path, monkeypatch):
    """Challenger alias must not be attached if register_mlflow fails or creates no new version."""
    import src.monitoring.trigger as _trigger_mod

    tag_calls = []
    monkeypatch.setattr(_trigger_mod, "_tag_challenger_in_mlflow", lambda **kw: tag_calls.append(kw) or "99")
    monkeypatch.setattr(_trigger_mod, "_latest_version_number", lambda: "3")

    def fail_reg(cmd, **kw):
        if "register_mlflow" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fail_reg)
    res_fail = execute_retraining_pipeline("t1", ["w1"], "abc", challenger_dir=tmp_path)
    assert res_fail["mlflow_model_version"] is None
    assert tag_calls == []

    # Now registration exits 0, but latest version stays "3" (before == after)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "ok", ""))
    res_same = execute_retraining_pipeline("t2", ["w1"], "abc", challenger_dir=tmp_path)
    assert res_same["mlflow_model_version"] is None
    assert tag_calls == []


def test_snapshot_from_monitoring_extracts_joined_rows(tmp_path):
    """Snapshot exports only joined rows in [start, end) with features and target."""
    import pandas as pd

    db_file = tmp_path / "m.sqlite3"
    contract = load_contract()
    pred_logger = PredictionLogger(db_file, "pseudonym_secret_key_12345678", contract=contract)
    feats = {
        "topic": "Chemical Kinetics",
        "selected_difficulty": "Medium",
        "recent_accuracy": 0.6,
        "mistake_count": 1,
        "hint_count": 2,
        "response_time_seconds": 25.0,
        "prior_interaction_count": 3,
        "previous_difficulty": "Medium",
    }
    for i in range(2):
        pred_logger.log(
            PredictionRecord(
                request_id=f"req_{i}",
                source="simulated",
                learner_id=f"lrn_{i}",
                learner_seq=i,
                question_id=f"q_{i}",
                features=feats,
                score=0.8,
                label=1,
                threshold=0.7,
                model_name="learner-difficulty",
                model_version="1",
                latency_ms=10.0,
                status="ok",
                occurred_at="2026-11-26T10:35:00.000Z",
            )
        )
    with db.session(db_file) as conn:
        conn.execute("UPDATE predictions SET label_observed = 0, label_status = 'joined' WHERE request_id = 'req_0'")

    snap_path, sha = snapshot_from_monitoring(
        db_file, "2026-11-26T10:30:00.000Z", "2026-11-26T10:45:00.000Z", out_dir=tmp_path / "snaps"
    )
    assert snap_path.exists()
    assert len(sha) == 64
    df = pd.read_csv(snap_path)
    assert len(df) == 1
    assert df["switch_to_lower_difficulty"].iloc[0] == 0
    assert df["hint_count"].iloc[0] == 2


def test_excluded_features_do_not_fire_trigger(trigger_config):
    """topic and prior_interaction_count alerts stay on dashboard but do not fire trigger."""
    report = {
        "window_id": "win_excluded_001",
        "n_predictions": 60,
        "n_labelled": 40,
        "alerts": [
            {"severity": "critical", "category": "drift", "metric": "CATEGORICAL_DRIFT:psi:topic", "value": 4.95, "threshold": 0.20},
            {"severity": "critical", "category": "drift", "metric": "FEATURE_DRIFT:psi:prior_interaction_count", "value": 0.75, "threshold": 0.20},
            {"severity": "critical", "category": "drift", "metric": "FEATURE_DRIFT:psi:response_time_seconds", "value": 0.25, "threshold": 0.20},
        ],
    }
    should_fire, cond, _ = evaluate_trigger_conditions(report, trigger_config)
    assert not should_fire
    assert cond is None


def test_baseline_fixture_does_not_fire_trigger(trigger_config):
    """Unshifted 60-row baseline fixture must not satisfy trigger conditions."""
    import joblib
    import pandas as pd
    from src.monitoring.drift_window import evaluate_window_drift
    from src.monitoring.drift_windows import load_reference_profile, load_thresholds
    from src.track_b.schema import FEATURE_COLUMNS

    df = pd.read_csv("tests/fixtures/monitoring/sample_window_baseline.csv")
    ref = load_reference_profile()
    thr = load_thresholds()
    scores = joblib.load("models/track_b/candidate/model.joblib").predict_proba(df[FEATURE_COLUMNS])[:, 1]
    res = evaluate_window_drift(df, ref, thr, prediction_scores=scores, window_id="win_fixture_base")
    normalized_alerts = [
        {
            "severity": "critical" if a["severity"] == "ALERT" else "warning",
            "category": "performance" if "PERFORMANCE" in a["type"] else "drift",
            "metric": f"{a['type']}:{a['metric']}:{a['feature']}" if "feature" in a else f"{a['type']}:{a['metric']}",
            "value": a["value"],
            "threshold": a["threshold"],
        }
        for a in res["alerts"]
    ]
    report = {
        "window_id": "win_fixture_base",
        "status": "alert" if normalized_alerts else "ok",
        "n_predictions": len(df),
        "n_labelled": 0,
        "prediction_psi": res["prediction_psi"],
        "alerts": normalized_alerts,
    }
    should_fire, cond, _ = evaluate_trigger_conditions(report, trigger_config)
    assert not should_fire
    assert cond is None


def test_small_window_f1_drop_does_not_fire_trigger(trigger_config):
    """300-row control windows have noisy F1 (sd ~0.07) and must not fire below min_labeled_for_perf."""
    noisy_small_window = {
        "window_id": "win_small_f1_noise",
        "n_predictions": 300,
        "n_labelled": 300,
        "alerts": [],
        "prediction_psi": 0.05,
        "performance": {"f1": 0.58, "roc_auc": 0.85},
    }
    should_fire, cond, _ = evaluate_trigger_conditions(noisy_small_window, trigger_config)
    assert not should_fire
    assert cond is None
