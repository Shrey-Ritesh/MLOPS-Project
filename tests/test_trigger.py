"""Unit tests for the automated retraining trigger engine."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.monitoring.trigger import (
    check_cooldown,
    evaluate_trigger_conditions,
    load_trigger_config,
    process_window_for_trigger,
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
            {"severity": "critical", "category": "drift", "metric": "topic", "value": 0.22, "threshold": 0.20},
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
