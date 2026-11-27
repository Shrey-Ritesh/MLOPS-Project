"""Unit and regression tests for sliding-window drift module and traffic generator."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring.drift_window import (
    compute_categorical_psi,
    compute_numeric_psi,
    compute_window_performance,
    evaluate_window_drift,
    psi_from_proportions,
)
from src.track_b.drift import psi_numeric
from scripts.simulate_traffic import format_prediction_payload


@pytest.fixture
def reference_profile():
    profile_path = Path("reports/monitoring/reference_profile.json")
    assert profile_path.exists(), "reference_profile.json must be generated first"
    return json.loads(profile_path.read_text(encoding="utf-8"))


@pytest.fixture
def thresholds():
    thresh_path = Path("configs/monitoring/thresholds.yaml")
    assert thresh_path.exists(), "thresholds.yaml must exist"
    return yaml.safe_load(thresh_path.read_text(encoding="utf-8"))


def test_psi_from_proportions_identical():
    p = np.array([0.2, 0.3, 0.5])
    assert psi_from_proportions(p, p) == pytest.approx(0.0, abs=1e-5)


def test_psi_from_proportions_different():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.5, 0.3, 0.2])
    psi = psi_from_proportions(p, q)
    assert psi > 0.1  # Significant shift


def test_evaluate_window_insufficient_data(reference_profile, thresholds):
    df_small = pd.read_csv("tests/fixtures/monitoring/sample_window_insufficient.csv")
    res = evaluate_window_drift(df_small, reference_profile, thresholds, window_id="win_test_small")
    assert res["insufficient_data"] is True
    assert res["status"] == "insufficient_data"
    assert res["drift"] == {}


def test_evaluate_window_baseline(reference_profile, thresholds):
    df_base = pd.read_csv("tests/fixtures/monitoring/sample_window_baseline.csv")
    res = evaluate_window_drift(df_base, reference_profile, thresholds, window_id="win_test_base")
    assert res["insufficient_data"] is False
    assert res["status"] == "ok"
    assert "recent_accuracy" in res["drift"]
    assert "topic" in res["drift"]
    # Baseline traffic should not trigger performance degradation alerts
    alert_types = [a["type"] for a in res["alerts"]]
    assert "PERFORMANCE_DEGRADATION" not in alert_types


def test_evaluate_window_shifted(reference_profile, thresholds):
    df_shift = pd.read_csv("tests/fixtures/monitoring/sample_window_shifted.csv")
    res = evaluate_window_drift(df_shift, reference_profile, thresholds, window_id="win_test_shift")
    assert res["insufficient_data"] is False
    assert "hint_count" in res["drift"]
    assert res["drift"]["hint_count"]["psi"] >= 0.0


def test_performance_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.92, 0.05])
    perf = compute_window_performance(y_true, y_score, threshold=0.5)
    assert perf is not None
    assert perf["f1"] == 1.0
    assert perf["roc_auc"] == 1.0
    assert perf["brier"] < 0.1


def test_drift_regression_against_phase1_report():
    """Regression test ensuring exact match with reports/track_b/drift_report.json."""
    drift_report_path = Path("reports/track_b/drift_report.json")
    assert drift_report_path.exists(), "drift_report.json must exist"
    p1_report = json.loads(drift_report_path.read_text(encoding="utf-8"))

    ref_df = pd.read_csv(p1_report["reference"]["dataset"])
    cur_df = pd.read_csv(p1_report["current"]["dataset"])

    # Verify that numeric PSI reproduces the recorded numbers
    for col, meta in p1_report["numeric_features"].items():
        psi_calc = psi_numeric(ref_df[col], cur_df[col], bins=10)
        assert round(psi_calc, 4) == meta["psi"], f"Regression failure for {col} PSI"


def test_simulate_traffic_payload_contract():
    """Verify that simulated traffic enforces Section 4.1 contract."""
    df = pd.read_csv("tests/fixtures/monitoring/sample_window_baseline.csv")
    payload = format_prediction_payload(df.iloc[0], mode="baseline")
    assert payload["source"] == "simulated"
    assert "prediction_id" in payload
    assert "features" in payload
    assert "selected_difficulty" in payload


def test_partial_ground_truth_labels_handled_safely(reference_profile, thresholds):
    """Ensure rows with None/NaN delayed labels do not crash performance computation."""
    baseline_df = pd.read_csv("tests/fixtures/monitoring/sample_window_baseline.csv")
    scores = pd.Series([0.8 if i % 2 == 0 else 0.2 for i in range(len(baseline_df))])
    # Mix valid labels and None (simulating pending delayed labels)
    labels = pd.Series([i % 2 if i < 30 else None for i in range(len(baseline_df))], dtype=object)

    rep = evaluate_window_drift(
        window_df=baseline_df,
        reference_profile=reference_profile,
        thresholds=thresholds,
        prediction_scores=scores,
        ground_truth_labels=labels,
        window_id="win_partial_labels",
    )
    assert rep["n_labelled"] == 30
    assert isinstance(rep["performance"], dict)
    assert "f1" in rep["performance"]


def test_drift_windows_bridge_edge_cases(monkeypatch):
    """Verify drift_windows bridge handles empty rows, corrupted features, and missing pandas cleanly."""
    from src.monitoring import drift_windows

    empty_res = drift_windows.evaluate_window([])
    assert empty_res["drift"] == "insufficient_data"
    assert empty_res["alerts"] == []

    # Insufficient predictions with malformed features string should still not crash
    small_rows = [{"features": "{bad-json", "score": 0.5, "label_status": "pending"} for _ in range(5)]
    small_res = drift_windows.evaluate_window(small_rows)
    assert small_res["drift"] == "insufficient_data"

    # Simulate missing optional config / dependency -> graceful not_configured fallback
    monkeypatch.setattr(drift_windows, "load_reference_profile", lambda *a, **kw: (_ for _ in ()).throw(ModuleNotFoundError("pandas")))
    fallback_res = drift_windows.evaluate_window(small_rows)
    assert fallback_res["drift"] == "not_configured"
    assert fallback_res["prediction_psi"] is None


def test_ks_only_alert_labelled_as_ks_not_psi(monkeypatch):
    """When PSI stays below warn threshold and only KS rejects, alert must report ks."""
    from src.monitoring import drift_windows

    ref_profile = {
        "model_id": "test_model",
        "dataset_sha256": "abc123",
        "decision_threshold": 0.7,
        "numeric_features": {
            "response_time_seconds": {
                "bin_edges": [-float("inf"), 10.0, float("inf")],
                "bin_proportions": [0.5, 0.5],
                "ks_reference_sample": [1.0] * 50 + [20.0] * 50,
            }
        },
        "categorical_features": {},
    }
    thresholds = {
        "window": {"min_predictions": 50, "min_labeled": 20},
        "feature_drift": {"psi_warn": 0.10, "psi_alert": 0.20, "ks_alpha": 0.01},
    }
    # Identical bin proportions (psi == 0.0), but clustered away from 1.0 and 20.0 so KS p < 0.01
    df = pd.DataFrame({"response_time_seconds": [9.0] * 30 + [11.0] * 30})
    res = evaluate_window_drift(df, ref_profile, thresholds, window_id="win_ks_only")
    assert res["drift"]["response_time_seconds"]["psi"] == pytest.approx(0.0, abs=1e-4)
    assert res["drift"]["response_time_seconds"]["status"] == "WARN"
    assert len(res["alerts"]) == 1
    alert = res["alerts"][0]
    assert alert["metric"] == "ks"
    assert alert["value"] < 0.01
    assert alert["threshold"] == 0.01

    monkeypatch.setattr(drift_windows, "load_reference_profile", lambda *a, **kw: ref_profile)
    monkeypatch.setattr(drift_windows, "load_thresholds", lambda *a, **kw: thresholds)
    rows = [{"features": {"response_time_seconds": float(v)}, "score": 0.5, "label_status": "pending"} for v in df["response_time_seconds"]]
    bridged = drift_windows.evaluate_window(rows)
    assert len(bridged["alerts"]) == 1
    norm = bridged["alerts"][0]
    assert norm["metric"] == "FEATURE_DRIFT:ks:response_time_seconds"
    assert "ks_p=" in norm["message"]
    assert ">= 0.1" not in norm["message"]
