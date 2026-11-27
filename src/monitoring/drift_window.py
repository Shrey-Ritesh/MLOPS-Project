"""Sliding-window drift detection and performance monitoring engine for Phase 3.

Reuses statistical formulas from src/track_b/drift.py and conforms to
the Phase 3 Section 4.2 monitoring window contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
import yaml

_EPS = 1e-4


def psi_from_proportions(reference: np.ndarray | list[float], current: np.ndarray | list[float]) -> float:
    """Population Stability Index between two aligned probability vectors with epsilon smoothing."""
    ref = np.clip(np.asarray(reference, dtype=float), _EPS, None)
    cur = np.clip(np.asarray(current, dtype=float), _EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compute_numeric_psi(
    series: pd.Series, bin_edges: list[float], reference_proportions: list[float]
) -> float:
    """Compute PSI for a numeric series against precomputed reference quantiles and proportions."""
    clean = series.dropna().to_numpy()
    if len(clean) == 0:
        return 0.0
    edges = np.asarray(bin_edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    cur_counts, _ = np.histogram(clean, bins=edges)
    cur_props = cur_counts / max(1, cur_counts.sum())
    return psi_from_proportions(reference_proportions, cur_props)


def compute_categorical_psi(
    series: pd.Series, reference_proportions: dict[str, float]
) -> float:
    """Compute PSI for categorical feature against reference category proportions."""
    clean = series.dropna().astype(str)
    all_cats = sorted(set(reference_proportions.keys()) | set(clean.unique()))
    ref = np.array([reference_proportions.get(c, 0.0) for c in all_cats], dtype=float)
    val_counts = clean.value_counts(normalize=True).to_dict()
    cur = np.array([val_counts.get(c, 0.0) for c in all_cats], dtype=float)
    return psi_from_proportions(ref, cur)


def compute_ks(series: pd.Series, reference_sample: list[float]) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test between current window and reference sample."""
    clean = series.dropna().to_numpy()
    if len(clean) == 0 or len(reference_sample) == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    res = stats.ks_2samp(reference_sample, clean)
    return {"statistic": round(float(res.statistic), 4), "p_value": float(res.pvalue)}


def compute_window_performance(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float] | None:
    """Compute F1, ROC-AUC, and Brier score when delayed ground truth labels exist."""
    y_t = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    y_s = pd.to_numeric(pd.Series(y_score), errors="coerce").to_numpy(dtype=float)
    if len(y_t) != len(y_s):
        raise ValueError(f"Length mismatch between y_true ({len(y_t)}) and y_score ({len(y_s)})")
    valid = ~np.isnan(y_t) & ~np.isnan(y_s)
    y_t_clean = y_t[valid].astype(int)
    y_s_clean = y_s[valid]

    if len(y_t_clean) < 5 or len(np.unique(y_t_clean)) < 2:
        return None

    y_pred = (y_s_clean >= threshold).astype(int)
    f1 = float(f1_score(y_t_clean, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_t_clean, y_s_clean))
    brier = float(brier_score_loss(y_t_clean, y_s_clean))

    return {
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
    }


def evaluate_window_drift(
    window_df: pd.DataFrame,
    reference_profile: dict[str, Any],
    thresholds: dict[str, Any],
    prediction_scores: np.ndarray | pd.Series | None = None,
    ground_truth_labels: np.ndarray | pd.Series | None = None,
    window_id: str = "window_unspecified",
) -> dict[str, Any]:
    """Evaluate sliding-window drift according to the Section 4.2 contract.

    Handles window size checks (marks insufficient_data if N < min_predictions),
    computes feature PSI and KS, prediction PSI, performance metrics, and alerts.
    """
    n_predictions = len(window_df)
    min_preds = thresholds.get("window", {}).get("min_predictions", 50)
    min_labeled = thresholds.get("window", {}).get("min_labeled", 20)

    # Contract requirement: Small windows are reported as insufficient_data, not as "no drift"
    if n_predictions < min_preds:
        return {
            "window_id": window_id,
            "n_predictions": n_predictions,
            "n_labelled": 0 if ground_truth_labels is None else int(pd.Series(ground_truth_labels).notna().sum()),
            "status": "insufficient_data",
            "insufficient_data": True,
            "drift": {},
            "prediction_psi": None,
            "performance": None,
            "alerts": [],
            "reference_dataset_sha256": reference_profile.get("dataset_sha256", "unknown"),
            "champion_version": reference_profile.get("model_id", "unknown"),
        }

    psi_warn = thresholds.get("feature_drift", {}).get("psi_warn", 0.10)
    psi_alert = thresholds.get("feature_drift", {}).get("psi_alert", 0.20)
    ks_alpha = thresholds.get("feature_drift", {}).get("ks_alpha", 0.01)

    drift_results: dict[str, dict[str, Any]] = {}
    alerts: list[dict[str, Any]] = []

    # 1. Numeric Feature Drift
    for col, meta in reference_profile.get("numeric_features", {}).items():
        if col in window_df.columns:
            psi = compute_numeric_psi(window_df[col], meta["bin_edges"], meta["bin_proportions"])
            ks_res = compute_ks(window_df[col], meta.get("ks_reference_sample", []))
            
            status = "ALERT" if psi >= psi_alert else "WARN" if psi >= psi_warn else "OK"
            alert_metric = "psi"
            alert_value = round(psi, 4)
            alert_threshold = psi_alert if status == "ALERT" else psi_warn
            if ks_res["p_value"] < ks_alpha and status == "OK":
                status = "WARN"
                alert_metric = "ks"
                alert_value = float(ks_res["p_value"])
                alert_threshold = float(ks_alpha)

            drift_results[col] = {
                "psi": round(psi, 4),
                "ks_stat": ks_res["statistic"],
                "ks_p": ks_res["p_value"],
                "status": status,
            }

            if status in ("WARN", "ALERT"):
                alerts.append({
                    "type": "FEATURE_DRIFT",
                    "feature": col,
                    "metric": alert_metric,
                    "value": alert_value,
                    "threshold": alert_threshold,
                    "severity": status,
                })

    # 2. Categorical Feature Drift
    for col, meta in reference_profile.get("categorical_features", {}).items():
        if col in window_df.columns:
            psi = compute_categorical_psi(window_df[col], meta["proportions"])
            status = "ALERT" if psi >= psi_alert else "WARN" if psi >= psi_warn else "OK"
            drift_results[col] = {
                "psi": round(psi, 4),
                "status": status,
            }
            if status in ("WARN", "ALERT"):
                alerts.append({
                    "type": "CATEGORICAL_DRIFT",
                    "feature": col,
                    "metric": "psi",
                    "value": round(psi, 4),
                    "threshold": psi_alert if status == "ALERT" else psi_warn,
                    "severity": status,
                })

    # 3. Prediction Drift (PSI)
    prediction_psi = None
    pred_meta = reference_profile.get("prediction_scores", {})
    if prediction_scores is not None and "bin_edges" in pred_meta:
        scores_series = pd.Series(prediction_scores)
        prediction_psi = round(
            compute_numeric_psi(scores_series, pred_meta["bin_edges"], pred_meta["bin_proportions"]), 4
        )
        pred_alert_thr = thresholds.get("prediction_drift", {}).get("psi_alert", 0.15)
        pred_warn_thr = thresholds.get("prediction_drift", {}).get("psi_warn", 0.10)
        
        if prediction_psi >= pred_alert_thr:
            alerts.append({
                "type": "PREDICTION_DRIFT",
                "metric": "prediction_psi",
                "value": prediction_psi,
                "threshold": pred_alert_thr,
                "severity": "ALERT",
            })
        elif prediction_psi >= pred_warn_thr:
            alerts.append({
                "type": "PREDICTION_DRIFT",
                "metric": "prediction_psi",
                "value": prediction_psi,
                "threshold": pred_warn_thr,
                "severity": "WARN",
            })

    # 4. Model Performance (if ground truth is present)
    n_labelled = 0
    performance_metrics = None
    if ground_truth_labels is not None:
        labels_series = pd.Series(ground_truth_labels)
        n_labelled = int(labels_series.notna().sum())
        if n_labelled >= min_labeled and prediction_scores is not None:
            thr = reference_profile.get("decision_threshold", 0.7)
            performance_metrics = compute_window_performance(labels_series, prediction_scores, threshold=thr)

            if performance_metrics is not None:
                base_f1 = thresholds.get("performance_degradation", {}).get("baseline_f1", 0.68)
                max_f1_drop = thresholds.get("performance_degradation", {}).get("max_f1_drop", 0.05)
                if (base_f1 - performance_metrics["f1"]) >= max_f1_drop:
                    alerts.append({
                        "type": "PERFORMANCE_DEGRADATION",
                        "metric": "f1_drop",
                        "value": round(base_f1 - performance_metrics["f1"], 4),
                        "threshold": max_f1_drop,
                        "severity": "ALERT",
                    })

    return {
        "window_id": window_id,
        "n_predictions": n_predictions,
        "n_labelled": n_labelled,
        "status": "ok",
        "insufficient_data": False,
        "drift": drift_results,
        "prediction_psi": prediction_psi,
        "performance": performance_metrics,
        "alerts": alerts,
        "reference_dataset_sha256": reference_profile.get("dataset_sha256", "unknown"),
        "champion_version": reference_profile.get("model_id", "unknown"),
    }
