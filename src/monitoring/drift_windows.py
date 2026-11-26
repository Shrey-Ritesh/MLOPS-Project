"""Drift evaluator bridge plugging into Person 1's drift_hook.py.

Conforms to Section 4.2 contract and connects src/monitoring/drift_window.py
with the scheduled monitoring window evaluator (src/monitoring/run_window.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE_PATH = Path("reports/monitoring/reference_profile.json")
DEFAULT_THRESHOLDS_PATH = Path("configs/monitoring/thresholds.yaml")


def load_reference_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Reference profile not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Thresholds config not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evaluate_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluator function called by src/monitoring/drift_hook.py.

    Args:
        rows: List of prediction records from the window. Each record contains
              keys: 'features' (dict), 'score', 'label_observed', 'label_status', etc.

    Returns:
        dict with keys: 'drift', 'prediction_psi', 'performance',
        'reference_dataset_sha256', and 'alerts'.
    """
    if not rows:
        return {
            "drift": "insufficient_data",
            "prediction_psi": None,
            "performance": "insufficient_data",
            "reference_dataset_sha256": None,
            "alerts": [],
        }

    try:
        import pandas as pd
        from src.monitoring.drift_window import evaluate_window_drift

        ref_profile = load_reference_profile()
        thresholds = load_thresholds()
    except Exception:
        # Fallback if configs or optional statistical/ML packages are missing
        return {
            "drift": "not_configured",
            "prediction_psi": None,
            "performance": "not_configured",
            "reference_dataset_sha256": None,
            "alerts": [],
        }

    # Flatten feature dicts into a DataFrame
    flat_records = []
    scores = []
    labels = []

    for r in rows:
        feats = r.get("features", {})
        if not isinstance(feats, dict):
            try:
                feats = json.loads(feats) if feats else {}
            except Exception:
                feats = {}

        record = dict(feats)
        score = r.get("score")
        if score is not None:
            record["score"] = float(score)
            scores.append(float(score))
        else:
            scores.append(float("nan"))

        # Delayed ground truth label if observed
        lbl = r.get("label_observed")
        if lbl is not None and r.get("label_status") == "joined":
            labels.append(int(lbl))
        else:
            labels.append(None)

        flat_records.append(record)

    window_df = pd.DataFrame(flat_records)
    scores_series = pd.Series(scores) if scores else None
    labels_series = pd.Series(labels) if any(l is not None for l in labels) else None

    eval_result = evaluate_window_drift(
        window_df=window_df,
        reference_profile=ref_profile,
        thresholds=thresholds,
        prediction_scores=scores_series,
        ground_truth_labels=labels_series,
    )

    if eval_result.get("insufficient_data"):
        return {
            "drift": "insufficient_data",
            "prediction_psi": None,
            "performance": "insufficient_data",
            "reference_dataset_sha256": eval_result.get("reference_dataset_sha256"),
            "alerts": [],
        }

    # Normalize alerts into drift_hook's expected schema
    normalized_alerts: list[dict[str, Any]] = []
    for a in eval_result.get("alerts", []):
        sev = "critical" if a.get("severity") == "ALERT" else "warning"
        category = "performance" if "PERFORMANCE" in a.get("type", "") else "drift"
        metric = f"{a.get('type')}:{a.get('feature', a.get('metric', 'value'))}"
        val = a.get("value", 0.0)
        thr = a.get("threshold", 0.0)
        msg = f"{metric} breached threshold ({val} >= {thr})"

        normalized_alerts.append({
            "severity": sev,
            "category": category,
            "metric": metric,
            "value": float(val),
            "threshold": float(thr),
            "message": msg,
        })

    perf = eval_result.get("performance")
    if perf is None:
        perf = "insufficient_data"

    return {
        "drift": eval_result.get("drift", {}),
        "prediction_psi": eval_result.get("prediction_psi"),
        "performance": perf,
        "reference_dataset_sha256": eval_result.get("reference_dataset_sha256"),
        "alerts": normalized_alerts,
    }
