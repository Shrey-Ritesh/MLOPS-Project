"""Shared feature pipeline, model construction and evaluation for the Track B demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.track_b.schema import (
    DIFFICULTY_ORDINAL,
    FEATURE_COLUMNS,
    TARGET,
    check_feature_list,
    validate_dataset,
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load a synthetic dataset plus its sidecar metadata; validate the schema first."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    problems = validate_dataset(df)
    if problems:
        raise ValueError(f"{csv_path} failed schema validation: {problems[:3]}")
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return df, meta


def split_by_learner(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split by learner using the test ids recorded at generation time."""
    test_ids = set(meta["test_learner_ids"])
    is_test = df["learner_id"].isin(test_ids)
    train, test = df[~is_test].copy(), df[is_test].copy()
    assert not set(train["learner_id"]) & set(test["learner_id"])
    return train, test


def _ordinal(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame[c].map(DIFFICULTY_ORDINAL).to_numpy() for c in frame.columns])


def _log1p(x):
    return np.log1p(x)


def apply_difficulty_boundary_guard(proba: np.ndarray, selected_difficulty: pd.Series) -> np.ndarray:
    """Business constraint: a learner at Easy difficulty cannot switch to a lower difficulty."""
    p = proba.copy()
    p[(selected_difficulty == "Easy").to_numpy()] = 0.0
    return p


def tune_threshold_on_train(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Find the decision threshold in [0.2, 0.85] that maximizes F1 on training data."""
    if grid is None:
        grid = np.arange(0.20, 0.86, 0.02)
    best_thr, best_f1 = 0.5, 0.0
    for thr in grid:
        f = f1_score(y_true, proba >= thr, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = float(round(thr, 2))
    return best_thr


def build_pipeline(model_cfg: dict, features: list[str] = FEATURE_COLUMNS) -> Pipeline:
    check_feature_list(features)
    model_type = model_cfg.get("type", "logistic_regression")

    if model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                 ["topic", "selected_difficulty", "previous_difficulty"]),
            ],
            remainder="passthrough",
        )
        clf = HistGradientBoostingClassifier(
            max_iter=model_cfg.get("max_iter", 250),
            class_weight=model_cfg.get("class_weight", "balanced"),
            l2_regularization=model_cfg.get("l2_regularization", 1.0),
            random_state=model_cfg.get("random_state", 20261015),
        )
        return Pipeline([("features", pre), ("model", clf)])

    elif model_type == "logistic_regression":
        pre = ColumnTransformer(
            [
                ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
                ("difficulty", FunctionTransformer(_ordinal), ["selected_difficulty", "previous_difficulty"]),
                ("log_numeric", Pipeline([("log", FunctionTransformer(_log1p)), ("scale", StandardScaler())]),
                 ["response_time_seconds", "prior_interaction_count"]),
                ("numeric", StandardScaler(), ["recent_accuracy", "mistake_count", "hint_count"]),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            C=model_cfg["C"],
            class_weight=model_cfg.get("class_weight"),
            max_iter=model_cfg["max_iter"],
            random_state=model_cfg["random_state"],
        )
        return Pipeline([("features", pre), ("model", clf)])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model: Pipeline, df: pd.DataFrame, threshold: float = 0.5, apply_guard: bool = False) -> dict:
    """Metrics for `model` on `df`, plus the majority-class reference on the same data."""
    X, y = df[FEATURE_COLUMNS], df[TARGET].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    if apply_guard and "selected_difficulty" in df.columns:
        proba = apply_difficulty_boundary_guard(proba, df["selected_difficulty"])
    pred = (proba >= threshold).astype(int)
    majority_label = int(round(y.mean()))  # 0 whenever positives are the minority
    maj_pred = np.full_like(y, majority_label)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_rows": int(len(df)),
        "n_learners": int(df["learner_id"].nunique()),
        "positive_rate": round(float(y.mean()), 4),
        "decision_threshold": round(float(threshold), 4),
        "boundary_guard_applied": bool(apply_guard),
        "model": {
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4) if len(set(y)) == 2 else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "majority_class_baseline": {
            "predicts": majority_label,
            "precision": round(float(precision_score(y, maj_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, maj_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, maj_pred, zero_division=0)), 4),
            "accuracy": round(float((maj_pred == y).mean()), 4),
        },
        "model_accuracy": round(float((pred == y).mean()), 4),
    }
