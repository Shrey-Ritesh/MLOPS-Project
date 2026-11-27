"""Champion vs Challenger model comparison and Promotion Gate v2 evaluator (Person 3).

Evaluates champion and challenger on identical rows across three datasets:
  (a) fixed original held-out test learners (original_test)
  (b) fixed shifted held-out test learners (shifted_test)
  (c) recent labelled live window from monitoring.sqlite3 (live_window)

Computes F1, ROC-AUC, precision, recall, calibration (Brier score), single-row
latency (p50/p95 ms), and serialized model size (MB), evaluates
``configs/track_b/promotion_gate_v2.yaml``, and optionally logs the report and
decision to MLflow.

Synthetic-data notice: Track B models are trained and evaluated on SYNTHETIC
learner data only; comparison results are not evidence about real students.

CLI usage::

    python -m src.track_b.model_comparison \
        --champion-dir models/track_b/baseline \
        --challenger-dir models/track_b/candidate \
        --monitoring-db data/state/monitoring.sqlite3 \
        --log-mlflow
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import yaml

from src.learner.progress_model import resolve_serving_config
from src.track_b import tracking
from src.track_b.modeling import (
    apply_difficulty_boundary_guard,
    file_sha256,
    load_dataset,
    split_by_learner,
)
from src.track_b.promotion_gate import gate_integrity
from src.track_b.schema import (
    FEATURE_COLUMNS,
    SchemaError,
    TARGET,
    check_feature_list,
    validate_dataset,
)

DEFAULT_GATE_V2 = Path("configs/track_b/promotion_gate_v2.yaml")
DEFAULT_EXPERIMENT_CFG = Path("configs/track_b/drift_experiment.yaml")
DEFAULT_COMPARISON_OUT = Path("reports/track_b/model_comparison_v2.json")
DEFAULT_DECISION_OUT = Path("reports/track_b/promotion_decision_v2.json")

SYNTHETIC_LABEL = (
    "Comparison and gate v2 decision on SYNTHETIC data only; not validated for real students."
)


class LoadedModelSpec:
    """Container for a loaded sklearn model + metadata needed for comparison."""

    def __init__(
        self,
        *,
        model: Any,
        model_id: str,
        model_version: str,
        threshold: float,
        apply_boundary_guard: bool,
        model_size_mb: float,
        card: dict[str, Any],
        source: str,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.model_version = model_version
        self.threshold = float(threshold)
        self.apply_boundary_guard = bool(apply_boundary_guard)
        self.model_size_mb = round(float(model_size_mb), 4)
        self.card = card
        self.source = source


def _estimate_model_size_mb(model: Any, model_file: Path | None = None) -> float:
    if model_file is not None and model_file.is_file():
        return model_file.stat().st_size / (1024 * 1024)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell() / (1024 * 1024)


def load_model_from_dir(model_dir: str | Path, *, version_label: str = "local") -> LoadedModelSpec:
    path = Path(model_dir)
    model_file = path / "model.joblib"
    card_file = path / "model_card.json"
    model = joblib.load(model_file)
    card = json.loads(card_file.read_text(encoding="utf-8")) if card_file.is_file() else {}
    threshold, apply_guard = resolve_serving_config(card=card)
    return LoadedModelSpec(
        model=model,
        model_id=str(card.get("model_id", path.name)),
        model_version=str(card.get("model_version", version_label)),
        threshold=threshold,
        apply_boundary_guard=apply_guard,
        model_size_mb=_estimate_model_size_mb(model, model_file),
        card=card,
        source=str(model_file),
    )


def load_model_from_mlflow(
    *,
    model_name: str = tracking.REGISTERED_MODEL_NAME,
    alias: str,
    tracking_uri: str | None = None,
) -> LoadedModelSpec:
    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or tracking.configure()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    mv = client.get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

    run = client.get_run(mv.run_id)
    params = run.data.params or {}
    mv_tags = dict(mv.tags or {})
    tags = {**(run.data.tags or {}), **mv_tags}
    card: dict[str, Any] = {
        "model_id": tags.get("model_id") or params.get("model_id") or run.info.run_name or f"{model_name}_v{mv.version}",
        "feature_columns": list(getattr(model, "feature_names_in_", FEATURE_COLUMNS)),
        "dataset_sha256": tags.get("dataset_sha256", ""),
        "config_sha256": tags.get("config_sha256", ""),
        "git_commit": tags.get("git_commit", ""),
        "seed": params.get("seed") or tags.get("seed") or "",
        "trigger_id": tags.get("trigger_id", ""),
        "trained_at_utc": dt.datetime.fromtimestamp(
            run.info.start_time / 1000.0, tz=dt.timezone.utc
        ).isoformat(timespec="seconds"),
        "test_learners_used_for_training": [],
    }
    threshold, apply_guard = resolve_serving_config(params=params, tags=mv_tags)
    return LoadedModelSpec(
        model=model,
        model_id=str(card["model_id"]),
        model_version=str(mv.version),
        threshold=threshold,
        apply_boundary_guard=apply_guard,
        model_size_mb=_estimate_model_size_mb(model),
        card=card,
        source=f"models:/{model_name}@{alias}",
    )


def load_live_window_from_db(
    db_path: str | Path,
    *,
    window_hours: float | None = None,
    source_filter: str | None = None,
) -> pd.DataFrame:
    """Load labelled predictions (label_status='joined') from monitoring.sqlite3 into a DataFrame."""
    path = Path(db_path)
    if not path.is_file():
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET, "prediction_id", "learner_key", "source"])

    clauses = ["label_status = 'joined'", "label_observed IS NOT NULL"]
    params: list[Any] = []
    if window_hours is not None and window_hours > 0:
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        clauses.append("occurred_at >= ?")
        params.append(cutoff)
    if source_filter:
        clauses.append("source = ?")
        params.append(source_filter)

    query = (
        "SELECT prediction_id, learner_key, source, features, label_observed "
        f"FROM predictions WHERE {' AND '.join(clauses)} ORDER BY occurred_at, prediction_id"
    )
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for pred_id, learner_key, source, feats_json, label_obs in rows:
        try:
            feats = json.loads(feats_json)
        except (TypeError, ValueError):
            continue
        if not isinstance(feats, dict) or not set(FEATURE_COLUMNS).issubset(feats):
            continue
        records.append({
            **{col: feats[col] for col in FEATURE_COLUMNS},
            TARGET: int(label_obs),
            "prediction_id": pred_id,
            "learner_key": learner_key,
            "source": source,
        })
    return pd.DataFrame(records, columns=[*FEATURE_COLUMNS, TARGET, "prediction_id", "learner_key", "source"])


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint of the exact feature+target rows evaluated."""
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = [c for c in [*FEATURE_COLUMNS, TARGET] if c in df.columns]
    payload = df[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _measure_single_row_latency(spec: LoadedModelSpec, df: pd.DataFrame, n_samples: int = 30) -> dict[str, float | None]:
    if df.empty:
        return {"p50_ms": None, "p95_ms": None, "n_timed": 0}
    sample_df = df[FEATURE_COLUMNS].iloc[: min(n_samples, len(df))]
    times_ms: list[float] = []
    for i in range(len(sample_df)):
        row_df = sample_df.iloc[i : i + 1]
        t0 = time.perf_counter()
        proba = spec.model.predict_proba(row_df)[:, 1]
        if spec.apply_boundary_guard and "selected_difficulty" in row_df.columns:
            apply_difficulty_boundary_guard(proba, row_df["selected_difficulty"])
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms, dtype=float)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "n_timed": int(len(times_ms)),
    }


def score_model_on_dataset(spec: LoadedModelSpec, df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one model on a DataFrame with FEATURE_COLUMNS and TARGET."""
    if df.empty:
        return {
            "n_rows": 0,
            "positive_rate": None,
            "decision_threshold": spec.threshold,
            "boundary_guard_applied": spec.apply_boundary_guard,
            "row_fingerprint": _dataset_fingerprint(df),
            "metrics": None,
        }

    X = df[FEATURE_COLUMNS]
    y = df[TARGET].to_numpy(dtype=int)
    proba = spec.model.predict_proba(X)[:, 1]
    if spec.apply_boundary_guard and "selected_difficulty" in df.columns:
        proba = apply_difficulty_boundary_guard(proba, df["selected_difficulty"])
    proba = np.clip(proba, 0.0, 1.0)
    pred = (proba >= spec.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both_classes = len(set(y.tolist())) == 2

    return {
        "n_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "decision_threshold": round(float(spec.threshold), 4),
        "boundary_guard_applied": bool(spec.apply_boundary_guard),
        "row_fingerprint": _dataset_fingerprint(df),
        "has_both_classes": has_both_classes,
        "metrics": {
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4) if has_both_classes else None,
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(y, proba)), 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }


def compare_models(
    champion: LoadedModelSpec,
    challenger: LoadedModelSpec,
    *,
    original_test: pd.DataFrame,
    shifted_test: pd.DataFrame,
    live_window: pd.DataFrame | None = None,
    challenger_original_test: pd.DataFrame | None = None,
    challenger_shifted_test: pd.DataFrame | None = None,
    challenger_live_window: pd.DataFrame | None = None,
    dataset_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score champion and challenger on identical rows across all three evaluation datasets."""
    live_df = live_window if live_window is not None else pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET])

    datasets_map = {
        "original_test": (original_test, challenger_original_test if challenger_original_test is not None else original_test),
        "shifted_test": (shifted_test, challenger_shifted_test if challenger_shifted_test is not None else shifted_test),
        "live_window": (live_df, challenger_live_window if challenger_live_window is not None else live_df),
    }
    eval_results: dict[str, Any] = {}
    identical_rows_ok = True

    for name, (champ_df, chall_df) in datasets_map.items():
        champ_res = score_model_on_dataset(champion, champ_df)
        chall_res = score_model_on_dataset(challenger, chall_df)
        rows_match = (
            champ_res["n_rows"] == chall_res["n_rows"]
            and champ_res["row_fingerprint"] == chall_res["row_fingerprint"]
        )
        if not rows_match:
            identical_rows_ok = False
        delta: dict[str, float | None] | None = None
        if champ_res["metrics"] and chall_res["metrics"]:
            delta = {}
            for k in ("f1", "roc_auc", "precision", "recall", "brier"):
                cv = chall_res["metrics"][k]
                bv = champ_res["metrics"][k]
                delta[k] = round(cv - bv, 4) if (cv is not None and bv is not None) else None
        eval_results[name] = {
            "n_rows": int(len(champ_df)),
            "row_fingerprint": champ_res["row_fingerprint"],
            "identical_rows": rows_match,
            "champion": champ_res,
            "challenger": chall_res,
            "challenger_minus_champion": delta,
        }

    champ_lat = _measure_single_row_latency(champion, shifted_test)
    chall_lat = _measure_single_row_latency(challenger, shifted_test)

    return {
        "label": SYNTHETIC_LABEL,
        "protocol": (
            "Both champion and challenger scored on identical rows across original_test, "
            "shifted_test, and recent labelled live_window."
        ),
        "compared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "identical_evaluation_rows": identical_rows_ok,
        "dataset_hashes": dataset_hashes or {},
        "champion": {
            "model_id": champion.model_id,
            "model_version": champion.model_version,
            "source": champion.source,
            "threshold": champion.threshold,
            "model_size_mb": champion.model_size_mb,
            "latency": champ_lat,
        },
        "challenger": {
            "model_id": challenger.model_id,
            "model_version": challenger.model_version,
            "source": challenger.source,
            "threshold": challenger.threshold,
            "model_size_mb": challenger.model_size_mb,
            "latency": chall_lat,
        },
        "datasets": eval_results,
    }


def evaluate_gate_v2(
    gate: dict[str, Any],
    comparison: dict[str, Any],
    checks: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Evaluate Promotion Gate v2 criteria and return (decision, criteria_results).

    Each criterion result has ``status`` in ``{"pass", "fail", "inconclusive"}`` and
    boolean ``passed`` (True only for ``"pass"``).
    """
    c = gate["criteria"]
    datasets = comparison["datasets"]
    results: list[dict[str, Any]] = []

    def _add(criterion: str, value: Any, threshold: Any, status: str, note: str = "") -> None:
        entry = {
            "criterion": criterion,
            "value": value,
            "threshold": threshold,
            "status": status,
            "passed": status == "pass",
        }
        if note:
            entry["note"] = note
        results.append(entry)

    # 1. Minimum predictive quality on shifted_test
    q = c["minimum_predictive_quality"]
    shifted_chall = datasets[q["dataset"]]["challenger"]["metrics"]
    q_val = shifted_chall[q["metric"]] if shifted_chall else None
    _add(
        "minimum_predictive_quality",
        q_val,
        q["minimum"],
        "pass" if (q_val is not None and q_val >= q["minimum"]) else "fail",
    )

    # 2. Improvement over champion on shifted_test
    imp = c["improvement_over_champion"]
    shifted_champ = datasets[imp["dataset"]]["champion"]["metrics"]
    if shifted_chall and shifted_champ:
        gain = round(shifted_chall[imp["metric"]] - shifted_champ[imp["metric"]], 4)
        _add("improvement_over_champion", gain, imp["minimum_gain"], "pass" if gain >= imp["minimum_gain"] else "fail")
    else:
        _add("improvement_over_champion", None, imp["minimum_gain"], "fail", "missing shifted_test metrics")

    # 3. Regression tolerance on original_test
    reg = c["regression_tolerance"]
    orig_champ = datasets[reg["dataset"]]["champion"]["metrics"]
    orig_chall = datasets[reg["dataset"]]["challenger"]["metrics"]
    if orig_champ and orig_chall:
        f1_drop = round(orig_champ["f1"] - orig_chall["f1"], 4)
        _add("regression_tolerance_f1", f1_drop, reg["max_f1_drop"], "pass" if f1_drop <= reg["max_f1_drop"] else "fail")
        if orig_champ["roc_auc"] is not None and orig_chall["roc_auc"] is not None:
            auc_drop = round(orig_champ["roc_auc"] - orig_chall["roc_auc"], 4)
            _add(
                "regression_tolerance_roc_auc",
                auc_drop,
                reg["max_roc_auc_drop"],
                "pass" if auc_drop <= reg["max_roc_auc_drop"] else "fail",
            )
        else:
            _add("regression_tolerance_roc_auc", None, reg["max_roc_auc_drop"], "fail", "missing roc_auc")
    else:
        _add("regression_tolerance_f1", None, reg["max_f1_drop"], "fail")
        _add("regression_tolerance_roc_auc", None, reg["max_roc_auc_drop"], "fail")

    # 4. Calibration tolerance (Brier score, lower is better)
    cal = c["calibration_tolerance"]
    cal_champ = datasets[cal["dataset"]]["champion"]["metrics"]
    cal_chall = datasets[cal["dataset"]]["challenger"]["metrics"]
    if cal_champ and cal_chall:
        brier_inc = round(cal_chall["brier"] - cal_champ["brier"], 4)
        _add(
            "calibration_tolerance_brier",
            brier_inc,
            cal["max_brier_increase"],
            "pass" if brier_inc <= cal["max_brier_increase"] else "fail",
        )
    else:
        _add("calibration_tolerance_brier", None, cal["max_brier_increase"], "fail")

    # 5. Live window quality (supports explicit INCONCLUSIVE when sparse/single-class)
    lw = c["live_window_quality"]
    live_block = datasets[lw["dataset"]]
    live_n = live_block["n_rows"]
    live_champ = live_block["champion"]
    live_chall = live_block["challenger"]
    if live_n < lw["min_labelled_rows"] or not live_chall.get("has_both_classes", False):
        _add(
            "live_window_quality",
            {"n_rows": live_n, "has_both_classes": live_chall.get("has_both_classes", False)},
            {"min_labelled_rows": lw["min_labelled_rows"], "max_f1_drop": lw["max_f1_drop"]},
            "inconclusive",
            f"live_window has {live_n} labelled rows (< {lw['min_labelled_rows']}) or single class",
        )
    else:
        live_f1_drop = round(live_champ["metrics"]["f1"] - live_chall["metrics"]["f1"], 4)
        _add(
            "live_window_quality",
            live_f1_drop,
            lw["max_f1_drop"],
            "pass" if live_f1_drop <= lw["max_f1_drop"] else "fail",
        )

    # 6. Operational constraints (latency & model size)
    ops = c["operational_constraints"]
    p95 = comparison["challenger"]["latency"].get("p95_ms")
    _add(
        "operational.latency_p95_ms",
        p95,
        ops["max_p95_latency_ms"],
        "pass" if (p95 is not None and p95 <= ops["max_p95_latency_ms"]) else "fail",
    )
    size_mb = comparison["challenger"]["model_size_mb"]
    _add(
        "operational.model_size_mb",
        size_mb,
        ops["max_model_size_mb"],
        "pass" if size_mb <= ops["max_model_size_mb"] else "fail",
    )

    # 7. Data validity (verify identical rows directly from per-dataset fingerprints, not just a flag)
    per_dataset_match = all(
        ds_block["champion"].get("n_rows") == ds_block["challenger"].get("n_rows")
        and ds_block["champion"].get("row_fingerprint") == ds_block["challenger"].get("row_fingerprint")
        and bool(ds_block["champion"].get("row_fingerprint"))
        for ds_block in datasets.values()
    )
    identical_ok = bool(checks.get("identical_evaluation_rows", True)) and per_dataset_match

    for name in ("schema_validation", "leakage_check", "learner_level_split"):
        ok = bool(checks.get(name, False))
        _add(f"data_validity.{name}", ok, "required", "pass" if ok else "fail")
    _add(
        "data_validity.identical_evaluation_rows",
        identical_ok,
        "required",
        "pass" if identical_ok else "fail",
        "" if identical_ok else "champion and challenger row fingerprints or counts differ",
    )

    # 8. Reproducibility fields
    present = checks.get("reproducibility_fields_present", {})
    _add(
        "reproducibility",
        present,
        c["reproducibility"]["required_fields"],
        "pass" if (present and all(present.values())) else "fail",
    )

    # 9. Gate integrity (thresholds must be committed unmodified BEFORE challenger was trained)
    integrity = checks.get("gate_integrity") or {}
    integrity_ok = bool(integrity.get("gate_file_unmodified")) and bool(
        integrity.get("committed_before_candidate_trained")
    )
    _add(
        "gate_integrity",
        {
            "gate_file_unmodified": bool(integrity.get("gate_file_unmodified", False)),
            "committed_before_candidate_trained": bool(integrity.get("committed_before_candidate_trained", False)),
        },
        "gate committed unmodified before challenger trained",
        "pass" if integrity_ok else "fail",
    )

    has_fail = any(r["status"] == "fail" for r in results)
    has_inconclusive = any(r["status"] == "inconclusive" for r in results)
    if has_fail:
        decision = "REJECT_CHALLENGER_RETAIN_CHAMPION"
    elif has_inconclusive:
        decision = "INCONCLUSIVE"
    else:
        decision = "PROMOTE_CHALLENGER_PENDING_APPROVAL"
    return decision, results


def log_comparison_to_mlflow(
    comparison: dict[str, Any],
    decision_doc: dict[str, Any],
    comparison_path: Path,
    decision_path: Path,
    *,
    tracking_uri: str | None = None,
) -> str:
    """Log comparison metrics, gate v2 decision, and JSON reports to MLflow; return run_id."""
    import mlflow

    uri = tracking_uri or tracking.configure()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(tracking.EXPERIMENT_NAME)

    run_name = (
        f"compare_{comparison['challenger']['model_id']}_vs_{comparison['champion']['model_id']}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "gate_id": decision_doc["gate_id"],
            "champion_model_id": comparison["champion"]["model_id"],
            "champion_version": comparison["champion"]["model_version"],
            "challenger_model_id": comparison["challenger"]["model_id"],
            "challenger_version": comparison["challenger"]["model_version"],
            "identical_evaluation_rows": comparison["identical_evaluation_rows"],
        })
        metrics: dict[str, float] = {
            "challenger_model_size_mb": float(comparison["challenger"]["model_size_mb"]),
            "champion_model_size_mb": float(comparison["champion"]["model_size_mb"]),
        }
        if comparison["challenger"]["latency"].get("p95_ms") is not None:
            metrics["challenger_latency_p95_ms"] = float(comparison["challenger"]["latency"]["p95_ms"])
        if comparison["champion"]["latency"].get("p95_ms") is not None:
            metrics["champion_latency_p95_ms"] = float(comparison["champion"]["latency"]["p95_ms"])

        for ds_name, ds_block in comparison["datasets"].items():
            for role in ("champion", "challenger"):
                m = ds_block[role].get("metrics")
                if not m:
                    continue
                for k in ("f1", "roc_auc", "precision", "recall", "brier"):
                    if m.get(k) is not None:
                        metrics[f"{ds_name}_{role}_{k}"] = float(m[k])
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "role": "model_comparison",
            "gate_id": decision_doc["gate_id"],
            "gate_decision": decision_doc["decision"],
            "failed_criteria": ",".join(decision_doc["failed_criteria"]) or "none",
            "inconclusive_criteria": ",".join(decision_doc["inconclusive_criteria"]) or "none",
            "data_kind": "SYNTHETIC - not validated for real students",
        })
        if comparison_path.is_file():
            mlflow.log_artifact(str(comparison_path))
        if decision_path.is_file():
            mlflow.log_artifact(str(decision_path))
        return run.info.run_id


def run_comparison_pipeline(
    *,
    champion: LoadedModelSpec,
    challenger: LoadedModelSpec,
    gate_path: Path = DEFAULT_GATE_V2,
    experiment_cfg_path: Path = DEFAULT_EXPERIMENT_CFG,
    monitoring_db: Path | None = None,
    live_window_hours: float | None = None,
    comparison_out: Path = DEFAULT_COMPARISON_OUT,
    decision_out: Path = DEFAULT_DECISION_OUT,
    log_mlflow: bool = False,
    tracking_uri: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run full 3-dataset comparison + Gate v2 evaluation and write JSON reports."""
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(experiment_cfg_path.read_text(encoding="utf-8"))

    orig_df, orig_meta = load_dataset(cfg["datasets"]["baseline"])
    shifted_df, shifted_meta = load_dataset(cfg["datasets"]["shifted"])
    _, orig_test = split_by_learner(orig_df, orig_meta)
    shifted_train, shifted_test = split_by_learner(shifted_df, shifted_meta)

    live_df = load_live_window_from_db(monitoring_db, window_hours=live_window_hours) if monitoring_db else pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET])

    dataset_hashes = {
        "original": file_sha256(cfg["datasets"]["baseline"]),
        "shifted": file_sha256(cfg["datasets"]["shifted"]),
    }

    comparison = compare_models(
        champion,
        challenger,
        original_test=orig_test,
        shifted_test=shifted_test,
        live_window=live_df,
        dataset_hashes=dataset_hashes,
    )

    schema_ok = not validate_dataset(orig_df) and not validate_dataset(shifted_df)
    try:
        check_feature_list(list(challenger.card.get("feature_columns") or FEATURE_COLUMNS))
        leakage_ok = True
    except SchemaError:
        leakage_ok = False
    split_ok = (
        not (set(shifted_train["learner_id"]) & set(shifted_test["learner_id"]))
        and not challenger.card.get("test_learners_used_for_training")
    )
    trained_at = challenger.card.get("trained_at_utc") or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    integrity = gate_integrity(gate_path, trained_at)

    req_fields = gate["criteria"]["reproducibility"]["required_fields"]
    checks = {
        "schema_validation": schema_ok,
        "leakage_check": leakage_ok,
        "learner_level_split": split_ok,
        "identical_evaluation_rows": comparison["identical_evaluation_rows"],
        "reproducibility_fields_present": {f: bool(challenger.card.get(f)) for f in req_fields},
        "gate_integrity": integrity,
    }

    decision_str, criteria_results = evaluate_gate_v2(gate, comparison, checks)

    decision_doc: dict[str, Any] = {
        "gate_id": gate["gate_id"],
        "decision": decision_str,
        "human_approval_required": bool(gate.get("governance", {}).get("human_approval_required", True)),
        "champion": {
            "model_id": champion.model_id,
            "model_version": champion.model_version,
        },
        "challenger": {
            "model_id": challenger.model_id,
            "model_version": challenger.model_version,
        },
        "criteria_results": criteria_results,
        "failed_criteria": [r["criterion"] for r in criteria_results if r["status"] == "fail"],
        "inconclusive_criteria": [r["criterion"] for r in criteria_results if r["status"] == "inconclusive"],
        "gate_integrity": integrity,
        "versions": {
            "gate_config_sha256": file_sha256(gate_path),
            "experiment_config_sha256": file_sha256(experiment_cfg_path),
            "baseline_dataset_sha256": dataset_hashes["original"],
            "shifted_dataset_sha256": dataset_hashes["shifted"],
            "challenger_git_commit": challenger.card.get("git_commit", ""),
            "challenger_seed": challenger.card.get("seed", ""),
        },
        "decided_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": SYNTHETIC_LABEL,
    }

    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    decision_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    decision_out.write_text(json.dumps(decision_doc, indent=2), encoding="utf-8")

    if log_mlflow:
        run_id = log_comparison_to_mlflow(
            comparison,
            decision_doc,
            comparison_out,
            decision_out,
            tracking_uri=tracking_uri,
        )
        decision_doc["mlflow_run_id"] = run_id
        decision_out.write_text(json.dumps(decision_doc, indent=2), encoding="utf-8")

    return comparison, decision_doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--champion-dir", type=Path, help="Local directory containing champion model.joblib + model_card.json")
    parser.add_argument("--challenger-dir", type=Path, help="Local directory containing challenger model.joblib + model_card.json")
    parser.add_argument("--champion-alias", default="production", help="MLflow alias for champion (used if --champion-dir omitted)")
    parser.add_argument("--challenger-alias", default="challenger", help="MLflow alias for challenger (used if --challenger-dir omitted)")
    parser.add_argument("--model-name", default=tracking.REGISTERED_MODEL_NAME)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE_V2)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CFG)
    parser.add_argument("--monitoring-db", type=Path, default=None, help="Path to monitoring.sqlite3 for live_window")
    parser.add_argument("--live-window-hours", type=float, default=None)
    parser.add_argument("--comparison-out", type=Path, default=DEFAULT_COMPARISON_OUT)
    parser.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION_OUT)
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args(argv)

    champion = (
        load_model_from_dir(args.champion_dir, version_label="champion")
        if args.champion_dir
        else load_model_from_mlflow(model_name=args.model_name, alias=args.champion_alias, tracking_uri=args.tracking_uri)
    )
    challenger = (
        load_model_from_dir(args.challenger_dir, version_label="challenger")
        if args.challenger_dir
        else load_model_from_mlflow(model_name=args.model_name, alias=args.challenger_alias, tracking_uri=args.tracking_uri)
    )

    _, decision = run_comparison_pipeline(
        champion=champion,
        challenger=challenger,
        gate_path=args.gate,
        experiment_cfg_path=args.config,
        monitoring_db=args.monitoring_db,
        live_window_hours=args.live_window_hours,
        comparison_out=args.comparison_out,
        decision_out=args.decision_out,
        log_mlflow=args.log_mlflow,
        tracking_uri=args.tracking_uri,
    )
    for r in decision["criteria_results"]:
        print(f"{r['status'].upper():12} {r['criterion']:34} value={r['value']} threshold={r['threshold']}")
    print("DECISION:", decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
