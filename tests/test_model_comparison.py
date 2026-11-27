"""Tests for Person 3 Model Comparison, Promotion Gate v2, and approval-gated registry ops."""
from __future__ import annotations

import json
import uuid

import joblib
import pandas as pd
import pytest
import yaml

from src.learner.progress_model import ProgressModel
from src.monitoring.label_joiner import join_labels
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.track_b.model_comparison import (
    evaluate_gate_v2,
    load_model_from_dir,
    run_comparison_pipeline,
)
from src.track_b.modeling import file_sha256
from src.track_b.registry_ops import (
    GovernanceApprovalError,
    promote_challenger,
    reject_challenger,
    rollback_production,
)

with open("configs/track_b/promotion_gate_v2.yaml", encoding="utf-8") as _fh:
    GATE_V2 = yaml.safe_load(_fh)

OK_CHECKS = {
    "schema_validation": True,
    "leakage_check": True,
    "learner_level_split": True,
    "identical_evaluation_rows": True,
    "reproducibility_fields_present": {
        f: True for f in GATE_V2["criteria"]["reproducibility"]["required_fields"]
    },
    "gate_integrity": {
        "gate_file_unmodified": True,
        "committed_before_candidate_trained": True,
    },
}


def _mock_comparison(
    *,
    chall_shifted_f1=0.70,
    champ_shifted_f1=0.65,
    chall_orig_f1=0.69,
    champ_orig_f1=0.70,
    chall_orig_auc=0.92,
    champ_orig_auc=0.93,
    chall_brier=0.14,
    champ_brier=0.15,
    live_n=60,
    live_both_classes=True,
    chall_live_f1=0.67,
    champ_live_f1=0.66,
    p95_ms=5.0,
    size_mb=1.2,
):
    def _ds(n, champ_m, chall_m, both=True):
        return {
            "n_rows": n,
            "row_fingerprint": "abc",
            "champion": {"n_rows": n, "row_fingerprint": "abc", "has_both_classes": both, "metrics": champ_m if n else None},
            "challenger": {"n_rows": n, "row_fingerprint": "abc", "has_both_classes": both, "metrics": chall_m if n else None},
        }

    return {
        "identical_evaluation_rows": True,
        "champion": {"model_id": "champ", "model_version": "1", "model_size_mb": 1.0, "latency": {"p95_ms": 4.0}},
        "challenger": {"model_id": "chall", "model_version": "2", "model_size_mb": size_mb, "latency": {"p95_ms": p95_ms}},
        "datasets": {
            "original_test": _ds(
                1000,
                {"f1": champ_orig_f1, "roc_auc": champ_orig_auc, "brier": 0.14},
                {"f1": chall_orig_f1, "roc_auc": chall_orig_auc, "brier": 0.14},
            ),
            "shifted_test": _ds(
                1000,
                {"f1": champ_shifted_f1, "roc_auc": 0.90, "brier": champ_brier},
                {"f1": chall_shifted_f1, "roc_auc": 0.92, "brier": chall_brier},
            ),
            "live_window": _ds(
                live_n,
                {"f1": champ_live_f1, "roc_auc": 0.85, "brier": 0.16},
                {"f1": chall_live_f1, "roc_auc": 0.86, "brier": 0.16},
                both=live_both_classes,
            ),
        },
    }


# ---- Gate v2 unit tests ----------------------------------------------------------------------

def test_gate_v2_promotes_pending_approval_when_all_criteria_pass():
    decision, results = evaluate_gate_v2(GATE_V2, _mock_comparison(), OK_CHECKS)
    assert decision == "PROMOTE_CHALLENGER_PENDING_APPROVAL"
    assert all(r["status"] == "pass" for r in results)


def test_gate_v2_returns_inconclusive_when_live_window_is_sparse():
    decision, results = evaluate_gate_v2(GATE_V2, _mock_comparison(live_n=12), OK_CHECKS)
    assert decision == "INCONCLUSIVE"
    inconclusive = [r["criterion"] for r in results if r["status"] == "inconclusive"]
    assert inconclusive == ["live_window_quality"]


@pytest.mark.parametrize(
    "kwargs, failed_criterion",
    [
        ({"chall_shifted_f1": 0.50, "champ_shifted_f1": 0.40}, "minimum_predictive_quality"),
        ({"chall_shifted_f1": 0.655, "champ_shifted_f1": 0.65}, "improvement_over_champion"),
        ({"chall_orig_f1": 0.60, "champ_orig_f1": 0.70}, "regression_tolerance_f1"),
        ({"chall_orig_auc": 0.88, "champ_orig_auc": 0.93}, "regression_tolerance_roc_auc"),
        ({"chall_brier": 0.20, "champ_brier": 0.15}, "calibration_tolerance_brier"),
        ({"chall_live_f1": 0.60, "champ_live_f1": 0.66}, "live_window_quality"),
        ({"p95_ms": 75.0}, "operational.latency_p95_ms"),
        ({"size_mb": 30.0}, "operational.model_size_mb"),
    ],
)
def test_gate_v2_rejects_each_failing_criterion(kwargs, failed_criterion):
    decision, results = evaluate_gate_v2(GATE_V2, _mock_comparison(**kwargs), OK_CHECKS)
    assert decision == "REJECT_CHALLENGER_RETAIN_CHAMPION"
    assert failed_criterion in [r["criterion"] for r in results if r["status"] == "fail"]


def test_gate_v2_rejects_even_when_live_window_is_sparse_if_hard_criterion_fails():
    decision, _ = evaluate_gate_v2(GATE_V2, _mock_comparison(live_n=0, chall_brier=0.25), OK_CHECKS)
    assert decision == "REJECT_CHALLENGER_RETAIN_CHAMPION"


def test_gate_v2_rejects_mismatched_rows_and_broken_gate_integrity():
    # Review #6a: mismatched row fingerprints between champion and challenger MUST fail the gate.
    comp = _mock_comparison()
    comp["datasets"]["shifted_test"]["challenger"]["row_fingerprint"] = "different-sha256"
    decision, results = evaluate_gate_v2(GATE_V2, comp, OK_CHECKS)
    assert decision == "REJECT_CHALLENGER_RETAIN_CHAMPION"
    assert "data_validity.identical_evaluation_rows" in [r["criterion"] for r in results if r["status"] == "fail"]

    # Review #6b: gate committed AFTER challenger was trained (or modified) MUST fail the gate.
    bad_integrity = {**OK_CHECKS, "gate_integrity": {"gate_file_unmodified": True, "committed_before_candidate_trained": False}}
    decision2, results2 = evaluate_gate_v2(GATE_V2, _mock_comparison(), bad_integrity)
    assert decision2 == "REJECT_CHALLENGER_RETAIN_CHAMPION"
    assert "gate_integrity" in [r["criterion"] for r in results2 if r["status"] == "fail"]


def test_serving_config_parity_between_app_and_gate():
    # Review #7: missing threshold / guard defaults must match ProgressModel (0.7, True).
    from src.learner.progress_model import DEFAULT_APPLY_BOUNDARY_GUARD, DEFAULT_THRESHOLD, resolve_serving_config

    thr, guard = resolve_serving_config()
    assert thr == DEFAULT_THRESHOLD == 0.7
    assert guard is DEFAULT_APPLY_BOUNDARY_GUARD is True


# ---- End-to-end 3-dataset comparison + MLflow + approval-gated promotion & rollback ----------

def _populate_live_window_db(db_path, n_learners: int = 12):
    """Populate a monitoring.sqlite3 DB with real Track B shifted rows and join delayed labels."""
    df = pd.read_csv("data/synthetic/learner_interactions_shifted_v1.csv")
    logger = PredictionLogger(db_path, "test-pseudonym-key-0123456789")
    learners = list(df["learner_id"].unique()[:n_learners])
    for lid in learners:
        group = df[df["learner_id"] == lid].head(8)
        for seq, (_, row) in enumerate(group.iterrows()):
            feats = {
                "topic": str(row["topic"]),
                "selected_difficulty": str(row["selected_difficulty"]),
                "recent_accuracy": float(row["recent_accuracy"]),
                "mistake_count": int(row["mistake_count"]),
                "hint_count": int(row["hint_count"]),
                "response_time_seconds": float(row["response_time_seconds"]),
                "prior_interaction_count": int(seq),
                "previous_difficulty": str(row["previous_difficulty"]),
            }
            logger.log(
                PredictionRecord(
                    request_id=uuid.uuid4().hex,
                    source="simulated",
                    learner_id=str(lid),
                    learner_seq=seq,
                    question_id=f"q-{seq}",
                    features=feats,
                    score=0.5,
                    label=0,
                    threshold=0.7,
                    model_name="learner-difficulty",
                    model_version="1",
                    latency_ms=2.0,
                    status="ok",
                )
            )
    res = join_labels(db_path)
    assert res.joined >= 60


def test_end_to_end_comparison_gate_v2_promotion_and_rollback(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from src.track_b.register_mlflow import TRUSTED_TYPES

    db_path = tmp_path / "monitoring.sqlite3"
    _populate_live_window_db(db_path)

    mlflow_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("track_b_learner_difficulty")
    client = MlflowClient(tracking_uri=mlflow_uri)

    # Register v1 (baseline) as current production and v2 (candidate HGB) as challenger.
    base_model = joblib.load("models/track_b/baseline/model.joblib")
    cand_model = joblib.load("models/track_b/candidate/model.joblib")
    with mlflow.start_run(run_name="baseline_logreg_v1"):
        mlflow.log_params({"model_id": "baseline_logreg_v1", "decision_threshold": 0.5, "apply_boundary_guard": False})
        v1 = str(
            mlflow.sklearn.log_model(
                base_model, name="model", registered_model_name="learner-difficulty", skops_trusted_types=TRUSTED_TYPES
            ).registered_model_version
        )
    with mlflow.start_run(run_name="candidate_hgb_shifted_v2"):
        mlflow.log_params({"model_id": "candidate_hgb_shifted_v2", "decision_threshold": 0.7, "apply_boundary_guard": True})
        v2 = str(
            mlflow.sklearn.log_model(
                cand_model, name="model", registered_model_name="learner-difficulty", skops_trusted_types=TRUSTED_TYPES
            ).registered_model_version
        )
    client.set_registered_model_alias("learner-difficulty", "production", v1)
    client.set_registered_model_alias("learner-difficulty", "challenger", v2)

    champion = load_model_from_dir("models/track_b/baseline", version_label=v1)
    challenger = load_model_from_dir("models/track_b/candidate", version_label=v2)
    # Simulate a Phase 3 challenger trained AFTER promotion_gate_v2.yaml was committed.
    challenger.card["trained_at_utc"] = "2099-01-01T00:00:00+00:00"

    comp_out = tmp_path / "comparison_v2.json"
    dec_out = tmp_path / "decision_v2.json"
    comparison, decision = run_comparison_pipeline(
        champion=champion,
        challenger=challenger,
        monitoring_db=db_path,
        comparison_out=comp_out,
        decision_out=dec_out,
        log_mlflow=True,
        tracking_uri=mlflow_uri,
    )

    assert comparison["identical_evaluation_rows"] is True
    assert comparison["datasets"]["live_window"]["n_rows"] >= 60
    assert comparison["datasets"]["shifted_test"]["challenger"]["metrics"]["brier"] is not None
    assert decision["decision"] == "PROMOTE_CHALLENGER_PENDING_APPROVAL"
    assert decision.get("mlflow_run_id")

    # 1. Promotion without a valid human approval file MUST fail.
    with pytest.raises(GovernanceApprovalError):
        promote_challenger(
            decision_path=dec_out,
            approval_path=tmp_path / "missing_approval.json",
            tracking_uri=mlflow_uri,
            history_path=tmp_path / "history.json",
        )

    # 1b. Review #5: failed promote (e.g. non-existent version string "challenger") MUST NOT archive live champion.
    bad_dec = tmp_path / "bad_dec.json"
    bad_dec.write_text(json.dumps({**decision, "challenger": {"model_id": "x", "model_version": "challenger"}}), encoding="utf-8")
    bad_appr = tmp_path / "bad_appr.json"
    bad_appr.write_text(
        json.dumps(
            {
                "approval_id": "appr-bad-001",
                "approved_by": "person4",
                "approved_at": "2026-11-26T12:00:00Z",
                "challenger_version": "challenger",
                "rationale": "Testing atomic refusal",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceApprovalError, match="does not exist"):
        promote_challenger(
            decision_path=bad_dec,
            approval_path=bad_appr,
            tracking_uri=mlflow_uri,
            history_path=tmp_path / "history.json",
        )
    # Live champion v1 must still be production and NOT tagged archived.
    mv1 = client.get_model_version("learner-difficulty", v1)
    assert mv1.tags.get("status") != "archived"
    assert str(client.get_model_version_by_alias("learner-difficulty", "production").version) == v1

    # 2. Record human approval (Person 4) and promote v1 -> v2.
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "approval_id": "appr-test-001",
                "approved_by": "person4-governance-lead",
                "approved_at": "2026-11-26T12:00:00Z",
                "challenger_version": v2,
                "gate_decision_sha256": file_sha256(dec_out),
                "rationale": "All Gate v2 criteria passed on synthetic test sets and live window.",
            }
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "history.json"
    prom_event = promote_challenger(
        decision_path=dec_out,
        approval_path=approval_path,
        tracking_uri=mlflow_uri,
        history_path=history_path,
    )
    assert prom_event["from_version"] == v1
    assert prom_event["to_version"] == v2
    assert str(client.get_model_version_by_alias("learner-difficulty", "production").version) == v2
    assert str(client.get_model_version_by_alias("learner-difficulty", "archived").version) == v1

    # 3. Live app ProgressModel sees v2, then rollback moves alias back to v1 without rebuild!
    pm = ProgressModel(
        tracking_uri=mlflow_uri,
        model_name="learner-difficulty",
        registry_alias="production",
        allow_local_fallback=False,
        refresh_seconds=0.0,
    )
    sample = {
        "topic": "Chemical Kinetics",
        "selected_difficulty": "Hard",
        "recent_accuracy": 0.4,
        "mistake_count": 2,
        "hint_count": 1,
        "response_time_seconds": 60.0,
        "prior_interaction_count": 3,
        "previous_difficulty": "Hard",
    }
    assert pm.predict(sample).model_version == v2

    rb_event = rollback_production(
        reason="Verification of zero-rebuild rollback",
        operator="person3",
        tracking_uri=mlflow_uri,
        history_path=history_path,
    )
    assert rb_event["from_version"] == v2
    assert rb_event["to_version"] == v1
    assert pm.predict(sample).model_version == v1

    # 3b. Review minor #1: running rollback twice must refuse and NEVER tag live production as "rolled_back".
    with pytest.raises(GovernanceApprovalError, match="already the active production version"):
        rollback_production(
            reason="Duplicate rollback attempt",
            operator="person3",
            tracking_uri=mlflow_uri,
            history_path=history_path,
        )
    assert client.get_model_version("learner-difficulty", v1).tags.get("status") == "production"

    # 4. Reject flow sets rejected_candidate without touching production (still v1).
    rej_event = reject_challenger(
        decision_path=dec_out,
        tracking_uri=mlflow_uri,
        history_path=history_path,
    )
    assert rej_event["alias"] == "rejected_candidate"
    assert str(client.get_model_version_by_alias("learner-difficulty", "production").version) == v1
    assert str(client.get_model_version_by_alias("learner-difficulty", "rejected_candidate").version) == v2
