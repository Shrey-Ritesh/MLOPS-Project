"""Log the Track B baseline and candidate to MLflow and register the models.

Reads the saved artifacts (models, reports, model cards); nothing is retrained, so the
DVC pipeline outputs are unchanged. Safe to re-run: runs are matched on the model
file's hash and skipped if already logged.

    python -m src.track_b.register_mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # then open http://localhost:5001
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

from src.track_b import tracking
from src.track_b.modeling import file_sha256

REPORTS = Path("reports/track_b")
TRUSTED_TYPES = ["src.track_b.modeling._log1p", "src.track_b.modeling._ordinal"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_run(client: MlflowClient, experiment_id: str, model_sha: str) -> str | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.model_sha256 = '{model_sha}' and attributes.status = 'FINISHED'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _metrics(prefix: str, block: dict) -> dict:
    m = block["model"]
    return {f"{prefix}_{k}": float(m[k]) for k in ("precision", "recall", "f1", "roc_auc")}


def log_model_run(client, experiment_id, model_dir: Path, run_name: str, params: dict, metrics: dict,
                  tags: dict, artifacts: list[Path]) -> tuple[str, str]:
    """Log one run with its model; return (run_id, model_version)."""
    model_file = model_dir / "model.joblib"
    sha = file_sha256(model_file)
    existing = _existing_run(client, experiment_id, sha)
    if existing:
        versions = client.search_model_versions(f"run_id='{existing}'")
        return existing, versions[0].version if versions else ""

    card = _read(model_dir / "model_card.json")
    model = joblib.load(model_file)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model_sha256": sha,
            "dataset_sha256": card["dataset_sha256"],
            "config_sha256": card["config_sha256"],
            "git_commit": card.get("git_commit", ""),
            "data_kind": "SYNTHETIC - not validated for real students",
            "loading_note": "pickled sklearn pipeline; load with the repo on PYTHONPATH (src.track_b.modeling)",
            **tags,
        })
        for path in artifacts:
            mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(model_dir / "model_card.json"))
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=tracking.REGISTERED_MODEL_NAME,
            # MLflow's default (skops) format rejects unknown types; trust only our two own helpers.
            skops_trusted_types=TRUSTED_TYPES,
        )
    return run.info.run_id, str(info.registered_model_version)


def main() -> None:
    uri = tracking.configure()
    client = MlflowClient()
    experiment_id = mlflow.get_experiment_by_name(tracking.EXPERIMENT_NAME).experiment_id

    base_eval = _read(REPORTS / "baseline_evaluation.json")
    shifted = _read(REPORTS / "shifted_evaluation.json")
    comparison = _read(REPORTS / "candidate_comparison.json")
    decision = _read(REPORTS / "promotion_decision.json")
    base_card = _read(Path("models/track_b/baseline/model_card.json"))
    cand_card = _read(Path("models/track_b/candidate/model_card.json"))

    base_metrics = {
        **_metrics("original_test", base_eval["test"]),
        **_metrics("shifted_test", shifted["shifted_test"]),
    }
    base_run, base_version = log_model_run(
        client, experiment_id, Path("models/track_b/baseline"), "baseline_logreg_v1",
        {"model_type": "logistic_regression", "C": 1.0, "class_weight": "balanced",
         "decision_threshold": base_card["decision_threshold"], "dataset": base_card["dataset"]},
        base_metrics, {"role": "baseline"},
        [REPORTS / "baseline_evaluation.json", REPORTS / "shifted_evaluation.json", REPORTS / "drift_report.json"],
    )

    cand_metrics = {
        **_metrics("shifted_test", comparison["candidate"]["shifted_test"]),
        **_metrics("original_test", comparison["candidate"]["original_test"]),
    }
    cand_model_id = cand_card.get("model_id", comparison["candidate"]["model_id"])
    cand_params = {
        "model_id": cand_model_id,
        "decision_threshold": cand_card["decision_threshold"],
        "apply_boundary_guard": cand_card.get("apply_boundary_guard", True),
        "dataset": cand_card["dataset"],
    }
    cand_run, cand_version = log_model_run(
        client, experiment_id, Path("models/track_b/candidate"), cand_model_id,
        cand_params,
        cand_metrics,
        {"role": "candidate", "gate_decision": decision["decision"],
         "gate_failed_criteria": ",".join(decision["failed_criteria"]) or "none"},
        [REPORTS / "candidate_comparison.json", REPORTS / "promotion_decision.json"],
    )

    # Registry state follows the gate decision: the baseline stays "production"; a candidate
    # that failed the gate is tagged rejected and gets no production alias.
    name = tracking.REGISTERED_MODEL_NAME
    promoted = decision["decision"] == "PROMOTE_CANDIDATE"
    client.set_model_version_tag(name, base_version, "role", "baseline")
    client.set_model_version_tag(name, cand_version, "role", "candidate")
    client.set_model_version_tag(name, cand_version, "gate_decision", decision["decision"])
    client.set_registered_model_alias(name, "production", cand_version if promoted else base_version)
    if not promoted:
        client.set_registered_model_alias(name, "rejected_candidate", cand_version)

    print(f"tracking store: {uri}")
    print(f"baseline  run {base_run[:8]}  model {name} v{base_version}  -> alias 'production'" if not promoted else "")
    print(f"candidate run {cand_run[:8]}  model {name} v{cand_version}  gate: {decision['decision']}")


if __name__ == "__main__":
    main()
