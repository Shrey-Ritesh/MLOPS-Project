"""Register precomputed NCERT retrieval comparison results in local MLflow.

This registers supplied metrics and provenance. It does not rerun retrieval,
embedding inference, Chroma indexing, or relevance evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from mlflow import MlflowClient

from src.retrieval_mlflow import tracking

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/retrieval_mlflow.yaml"
METRICS_PATH = ROOT / "configs/retrieval_metrics_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_mlflow_metrics(metrics: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Normalize metric keys to MLflow-safe names and preserve the mapping."""
    normalized_metrics: dict[str, float] = {}
    name_mapping: dict[str, str] = {}

    for original_name, value in metrics.items():
        if not isinstance(original_name, str) or not original_name.strip():
            raise ValueError("Metric names must be non-empty strings.")

        # MLflow disallows '@'; make the cutoff explicit and use stable snake case.
        normalized_name = original_name.replace("@", "_at_").lower()
        if not re.fullmatch(r"[A-Za-z0-9_./ -]+", normalized_name):
            raise ValueError(
                f"Metric name {original_name!r} cannot be converted to a valid MLflow key."
            )
        if normalized_name in normalized_metrics:
            raise ValueError(
                f"Metric name collision after normalization: {original_name!r} "
                f"maps to {normalized_name!r}."
            )

        normalized_metrics[normalized_name] = float(value)
        name_mapping[original_name] = normalized_name

    return normalized_metrics, name_mapping


def checksum_file(path: Path, algorithm: str) -> str:
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}") from exc

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_tree(folder: Path, algorithm: str) -> tuple[dict[str, str], str]:
    file_hashes = {
        path.relative_to(folder).as_posix(): checksum_file(path, algorithm)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    tree_payload = json.dumps(
        file_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    try:
        tree_digest = hashlib.new(algorithm, tree_payload).hexdigest()
    except ValueError as exc:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}") from exc
    return file_hashes, tree_digest


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def pinned_revision(model_cfg: dict[str, Any]) -> str:
    revision = model_cfg.get("model_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision.lower()
    ):
        raise ValueError(
            f"{model_cfg['model_name']}: model_revision must be the exact "
            "40-character Hugging Face commit SHA. Branch names, tags and null "
            "are not accepted."
        )
    return revision


def validate_model_artifact_policy(policy: dict[str, Any]) -> None:
    """Fail early when the configured artifact policy is unsupported or unsafe."""
    if policy.get("source") != "huggingface_hub_snapshot":
        raise ValueError("model_artifacts.source must be 'huggingface_hub_snapshot'.")
    if policy.get("include_all_repository_files") is not True:
        raise ValueError(
            "include_all_repository_files must be true to preserve weights, "
            "tokenizer, configuration and remote model code."
        )
    if policy.get("checksum_scope") != "per_file_and_snapshot_tree":
        raise ValueError(
            "Only checksum_scope='per_file_and_snapshot_tree' is implemented."
        )
    if policy.get("download_directory") != "temporary":
        raise ValueError(
            "Only download_directory='temporary' is supported; model snapshots "
            "must not be written into the Git worktree."
        )
    artifact_path = policy.get("mlflow_artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError("model_artifacts.mlflow_artifact_path must be non-empty.")
    algorithm = policy.get("checksum_algorithm")
    if not isinstance(algorithm, str):
        raise ValueError("model_artifacts.checksum_algorithm must be a string.")
    try:
        hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported model_artifacts.checksum_algorithm: {algorithm}"
        ) from exc


def download_snapshot(
    model_cfg: dict[str, Any],
    destination: Path,
    artifact_policy: dict[str, Any],
) -> dict[str, Any]:
    """Download a pinned full HF repository snapshot and hash its model files."""
    revision = pinned_revision(model_cfg)
    info = HfApi().model_info(
        repo_id=model_cfg["model_name"],
        revision=revision,
    )
    if info.sha != revision:
        raise RuntimeError(
            f"Requested revision {revision}, Hugging Face resolved {info.sha}."
        )

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_cfg["model_name"],
            revision=revision,
            local_dir=str(destination),
        )
    )

    # Local-dir download bookkeeping is not part of the model snapshot.
    shutil.rmtree(snapshot_path / ".cache", ignore_errors=True)

    algorithm = artifact_policy["checksum_algorithm"]
    file_hashes, tree_digest = checksum_tree(snapshot_path, algorithm)
    return {
        "repo_id": model_cfg["model_name"],
        "revision": revision,
        "resolved_revision": info.sha,
        "checksum_algorithm": algorithm,
        "files_sha256": file_hashes,
        "snapshot_tree_sha256": tree_digest,
    }


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    metrics_doc = read_json(METRICS_PATH)
    artifact_policy = config.get("model_artifacts", {})
    validate_model_artifact_policy(artifact_policy)

    tracking_uri = tracking.configure(
        config["experiment"]["name"], config["tracking"]["default_uri"]
    )
    experiment = mlflow.get_experiment_by_name(config["experiment"]["name"])
    client = MlflowClient()
    git_commit = get_git_commit()

    eval_meta = {
        key: value for key, value in metrics_doc.items() if key != "models"
    }
    artifact_path = artifact_policy["mlflow_artifact_path"]
    checksum_algorithm = artifact_policy["checksum_algorithm"]

    for model_key, model_cfg in config["models"].items():
        source_paths = [
            CONFIG_PATH,
            METRICS_PATH,
            ROOT / model_cfg["config_path"],
            ROOT / model_cfg["index_manifest"],
            ROOT / model_cfg["retrieval_summary"],
            ROOT / model_cfg["retrieval_results"],
            ROOT / model_cfg["judged_benchmark"],
        ]
        missing = [str(path) for path in source_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required registration inputs:\n" + "\n".join(missing)
            )

        revision = pinned_revision(model_cfg)
        input_hashes = {
            path.relative_to(ROOT).as_posix(): checksum_file(
                path, checksum_algorithm
            )
            for path in source_paths
        }

        # Temporary snapshots are logged to MLflow and removed after each run.
        with tempfile.TemporaryDirectory(
            prefix=f"retrieval-{model_key}-"
        ) as temporary_directory:
            snapshot_dir = Path(temporary_directory) / "snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            model_manifest = download_snapshot(
                model_cfg, snapshot_dir, artifact_policy
            )
            mlflow_metrics, metric_name_mapping = prepare_mlflow_metrics(
                metrics_doc["models"][model_key]
            )
            fingerprint_payload = {
                "model": model_manifest,
                "model_config": model_cfg,
                "metrics": metrics_doc["models"][model_key],
                "mlflow_metric_name_mapping": metric_name_mapping,
                "inputs": input_hashes,
                "evaluation": eval_meta,
                "artifact_policy": artifact_policy,
            }
            fingerprint = hashlib.new(
                checksum_algorithm,
                json.dumps(
                    fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            ).hexdigest()

            existing = client.search_runs(
                [experiment.experiment_id],
                filter_string=(
                    f"tags.registration_fingerprint = '{fingerprint}'"
                ),
                max_results=1,
            )
            if existing:
                print(
                    f"{model_key}: unchanged; existing run "
                    f"{existing[0].info.run_id} (skipped)"
                )
                continue

            with mlflow.start_run(
                run_name=f"{model_key}_retrieval_v1"
            ) as run:
                params = {
                    "model_key": model_key,
                    "provider": model_cfg["provider"],
                    "model_name": model_cfg["model_name"],
                    "model_revision": revision,
                    "tokenizer_repository": (
                        model_cfg["model_name"]
                        if model_cfg["tokenizer"]["repository"] == "same_as_model"
                        else model_cfg["tokenizer"]["repository"]
                    ),
                    "tokenizer_revision": revision,
                    "tokenizer_loader": model_cfg["tokenizer"]["loader"],
                    "dimension": int(model_cfg["dimension"]),
                    "max_seq_length": int(model_cfg["max_seq_length"]),
                    "pooling": model_cfg["pooling"],
                    "normalize_embeddings": str(
                        bool(model_cfg["normalize_embeddings"])
                    ).lower(),
                    "collection": model_cfg["collection"],
                    "corpus": eval_meta["corpus"],
                    "query_count": int(eval_meta["query_count"]),
                    "metric_cutoff": int(eval_meta["metric_cutoff"]),
                    "checksum_algorithm": checksum_algorithm,
                    "model_artifact_source": artifact_policy["source"],
                    "model_artifact_path": artifact_policy["mlflow_artifact_path"],
                    "checksum_scope": artifact_policy["checksum_scope"],
                }
                if model_key == "chembed":
                    params.update({
                        "document_prefix": model_cfg["document_prefix"],
                        "query_prefix": model_cfg["query_prefix"],
                        "query_task": model_cfg["query_task"],
                        "trust_remote_code": str(
                            bool(model_cfg["trust_remote_code"])
                        ).lower(),
                        "use_safetensors": str(
                            bool(model_cfg["use_safetensors"])
                        ).lower(),
                    })

                mlflow.log_params(params)
                mlflow.log_metrics(mlflow_metrics)
                mlflow.set_tags({
                    "git_commit": git_commit,
                    "registration_fingerprint": fingerprint,
                    "benchmark_id": eval_meta["benchmark_id"],
                    "benchmark_version": eval_meta["benchmark_version"],
                    "corpus": eval_meta["corpus"],
                    "retrieval_mode": eval_meta["retrieval_mode"],
                    "query_composition": eval_meta["query_composition"],
                    "reranking": "false",
                    "recall_scope": "judged_pool_not_exhaustive_corpus",
                    "model_snapshot_tree_sha256": model_manifest[
                        "snapshot_tree_sha256"
                    ],
                    "model_revision": revision,
                    "revision_evidence": model_cfg.get(
                        "revision_evidence", {}
                    ).get("confidence", "not_recorded"),
                    "metric_source": "committed supplied evaluation results",
                })

                for source_path in source_paths:
                    mlflow.log_artifact(
                        str(source_path), artifact_path="evaluation_inputs"
                    )

                mlflow.log_dict(
                    model_cfg, "provenance/model_config.json"
                )
                mlflow.log_dict(
                    artifact_policy, "provenance/model_artifact_policy.json"
                )
                mlflow.log_dict(
                    eval_meta, "provenance/evaluation_protocol.json"
                )
                mlflow.log_dict(
                    metric_name_mapping,
                    "provenance/mlflow_metric_name_mapping.json",
                )
                mlflow.log_dict(
                    model_manifest,
                    "provenance/model_snapshot_manifest.json",
                )
                mlflow.log_artifacts(
                    str(snapshot_dir), artifact_path=artifact_path
                )

                print(f"{model_key}: logged run {run.info.run_id}")

    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {config['experiment']['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
