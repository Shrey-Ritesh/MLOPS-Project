import pytest


def test_model_revision_requires_immutable_sha():
    from src.retrieval_mlflow.register import pinned_revision

    with pytest.raises(ValueError):
        pinned_revision({"model_name": "example/model", "model_revision": "main"})
    with pytest.raises(ValueError):
        pinned_revision({"model_name": "example/model", "model_revision": None})


def test_model_revision_accepts_40_character_sha():
    from src.retrieval_mlflow.register import pinned_revision

    sha = "a" * 40
    assert pinned_revision({"model_name": "example/model", "model_revision": sha}) == sha


def test_tracking_uri_uses_environment_override(monkeypatch):
    pytest.importorskip("mlflow")
    from src.retrieval_mlflow import tracking

    captured = {}
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: captured.update(uri=uri))
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: captured.update(experiment=name))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    assert tracking.configure() == "http://localhost:5000"
    assert captured == {
        "uri": "http://localhost:5000",
        "experiment": tracking.EXPERIMENT_NAME,
    }

def test_model_artifact_policy_requires_temporary_full_snapshot_and_checksums():
    from src.retrieval_mlflow.register import validate_model_artifact_policy

    validate_model_artifact_policy({
        "source": "huggingface_hub_snapshot",
        "include_all_repository_files": True,
        "checksum_algorithm": "sha256",
        "checksum_scope": "per_file_and_snapshot_tree",
        "download_directory": "temporary",
        "mlflow_artifact_path": "embedding_model_snapshot",
    })

def test_mlflow_metric_names_are_normalized_and_original_mapping_preserved():
    from src.retrieval_mlflow.register import prepare_mlflow_metrics

    normalized, mapping = prepare_mlflow_metrics({
        "Hit@5": 1.0,
        "MRR@5": 0.95,
        "Precision@5": 0.86,
        "Recall@5": 0.38,
        "nDCG@5": 0.64,
    })

    assert normalized == {
        "hit_at_5": 1.0,
        "mrr_at_5": 0.95,
        "precision_at_5": 0.86,
        "recall_at_5": 0.38,
        "ndcg_at_5": 0.64,
    }

    assert mapping["Hit@5"] == "hit_at_5"
    assert mapping["nDCG@5"] == "ndcg_at_5"

