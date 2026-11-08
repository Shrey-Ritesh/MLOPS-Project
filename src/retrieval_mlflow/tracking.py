"""MLflow tracking setup for the retrieval comparison, following Track B conventions."""
from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "chemistry_ncert_retrieval_comparison"


def configure(
    experiment_name: str | None = None,
    default_uri: str = DEFAULT_TRACKING_URI,
) -> str:
    """Use the configured local SQLite store unless MLFLOW_TRACKING_URI overrides it."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name or EXPERIMENT_NAME)
    return uri
