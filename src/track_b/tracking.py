"""MLflow tracking setup shared by the Track B scripts.

Default is a LOCAL SQLite store (mlflow.db) in the repo root. To use a shared server later,
set MLFLOW_TRACKING_URI (for example http://<server>:5000); no code changes are needed.
"""

from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "track_b_learner_difficulty"
REGISTERED_MODEL_NAME = "learner-difficulty"


def configure() -> str:
    """Point MLflow at the configured store and select the experiment; return the URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri
