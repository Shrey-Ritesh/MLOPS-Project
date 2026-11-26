"""Guard the MLflow service settings that matter for safety and for the other containers."""
from __future__ import annotations

import re

import yaml

COMPOSE = yaml.safe_load(open("docker-compose.yml", encoding="utf-8"))
MLFLOW = COMPOSE["services"]["mlflow"]


def _arg(name: str) -> str:
    return next(a.split("=", 1)[1] for a in MLFLOW["command"] if a.startswith(f"--{name}="))


def test_mlflow_is_never_published_beyond_the_server():
    assert MLFLOW["ports"] == ["127.0.0.1:5001:5000"]


def test_containers_can_reach_mlflow_by_service_name():
    # MLflow 3.x answers 403 to unknown Host headers, and "mlflow:5000" is unknown by default.
    allowed = _arg("allowed-hosts").split(",")
    assert "mlflow:5000" in allowed and "*" not in allowed
    for service in ("app", "monitor"):
        assert COMPOSE["services"][service]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_registry_lives_on_a_named_volume_with_one_worker():
    assert MLFLOW["volumes"] == ["mlflow_data:/mlflow"] and "mlflow_data" in COMPOSE["volumes"]
    assert _arg("backend-store-uri") == "sqlite:////mlflow/mlflow.db"
    assert _arg("workers") == "1"  # SQLite backend: one writer process avoids lock errors


def test_server_image_matches_the_pinned_client_version():
    pinned = re.search(r"^mlflow==(\S+)$", open("requirements.txt", encoding="utf-8").read(), re.M).group(1)
    assert MLFLOW["image"] == f"ghcr.io/mlflow/mlflow:v{pinned}"


def test_deploy_script_starts_mlflow():
    script = open("scripts/deploy.sh", encoding="utf-8").read()
    assert "up -d --no-build chroma mlflow api app monitor" in script
