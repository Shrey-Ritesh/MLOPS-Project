"""Plug point for Person 2's drift and performance evaluation.

The window report shape (§4.2) has fixed ``drift``, ``prediction_psi`` and ``performance``
sections. Person 1's job fills health/validity; Person 2 provides a function

    evaluate_window(rows: list[dict]) -> dict

in ``src/monitoring/drift_windows.py`` returning any of these keys:

    {
      "drift": {feature: {"psi": float, "ks_stat": float | None, "ks_p": float | None, "status": str}},
      "prediction_psi": float | None,
      "performance": {"f1": float, "roc_auc": float, "brier": float} | "insufficient_data",
      "reference_dataset_sha256": str,
      "alerts": [{"severity": "warning|critical", "category": "drift|performance",
                  "metric": str, "value": float, "threshold": float, "message": str}],
    }

``rows`` are the window's prediction rows (dicts with the ``predictions`` columns, ``features``
already parsed). Until that module exists the sections are reported as ``not_configured`` so
the report never claims "no drift" when drift was not measured.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from src.monitoring.app_logging import log_event

EVALUATOR_MODULE = "src.monitoring.drift_windows"
EVALUATOR_FUNCTION = "evaluate_window"

NOT_CONFIGURED: dict[str, Any] = {
    "drift": "not_configured",
    "prediction_psi": "not_configured",
    "performance": "not_configured",
    "reference_dataset_sha256": None,
    "alerts": [],
}

Evaluator = Callable[[list[dict[str, Any]]], dict[str, Any]]


def load_evaluator() -> Evaluator | None:
    try:
        module = importlib.import_module(EVALUATOR_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == EVALUATOR_MODULE:
            return None
        raise
    return getattr(module, EVALUATOR_FUNCTION)


def evaluate(rows: list[dict[str, Any]], evaluator: Evaluator | None) -> dict[str, Any]:
    """Run the evaluator; a crash is reported in the window, not allowed to kill monitoring."""
    if evaluator is None:
        return dict(NOT_CONFIGURED)
    try:
        result = evaluator(rows) or {}
    except Exception as exc:
        log_event("drift_evaluation_failed", service="monitoring", error_type=type(exc).__name__)
        return {**NOT_CONFIGURED, "drift": "error", "prediction_psi": "error", "performance": "error",
                "error_type": type(exc).__name__}
    merged = {**NOT_CONFIGURED, **result}
    merged["alerts"] = list(result.get("alerts") or [])
    return merged
