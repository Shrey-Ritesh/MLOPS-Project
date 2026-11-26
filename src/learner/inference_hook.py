"""Live inference hook connecting the practice flow (and traffic generator) to ProgressModel and PredictionLogger.

Provides:
- ``build_learner_features(...)``: builds the 8 Track B features strictly from pre-submission
  state (no future leakage).
- ``InferenceService``: wraps ``ProgressModel`` + optional ``PredictionLogger`` so both
  ``app.py`` (``source="live"``) and Person 2's simulated traffic generator
  (``source="simulated"``) run through the exact same inference and logging path.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

from src.learner.history import AttemptRecord
from src.learner.progress_model import PredictionResult, ProgressModel
from src.monitoring.app_logging import new_request_id
from src.monitoring.prediction_log import (
    OutcomeRecord,
    PredictionLogConfigError,
    PredictionLogger,
    PredictionRecord,
)

logger = logging.getLogger(__name__)

VALID_DIFFICULTIES = ("Easy", "Medium", "Hard")


def build_learner_features(
    *,
    topic: str,
    selected_difficulty: str,
    prior_attempts: Sequence[AttemptRecord],
    hint_count: int,
    response_time_seconds: float,
    previous_difficulty: str | None = None,
    recent_window: int = 5,
) -> dict[str, Any]:
    """Construct the 8 Track B features using only pre-submission learner state.

    ``prior_attempts`` must contain only attempts recorded *before* the current submission
    so there is no target or future leakage.
    """
    prior_count = len(prior_attempts)
    # list_for_learner returns newest-first; take up to `recent_window` most recent attempts.
    recent = list(prior_attempts)[:recent_window]
    scored_values: list[float] = []
    mistakes = 0
    for att in recent:
        if att.outcome == "correct":
            scored_values.append(1.0)
        elif att.outcome == "incorrect":
            scored_values.append(0.0)
            mistakes += 1
        elif att.outcome == "partial":
            scored_values.append(0.5)

    recent_accuracy = round(sum(scored_values) / len(scored_values), 4) if scored_values else 0.5
    diff = selected_difficulty if selected_difficulty in VALID_DIFFICULTIES else "Medium"
    prev_diff = previous_difficulty if previous_difficulty in VALID_DIFFICULTIES else diff
    clamped_rt = round(min(3600.0, max(1.0, float(response_time_seconds))), 2)
    clamped_hints = max(0, min(50, int(hint_count)))

    return {
        "topic": str(topic or "unknown"),
        "selected_difficulty": diff,
        "recent_accuracy": float(recent_accuracy),
        "mistake_count": int(min(50, mistakes)),
        "hint_count": clamped_hints,
        "response_time_seconds": clamped_rt,
        "prior_interaction_count": int(prior_count),
        "previous_difficulty": prev_diff,
    }


@dataclass(frozen=True)
class InferenceEvent:
    request_id: str
    prediction_id: str | None
    features: dict[str, Any]
    result: PredictionResult


class InferenceService:
    """Shared inference + prediction logging entry point for live app and simulated traffic."""

    def __init__(
        self,
        model: ProgressModel | None = None,
        prediction_logger: PredictionLogger | None = None,
    ) -> None:
        self.model = model or ProgressModel.from_env()
        self.prediction_logger = prediction_logger

    @classmethod
    def from_env(cls) -> "InferenceService":
        model = ProgressModel.from_env()
        pred_logger: PredictionLogger | None = None
        try:
            pred_logger = PredictionLogger.from_env()
        except PredictionLogConfigError as exc:
            # App must still function locally when LOG_PSEUDONYM_KEY is unset; log once clearly.
            logger.warning("PredictionLogger disabled (config missing): %s", exc)
        return cls(model=model, prediction_logger=pred_logger)

    def predict_and_log(
        self,
        *,
        learner_id: str,
        learner_seq: int,
        question_id: str,
        features: Mapping[str, Any],
        source: str = "live",
        request_id: str | None = None,
    ) -> InferenceEvent:
        """Run ProgressModel.predict() and write one PredictionRecord to the monitoring DB."""
        req_id = request_id or new_request_id()
        result = self.model.predict(features)

        prediction_id: str | None = None
        if self.prediction_logger is not None:
            record = PredictionRecord(
                request_id=req_id,
                source=source,
                learner_id=learner_id,
                learner_seq=int(learner_seq),
                question_id=question_id,
                features=dict(features),
                score=result.score,
                label=result.label,
                threshold=result.threshold,
                model_name=result.model_name,
                model_version=result.model_version,
                model_id=result.model_id,
                registry_alias=result.registry_alias,
                fallback=result.fallback,
                latency_ms=result.latency_ms,
                status=result.status,
                error_type=result.error_type,
            )
            prediction_id = self.prediction_logger.log(record)

        return InferenceEvent(
            request_id=req_id,
            prediction_id=prediction_id,
            features=dict(features),
            result=result,
        )

    def log_outcome(
        self,
        *,
        learner_id: str,
        topic: str,
        question_id: str,
        outcome: str,
        source: str = "live",
        request_id: str | None = None,
    ) -> str | None:
        """Forward an evaluated answer outcome to PredictionLogger (no-op if logger disabled)."""
        if self.prediction_logger is None:
            return None
        return self.prediction_logger.log_outcome(
            OutcomeRecord(
                source=source,
                learner_id=learner_id,
                topic=topic,
                question_id=question_id,
                outcome=outcome,
                request_id=request_id,
            )
        )
