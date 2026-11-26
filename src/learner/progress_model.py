"""Live inference wrapper for the Track B learner-progress model (Person 3).

Responsibilities:
- Load the champion model by MLflow registry alias (default ``learner-difficulty@production``),
  with automatic fallback to a local joblib artefact (``models/track_b/candidate/model.joblib``)
  and a deterministic rule-based fallback if neither is available or inference times out.
- Enforce the exact feature-order contract directly against ``model.feature_names_in_``
  (fails loudly on mismatch with ``FeatureContractMismatchError``).
- Support hot-reloading when the MLflow ``production`` alias moves (promotion or rollback)
  without rebuilding or restarting the app.
- Return structured metadata matching the §4.1 prediction log contract:
  ``score``, ``label``, ``threshold``, ``model_name``, ``model_version``, ``model_id``,
  ``registry_alias``, ``feature_hash``, ``latency_ms``, ``status``, ``fallback``, ``error_type``.

Synthetic-data notice: Track B models are trained on SYNTHETIC learner data only and
are not validated for real students.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import joblib
import pandas as pd

from src.monitoring.contract import Contract, load_contract
from src.monitoring.prediction_log import canonical_features, feature_hash
from src.track_b.modeling import apply_difficulty_boundary_guard
from src.track_b.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "learner-difficulty"
DEFAULT_REGISTRY_ALIAS = "production"
DEFAULT_LOCAL_MODEL_DIR = Path("models/track_b/candidate")
DEFAULT_THRESHOLD = 0.7
DEFAULT_TIMEOUT_MS = 500.0
DEFAULT_REFRESH_SECONDS = 60.0

SYNTHETIC_NOTICE = (
    "Estimate from synthetic-trained model (Track B); not validated for real students."
)


class FeatureContractMismatchError(ValueError):
    """Raised loudly when a loaded model's feature order disagrees with the contract."""


@dataclass(frozen=True)
class PredictionResult:
    score: float
    label: int
    threshold: float
    model_name: str
    model_version: str
    model_id: str | None
    registry_alias: str | None
    feature_hash: str
    latency_ms: float
    status: str  # ok | timeout | error | fallback
    fallback: bool
    error_type: str | None = None
    reason: str = ""
    synthetic_notice: str = SYNTHETIC_NOTICE
    source_backend: str = "mlflow"  # mlflow | local_joblib | rule_fallback


def _validate_model_feature_order(model: Any, expected_features: tuple[str, ...]) -> None:
    """Fail loudly if the loaded sklearn estimator's feature_names_in_ mismatches the contract."""
    actual = getattr(model, "feature_names_in_", None)
    if actual is None:
        raise FeatureContractMismatchError(
            "Loaded model does not expose feature_names_in_; cannot verify feature contract."
        )
    actual_tuple = tuple(str(c) for c in actual)
    if actual_tuple != expected_features:
        raise FeatureContractMismatchError(
            f"Model feature order mismatch: expected {list(expected_features)}, "
            f"model has {list(actual_tuple)}"
        )


def _rule_based_fallback(features: Mapping[str, Any]) -> tuple[float, int, float, str]:
    """Deterministic fallback rule when no ML model is available or inference times out."""
    threshold = 0.60
    difficulty = str(features.get("selected_difficulty", "Medium"))
    if difficulty == "Easy":
        return 0.0, 0, threshold, "Already at Easy difficulty; rule-based fallback keeps current level."

    recent_acc = float(features.get("recent_accuracy", 0.5))
    mistakes = int(features.get("mistake_count", 0))
    hints = int(features.get("hint_count", 0))
    score = min(1.0, max(0.0, (1.0 - recent_acc) * 0.55 + min(mistakes, 4) * 0.12 + min(hints, 3) * 0.08))
    score = round(score, 4)
    label = int(score >= threshold)
    if label == 1:
        reason = (
            f"Rule-based fallback estimate: lower accuracy ({recent_acc:.0%}) with "
            f"{mistakes} mistake(s) and {hints} hint(s) suggests stepping down from {difficulty}."
        )
    else:
        reason = (
            f"Rule-based fallback estimate: current signals ({recent_acc:.0%} accuracy) "
            f"support staying at {difficulty} difficulty."
        )
    return score, label, threshold, reason


def _build_reason(features: Mapping[str, Any], score: float, label: int, threshold: float) -> str:
    difficulty = str(features.get("selected_difficulty", "Medium"))
    topic = str(features.get("topic", "Chemistry"))
    recent_acc = float(features.get("recent_accuracy", 0.0))
    mistakes = int(features.get("mistake_count", 0))
    hints = int(features.get("hint_count", 0))
    if difficulty == "Easy":
        return (
            f"Already at Easy difficulty on {topic} (switch-down score {score:.2f} < {threshold:.2f}; "
            "boundary guard active)."
        )
    if label == 1:
        return (
            f"Model estimates a {score:.0%} likelihood of stepping down from {difficulty} on {topic} "
            f"(recent accuracy {recent_acc:.0%}, {mistakes} mistake(s), {hints} hint(s))."
        )
    return (
        f"Model estimates staying at {difficulty} on {topic} "
        f"(step-down score {score:.0%} below {threshold:.0%} threshold; recent accuracy {recent_acc:.0%})."
    )


class ProgressModel:
    """Versioned learner-progress inference wrapper with MLflow alias resolution and fallback."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        registry_alias: str = DEFAULT_REGISTRY_ALIAS,
        tracking_uri: str | None = None,
        local_model_dir: str | Path | None = DEFAULT_LOCAL_MODEL_DIR,
        contract: Contract | None = None,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        allow_local_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.registry_alias = registry_alias
        self.tracking_uri = tracking_uri if tracking_uri is not None else os.environ.get("MLFLOW_TRACKING_URI")
        self.local_model_dir = Path(local_model_dir) if local_model_dir is not None else None
        self.contract = contract or load_contract()
        self.expected_features: tuple[str, ...] = tuple(self.contract.feature_names or FEATURE_COLUMNS)
        self.timeout_ms = float(timeout_ms)
        self.refresh_seconds = float(refresh_seconds)
        self.allow_local_fallback = allow_local_fallback

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="progress-model")
        self._model: Any = None
        self._loaded_version: str = "fallback-v1"
        self._loaded_model_id: str | None = None
        self._loaded_model_name: str = "rule_fallback"
        self._loaded_alias: str | None = None
        self._threshold: float = DEFAULT_THRESHOLD
        self._apply_boundary_guard: bool = True
        self._source_backend: str = "rule_fallback"
        self._last_load_error: str | None = None
        self._last_checked_monotonic: float = 0.0

        self.refresh(force=True)

    @classmethod
    def from_env(cls) -> "ProgressModel":
        local_dir = os.environ.get("PROGRESS_MODEL_LOCAL_DIR", str(DEFAULT_LOCAL_MODEL_DIR))
        return cls(
            model_name=os.environ.get("PROGRESS_MODEL_NAME", DEFAULT_MODEL_NAME),
            registry_alias=os.environ.get("PROGRESS_MODEL_ALIAS", DEFAULT_REGISTRY_ALIAS),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            local_model_dir=local_dir if local_dir else None,
            timeout_ms=float(os.environ.get("PROGRESS_MODEL_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))),
            refresh_seconds=float(os.environ.get("PROGRESS_MODEL_REFRESH_SECONDS", str(DEFAULT_REFRESH_SECONDS))),
        )

    @property
    def current_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_name": self._loaded_model_name,
                "model_version": self._loaded_version,
                "model_id": self._loaded_model_id,
                "registry_alias": self._loaded_alias,
                "threshold": self._threshold,
                "source_backend": self._source_backend,
                "expected_features": list(self.expected_features),
            }

    def _try_load_from_mlflow(self) -> bool:
        """Attempt to load the model version pointed to by ``registry_alias`` from MLflow.

        Raises ``FeatureContractMismatchError`` loudly if the loaded model violates the contract.
        Returns True if a valid MLflow model is active, False if MLflow is unavailable.
        """
        if not self.tracking_uri:
            return False
        try:
            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            mv = client.get_model_version_by_alias(self.model_name, self.registry_alias)
            version_str = str(mv.version)

            # Fast path: alias still points to the version already loaded in memory.
            if (
                self._model is not None
                and self._source_backend == "mlflow"
                and self._loaded_version == version_str
            ):
                return True

            mlflow.set_tracking_uri(self.tracking_uri)
            model_uri = f"models:/{self.model_name}@{self.registry_alias}"
            loaded = mlflow.sklearn.load_model(model_uri)
        except FeatureContractMismatchError:
            raise
        except Exception as exc:
            self._last_load_error = type(exc).__name__
            logger.warning(
                "MLflow model load failed (%s@%s): %s",
                self.model_name,
                self.registry_alias,
                type(exc).__name__,
            )
            return False

        # Contract check MUST fail loudly even when loaded from MLflow.
        _validate_model_feature_order(loaded, self.expected_features)

        # Read threshold / model_id from MLflow run params & version tags when present.
        threshold = DEFAULT_THRESHOLD
        apply_guard = True
        model_id = (mv.tags or {}).get("model_id")
        try:
            run = client.get_run(mv.run_id)
            params = run.data.params or {}
            if "decision_threshold" in params:
                threshold = float(params["decision_threshold"])
            if "apply_boundary_guard" in params:
                apply_guard = str(params["apply_boundary_guard"]).strip().lower() in {"1", "true", "yes"}
            if not model_id:
                model_id = params.get("model_id") or run.info.run_name
        except Exception:
            pass

        self._model = loaded
        self._loaded_model_name = self.model_name
        self._loaded_version = version_str
        self._loaded_model_id = model_id
        self._loaded_alias = self.registry_alias
        self._threshold = threshold
        self._apply_boundary_guard = apply_guard
        self._source_backend = "mlflow"
        self._last_load_error = None
        return True

    def _try_load_from_local(self) -> bool:
        """Load the committed local joblib artefact + model_card.json as fallback."""
        if not self.allow_local_fallback or self.local_model_dir is None:
            return False
        model_file = self.local_model_dir / "model.joblib"
        if not model_file.is_file():
            self._last_load_error = "FileNotFoundError"
            return False

        try:
            loaded = joblib.load(model_file)
        except Exception as exc:
            self._last_load_error = type(exc).__name__
            logger.warning("Local model load failed (%s): %s", model_file, type(exc).__name__)
            return False

        # Contract check MUST fail loudly.
        _validate_model_feature_order(loaded, self.expected_features)

        card: dict[str, Any] = {}
        card_file = self.local_model_dir / "model_card.json"
        if card_file.is_file():
            try:
                card = json.loads(card_file.read_text(encoding="utf-8"))
            except Exception:
                card = {}

        self._model = loaded
        self._loaded_model_name = self.model_name
        self._loaded_version = str(card.get("model_version", "local-joblib"))
        self._loaded_model_id = card.get("model_id")
        self._loaded_alias = self.registry_alias
        self._threshold = float(card.get("decision_threshold", DEFAULT_THRESHOLD))
        self._apply_boundary_guard = bool(card.get("apply_boundary_guard", True))
        self._source_backend = "local_joblib"
        self._last_load_error = None
        return True

    def refresh(self, *, force: bool = False) -> bool:
        """Check whether the registry alias changed and reload if needed."""
        now = time.monotonic()
        with self._lock:
            if not force and self.refresh_seconds > 0 and (now - self._last_checked_monotonic) < self.refresh_seconds:
                return self._model is not None
            self._last_checked_monotonic = now

            if self._try_load_from_mlflow():
                return True
            if self._source_backend == "local_joblib" and self._model is not None:
                return True
            if self._try_load_from_local():
                return True

            self._model = None
            self._loaded_model_name = "rule_fallback"
            self._loaded_version = "fallback-v1"
            self._loaded_model_id = None
            self._loaded_alias = None
            self._threshold = 0.60
            self._apply_boundary_guard = True
            self._source_backend = "rule_fallback"
            return False

    def _score_row(self, model: Any, frame: pd.DataFrame, apply_guard: bool) -> float:
        proba = model.predict_proba(frame)[:, 1]
        if apply_guard and "selected_difficulty" in frame.columns:
            proba = apply_difficulty_boundary_guard(proba, frame["selected_difficulty"])
        return float(min(1.0, max(0.0, proba[0])))

    def predict(self, features: Mapping[str, Any]) -> PredictionResult:
        """Run one versioned prediction with contract validation, timeout, and rule fallback."""
        ordered = canonical_features(features, self.contract)
        fhash = feature_hash(ordered)

        self.refresh(force=False)
        with self._lock:
            model = self._model
            model_name = self._loaded_model_name
            model_version = self._loaded_version
            model_id = self._loaded_model_id
            registry_alias = self._loaded_alias
            threshold = self._threshold
            apply_guard = self._apply_boundary_guard
            source_backend = self._source_backend
            load_error = self._last_load_error

        t0 = time.perf_counter()
        if model is None:
            score, label, fb_thr, reason = _rule_based_fallback(ordered)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return PredictionResult(
                score=score,
                label=label,
                threshold=fb_thr,
                model_name="rule_fallback",
                model_version="fallback-v1",
                model_id=None,
                registry_alias=None,
                feature_hash=fhash,
                latency_ms=latency_ms,
                status="fallback",
                fallback=True,
                error_type=load_error or "ModelUnavailable",
                reason=reason,
                source_backend="rule_fallback",
            )

        frame = pd.DataFrame([ordered], columns=list(self.expected_features))
        future = self._executor.submit(self._score_row, model, frame, apply_guard)
        try:
            score = future.result(timeout=self.timeout_ms / 1000.0)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            label = int(score >= threshold)
            return PredictionResult(
                score=round(score, 6),
                label=label,
                threshold=threshold,
                model_name=model_name,
                model_version=model_version,
                model_id=model_id,
                registry_alias=registry_alias,
                feature_hash=fhash,
                latency_ms=latency_ms,
                status="ok",
                fallback=False,
                error_type=None,
                reason=_build_reason(ordered, score, label, threshold),
                source_backend=source_backend,
            )
        except FuturesTimeoutError:
            future.cancel()
            score, label, fb_thr, reason = _rule_based_fallback(ordered)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return PredictionResult(
                score=score,
                label=label,
                threshold=fb_thr,
                model_name="rule_fallback",
                model_version="fallback-v1",
                model_id=None,
                registry_alias=registry_alias,
                feature_hash=fhash,
                latency_ms=latency_ms,
                status="timeout",
                fallback=True,
                error_type="TimeoutError",
                reason=f"Inference timed out (> {self.timeout_ms:.0f} ms). {reason}",
                source_backend="rule_fallback",
            )
        except Exception as exc:
            score, label, fb_thr, reason = _rule_based_fallback(ordered)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return PredictionResult(
                score=score,
                label=label,
                threshold=fb_thr,
                model_name="rule_fallback",
                model_version="fallback-v1",
                model_id=None,
                registry_alias=registry_alias,
                feature_hash=fhash,
                latency_ms=latency_ms,
                status="fallback",
                fallback=True,
                error_type=type(exc).__name__,
                reason=reason,
                source_backend="rule_fallback",
            )
