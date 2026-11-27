"""Live inference wrapper for the Track B learner-progress model (Person 3).

Responsibilities:
- Load the champion model by MLflow registry alias (default ``learner-difficulty@production``),
  with automatic fallback to a local joblib artefact (``models/track_b/candidate/model.joblib``)
  and a deterministic rule-based fallback if neither is available or inference times out.
- Enforce the exact feature-order contract directly against ``model.feature_names_in_``
  (fails loudly on mismatch with ``FeatureContractMismatchError`` at load time).
- On contract violations at ``predict()`` time (e.g. during a model contract switch),
  return a safe rule-based fallback result and log a structured error event instead of crashing.
- Keep the already-loaded MLflow production model during brief MLflow outages instead of
  downgrading to a local file or rule fallback.
- Refresh MLflow aliases in the background with short HTTP timeouts and zero retries so
  live practice submissions never freeze when MLflow is down.
- Support hot-reloading when the MLflow ``production`` alias moves (promotion or rollback)
  without rebuilding or restarting the app.
- Return structured metadata matching the §4.1 prediction log contract.

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

from src.monitoring.app_logging import log_event
from src.monitoring.contract import Contract, load_contract
from src.monitoring.prediction_log import PredictionRecordError, canonical_features, feature_hash
from src.track_b.modeling import apply_difficulty_boundary_guard
from src.track_b.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "learner-difficulty"
DEFAULT_REGISTRY_ALIAS = "production"
DEFAULT_LOCAL_MODEL_DIR = Path("models/track_b/candidate")
DEFAULT_THRESHOLD = 0.7
DEFAULT_APPLY_BOUNDARY_GUARD = True
DEFAULT_TIMEOUT_MS = 500.0
DEFAULT_REFRESH_SECONDS = 60.0
DEFAULT_MLFLOW_HTTP_TIMEOUT_SECONDS = 3

SYNTHETIC_NOTICE = (
    "Estimate from synthetic-trained model (Track B); not validated for real students."
)


class FeatureContractMismatchError(ValueError):
    """Raised loudly when a loaded model's feature order disagrees with the contract."""


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def resolve_serving_config(
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    card: Mapping[str, Any] | None = None,
) -> tuple[float, bool]:
    """Single source of truth for (decision_threshold, apply_boundary_guard).

    Shared by ProgressModel (live serving) and model_comparison.py (Gate v2) so
    the champion is always evaluated identically to how the app serves it.
    """
    params = params or {}
    tags = tags or {}
    card = card or {}

    raw_thr = params.get("decision_threshold")
    if raw_thr is None:
        raw_thr = tags.get("decision_threshold")
    if raw_thr is None:
        raw_thr = card.get("decision_threshold")
    threshold = float(raw_thr) if raw_thr is not None else DEFAULT_THRESHOLD

    raw_guard = params.get("apply_boundary_guard")
    if raw_guard is None:
        raw_guard = tags.get("apply_boundary_guard")
    if raw_guard is None:
        raw_guard = card.get("apply_boundary_guard")
    apply_guard = _parse_bool(raw_guard, DEFAULT_APPLY_BOUNDARY_GUARD)

    return threshold, apply_guard


def _configure_short_mlflow_timeouts() -> None:
    """Ensure MLflow HTTP calls fail fast instead of blocking live requests for minutes."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(DEFAULT_MLFLOW_HTTP_TIMEOUT_SECONDS))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")


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


@dataclass(frozen=True)
class _LoadedCandidate:
    model: Any
    model_name: str
    model_version: str
    model_id: str | None
    registry_alias: str | None
    threshold: float
    apply_boundary_guard: bool
    source_backend: str


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
    """Deterministic fallback rule when no ML model is available, contract mismatches, or inference times out."""
    threshold = 0.60
    difficulty = str(features.get("selected_difficulty", "Medium"))
    if difficulty == "Easy":
        return 0.0, 0, threshold, "Already at Easy difficulty; rule-based fallback keeps current level."

    try:
        recent_acc = float(features.get("recent_accuracy", 0.5))
    except (TypeError, ValueError):
        recent_acc = 0.5
    try:
        mistakes = int(features.get("mistake_count", 0))
    except (TypeError, ValueError):
        mistakes = 0
    try:
        hints = int(features.get("hint_count", 0))
    except (TypeError, ValueError):
        hints = 0

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
        background_refresh: bool = True,
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
        self.background_refresh = background_refresh

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="progress-model")
        self._refresh_in_flight = False
        self._has_initialized = False

        self._model: Any = None
        self._loaded_version: str = "fallback-v1"
        self._loaded_model_id: str | None = None
        self._loaded_model_name: str = "rule_fallback"
        self._loaded_alias: str | None = None
        self._threshold: float = DEFAULT_THRESHOLD
        self._apply_boundary_guard: bool = DEFAULT_APPLY_BOUNDARY_GUARD
        self._source_backend: str = "rule_fallback"
        self._last_load_error: str | None = None
        self._last_checked_monotonic: float = 0.0

        self.refresh(force=True)
        self._has_initialized = True

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
                "apply_boundary_guard": self._apply_boundary_guard,
                "source_backend": self._source_backend,
                "expected_features": list(self.expected_features),
            }

    def _fetch_mlflow_candidate(self, current_mlflow_version: str | None) -> tuple[_LoadedCandidate | str | None, str | None]:
        """Run MLflow lookup + model load WITHOUT holding self._lock.

        Returns:
          ("unchanged", None) if alias still points to current_mlflow_version
          (_LoadedCandidate, None) if a new MLflow model version was loaded
          (None, error_type) if MLflow is unavailable or lookup failed
        Raises FeatureContractMismatchError loudly if the loaded model violates the contract.
        """
        if not self.tracking_uri:
            return None, None
        _configure_short_mlflow_timeouts()
        try:
            import mlflow
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            mv = client.get_model_version_by_alias(self.model_name, self.registry_alias)
            version_str = str(mv.version)

            if current_mlflow_version is not None and current_mlflow_version == version_str:
                return "unchanged", None

            mlflow.set_tracking_uri(self.tracking_uri)
            model_uri = f"models:/{self.model_name}@{self.registry_alias}"
            loaded = mlflow.sklearn.load_model(model_uri)
        except FeatureContractMismatchError:
            raise
        except Exception as exc:
            err_name = type(exc).__name__
            logger.warning(
                "MLflow model load failed (%s@%s): %s",
                self.model_name,
                self.registry_alias,
                err_name,
            )
            return None, err_name

        # Contract check MUST fail loudly even when loaded from MLflow.
        _validate_model_feature_order(loaded, self.expected_features)

        params: dict[str, Any] = {}
        mv_tags: dict[str, Any] = dict(mv.tags or {})
        model_id = mv_tags.get("model_id")
        try:
            run = client.get_run(mv.run_id)
            params = dict(run.data.params or {})
            if not model_id:
                model_id = params.get("model_id") or run.info.run_name
        except Exception:
            pass

        threshold, apply_guard = resolve_serving_config(params=params, tags=mv_tags)
        return (
            _LoadedCandidate(
                model=loaded,
                model_name=self.model_name,
                model_version=version_str,
                model_id=model_id,
                registry_alias=self.registry_alias,
                threshold=threshold,
                apply_boundary_guard=apply_guard,
                source_backend="mlflow",
            ),
            None,
        )

    def _fetch_local_candidate(self) -> tuple[_LoadedCandidate | None, str | None]:
        """Load the committed local joblib artefact + model_card.json WITHOUT holding self._lock."""
        if not self.allow_local_fallback or self.local_model_dir is None:
            return None, None
        model_file = self.local_model_dir / "model.joblib"
        if not model_file.is_file():
            return None, "FileNotFoundError"

        try:
            loaded = joblib.load(model_file)
        except Exception as exc:
            err_name = type(exc).__name__
            logger.warning("Local model load failed (%s): %s", model_file, err_name)
            return None, err_name

        # Contract check MUST fail loudly.
        _validate_model_feature_order(loaded, self.expected_features)

        card: dict[str, Any] = {}
        card_file = self.local_model_dir / "model_card.json"
        if card_file.is_file():
            try:
                card = json.loads(card_file.read_text(encoding="utf-8"))
            except Exception:
                card = {}

        threshold, apply_guard = resolve_serving_config(card=card)
        return (
            _LoadedCandidate(
                model=loaded,
                model_name=self.model_name,
                model_version=str(card.get("model_version", "local-joblib")),
                model_id=card.get("model_id"),
                # Local file fallback is NOT served from the MLflow registry; keep registry_alias=None
                # so monitoring can distinguish local fallback from promoted registry models.
                registry_alias=None,
                threshold=threshold,
                apply_boundary_guard=apply_guard,
                source_backend="local_joblib",
            ),
            None,
        )

    def _apply_candidate(self, cand: _LoadedCandidate) -> None:
        with self._lock:
            self._model = cand.model
            self._loaded_model_name = cand.model_name
            self._loaded_version = cand.model_version
            self._loaded_model_id = cand.model_id
            self._loaded_alias = cand.registry_alias
            self._threshold = cand.threshold
            self._apply_boundary_guard = cand.apply_boundary_guard
            self._source_backend = cand.source_backend
            self._last_load_error = None

    def _run_refresh_work(self) -> bool:
        """Perform refresh I/O outside the lock, then atomically swap state under the lock."""
        with self._lock:
            current_mlflow_ver = self._loaded_version if (self._source_backend == "mlflow" and self._model is not None) else None
            has_local_model = self._source_backend == "local_joblib" and self._model is not None

        mlflow_result, mlflow_err = self._fetch_mlflow_candidate(current_mlflow_ver)
        if mlflow_result == "unchanged":
            with self._lock:
                self._last_checked_monotonic = time.monotonic()
            return True
        if isinstance(mlflow_result, _LoadedCandidate):
            self._apply_candidate(mlflow_result)
            with self._lock:
                self._last_checked_monotonic = time.monotonic()
            return True

        # Brief MLflow outage: KEEP the already-loaded promoted MLflow model instead of
        # downgrading to a local file or rule fallback.
        if current_mlflow_ver is not None:
            with self._lock:
                self._last_load_error = mlflow_err
                self._last_checked_monotonic = time.monotonic()
            log_event(
                "mlflow_refresh_failed_keeping_cached_model",
                level=logging.WARNING,
                model_name=self.model_name,
                model_version=current_mlflow_ver,
                error_type=mlflow_err or "MLflowUnavailable",
            )
            return True

        # No MLflow model loaded yet; keep existing local joblib if already loaded, else load it.
        if has_local_model:
            with self._lock:
                self._last_checked_monotonic = time.monotonic()
            return True

        local_cand, local_err = self._fetch_local_candidate()
        if local_cand is not None:
            self._apply_candidate(local_cand)
            with self._lock:
                self._last_checked_monotonic = time.monotonic()
            return True

        with self._lock:
            self._model = None
            self._loaded_model_name = "rule_fallback"
            self._loaded_version = "fallback-v1"
            self._loaded_model_id = None
            self._loaded_alias = None
            self._threshold = 0.60
            self._apply_boundary_guard = True
            self._source_backend = "rule_fallback"
            self._last_load_error = mlflow_err or local_err or "ModelUnavailable"
            self._last_checked_monotonic = time.monotonic()
        return False

    def _background_refresh_worker(self) -> None:
        try:
            self._run_refresh_work()
        except Exception as exc:
            log_event("background_refresh_failed", level=logging.ERROR, error_type=type(exc).__name__)
        finally:
            with self._lock:
                self._refresh_in_flight = False

    def refresh(self, *, force: bool = False) -> bool:
        """Check whether the registry alias changed and reload if needed.

        When ``force=False`` and ``background_refresh=True`` (the live-request path),
        expired TTL refreshes run asynchronously so practice submissions never block
        on MLflow network timeouts.
        """
        now = time.monotonic()
        with self._lock:
            if not force and self.refresh_seconds > 0 and (now - self._last_checked_monotonic) < self.refresh_seconds:
                return self._model is not None

            if not force and self.background_refresh and self._has_initialized and self.refresh_seconds > 0:
                if not self._refresh_in_flight:
                    self._refresh_in_flight = True
                    self._last_checked_monotonic = now
                    t = threading.Thread(target=self._background_refresh_worker, daemon=True, name="progress-model-refresh")
                    t.start()
                return self._model is not None

        return self._run_refresh_work()

    def _score_row(self, model: Any, frame: pd.DataFrame, apply_guard: bool) -> float:
        proba = model.predict_proba(frame)[:, 1]
        if apply_guard and "selected_difficulty" in frame.columns:
            proba = apply_difficulty_boundary_guard(proba, frame["selected_difficulty"])
        return float(min(1.0, max(0.0, proba[0])))

    def predict(self, features: Mapping[str, Any]) -> PredictionResult:
        """Run one versioned prediction with contract validation, timeout, and rule fallback."""
        t0 = time.perf_counter()
        try:
            ordered = canonical_features(features, self.contract)
        except PredictionRecordError as exc:
            err_type = type(exc).__name__
            log_event(
                "prediction_contract_error",
                level=logging.ERROR,
                model_name=self.model_name,
                error_type=err_type,
            )
            score, label, fb_thr, reason = _rule_based_fallback(features)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return PredictionResult(
                score=score,
                label=label,
                threshold=fb_thr,
                model_name="rule_fallback",
                model_version="fallback-v1",
                model_id=None,
                registry_alias=None,
                feature_hash=feature_hash(features),
                latency_ms=latency_ms,
                status="fallback",
                fallback=True,
                error_type=err_type,
                reason=f"Feature contract mismatch ({err_type}). {reason}",
                source_backend="rule_fallback",
            )

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
