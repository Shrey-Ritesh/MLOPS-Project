"""Prediction logger: writes exactly one ``predictions`` row per model call.

Usage from the inference hook (Person 3)::

    logger = PredictionLogger.from_env()            # once per process
    logger.log(PredictionRecord(
        request_id=request_id, source="live", learner_id=user_learner_id, learner_seq=n_prior,
        question_id=qid, features=features_dict, score=0.81, label=1, threshold=0.7,
        model_name="learner-difficulty", model_version="3", model_id="candidate_hgb_shifted_v2",
        registry_alias="production", fallback=False, latency_ms=12.4, status="ok",
    ))

``log`` never raises into the app: a failed write is reported as a structured
``prediction_log_write_failed`` event (counted by monitoring) and returns ``None``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.monitoring import db
from src.monitoring.app_logging import log_event, utc_now_iso
from src.monitoring.schema import FEATURE_COLUMNS, SCHEMA_VERSION, SOURCES, STATUSES

PSEUDONYM_ENV = "LOG_PSEUDONYM_KEY"
_PLACEHOLDERS = {"", "changeme", "your_log_pseudonym_key_here"}
MIN_KEY_LENGTH = 16


class PredictionLogConfigError(RuntimeError):
    """The logger cannot run safely (for example, no pseudonym key)."""


class PredictionRecordError(ValueError):
    """A record breaks the contract; it is rejected rather than written."""


def pseudonymize(learner_id: str, key: str) -> str:
    """Stable pseudonym for a learner: HMAC-SHA256, first 16 hex chars. Irreversible without the key."""
    if not learner_id:
        raise PredictionRecordError("learner_id must be non-empty")
    return hmac.new(key.encode("utf-8"), learner_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def canonical_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Return features in contract order; extra or missing keys are a contract violation."""
    keys = set(features)
    expected = set(FEATURE_COLUMNS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PredictionRecordError(f"feature mismatch: missing={missing} extra={extra}")
    return {name: features[name] for name in FEATURE_COLUMNS}


def feature_hash(features: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(features), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PredictionRecord:
    request_id: str
    source: str
    learner_id: str  # raw id from the app; only its pseudonym is stored
    learner_seq: int
    question_id: str
    features: Mapping[str, Any]
    model_name: str
    model_version: str
    latency_ms: float
    status: str
    fallback: bool = False
    score: float | None = None
    label: int | None = None
    threshold: float | None = None
    model_id: str | None = None
    registry_alias: str | None = None
    error_type: str | None = None
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: str = field(default_factory=utc_now_iso)

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise PredictionRecordError(f"source must be one of {SOURCES}")
        if self.status not in STATUSES:
            raise PredictionRecordError(f"status must be one of {STATUSES}")
        if not isinstance(self.learner_seq, int) or self.learner_seq < 0:
            raise PredictionRecordError("learner_seq must be a non-negative int")
        if not self.request_id or not self.question_id:
            raise PredictionRecordError("request_id and question_id are required")
        if self.learner_id and (self.learner_id in self.request_id or self.learner_id in self.question_id):
            # request_id/question_id are stored as-is; a learner id inside them would undo the pseudonym
            raise PredictionRecordError("request_id/question_id must not contain the learner id")
        if not self.model_name or not self.model_version:
            raise PredictionRecordError("model_name and model_version are required")
        if self.latency_ms is None or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise PredictionRecordError("latency_ms must be a finite, non-negative number")
        if self.status == "ok":
            if self.score is None or self.label is None:
                raise PredictionRecordError("status 'ok' requires score and label")
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise PredictionRecordError("score must be within [0, 1]")
        if self.label not in (None, 0, 1):
            raise PredictionRecordError("label must be 0, 1 or None")
        if self.status == "fallback" and not self.fallback:
            raise PredictionRecordError("status 'fallback' requires fallback=True")


class PredictionLogger:
    def __init__(self, db_path: str | Path, pseudonym_key: str) -> None:
        if (pseudonym_key or "").strip().lower() in _PLACEHOLDERS or len(pseudonym_key) < MIN_KEY_LENGTH:
            raise PredictionLogConfigError(
                f"{PSEUDONYM_ENV} must be set to a secret of at least {MIN_KEY_LENGTH} characters"
            )
        self.db_path = str(db_path)
        self._key = pseudonym_key
        with db.session(self.db_path) as conn:
            db.ensure_schema(conn)

    @classmethod
    def from_env(cls) -> "PredictionLogger":
        return cls(db.db_path_from_env(), os.environ.get(PSEUDONYM_ENV, ""))

    def build_row(self, record: PredictionRecord) -> dict[str, Any]:
        record.validate()
        features = canonical_features(record.features)
        return {
            "prediction_id": record.prediction_id,
            "schema_version": SCHEMA_VERSION,
            "occurred_at": record.occurred_at,
            "request_id": record.request_id,
            "source": record.source,
            "learner_key": pseudonymize(record.learner_id, self._key),
            "learner_seq": record.learner_seq,
            "question_id": record.question_id,
            "topic": str(features["topic"] or "unknown"),
            "features": json.dumps(features, separators=(",", ":"), default=str),
            "feature_hash": feature_hash(features),
            "score": record.score,
            "label": record.label,
            "threshold": record.threshold,
            "model_name": record.model_name,
            "model_version": str(record.model_version),
            "model_id": record.model_id,
            "registry_alias": record.registry_alias,
            "fallback": int(bool(record.fallback)),
            "latency_ms": float(record.latency_ms),
            "status": record.status,
            "error_type": record.error_type,
        }

    def log(self, record: PredictionRecord) -> str | None:
        """Write the record; return its prediction_id, or None if it could not be written."""
        try:
            row = self.build_row(record)
            columns = ", ".join(row)
            placeholders = ", ".join(f":{name}" for name in row)
            with db.session(self.db_path) as conn:
                conn.execute(f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", row)
        except Exception as exc:  # never break the practice flow because logging failed
            log_event(
                "prediction_log_write_failed",
                level=logging.ERROR,
                request_id=getattr(record, "request_id", None),
                source=getattr(record, "source", None),
                error_type=type(exc).__name__,
            )
            return None
        log_event(
            "prediction",
            request_id=row["request_id"],
            prediction_id=row["prediction_id"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            status=row["status"],
            fallback=bool(row["fallback"]),
            latency_ms=round(row["latency_ms"], 3),
            source=row["source"],
        )
        return row["prediction_id"]
