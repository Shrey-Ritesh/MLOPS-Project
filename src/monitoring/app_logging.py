"""Structured JSON log lines on stdout (collected by Docker).

One line per event, e.g.::

    {"ts": "2026-11-26T09:14:03.121Z", "level": "INFO", "service": "app", "event": "prediction",
     "request_id": "...", "model_version": "3", "status": "ok", "latency_ms": 12.4, "source": "live"}

Only operational fields go here. Learner identifiers, feature values, question or answer text
and exception messages are never logged: ``log_event`` drops any field not in ``ALLOWED_FIELDS``.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "jee_tutor.events"

ALLOWED_FIELDS = frozenset({
    "request_id", "prediction_id", "window_id", "trigger_id",
    "model_name", "model_version", "registry_alias", "status", "error_type",
    "latency_ms", "source", "fallback", "count", "path", "reason",
})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_request_id() -> str:
    return uuid.uuid4().hex


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stdout`` is at emit time (survives stream swaps, e.g. in tests)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": utc_now_iso(),
            "level": record.levelname,
            "service": getattr(record, "service", self.service),
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, separators=(",", ":"), default=str)


def get_event_logger(service: str = "app") -> logging.Logger:
    """Return the shared event logger, attaching a JSON stdout handler once."""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_jee_json", False) for h in logger.handlers):
        handler = _StdoutHandler(sys.stdout)
        handler.setFormatter(JsonFormatter(service))
        handler._jee_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, service: str = "app", **fields: Any) -> dict[str, Any]:
    """Emit one JSON event line. Returns the fields actually logged (useful in tests)."""
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS and v is not None}
    get_event_logger(service).log(level, event, extra={"event": event, "service": service, "fields": safe})
    return safe
