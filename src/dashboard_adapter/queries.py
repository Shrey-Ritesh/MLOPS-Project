
"""Read-only data access for the Grafana dashboard adapter."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.monitoring.db import db_path_from_env

DEFAULT_TRIGGER_EVENTS_PATH = Path("reports/monitoring/trigger_events.jsonl")
DEFAULT_TRIGGER_STATE_PATH = Path("reports/monitoring/trigger_state.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object without creating or modifying the source file."""
    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Read the most recent trigger events from a JSONL audit log."""
    if not path.is_file():
        return []

    events: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(event, dict):
                    events.append(event)

    except OSError:
        return []

    return events[-limit:]


def _normalise_status(value: Any) -> str | None:
    """Normalise report status values to lowercase strings."""
    if not isinstance(value, str):
        return None

    status = value.strip().lower()

    aliases = {
        "ok": "ok",
        "alert": "alert",
        "warn": "warning",
        "warning": "warning",
        "insufficient_data": "insufficient_data",
        "unavailable": "unavailable",
    }

    return aliases.get(status, status or None)


def _cooldown_snapshot(state: dict[str, Any] | None) -> dict[str, Any]:
    """Derive cooldown status from persisted trigger state."""
    state = state or {}
    cooldown_until = state.get("cooldown_until")
    active = False
    remaining_seconds = 0

    if cooldown_until:
        try:
            parsed = datetime.fromisoformat(
                str(cooldown_until).replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            remaining_seconds = max(
                0,
                int(
                    (
                        parsed.astimezone(timezone.utc)
                        - datetime.now(timezone.utc)
                    ).total_seconds()
                ),
            )
            active = remaining_seconds > 0

        except (TypeError, ValueError):
            pass

    return {
        "cooldown_active": active,
        "cooldown_until": cooldown_until,
        "cooldown_remaining_seconds": remaining_seconds,
        "runs_today": state.get("runs_today", 0),
        "total_runs": state.get("total_runs", 0),
        "last_fired_at": state.get("last_fired_at"),
    }


def _extract_feature_drift(drift: Any) -> list[dict[str, Any]]:
    """
    Extract PSI metrics from both current and legacy report structures.

    Current structure:
        {
            "hint_count": {"psi": 8.252, "status": "ALERT"}
        }

    Legacy structure:
        {
            "metrics": [
                {"feature": "hint_count", "psi": 8.252}
            ]
        }
    """
    results: dict[str, dict[str, Any]] = {}

    if not isinstance(drift, dict):
        return []

    # Current report format: the metric name is the dictionary key.
    for key, node in drift.items():
        if not isinstance(node, dict):
            continue

        metric = node.get("metric") or node.get("feature") or key
        value = node.get("psi")

        if value is None:
            value = node.get("value")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            results[str(metric)] = {
                "metric": str(metric),
                "value": float(value),
                "severity": (
                    node.get("severity")
                    or _normalise_status(node.get("status"))
                ),
                "threshold": node.get("threshold"),
            }

    # Legacy fallback: recursively find explicitly named metrics.
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            metric = node.get("metric") or node.get("feature")
            value = node.get("psi")

            if value is None:
                value = node.get("value")

            if (
                metric
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                results.setdefault(
                    str(metric),
                    {
                        "metric": str(metric),
                        "value": float(value),
                        "severity": (
                            node.get("severity")
                            or _normalise_status(node.get("status"))
                        ),
                        "threshold": node.get("threshold"),
                    },
                )

            for child in node.values():
                walk(child)

        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(drift)

    return list(results.values())


def _derive_drift_status(drift: Any) -> str:
    """Derive overall drift status from the nested metric statuses."""
    if isinstance(drift, str):
        return _normalise_status(drift) or "unavailable"

    if not isinstance(drift, dict):
        return "unavailable"

    explicit_status = _normalise_status(drift.get("status"))
    if explicit_status:
        return explicit_status

    statuses = [
        _normalise_status(node.get("status"))
        for node in drift.values()
        if isinstance(node, dict) and node.get("status") is not None
    ]

    statuses = [status for status in statuses if status]

    if "alert" in statuses:
        return "alert"

    if "warning" in statuses:
        return "warning"

    if statuses and all(status == "ok" for status in statuses):
        return "ok"

    return "available"


def _derive_performance_status(
    performance: Any,
    report_alerts: Any,
) -> str:
    """Derive performance evaluation status from metrics and report alerts."""
    if isinstance(performance, str):
        return _normalise_status(performance) or "unavailable"

    if not isinstance(performance, dict):
        return "unavailable"

    explicit_status = _normalise_status(performance.get("status"))
    if explicit_status:
        return explicit_status

    metric_names = ("f1", "roc_auc", "brier")
    metrics_available = all(
        isinstance(performance.get(name), (int, float))
        and not isinstance(performance.get(name), bool)
        for name in metric_names
    )

    if not metrics_available:
        return "insufficient_data"

    if isinstance(report_alerts, list):
        for alert in report_alerts:
            if (
                isinstance(alert, dict)
                and str(alert.get("category", "")).lower() == "performance"
            ):
                return "alert"

    return "ok"


def get_dashboard_snapshot(
    db_path: str | Path | None = None,
    trigger_events_path: str | Path | None = None,
    trigger_state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch monitoring, drift, performance and trigger data read-only."""

    path = Path(db_path or db_path_from_env()).resolve()

    events_path = Path(
        trigger_events_path
        or os.environ.get(
            "DASHBOARD_TRIGGER_EVENTS_PATH",
            DEFAULT_TRIGGER_EVENTS_PATH,
        )
    )

    state_path = Path(
        trigger_state_path
        or os.environ.get(
            "DASHBOARD_TRIGGER_STATE_PATH",
            DEFAULT_TRIGGER_STATE_PATH,
        )
    )

    snapshot: dict[str, Any] = {
        "total_predictions": 0,
        "total_monitoring_windows": 0,
        "latest_window_status": None,
        "latest_report": None,
        "drift_status": "unavailable",
        "feature_drift_alerts": [],
        "performance_status": "unavailable",
        "performance": {
            "f1": None,
            "roc_auc": None,
            "brier": None,
        },
        "unacknowledged_alerts": {
            "warning": 0,
            "critical": 0,
        },
        "trigger_events": [],
        "trigger_state": {},
        "cooldown": _cooldown_snapshot(None),
    }

    # Read the monitoring database without creating or modifying it.
    if path.is_file():
        db_uri = f"file:{quote(path.as_posix(), safe='/')}?mode=ro&immutable=1"
        
        try:
            with sqlite3.connect(db_uri, uri=True, timeout=5) as conn:
                snapshot["total_predictions"] = conn.execute(
                    "SELECT COUNT(*) FROM predictions"
                ).fetchone()[0]

                snapshot["total_monitoring_windows"] = conn.execute(
                    "SELECT COUNT(*) FROM monitoring_windows"
                ).fetchone()[0]

                latest_window = conn.execute(
                    """
                    SELECT status, report
                    FROM monitoring_windows
                    ORDER BY window_end DESC
                    LIMIT 1
                    """
                ).fetchone()

                if latest_window:
                    snapshot["latest_window_status"] = latest_window[0]

                    try:
                        report = json.loads(latest_window[1])
                        if isinstance(report, dict):
                            snapshot["latest_report"] = report
                    except (TypeError, json.JSONDecodeError):
                        pass

                alert_rows = conn.execute(
                    """
                    SELECT severity, COUNT(*)
                    FROM alerts
                    WHERE acknowledged = 0
                    GROUP BY severity
                    """
                ).fetchall()

                for severity, count in alert_rows:
                    severity = str(severity).lower()

                    if severity in snapshot["unacknowledged_alerts"]:
                        snapshot["unacknowledged_alerts"][severity] = count

        except sqlite3.Error:
            # Keep the adapter available if the DB is unavailable or busy.
            pass

    # Extract fields from the latest monitoring report.
    report = snapshot.get("latest_report") or {}

    drift_data = report.get("drift")
    snapshot["drift_status"] = _derive_drift_status(drift_data)
    snapshot["feature_drift_alerts"] = _extract_feature_drift(drift_data)

    performance_data = report.get("performance")
    report_alerts = report.get("alerts")

    snapshot["performance_status"] = _derive_performance_status(
        performance_data,
        report_alerts,
    )

    if isinstance(performance_data, dict):
        snapshot["performance"] = {
            "f1": performance_data.get("f1"),
            "roc_auc": performance_data.get("roc_auc"),
            "brier": performance_data.get("brier"),
        }

    # Trigger state/events are JSON artefacts written by monitoring.trigger.
    trigger_state = _read_json(state_path) or {}

    snapshot["trigger_state"] = trigger_state
    snapshot["cooldown"] = _cooldown_snapshot(trigger_state)
    snapshot["trigger_events"] = _read_jsonl(events_path)

    return snapshot
