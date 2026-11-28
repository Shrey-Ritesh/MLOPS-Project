"""Prometheus metrics exported by the read-only dashboard adapter."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.dashboard_adapter.queries import get_dashboard_snapshot

# HTTP request metrics
HTTP_REQUESTS_TOTAL = Counter(
    "dashboard_adapter_http_requests_total",
    "Total HTTP requests handled by the dashboard adapter",
    ["method", "endpoint", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "dashboard_adapter_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Monitoring metrics
MONITORING_WINDOWS_TOTAL = Gauge(
    "dashboard_monitoring_windows_total",
    "Total number of monitoring windows recorded",
)
ACTIVE_ALERTS = Gauge(
    "dashboard_active_alerts",
    "Number of currently unacknowledged monitoring alerts",
)

# Model metrics
MODEL_PREDICTIONS_TOTAL = Gauge(
    "dashboard_model_predictions_total",
    "Total number of model predictions",
)
MODEL_DRIFT_SCORE = Gauge(
    "dashboard_model_drift_score",
    "Latest prediction-level population stability index",
)

# Existing dashboard metrics
MONITORING_WINDOW_STATUS = Gauge(
    "dashboard_monitoring_window_status",
    "Status of the latest monitoring window (ok=0, alert=1, insufficient_data=2)",
)
LATEST_P95_LATENCY = Gauge(
    "dashboard_latest_p95_latency_ms",
    "P95 inference latency from the latest monitoring window in milliseconds",
)
LATEST_FALLBACK_RATE = Gauge(
    "dashboard_latest_fallback_rate",
    "Fallback rate from the latest monitoring window",
)
LATEST_MEAN_PREDICTION_SCORE = Gauge(
    "dashboard_latest_mean_prediction_score",
    "Mean prediction score from the latest monitoring window",
)
PREDICTION_SCORE_BIN_COUNT = Gauge(
    "dashboard_prediction_score_bin_count",
    "Prediction count per score bin in the latest monitoring window",
    ["bin"],
)
UNACKNOWLEDGED_ALERTS = Gauge(
    "dashboard_unacknowledged_alerts",
    "Number of unacknowledged monitoring alerts by severity",
    ["severity"],
)
MONITORING_WINDOW_AVAILABLE = Gauge(
    "dashboard_monitoring_window_available",
    "Whether a monitoring window is available (1=yes, 0=no)",
)
LATEST_REPORT_AVAILABLE = Gauge(
    "dashboard_latest_report_available",
    "Whether a monitoring report is available (1=yes, 0=no)",
)
DRIFT_SCORE_AVAILABLE = Gauge(
    "dashboard_drift_score_available",
    "Whether a numeric prediction PSI is available (1=yes, 0=no)",
)

# Drift monitoring
DRIFT_EVALUATION_STATUS = Gauge(
    "dashboard_drift_evaluation_status",
    "Drift evaluation status (ok=0, alert=1, insufficient_data=2, unavailable=-1)",
)

FEATURE_DRIFT_PSI = Gauge(
    "dashboard_feature_drift_psi",
    "Feature-level population stability index",
    ["feature"],
)

FEATURE_DRIFT_ALERT = Gauge(
    "dashboard_feature_drift_alert",
    "Whether a feature has a drift alert (1=yes, 0=no)",
    ["feature"],
)

# Model performance monitoring
PERFORMANCE_EVALUATION_STATUS = Gauge(
    "dashboard_performance_evaluation_status",
    "Performance evaluation status (ok=0, alert=1, insufficient_data=2, unavailable=-1)",
)

LATEST_MODEL_F1 = Gauge(
    "dashboard_latest_model_f1",
    "Latest model F1 score",
)

LATEST_MODEL_ROC_AUC = Gauge(
    "dashboard_latest_model_roc_auc",
    "Latest model ROC-AUC score",
)

LATEST_MODEL_BRIER = Gauge(
    "dashboard_latest_model_brier",
    "Latest model Brier score",
)

# Retraining trigger monitoring
RETRAINING_TRIGGER_TOTAL = Gauge(
    "dashboard_retraining_trigger_total",
    "Total number of retraining triggers recorded",
)

RETRAINING_COOLDOWN_ACTIVE = Gauge(
    "dashboard_retraining_cooldown_active",
    "Whether retraining cooldown is active (1=yes, 0=no)",
)

RETRAINING_COOLDOWN_REMAINING = Gauge(
    "dashboard_retraining_cooldown_remaining_seconds",
    "Remaining retraining cooldown in seconds",
)

RETRAINING_RUNS_TODAY = Gauge(
    "dashboard_retraining_runs_today",
    "Number of retraining trigger runs today",
)

LAST_RETRAINING_TRIGGER_STATUS = Gauge(
    "dashboard_last_retraining_trigger_status",
    "Status of the latest retraining trigger (dry_run=1, executed=2, blocked=3, other=0)",
)


def get_metrics():
    """Generate the Prometheus exposition payload."""
    refresh_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST


def _set_or_nan(metric: Gauge, value) -> None:
    """Set a numeric metric, or NaN when its source value is unavailable."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        metric.set(value)
    else:
        metric.set(float("nan"))


def refresh_metrics() -> None:
    """Refresh all adapter gauges from one consistent SQLite snapshot."""
    snapshot = get_dashboard_snapshot()

    MODEL_PREDICTIONS_TOTAL.set(snapshot["total_predictions"])
    MONITORING_WINDOWS_TOTAL.set(snapshot["total_monitoring_windows"])

    status_mapping = {
        "ok": 0,
        "alert": 1,
        "insufficient_data": 2,
    }

    evaluation_status_mapping = {
        "ok": 0,
        "alert": 1,
        "insufficient_data": 2,
        "unavailable": -1,
    }

    window_status = snapshot["latest_window_status"]
    MONITORING_WINDOW_AVAILABLE.set(1 if window_status is not None else 0)
    MONITORING_WINDOW_STATUS.set(status_mapping.get(window_status, -1))

    report = snapshot.get("latest_report")
    report_available = isinstance(report, dict)
    LATEST_REPORT_AVAILABLE.set(1 if report_available else 0)

    # Reset report-derived gauges on every scrape to prevent stale values.
    _set_or_nan(LATEST_P95_LATENCY, None)
    _set_or_nan(LATEST_FALLBACK_RATE, None)
    _set_or_nan(LATEST_MEAN_PREDICTION_SCORE, None)
    _set_or_nan(MODEL_DRIFT_SCORE, None)
    DRIFT_SCORE_AVAILABLE.set(0)
    PREDICTION_SCORE_BIN_COUNT.clear()

    if report_available:
        health = report.get("health") or {}
        distribution = report.get("prediction_distribution") or {}

        _set_or_nan(LATEST_P95_LATENCY, health.get("p95"))
        _set_or_nan(LATEST_FALLBACK_RATE, health.get("fallback_rate"))
        _set_or_nan(LATEST_MEAN_PREDICTION_SCORE, distribution.get("mean_score"))

        bins = distribution.get("score_bins") or []
        counts = distribution.get("score_counts") or []
        for index, count in enumerate(counts):
            if index + 1 < len(bins):
                label = f"{bins[index]}-{bins[index + 1]}"
                if isinstance(count, (int, float)) and not isinstance(count, bool):
                    PREDICTION_SCORE_BIN_COUNT.labels(bin=label).set(count)

        drift_score = report.get("prediction_psi")
        if isinstance(drift_score, (int, float)) and not isinstance(drift_score, bool):
            MODEL_DRIFT_SCORE.set(drift_score)
            DRIFT_SCORE_AVAILABLE.set(1)

        # Drift evaluation status
    drift_status = snapshot.get("drift_status", "unavailable")
    DRIFT_EVALUATION_STATUS.set(
        evaluation_status_mapping.get(drift_status, -1)
    )

    # Feature-level drift metrics
    FEATURE_DRIFT_PSI.clear()
    FEATURE_DRIFT_ALERT.clear()

    for item in snapshot.get("feature_drift_alerts", []):
        feature = str(item.get("metric") or item.get("feature") or "unknown")
        value = item.get("value")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            FEATURE_DRIFT_PSI.labels(feature=feature).set(value)

        severity = str(item.get("severity", "")).lower()
        FEATURE_DRIFT_ALERT.labels(feature=feature).set(
            1 if severity in ("warning", "critical", "alert") else 0
        )

    # Model performance metrics
    performance_status = snapshot.get(
        "performance_status", "unavailable"
    )
    PERFORMANCE_EVALUATION_STATUS.set(
        evaluation_status_mapping.get(performance_status, -1)
    )

    performance = snapshot.get("performance") or {}

    _set_or_nan(LATEST_MODEL_F1, performance.get("f1"))
    _set_or_nan(LATEST_MODEL_ROC_AUC, performance.get("roc_auc"))
    _set_or_nan(LATEST_MODEL_BRIER, performance.get("brier"))

    # Retraining trigger metrics
        # Retraining trigger metrics
    trigger_events = snapshot.get("trigger_events", [])
    trigger_state = snapshot.get("trigger_state", {})
    cooldown = snapshot.get("cooldown", {})

    RETRAINING_TRIGGER_TOTAL.set(
        trigger_state.get("total_runs", 0)
    )

    RETRAINING_COOLDOWN_ACTIVE.set(
        1 if cooldown.get("cooldown_active", False) else 0
    )

    RETRAINING_COOLDOWN_REMAINING.set(
        cooldown.get("cooldown_remaining_seconds", 0)
    )

    RETRAINING_RUNS_TODAY.set(
        trigger_state.get("runs_today", 0)
    )

    latest_trigger = trigger_events[-1] if trigger_events else {}
    trigger_status = str(latest_trigger.get("status", "")).lower()

    trigger_status_mapping = {
        "dry_run": 1,
        "executed": 2,
        "blocked": 3,
        "success": 4
    }

    LAST_RETRAINING_TRIGGER_STATUS.set(
        trigger_status_mapping.get(trigger_status, 0)
    )

    alerts = snapshot["unacknowledged_alerts"]
    warning_count = alerts.get("warning", 0)
    critical_count = alerts.get("critical", 0)
    UNACKNOWLEDGED_ALERTS.labels(severity="warning").set(warning_count)
    UNACKNOWLEDGED_ALERTS.labels(severity="critical").set(critical_count)
    ACTIVE_ALERTS.set(warning_count + critical_count)
