"""Evaluate one monitoring window over the prediction log and record its report and alerts.

For window [start, end) it computes:
  * health:     p50/p95 latency, error / timeout / fallback rates          (Person 1)
  * validity:   schema failures, impossible values, unseen topics           (Person 1)
  * prediction distribution: score histogram and positive rate              (Person 1)
  * drift / prediction PSI / performance via ``drift_hook``                (Person 2)

and writes:
  * a ``monitoring_windows`` row with the full report JSON,
  * one ``alerts`` row per threshold breach,
  * ``<output_dir>/windows/<window_id>/report.json`` and ``predictions.csv`` (fixed snapshot).

Windows with too few predictions are ``insufficient_data`` for rate checks, never "healthy".

    python -m src.monitoring.run_window --start 2026-11-26T09:00:00Z --end 2026-11-26T09:15:00Z
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.monitoring import db, drift_hook
from src.monitoring.app_logging import log_event
from src.monitoring.contract import Contract, load_contract
from src.monitoring.schema import SCHEMA_VERSION

DEFAULT_CONFIG = "configs/monitoring/health.yaml"
REPORT_VERSION = "1.1.0"  # 1.1.0: contract_id, n_label_excluded, validity.contract_mismatch
INSUFFICIENT = "insufficient_data"


# ---- helpers ---------------------------------------------------------------------------------

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def window_id_for(start: datetime, end: datetime) -> str:
    minutes = int((end - start).total_seconds() // 60)
    return f"w-{start.astimezone(timezone.utc):%Y%m%dT%H%MZ}-{minutes}m"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Breach:
    severity: str
    category: str
    metric: str
    value: float
    threshold: float


def _check(category: str, metric: str, value: float | None, levels: dict[str, float]) -> Breach | None:
    """Critical if value >= critical; warning if value > warning (so a warning of 0 means 'any')."""
    if value is None:
        return None
    if value >= levels["critical"]:
        return Breach("critical", category, metric, value, levels["critical"])
    if value > levels["warning"]:
        return Breach("warning", category, metric, value, levels["warning"])
    return None


# ---- data ------------------------------------------------------------------------------------

def fetch_rows(conn, start: datetime, end: datetime) -> list[dict[str, Any]]:
    conn.row_factory = _dict_factory
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE occurred_at >= ? AND occurred_at < ? ORDER BY occurred_at",
            (iso(start), iso(end)),
        ).fetchall()
    finally:
        conn.row_factory = None
    for row in rows:
        try:
            row["features"] = json.loads(row["features"])
        except (TypeError, ValueError):
            row["features"] = None
    return rows


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


# ---- metrics ---------------------------------------------------------------------------------

def compute_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    latencies = np.array([r["latency_ms"] for r in rows if r["latency_ms"] is not None], dtype=float)
    count = lambda status: sum(1 for r in rows if r["status"] == status)  # noqa: E731
    by_source: dict[str, int] = {}
    by_version: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_version[r["model_version"]] = by_version.get(r["model_version"], 0) + 1
    return {
        "p50": round(float(np.percentile(latencies, 50)), 3) if latencies.size else None,
        "p95": round(float(np.percentile(latencies, 95)), 3) if latencies.size else None,
        "error_rate": round(count("error") / n, 4) if n else None,
        "timeout_rate": round(count("timeout") / n, 4) if n else None,
        "fallback_rate": round(sum(1 for r in rows if r["fallback"]) / n, 4) if n else None,
        "by_source": by_source,
        "by_model_version": by_version,
    }


def compute_validity(rows: list[dict[str, Any]], contract: Contract) -> dict[str, Any]:
    """Check every row against the contract's feature types and limits.

    A row made under another contract (e.g. during a model switch) counts as a schema failure and
    is reported separately as ``contract_mismatch``. "Unseen" = a value outside a feature's
    vocabulary (today only ``topic`` has one).
    """
    names = set(contract.feature_names)
    schema_failures = out_of_range = unseen = mismatch = 0
    unseen_names: dict[str, int] = {}
    for r in rows:
        feats = r["features"]
        if r.get("contract_id") != contract.contract_id:
            mismatch += 1
            schema_failures += 1
            continue
        if not isinstance(feats, dict) or set(feats) != names or r.get("schema_version") != SCHEMA_VERSION:
            schema_failures += 1
            continue
        problems = {f.name: f.check(feats.get(f.name)) for f in contract.features}
        out_of_range += any(p == "out_of_range" for p in problems.values())
        unseen_values = [str(feats.get(name) or "unknown") for name, p in problems.items() if p == "unseen"]
        if unseen_values:
            unseen += 1
            for value in unseen_values:
                unseen_names[value] = unseen_names.get(value, 0) + 1
    n = len(rows)
    rate = lambda k: round(k / n, 4) if n else None  # noqa: E731
    return {
        "contract_mismatch": mismatch,
        "schema_failures": schema_failures,
        "out_of_range": out_of_range,
        "unseen_topics": unseen,
        "schema_failure_rate": rate(schema_failures),
        "out_of_range_rate": rate(out_of_range),
        "unseen_topic_rate": rate(unseen),
        "unseen_topic_names": dict(sorted(unseen_names.items(), key=lambda kv: -kv[1])[:10]),
    }


def compute_prediction_distribution(rows: list[dict[str, Any]], bins: list[float]) -> dict[str, Any]:
    scores = np.array([r["score"] for r in rows if r["score"] is not None], dtype=float)
    labels = [r["label"] for r in rows if r["label"] is not None]
    counts, _ = np.histogram(scores, bins=np.array(bins, dtype=float)) if scores.size else (np.zeros(len(bins) - 1), None)
    return {
        "n_scored": int(scores.size),
        "mean_score": round(float(scores.mean()), 4) if scores.size else None,
        "positive_rate": round(sum(labels) / len(labels), 4) if labels else None,
        "score_bins": bins,
        "score_counts": [int(c) for c in counts],
    }


# ---- window ----------------------------------------------------------------------------------

def evaluate_window(
    db_path: str | Path,
    start: datetime,
    end: datetime,
    *,
    config: dict[str, Any] | None = None,
    evaluator: drift_hook.Evaluator | None | str = "auto",
    output_dir: str | Path | None = None,
    now: datetime | None = None,
    contract: Contract | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    contract = contract or load_contract()
    now = now or datetime.now(timezone.utc)
    wid = window_id_for(start, end)
    if evaluator == "auto":
        evaluator = drift_hook.load_evaluator()

    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        rows = fetch_rows(conn, start, end)

    n = len(rows)
    n_labelled = sum(1 for r in rows if r["label_status"] == "joined")
    enough = n >= cfg["min_predictions_for_rates"]

    health = compute_health(rows)
    validity = compute_validity(rows, contract)
    distribution = compute_prediction_distribution(rows, cfg["prediction_distribution"]["score_bins"])
    model_part = drift_hook.evaluate(rows, evaluator)

    breaches: list[Breach] = []
    if enough:
        for metric in ("error_rate", "timeout_rate", "fallback_rate"):
            breaches.append(_check("health", metric, health[metric], cfg["health"][metric]))
        breaches.append(_check("health", "latency_p95_ms", health["p95"], cfg["health"]["latency_p95_ms"]))
        for metric in ("schema_failure_rate", "out_of_range_rate", "unseen_topic_rate"):
            breaches.append(_check("validity", metric, validity[metric], cfg["validity"][metric]))
    breaches = [b for b in breaches if b]

    alerts = [
        {"severity": b.severity, "category": b.category, "metric": b.metric, "value": b.value,
         "threshold": b.threshold,
         "message": f"{b.metric}={b.value} breached {b.severity} threshold {b.threshold}"}
        for b in breaches
    ] + [_normalise_alert(a) for a in model_part["alerts"]]

    if not enough:
        status = INSUFFICIENT
    else:
        status = "alert" if alerts else "ok"

    champion_versions = sorted(health["by_model_version"], key=lambda v: -health["by_model_version"][v])
    report = {
        "report_version": REPORT_VERSION,
        "window_id": wid,
        "contract_id": contract.contract_id,
        "start": iso(start),
        "end": iso(end),
        "evaluated_at": iso(now),
        "status": status,
        "n_predictions": n,
        "n_labelled": n_labelled,
        "n_label_pending": sum(1 for r in rows if r["label_status"] == "pending"),
        "n_label_expired": sum(1 for r in rows if r["label_status"] == "expired"),
        "n_label_excluded": sum(1 for r in rows if r["label_status"] == "excluded"),
        "health": health if enough else {**health, "status": INSUFFICIENT},
        "validity": validity if enough else {**validity, "status": INSUFFICIENT},
        "prediction_distribution": distribution,
        "drift": model_part["drift"],
        "prediction_psi": model_part["prediction_psi"],
        "performance": model_part["performance"],
        "alerts": alerts,
        "reference_dataset_sha256": model_part.get("reference_dataset_sha256"),
        "champion_version": champion_versions[0] if champion_versions else None,
        "thresholds": {"min_predictions_for_rates": cfg["min_predictions_for_rates"],
                       "config_version": cfg.get("version")},
    }
    if "error_type" in model_part:
        report["drift_error_type"] = model_part["error_type"]

    _persist(db_path, report, alerts, now)
    if output_dir:
        _write_snapshot(Path(output_dir), report, rows, contract)
    log_event("window_evaluated", service="monitoring", window_id=wid, window_status=status,
              count=n, reason=f"labelled={n_labelled} alerts={len(alerts)}")
    for a in alerts:
        log_event("alert_raised", level=logging.WARNING if a["severity"] == "warning" else logging.ERROR,
                  service="monitoring", window_id=wid, severity=a["severity"], category=a["category"],
                  metric=a["metric"], value=a["value"], threshold=a["threshold"])
    return report


def _normalise_alert(alert: dict[str, Any]) -> dict[str, Any]:
    severity = alert.get("severity", "warning")
    return {
        "severity": severity if severity in ("warning", "critical") else "warning",
        "category": str(alert.get("category", "drift")),
        "metric": str(alert.get("metric", "unknown")),
        "value": alert.get("value"),
        "threshold": alert.get("threshold"),
        "message": str(alert.get("message") or f"{alert.get('metric')} breached threshold"),
    }


def _persist(db_path, report: dict[str, Any], alerts: list[dict[str, Any]], now: datetime) -> None:
    """Idempotent per window: re-evaluating a window replaces its report and its alerts."""
    wid = report["window_id"]
    with db.session(db_path) as conn:
        conn.execute("DELETE FROM alerts WHERE window_id = ?", (wid,))
        conn.execute(
            """INSERT OR REPLACE INTO monitoring_windows
               (window_id, window_start, window_end, evaluated_at, n_predictions, n_labelled, status, report)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (wid, report["start"], report["end"], report["evaluated_at"], report["n_predictions"],
             report["n_labelled"], report["status"], json.dumps(report, separators=(",", ":"), default=str)),
        )
        conn.executemany(
            """INSERT INTO alerts (alert_id, created_at, window_id, severity, category, metric, value, threshold, message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(str(uuid.uuid4()), iso(now), wid, a["severity"], a["category"], a["metric"],
              a["value"], a["threshold"], a["message"]) for a in alerts],
        )


BASE_COLUMNS = ("prediction_id", "occurred_at", "source", "contract_id", "learner_key", "learner_seq", "question_id")
TAIL_COLUMNS = ("score", "label", "threshold", "model_name", "model_version", "fallback", "latency_ms", "status",
                "label_observed", "label_status")


def _write_snapshot(output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]],
                    contract: Contract) -> None:
    folder = output_dir / "windows" / report["window_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    columns = (*BASE_COLUMNS, *contract.feature_names, *TAIL_COLUMNS)
    with open(folder / "predictions.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = {**r, **(r["features"] if isinstance(r["features"], dict) else {})}
            writer.writerow(flat)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=os.environ.get("MONITORING_OUTPUT_DIR"))
    args = parser.parse_args(argv)
    report = evaluate_window(args.db, parse_iso(args.start), parse_iso(args.end),
                             config=load_config(args.config), output_dir=args.output_dir)
    print(json.dumps({k: report[k] for k in ("window_id", "status", "n_predictions", "n_labelled")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
