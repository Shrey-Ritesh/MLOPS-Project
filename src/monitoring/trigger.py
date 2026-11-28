"""Automated Retraining Trigger Engine for Phase 3 (Owner: Person 2).

Conforms to Section 4.3 contract:
Evaluates monitoring window reports against configs/monitoring/retrain_trigger.yaml.
When drift/degradation criteria hold, it snapshots labelled monitoring data for lineage,
executes the configured retraining pipeline (dvc repro train_challenger), logs lineage
and registers the challenger model in MLflow, and enforces cooldown constraints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
import uuid
import yaml

from src.monitoring import db
from src.monitoring.run_window import iso, parse_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RetrainTrigger")

DEFAULT_CONFIG_PATH = Path("configs/monitoring/retrain_trigger.yaml")
DEFAULT_THRESHOLDS_PATH = Path("configs/monitoring/thresholds.yaml")
DEFAULT_STATE_PATH = Path("reports/monitoring/trigger_state.json")
DEFAULT_EVENTS_LOG = Path("reports/monitoring/trigger_events.jsonl")
# Overridable so the monitor container can keep retraining outputs on its persistent volume.
DEFAULT_CHALLENGER_DIR = Path(os.environ.get("TRIGGER_CHALLENGER_DIR", "models/track_b/challenger"))
DEFAULT_CHALLENGER_COMPARISON = Path(
    os.environ.get("TRIGGER_CHALLENGER_COMPARISON", "reports/track_b/challenger_comparison.json")
)
DEFAULT_SNAPSHOT_DIR = Path(os.environ.get("TRIGGER_SNAPSHOT_DIR", "data/snapshots"))


def load_trigger_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trigger config not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_baseline_f1(thresholds_path: Path = DEFAULT_THRESHOLDS_PATH) -> float:
    """Read champion baseline F1 from thresholds.yaml to stay aligned with drift_window.py."""
    if not thresholds_path.exists():
        return 0.6828
    thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}
    return float(thresholds.get("performance_degradation", {}).get("baseline_f1", 0.6828))


def load_trigger_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_fired_at": None,
            "cooldown_until": None,
            "runs_today": 0,
            "last_run_date": None,
            "total_runs": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_trigger_state(state: dict[str, Any], path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_trigger_event(event: dict[str, Any], log_path: Path | None = None) -> None:
    target_path = DEFAULT_EVENTS_LOG if log_path is None else log_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def check_cooldown(state: dict[str, Any], now: datetime, max_runs_per_day: int = 3) -> tuple[bool, str]:
    """Check whether cooldown or daily ceiling prevents triggering."""
    cooldown_str = state.get("cooldown_until")
    if cooldown_str:
        cooldown_until = parse_iso(cooldown_str)
        if now < cooldown_until:
            remaining = int((cooldown_until - now).total_seconds())
            return False, f"Cooldown active until {cooldown_str} ({remaining}s remaining)"

    today_str = now.strftime("%Y-%m-%d")
    if state.get("last_run_date") == today_str:
        if state.get("runs_today", 0) >= max_runs_per_day:
            return False, f"Daily limit reached (max {max_runs_per_day} runs per day)"
    return True, "Cooldown check passed"


def evaluate_trigger_conditions(
    report: dict[str, Any],
    cfg: dict[str, Any],
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
) -> tuple[bool, str | None, list[str]]:
    """Evaluate whether window report justifies firing retraining trigger."""
    guardrails = cfg.get("guardrails", {})
    min_records = guardrails.get("min_window_records", 50)
    if report.get("status") == "insufficient_data" or report.get("n_predictions", 0) < min_records:
        return False, None, []

    conds = cfg.get("trigger_conditions", {})
    min_feature_alerts = conds.get("min_feature_alerts", 2)
    pred_psi_alert = conds.get("prediction_psi_alert", 0.20)
    perf_f1_drop = conds.get("performance_f1_drop", 0.05)
    min_labeled = conds.get("min_labeled_for_perf", 30)

    alerts = report.get("alerts", [])
    has_health_outage = any(
        a.get("category") == "health"
        and a.get("severity") in ("critical", "ALERT")
        and a.get("metric") in ("fallback_rate", "error_rate", "timeout_rate")
        for a in alerts
    )
    feature_alerts = [
        a
        for a in alerts
        if a.get("category") == "drift"
        and not str(a.get("metric", "")).startswith("PREDICTION_DRIFT")
        and a.get("severity") in ("critical", "ALERT")
    ]

    # Rule 1: Feature drift count (excludes prediction-drift alerts)
    if len(feature_alerts) >= min_feature_alerts:
        features_drifted = [str(a.get("metric")) for a in feature_alerts]
        return True, f"feature_drift_exceeded_threshold ({', '.join(features_drifted)})", [report.get("window_id", "unknown")]

    # Rule 2: Prediction PSI (suppressed during inference outage/fallback incidents)
    pred_psi = report.get("prediction_psi")
    if not has_health_outage and isinstance(pred_psi, (int, float)) and pred_psi >= pred_psi_alert:
        return True, f"prediction_psi_exceeded_threshold ({pred_psi} >= {pred_psi_alert})", [report.get("window_id", "unknown")]

    # Rule 3: Performance degradation (if enough labels exist and not an inference outage)
    perf = report.get("performance")
    n_labeled = report.get("n_labelled", 0)
    if not has_health_outage and isinstance(perf, dict) and "f1" in perf and n_labeled >= min_labeled:
        base_f1 = load_baseline_f1(thresholds_path)
        f1_drop = base_f1 - perf["f1"]
        if f1_drop >= perf_f1_drop:
            return True, f"performance_f1_drop_exceeded_threshold ({f1_drop:.4f} >= {perf_f1_drop})", [report.get("window_id", "unknown")]

    return False, None, []


def snapshot_data(source_csv: Path, out_dir: Path | None = None) -> tuple[Path, str]:
    """Create reproducible data snapshot from a CSV and return (path, sha256)."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = out_dir / f"snapshot_{stamp}.csv"
    data_bytes = source_csv.read_bytes()
    snap_path.write_bytes(data_bytes)
    sha256 = hashlib.sha256(data_bytes).hexdigest()
    return snap_path, sha256


def snapshot_from_monitoring(
    db_path: Path | str,
    start: str,
    end: str,
    out_dir: Path | None = None,
) -> tuple[Path, str]:
    """Snapshot labelled prediction rows for [start, end) from the monitoring DB."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_SNAPSHOT_DIR
    import pandas as pd

    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        df = pd.read_sql_query(
            "SELECT features, label_observed FROM predictions "
            "WHERE label_status = 'joined' AND occurred_at >= ? AND occurred_at < ?",
            conn,
            params=(start, end),
        )

    if df.empty:
        raise ValueError(f"No joined prediction rows found in [{start}, {end}) in {db_path}")

    flat = pd.json_normalize(df["features"].map(json.loads))
    flat["switch_to_lower_difficulty"] = df["label_observed"].astype(int).to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = out_dir / f"snapshot_{stamp}_{uuid.uuid4().hex[:6]}.csv"
    csv_bytes = flat.to_csv(index=False).encode("utf-8")
    snap_path.write_bytes(csv_bytes)
    sha256 = hashlib.sha256(csv_bytes).hexdigest()
    return snap_path, sha256


def _latest_version_number() -> str | None:
    """Return the highest registered MLflow model version number, or None if unavailable."""
    try:
        from mlflow import MlflowClient
        from src.track_b import tracking

        tracking.configure()
        client = MlflowClient()
        versions = list(client.search_model_versions(f"name='{tracking.REGISTERED_MODEL_NAME}'"))
        if not versions:
            return None
        return str(max(int(v.version) for v in versions))
    except Exception as exc:
        logger.warning("Could not query MLflow model versions (%s)", exc)
        return None


def _tag_challenger_in_mlflow(
    version: str,
    trigger_id: str,
    evidence_windows: list[str],
    snapshot_sha: str,
) -> str | None:
    """Attach the 'challenger' alias and trigger lineage tags to a specific MLflow model version."""
    try:
        from mlflow import MlflowClient
        from src.track_b import tracking

        tracking.configure()
        client = MlflowClient()
        client.set_registered_model_alias(tracking.REGISTERED_MODEL_NAME, "challenger", str(version))
        client.set_model_version_tag(tracking.REGISTERED_MODEL_NAME, str(version), "trigger_id", trigger_id)
        client.set_model_version_tag(tracking.REGISTERED_MODEL_NAME, str(version), "snapshot_sha256", snapshot_sha)
        client.set_model_version_tag(
            tracking.REGISTERED_MODEL_NAME, str(version), "evidence_window_ids", ",".join(evidence_windows)
        )
        logger.info("Set MLflow alias 'challenger' -> %s v%s", tracking.REGISTERED_MODEL_NAME, version)
        return str(version)
    except Exception as exc:
        logger.warning("Could not tag challenger alias in MLflow (%s)", exc)
        return None


def execute_retraining_pipeline(
    trigger_id: str,
    evidence_windows: list[str],
    snapshot_sha: str,
    cfg: dict[str, Any] | None = None,
    challenger_dir: Path | None = None,
    comparison_out: Path | None = None,
) -> dict[str, Any]:
    """Execute configured retraining stage and register new challenger version in MLflow."""
    challenger_dir = Path(challenger_dir) if challenger_dir is not None else DEFAULT_CHALLENGER_DIR
    comparison_out = Path(comparison_out) if comparison_out is not None else DEFAULT_CHALLENGER_COMPARISON
    logger.info("Executing retraining pipeline for trigger %s...", trigger_id)

    pipeline_cmd = (cfg or {}).get("trigger_policy", {}).get("pipeline_command", "dvc repro train_challenger")
    cmd = shlex.split(pipeline_cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Pipeline command completed:\n%s", proc.stdout)
    except Exception as e:
        logger.warning("Pipeline command %r failed (%s). Running fallback challenger training...", cmd, e)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.track_b.train_candidate",
                "--output-dir",
                str(challenger_dir),
                "--comparison-out",
                str(comparison_out),
            ],
            check=True,
        )

    before = _latest_version_number()
    reg_cmd = [
        sys.executable,
        "-m",
        "src.track_b.register_mlflow",
        "--challenger-only",
        "--candidate-dir",
        str(challenger_dir),
        "--comparison-report",
        str(comparison_out),
        "--trigger-id",
        trigger_id,
        "--snapshot-sha256",
        snapshot_sha,
    ]
    reg_proc = subprocess.run(reg_cmd, capture_output=True, text=True)
    registered_version: str | None = None
    if reg_proc.returncode == 0:
        logger.info("MLflow registration result:\n%s", reg_proc.stdout.strip())
        after = _latest_version_number()
        if after and after != before:
            registered_version = _tag_challenger_in_mlflow(
                version=after,
                trigger_id=trigger_id,
                evidence_windows=evidence_windows,
                snapshot_sha=snapshot_sha,
            )
    else:
        logger.warning(
            "MLflow registration exited with %s (stderr: %s)",
            reg_proc.returncode,
            (reg_proc.stderr or "").strip(),
        )

    challenger_card_path = challenger_dir / "model_card.json"
    if challenger_card_path.exists():
        card = json.loads(challenger_card_path.read_text(encoding="utf-8"))
        model_version = card.get("model_id", "challenger_v2")
    else:
        model_version = "challenger_v2"

    return {
        "pipeline_run_id": f"run_{uuid.uuid4().hex[:12]}",
        "challenger_model_version": model_version,
        "mlflow_model_version": registered_version,
    }


def _create_trigger_snapshot(
    report: dict[str, Any],
    db_path: Path | str | None,
) -> tuple[Path, str, str]:
    """Create snapshot from monitoring DB joined rows when available, else fallback dataset."""
    start = report.get("start")
    end = report.get("end")
    candidate_db = db_path if db_path is not None else Path(db.db_path_from_env())
    if start and end and (db_path is not None or Path(candidate_db).exists()):
        try:
            snap_path, snap_sha = snapshot_from_monitoring(candidate_db, start, end)
            return snap_path, snap_sha, "monitoring_db"
        except Exception as exc:
            logger.warning("Monitoring DB snapshot unavailable (%s); falling back to shifted CSV.", exc)

    source_dataset = Path("data/synthetic/learner_interactions_shifted_v1.csv")
    snap_path, snap_sha = snapshot_data(source_dataset)
    return snap_path, snap_sha, "static_fallback"


def process_window_for_trigger(
    report: dict[str, Any],
    cfg_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path | None = None,
    events_path: Path | None = None,
    simulated_mode: bool = False,
    execute_pipeline: bool = True,
    db_path: Path | str | None = None,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
) -> dict[str, Any] | None:
    """Evaluate a window report and trigger retraining if policy matches."""
    cfg = load_trigger_config(cfg_path)
    if not cfg.get("trigger_policy", {}).get("enabled", True):
        logger.info("Retraining trigger policy is disabled in config.")
        return None

    target_state_path = DEFAULT_STATE_PATH if state_path is None else state_path
    target_events_path = DEFAULT_EVENTS_LOG if events_path is None else events_path
    now = parse_iso(report["end"]) if report.get("end") else datetime.now(timezone.utc)

    # Evaluate trigger conditions BEFORE cooldown so clean windows are not counted as blocked
    should_fire, condition_met, evidence_windows = evaluate_trigger_conditions(
        report, cfg, thresholds_path=thresholds_path
    )
    if not should_fire:
        logger.info("Window %s: no trigger conditions met.", report.get("window_id"))
        return None

    state = load_trigger_state(target_state_path)

    # Check cooldown and daily cap from config
    guardrails = cfg.get("guardrails", {})
    max_runs_per_day = guardrails.get("max_runs_per_day", 3)
    can_run, reason = check_cooldown(state, now, max_runs_per_day=max_runs_per_day)
    if not can_run:
        logger.info("Trigger skipped: %s", reason)
        return {"status": "cooldown_blocked", "reason": reason}

    # Condition met and cooldown clear -> fire trigger
    trigger_id = f"trig_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    logger.warning(">>> RETRAINING TRIGGER FIRED: %s (%s) <<<", trigger_id, condition_met)

    _, snap_sha, snapshot_source = _create_trigger_snapshot(report, db_path)

    # Calculate cooldown from config (window-time based)
    window_minutes = int(os.environ.get("MONITORING_WINDOW_MINUTES", 15))
    if simulated_mode:
        cooldown_secs = int(guardrails.get("simulated_cooldown_windows", 5)) * window_minutes * 60
    else:
        cooldown_secs = int(guardrails.get("cooldown_seconds", 21600))
    cooldown_until = now + timedelta(seconds=cooldown_secs)

    try:
        if execute_pipeline:
            pipeline_res = execute_retraining_pipeline(
                trigger_id=trigger_id,
                evidence_windows=evidence_windows,
                snapshot_sha=snap_sha,
                cfg=cfg,
            )
        else:
            pipeline_res = {
                "pipeline_run_id": "dry_run",
                "challenger_model_version": None,
                "mlflow_model_version": None,
            }

        note = (
            "Snapshot recorded for lineage; candidate retrained on the documented shifted dataset"
            if execute_pipeline
            else "Snapshot recorded for lineage; pipeline execution skipped (dry_run)"
        )
        event = {
            "trigger_id": trigger_id,
            "fired_at": iso(now),
            "condition_met": condition_met,
            "evidence_window_ids": evidence_windows,
            "data_snapshot_sha256": snap_sha,
            "snapshot_source": snapshot_source,
            "retraining_dataset_note": note,
            "cooldown_until": iso(cooldown_until),
            "pipeline_run_id": pipeline_res["pipeline_run_id"],
            "challenger_model_version": pipeline_res["challenger_model_version"],
            "mlflow_model_version": pipeline_res.get("mlflow_model_version"),
            "status": "success",
        }
    except Exception as e:
        logger.error("Retraining execution failed: %s", e)
        event = {
            "trigger_id": trigger_id,
            "fired_at": iso(now),
            "condition_met": condition_met,
            "evidence_window_ids": evidence_windows,
            "data_snapshot_sha256": snap_sha,
            "snapshot_source": snapshot_source,
            "cooldown_until": iso(cooldown_until),
            "status": f"failed: {e}",
        }

    # Record state and event
    today_str = now.strftime("%Y-%m-%d")
    state["last_fired_at"] = iso(now)
    state["cooldown_until"] = iso(cooldown_until)
    state["runs_today"] = state.get("runs_today", 0) + 1 if state.get("last_run_date") == today_str else 1
    state["last_run_date"] = today_str
    state["total_runs"] = state.get("total_runs", 0) + 1
    save_trigger_state(state, target_state_path)

    log_trigger_event(event, log_path=target_events_path)
    logger.info("Trigger event recorded: %s", event["trigger_id"])
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-report", type=Path, required=True,
                        help="Path to window report JSON to evaluate")
    parser.add_argument("--db", type=Path, default=None,
                        help="Monitoring SQLite DB path for snapshotting joined rows")
    parser.add_argument("--state-path", type=Path, default=None,
                        help="Trigger state JSON path")
    parser.add_argument("--events-path", type=Path, default=None,
                        help="Trigger events JSONL path")
    parser.add_argument("--simulated", action="store_true",
                        help="Enable simulated mode with window-based cooldown")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip executing DVC/MLflow retraining pipeline")
    args = parser.parse_args()

    report = json.loads(args.window_report.read_text(encoding="utf-8"))
    event = process_window_for_trigger(
        report,
        state_path=args.state_path,
        events_path=args.events_path,
        simulated_mode=args.simulated,
        execute_pipeline=not args.dry_run,
        db_path=args.db,
    )
    if event:
        print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
