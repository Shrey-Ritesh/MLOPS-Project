"""Automated Retraining Trigger Engine for Phase 3 (Owner: Person 2).

Conforms to Section 4.3 contract:
Evaluates monitoring window reports against configs/monitoring/retrain_trigger.yaml.
When drift/degradation criteria hold, it snapshots data, executes the retraining
pipeline (dvc repro candidate), logs lineage to MLflow, registers the challenger model,
and enforces cooldown constraints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid
import yaml

from src.monitoring import db
from src.monitoring.run_window import iso, parse_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RetrainTrigger")

DEFAULT_CONFIG_PATH = Path("configs/monitoring/retrain_trigger.yaml")
DEFAULT_STATE_PATH = Path("reports/monitoring/trigger_state.json")
DEFAULT_EVENTS_LOG = Path("reports/monitoring/trigger_events.jsonl")


def load_trigger_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trigger config not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
        a for a in alerts if a.get("category") == "drift" and a.get("severity") in ("critical", "ALERT")
    ]

    # Rule 1: Feature drift count
    if len(feature_alerts) >= min_feature_alerts:
        features_drifted = [a.get("metric") for a in feature_alerts]
        return True, f"feature_drift_exceeded_threshold ({', '.join(features_drifted)})", [report.get("window_id", "unknown")]

    # Rule 2: Prediction PSI (suppressed during inference outage/fallback incidents)
    pred_psi = report.get("prediction_psi")
    if not has_health_outage and isinstance(pred_psi, (int, float)) and pred_psi >= pred_psi_alert:
        return True, f"prediction_psi_exceeded_threshold ({pred_psi} >= {pred_psi_alert})", [report.get("window_id", "unknown")]

    # Rule 3: Performance degradation (if enough labels exist and not an inference outage)
    perf = report.get("performance")
    n_labeled = report.get("n_labelled", 0)
    if not has_health_outage and isinstance(perf, dict) and "f1" in perf and n_labeled >= min_labeled:
        base_f1 = 0.6828  # Champion baseline F1
        f1_drop = base_f1 - perf["f1"]
        if f1_drop >= perf_f1_drop:
            return True, f"performance_f1_drop_exceeded_threshold ({f1_drop:.4f} >= {perf_f1_drop})", [report.get("window_id", "unknown")]

    return False, None, []


def snapshot_data(source_csv: Path, out_dir: Path = Path("data/snapshots")) -> tuple[Path, str]:
    """Create reproducible data snapshot and return (path, sha256)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = out_dir / f"snapshot_{stamp}.csv"
    data_bytes = source_csv.read_bytes()
    snap_path.write_bytes(data_bytes)
    sha256 = hashlib.sha256(data_bytes).hexdigest()
    return snap_path, sha256


def execute_retraining_pipeline(
    trigger_id: str,
    evidence_windows: list[str],
    snapshot_sha: str,
) -> dict[str, Any]:
    """Execute dvc repro candidate and register challenger in MLflow."""
    logger.info("Executing retraining pipeline for trigger %s...", trigger_id)

    # 1. Run DVC repro candidate
    cmd = ["dvc", "repro", "candidate"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("DVC repro completed:\n%s", proc.stdout)
    except Exception as e:
        logger.warning("dvc repro failed (%s). Running fallback candidate training script...", e)
        # Fallback to direct python script invocation using active interpreter
        subprocess.run([sys.executable, "-m", "src.track_b.train_candidate"], check=True)

    # 2. Register challenger in MLflow (best-effort; do not crash if MLflow store is unavailable)
    reg_cmd = [sys.executable, "-m", "src.track_b.register_mlflow"]
    reg_proc = subprocess.run(reg_cmd, capture_output=True, text=True)
    if reg_proc.returncode == 0:
        logger.info("MLflow registration result:\n%s", reg_proc.stdout.strip())
    else:
        logger.warning(
            "MLflow registration exited with %s (stderr: %s)",
            reg_proc.returncode,
            (reg_proc.stderr or "").strip(),
        )

    registered_version = _tag_challenger_in_mlflow(
        trigger_id=trigger_id,
        evidence_windows=evidence_windows,
        snapshot_sha=snapshot_sha,
    )

    # 3. Read challenger model card to get version and run ID
    challenger_card_path = Path("models/track_b/candidate/model_card.json")
    if challenger_card_path.exists():
        card = json.loads(challenger_card_path.read_text(encoding="utf-8"))
        model_version = card.get("model_id", "candidate_v2")
    else:
        model_version = "candidate_v2"

    return {
        "pipeline_run_id": f"run_{uuid.uuid4().hex[:12]}",
        "challenger_model_version": model_version,
        "mlflow_model_version": registered_version,
    }


def _tag_challenger_in_mlflow(
    trigger_id: str,
    evidence_windows: list[str],
    snapshot_sha: str,
) -> str | None:
    """Attach the 'challenger' alias and trigger lineage tags in MLflow (best-effort)."""
    try:
        from mlflow import MlflowClient
        from src.track_b import tracking

        tracking.configure()
        client = MlflowClient()
        versions = list(client.search_model_versions(f"name='{tracking.REGISTERED_MODEL_NAME}'"))
        if not versions:
            return None
        versions.sort(key=lambda v: int(v.version), reverse=True)
        candidate_versions = [v for v in versions if (v.tags or {}).get("role") == "candidate"]
        target = candidate_versions[0] if candidate_versions else versions[0]

        client.set_registered_model_alias(tracking.REGISTERED_MODEL_NAME, "challenger", target.version)
        client.set_model_version_tag(tracking.REGISTERED_MODEL_NAME, target.version, "trigger_id", trigger_id)
        client.set_model_version_tag(tracking.REGISTERED_MODEL_NAME, target.version, "snapshot_sha256", snapshot_sha)
        client.set_model_version_tag(
            tracking.REGISTERED_MODEL_NAME, target.version, "evidence_window_ids", ",".join(evidence_windows)
        )
        logger.info("Set MLflow alias 'challenger' -> %s v%s", tracking.REGISTERED_MODEL_NAME, target.version)
        return str(target.version)
    except Exception as exc:
        logger.warning("Could not tag challenger alias in MLflow (%s)", exc)
        return None


def process_window_for_trigger(
    report: dict[str, Any],
    cfg_path: Path = DEFAULT_CONFIG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    simulated_mode: bool = False,
) -> dict[str, Any] | None:
    """Evaluate a window report and trigger retraining if policy matches."""
    cfg = load_trigger_config(cfg_path)
    if not cfg.get("trigger_policy", {}).get("enabled", True):
        logger.info("Retraining trigger policy is disabled in config.")
        return None

    now = datetime.now(timezone.utc)
    state = load_trigger_state(state_path)

    # Check cooldown and daily cap from config
    guardrails = cfg.get("guardrails", {})
    max_runs_per_day = guardrails.get("max_runs_per_day", 3)
    can_run, reason = check_cooldown(state, now, max_runs_per_day=max_runs_per_day)
    if not can_run:
        logger.info("Trigger skipped: %s", reason)
        return {"status": "cooldown_blocked", "reason": reason}

    # Evaluate conditions
    should_fire, condition_met, evidence_windows = evaluate_trigger_conditions(report, cfg)
    if not should_fire:
        logger.info("Window %s: no trigger conditions met.", report.get("window_id"))
        return None

    # Condition met! Fire trigger
    trigger_id = f"trig_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    logger.warning(">>> RETRAINING TRIGGER FIRED: %s (%s) <<<", trigger_id, condition_met)

    # Snapshot training data
    source_dataset = Path("data/synthetic/learner_interactions_shifted_v1.csv")
    snap_path, snap_sha = snapshot_data(source_dataset)

    # Calculate cooldown
    cooldown_secs = 60 if simulated_mode else guardrails.get("cooldown_seconds", 21600)
    cooldown_until = now + timedelta(seconds=cooldown_secs)

    try:
        pipeline_res = execute_retraining_pipeline(
            trigger_id=trigger_id,
            evidence_windows=evidence_windows,
            snapshot_sha=snap_sha,
        )

        event = {
            "trigger_id": trigger_id,
            "fired_at": iso(now),
            "condition_met": condition_met,
            "evidence_window_ids": evidence_windows,
            "data_snapshot_sha256": snap_sha,
            "cooldown_until": iso(cooldown_until),
            "pipeline_run_id": pipeline_res["pipeline_run_id"],
            "challenger_model_version": pipeline_res["challenger_model_version"],
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
    save_trigger_state(state, state_path)

    log_trigger_event(event)
    logger.info("Trigger event recorded: %s", event["trigger_id"])
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-report", type=Path, required=True,
                        help="Path to window report JSON to evaluate")
    parser.add_argument("--simulated", action="store_true",
                        help="Enable simulated mode with shorter cooldown")
    args = parser.parse_args()

    report = json.loads(args.window_report.read_text(encoding="utf-8"))
    event = process_window_for_trigger(report, simulated_mode=args.simulated)
    if event:
        print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
