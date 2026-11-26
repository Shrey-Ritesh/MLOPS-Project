"""End-to-End Drift & Degradation Experiment Runner for Phase 3 (Owner: Person 2).

Executes the Day 4 controlled experiment:
  1. Control phase: Streams representative baseline traffic scored by the champion model
     -> verifies 0 false-alarm drift/performance alerts and 0 false-alarm triggers.
  2. Shift injection: Injects documented input shift (response_time_seconds * 0.45,
     hint_count + 2) and concept degradation at T_shift -> verifies separate data-drift
     and performance-degradation detection.
  3. Automatic Retraining Trigger: Fires trigger without manual intervention on the first
     shifted window, creates data snapshot, runs candidate training, logs lineage, and
     enforces cooldown on the subsequent shifted window.
  4. Generates formal detection-delay & drift vs. performance experiment report (JSON + Markdown).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.monitoring import db, detection_report, run_window
from src.monitoring.contract import load_contract
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.monitoring.trigger import (
    DEFAULT_EVENTS_LOG,
    DEFAULT_STATE_PATH,
    process_window_for_trigger,
)
from src.track_b.modeling import apply_difficulty_boundary_guard
from src.track_b.schema import FEATURE_COLUMNS, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriftExperiment")

WINDOW_SAMPLE_SIZE = 450


def _score_with_champion(
    df: pd.DataFrame,
    model_path: Path = Path("models/track_b/candidate/model.joblib"),
    card_path: Path = Path("models/track_b/candidate/model_card.json"),
) -> tuple[np.ndarray, float, str]:
    """Score dataframe rows using the checked-in champion model."""
    model = joblib.load(model_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    threshold = float(card.get("decision_threshold", 0.7))
    apply_guard = bool(card.get("apply_boundary_guard", True))
    model_id = str(card.get("model_id", "candidate_hgb_shifted_v2"))

    scores = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    if apply_guard:
        scores = apply_difficulty_boundary_guard(scores, df["selected_difficulty"])
    return np.asarray(scores, dtype=float), threshold, model_id


def _log_window_records(
    pred_logger: PredictionLogger,
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    model_id: str,
    window_start: datetime,
    seq_offset: int,
    prefix: str,
    latency_ms: float,
) -> None:
    """Write a window of synthetic prediction records + joined labels."""
    step_seconds = (14.0 * 60.0) / max(1, len(df))
    for idx, (_, row) in enumerate(df.iterrows()):
        ts = (window_start + timedelta(seconds=idx * step_seconds)).isoformat()
        score = float(scores[idx])
        pred_label = int(score >= threshold)
        # Keep true ground truth temporarily in label_observed via DB update below
        rec = PredictionRecord(
            request_id=f"req_{prefix}_{uuid.uuid4().hex[:8]}",
            source="simulated",
            learner_id=str(row["learner_id"]),
            learner_seq=seq_offset + idx,
            question_id=f"q_{prefix}_{idx}",
            features={col: row[col] for col in FEATURE_COLUMNS if col in row},
            score=score,
            label=int(labels[idx]),  # store ground truth in log for label-join simulation
            threshold=threshold,
            model_name="learner-difficulty",
            model_version="1",
            model_id=model_id,
            registry_alias="production",
            latency_ms=latency_ms,
            status="ok",
            occurred_at=ts,
        )
        pred_logger.log(rec)
        _ = pred_label


def run_experiment(
    db_path: Path = Path("data/monitoring/drift_experiment.sqlite3"),
    report_out_dir: Path = Path("reports/drift"),
    reset_trigger_logs: bool = True,
) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    if reset_trigger_logs:
        if DEFAULT_STATE_PATH.exists():
            DEFAULT_STATE_PATH.unlink()
        if DEFAULT_EVENTS_LOG.exists():
            DEFAULT_EVENTS_LOG.unlink()

    with db.session(db_path) as conn:
        db.ensure_schema(conn)

    contract = load_contract()
    logger.info("Loaded monitoring contract: %s", contract.contract_id)

    # Simulation timeline (4 non-overlapping 15-minute windows):
    # Window 1 (Control): 10:00 - 10:15
    # Window 2 (Control): 10:15 - 10:30
    # Shift injected at:  10:30:00Z
    # Window 3 (Shifted): 10:30 - 10:45 (Detection + autonomous trigger firing)
    # Window 4 (Shifted): 10:45 - 11:00 (Sustained shift + cooldown enforcement)
    t0 = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)
    shift_time = t0 + timedelta(minutes=30)

    ref_df = pd.read_csv("data/synthetic/learner_interactions_shifted_v1.csv")
    base_df = pd.read_csv("data/synthetic/learner_interactions_v1.csv")

    key = "pseudonym_secret_key_12345678"
    pred_logger = PredictionLogger(db_path=db_path, pseudonym_key=key, contract=contract)

    # 1. Control Phase Windows (sampled randomly from champion reference distribution to avoid order skew)
    logger.info("Populating Control Phase traffic (2 windows, N=%d each)...", WINDOW_SAMPLE_SIZE)
    ctrl_w1 = ref_df.sample(n=WINDOW_SAMPLE_SIZE, random_state=101).reset_index(drop=True)
    ctrl_w2 = ref_df.sample(n=WINDOW_SAMPLE_SIZE, random_state=202).reset_index(drop=True)

    scores_w1, threshold, model_id = _score_with_champion(ctrl_w1)
    scores_w2, _, _ = _score_with_champion(ctrl_w2)

    _log_window_records(
        pred_logger, ctrl_w1, scores_w1, ctrl_w1[TARGET].to_numpy(),
        threshold, model_id, t0, seq_offset=0, prefix="ctrl1", latency_ms=15.0,
    )
    _log_window_records(
        pred_logger, ctrl_w2, scores_w2, ctrl_w2[TARGET].to_numpy(),
        threshold, model_id, t0 + timedelta(minutes=15), seq_offset=1000, prefix="ctrl2", latency_ms=16.0,
    )

    # 2. Shifted Phase Windows (documented extra input shift + concept degradation)
    logger.info("Populating Shifted Phase traffic (2 windows after %s)...", shift_time.isoformat())
    rng = np.random.default_rng(33)

    shift_w3 = base_df.sample(n=WINDOW_SAMPLE_SIZE, random_state=303).reset_index(drop=True).copy()
    shift_w3["response_time_seconds"] = (shift_w3["response_time_seconds"] * 0.45).round(2)
    shift_w3["hint_count"] = shift_w3["hint_count"] + 2
    scores_w3, _, _ = _score_with_champion(shift_w3)
    labels_w3 = shift_w3[TARGET].to_numpy().copy()
    flip_w3 = (scores_w3 < threshold) & (rng.random(len(labels_w3)) < 0.35)
    labels_w3[flip_w3] = 1

    shift_w4 = base_df.sample(n=WINDOW_SAMPLE_SIZE, random_state=404).reset_index(drop=True).copy()
    shift_w4["response_time_seconds"] = (shift_w4["response_time_seconds"] * 0.45).round(2)
    shift_w4["hint_count"] = shift_w4["hint_count"] + 2
    scores_w4, _, _ = _score_with_champion(shift_w4)
    labels_w4 = shift_w4[TARGET].to_numpy().copy()
    flip_w4 = (scores_w4 < threshold) & (rng.random(len(labels_w4)) < 0.35)
    labels_w4[flip_w4] = 1

    _log_window_records(
        pred_logger, shift_w3, scores_w3, labels_w3,
        threshold, model_id, shift_time, seq_offset=2000, prefix="shift3", latency_ms=35.0,
    )
    _log_window_records(
        pred_logger, shift_w4, scores_w4, labels_w4,
        threshold, model_id, shift_time + timedelta(minutes=15), seq_offset=3000, prefix="shift4", latency_ms=38.0,
    )

    # 3. Join delayed ground-truth labels for evaluation
    with db.session(db_path) as conn:
        conn.execute("UPDATE predictions SET label_observed = label, label_status = 'joined'")

    # 4. Evaluate Monitoring Windows & Run Policy-Driven Trigger
    windows = [
        ("control", t0, t0 + timedelta(minutes=15)),
        ("control", t0 + timedelta(minutes=15), t0 + timedelta(minutes=30)),
        ("shifted", shift_time, shift_time + timedelta(minutes=15)),
        ("shifted", shift_time + timedelta(minutes=15), shift_time + timedelta(minutes=30)),
    ]

    evaluated_reports = []
    control_triggers = []
    shifted_triggers = []

    for phase, w_start, w_end in windows:
        logger.info("Evaluating %s window [%s - %s]...", phase, w_start.isoformat(), w_end.isoformat())
        rep = run_window.evaluate_window(
            db_path=str(db_path),
            start=w_start,
            end=w_end,
            contract=contract,
            now=w_end + timedelta(minutes=1),
        )
        evaluated_reports.append(rep)

        trig_event = process_window_for_trigger(rep, simulated_mode=True)
        if trig_event:
            if phase == "control":
                control_triggers.append(trig_event)
            else:
                shifted_triggers.append(trig_event)

    # 5. Build official detection-delay report (Person 1's detection_report.py)
    det_report = detection_report.build_report(db_path=db_path, shift_at=shift_time)

    # Compute genuine trigger status
    fired_control = [e for e in control_triggers if e.get("status") != "cooldown_blocked"]
    fired_shifted = [e for e in shifted_triggers if e.get("status") != "cooldown_blocked"]
    cooldown_blocked = [e for e in shifted_triggers if e.get("status") == "cooldown_blocked"]

    data_det = det_report.get("data_drift_detection")
    perf_det = det_report.get("performance_detection")
    before_perf = det_report["performance_trend"]["before_shift"]
    after_perf = det_report["performance_trend"]["after_shift"]

    experiment_summary = {
        "experiment_id": f"exp_live_drift_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
        "conducted_at": datetime.now(timezone.utc).isoformat(),
        "shift_injected_at": shift_time.isoformat(),
        "window_sample_size": WINDOW_SAMPLE_SIZE,
        "phases": {
            "control": {
                "windows_evaluated": det_report["windows"]["control"],
                "false_alarm_drift_windows": len(
                    det_report["control_phase"]["false_alarm_windows_drift_or_performance"]
                ),
                "trigger_fired": len(fired_control) > 0,
            },
            "shifted": {
                "windows_evaluated": det_report["windows"]["shifted"],
                "data_drift_detected": data_det is not None,
                "first_data_drift_metric": data_det["metric"] if data_det else None,
                "data_drift_delay_minutes": data_det["delay_minutes"] if data_det else None,
                "performance_degradation_detected": perf_det is not None,
                "first_performance_metric": perf_det["metric"] if perf_det else None,
                "performance_delay_minutes": perf_det["delay_minutes"] if perf_det else None,
                "trigger_fired": len(fired_shifted) > 0,
                "cooldown_blocked_count": len(cooldown_blocked),
            },
        },
        "trigger_results": shifted_triggers,
        "detection_report_raw": det_report,
        "limitations": "Conducted on SYNTHETIC learner data only; findings not validated for real students.",
    }

    # Save JSON report
    json_path = report_out_dir / "live_drift_experiment_report.json"
    json_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    logger.info("Saved JSON report to %s", json_path)

    primary_trigger = fired_shifted[0] if fired_shifted else (shifted_triggers[0] if shifted_triggers else {})

    # Save Markdown report
    md_content = f"""# Drift and Performance Degradation Live Experiment Report

**Experiment ID:** `{experiment_summary['experiment_id']}`  
**Shift Injected At:** `{shift_time.isoformat()}`  
**Conducted At:** `{experiment_summary['conducted_at']}`  
**Owner:** Person 2 (ML: Drift & Retraining)  

---

## 1. Executive Summary & Hypothesis

- **Hypothesis:** When synthetic traffic transitions from champion-baseline patterns to an injected input shift + concept degradation, sliding-window monitoring will distinguish control vs. shifted regimes with 0 false-alarm triggers during control, detect data drift and performance degradation separately within $\\le 1$ window of shift onset, and trigger automated retraining with cooldown enforcement.
- **Control Phase:** {det_report['windows']['control']} windows evaluated, **{len(det_report['control_phase']['false_alarm_windows_drift_or_performance'])}** false-alarm drift/performance windows, **{len(fired_control)}** false-alarm retraining triggers.
- **Shifted Phase:**
  - **Data Drift:** `{data_det['metric'] if data_det else 'none'}` detected in `{data_det['first_window'] if data_det else 'N/A'}` (delay: **{data_det['delay_minutes'] if data_det else 'N/A'} min**).
  - **Performance Degradation:** `{perf_det['metric'] if perf_det else 'none'}` detected in `{perf_det['first_window'] if perf_det else 'N/A'}` (delay: **{perf_det['delay_minutes'] if perf_det else 'N/A'} min**).
  - **Retraining Trigger:** Fired autonomously on `{primary_trigger.get('evidence_window_ids', ['N/A'])[0] if primary_trigger else 'N/A'}`; subsequent window blocked by cooldown ({len(cooldown_blocked)} blocked).

---

## 2. Injected Shifts & Methodology

1. **Control Baseline (`10:00Z - 10:30Z`):** Random windows ($N={WINDOW_SAMPLE_SIZE}$ per 15-min window) drawn from champion reference distribution (`learner_interactions_shifted_v1.csv`) and scored by `{model_id}`.
2. **Extra Input Shift (`>= 10:30Z`):** Applied documented input shift (`response_time_seconds * 0.45`, `hint_count + 2`) on baseline interactions ($N={WINDOW_SAMPLE_SIZE}$ per window), because raw `v1` vs `shifted_v1` input PSI alone is $\\le 0.056$.
3. **Concept Degradation (`>= 10:30Z`):** Flipped 35% of negative-predicted outcomes to `1` to model unseen learner struggle under the shifted regime.

---

## 3. Data Drift vs. Performance Degradation (Kept Separate)

| Signal Category | First Alert Metric | First Window Detected | Detection Delay (min) | Mean F1 (Before -> After) |
| :--- | :--- | :--- | :--- | :--- |
| **Data Drift** | `{data_det['metric'] if data_det else 'N/A'}` | `{data_det['first_window'] if data_det else 'N/A'}` | **{data_det['delay_minutes'] if data_det else 'N/A'} min** | — |
| **Performance** | `{perf_det['metric'] if perf_det else 'N/A'}` | `{perf_det['first_window'] if perf_det else 'N/A'}` | **{perf_det['delay_minutes'] if perf_det else 'N/A'} min** | {before_perf.get('mean_f1')} -> {after_perf.get('mean_f1')} |

- **ROC-AUC (Before -> After):** {before_perf.get('mean_roc_auc')} -> {after_perf.get('mean_roc_auc')}
- **Brier Score (Before -> After):** {before_perf.get('mean_brier')} -> {after_perf.get('mean_brier')}

---

## 4. Retraining Trigger Execution & Lineage

```json
{json.dumps(primary_trigger, indent=2)}
```

- **Cooldown Verification:** Window 4 (`10:45Z - 11:00Z`) returned `{cooldown_blocked[0].get('status') if cooldown_blocked else 'N/A'}` (`{cooldown_blocked[0].get('reason') if cooldown_blocked else 'N/A'}`).

---

## 5. Limitations & Ethical Boundary

> [!NOTE]
> **Synthetic Data Disclaimer:** All traffic, learner patterns, injected shifts, and evaluation results are synthetic simulations. These findings demonstrate monitoring and automated retraining governance, not clinical conclusions about human JEE learners.
"""

    md_path = report_out_dir / "live_drift_experiment_report.md"
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Saved Markdown report to %s", md_path)

    return experiment_summary


if __name__ == "__main__":
    run_experiment()
