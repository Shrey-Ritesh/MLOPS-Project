# Drift and Performance Degradation Live Experiment Report

**Experiment ID:** `exp_live_drift_20261127_161910`  
**Shift Injected At:** `2026-11-26T10:30:00+00:00`  
**Conducted At:** `2026-11-27T16:19:10.388677+00:00`  
**Owner:** Person 2 (ML: Drift & Retraining)  

---

## 1. Executive Summary & Hypothesis

- **Hypothesis:** When synthetic traffic transitions from champion-baseline patterns to an injected input shift + concept degradation, sliding-window monitoring will distinguish control vs. shifted regimes with 0 false-alarm triggers during control, detect data drift and performance degradation separately within $\le 1$ window of shift onset, and trigger automated retraining with cooldown enforcement.
- **Control Phase:** 2 windows evaluated, no drift that would trigger a retrain in the control phase (**0** false-alarm retraining triggers; dashboard `status: alert` on `topic` and `prior_interaction_count` remains visible when `N >= 200` but is excluded from trigger firing).
- **Shifted Phase:**
  - **Data Drift:** `FEATURE_DRIFT:psi:hint_count` detected in `w-20261126T1030Z-15m` (delay: **16.0 min**).
  - **Performance Degradation:** `PERFORMANCE_DEGRADATION:f1_drop` detected in `w-20261126T1030Z-15m` (delay: **16.0 min**).
  - **Retraining Trigger:** Fired autonomously on `w-20261126T1030Z-15m`; subsequent window blocked by cooldown (1 blocked).

---

## 2. Injected Shifts & Methodology

1. **Control Baseline (`10:00Z - 10:30Z`):** Random windows ($N=450$ per 15-min window) drawn from champion reference distribution (`learner_interactions_shifted_v1.csv`) and scored by `candidate_hgb_shifted_v2`.
2. **Extra Input Shift (`>= 10:30Z`):** Applied documented input shift (`response_time_seconds * 0.45`, `hint_count + 2`) on baseline interactions ($N=450$ per window), because raw `v1` vs `shifted_v1` input PSI alone is $\le 0.056$.
3. **Concept Degradation (`>= 10:30Z`):** Flipped 35% of negative-predicted outcomes to `1` to model unseen learner struggle under the shifted regime.

---

## 3. Data Drift vs. Performance Degradation (Kept Separate)

| Signal Category | First Alert Metric | First Window Detected | Detection Delay (min) | Mean F1 (Before -> After) |
| :--- | :--- | :--- | :--- | :--- |
| **Data Drift** | `FEATURE_DRIFT:psi:hint_count` | `w-20261126T1030Z-15m` | **16.0 min** | — |
| **Performance** | `PERFORMANCE_DEGRADATION:f1_drop` | `w-20261126T1030Z-15m` | **16.0 min** | 0.7533 -> 0.3904 |

- **ROC-AUC (Before -> After):** 0.9539 -> 0.6361
- **Brier Score (Before -> After):** 0.0903 -> 0.299

---

## 4. Retraining Trigger Execution & Lineage

```json
{
  "trigger_id": "trig_20261126_104500_536c35",
  "fired_at": "2026-11-26T10:45:00.000Z",
  "condition_met": "feature_drift_exceeded_threshold (FEATURE_DRIFT:psi:hint_count, FEATURE_DRIFT:psi:response_time_seconds)",
  "evidence_window_ids": [
    "w-20261126T1030Z-15m"
  ],
  "data_snapshot_sha256": "2ef6dee147ea9f5bd9211f3e0d8b4bc6e17c3cdc18fd1c5ab37265597fdcfc67",
  "snapshot_source": "monitoring_db",
  "retraining_dataset_note": "Snapshot recorded for lineage; pipeline execution skipped (dry_run)",
  "cooldown_until": "2026-11-26T12:00:00.000Z",
  "pipeline_run_id": "dry_run",
  "challenger_model_version": null,
  "mlflow_model_version": null,
  "status": "success"
}
```

- **Cooldown Verification:** Window 4 (`10:45Z - 11:00Z`) returned `cooldown_blocked` (`Cooldown active until 2026-11-26T12:00:00.000Z (3600s remaining)`).
- **Snapshot & Retraining Lineage:** Snapshot recorded for lineage from joined monitoring rows (`snapshot_source=monitoring_db`, `data_snapshot_sha256=2ef6dee147ea9f5bd9211f3e0d8b4bc6e17c3cdc18fd1c5ab37265597fdcfc67`); candidate retrained on the documented shifted dataset (`learner_interactions_shifted_v1.csv`) in real trigger runs (`execute_pipeline=False` dry-run in this experiment to preserve the committed champion). Direct training on DB snapshots is deferred as future work.

---

## 5. Limitations & Ethical Boundary

> [!NOTE]
> **Synthetic Data Disclaimer:** All traffic, learner patterns, injected shifts, and evaluation results are synthetic simulations. These findings demonstrate monitoring and automated retraining governance, not clinical conclusions about human JEE learners.
