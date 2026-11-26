# Drift and Performance Degradation Live Experiment Report

**Experiment ID:** `exp_live_drift_20261126_164148`  
**Shift Injected At:** `2026-11-26T10:30:00+00:00`  
**Conducted At:** `2026-11-26T16:41:48.840493+00:00`  
**Owner:** Person 2 (ML: Drift & Retraining)  

---

## 1. Executive Summary & Hypothesis

- **Hypothesis:** When synthetic traffic transitions from champion-baseline patterns to an injected input shift + concept degradation, sliding-window monitoring will distinguish control vs. shifted regimes with 0 false-alarm triggers during control, detect data drift and performance degradation separately within $\le 1$ window of shift onset, and trigger automated retraining with cooldown enforcement.
- **Control Phase:** 2 windows evaluated, **0** false-alarm drift/performance windows, **0** false-alarm retraining triggers.
- **Shifted Phase:**
  - **Data Drift:** `FEATURE_DRIFT:hint_count` detected in `w-20261126T1030Z-15m` (delay: **16.0 min**).
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
| **Data Drift** | `FEATURE_DRIFT:hint_count` | `w-20261126T1030Z-15m` | **16.0 min** | — |
| **Performance** | `PERFORMANCE_DEGRADATION:f1_drop` | `w-20261126T1030Z-15m` | **16.0 min** | 0.7535 -> 0.3906 |

- **ROC-AUC (Before -> After):** 0.954 -> 0.6357
- **Brier Score (Before -> After):** 0.0903 -> 0.2993

---

## 4. Retraining Trigger Execution & Lineage

```json
{
  "trigger_id": "trig_20261126_164133_d3453a",
  "fired_at": "2026-11-26T16:41:33.843Z",
  "condition_met": "feature_drift_exceeded_threshold (FEATURE_DRIFT:hint_count, FEATURE_DRIFT:response_time_seconds)",
  "evidence_window_ids": [
    "w-20261126T1030Z-15m"
  ],
  "data_snapshot_sha256": "52876f92add9058271b7d42bccb7c83e197815a19be7184642ab8e8e24fbafc0",
  "cooldown_until": "2026-11-26T16:42:33.843Z",
  "pipeline_run_id": "run_6b58bc69a4e8",
  "challenger_model_version": "candidate_hgb_shifted_v2",
  "status": "success"
}
```

- **Cooldown Verification:** Window 4 (`10:45Z - 11:00Z`) returned `cooldown_blocked` (`Cooldown active until 2026-11-26T16:42:33.843Z (45s remaining)`).

---

## 5. Limitations & Ethical Boundary

> [!NOTE]
> **Synthetic Data Disclaimer:** All traffic, learner patterns, injected shifts, and evaluation results are synthetic simulations. These findings demonstrate monitoring and automated retraining governance, not clinical conclusions about human JEE learners.
