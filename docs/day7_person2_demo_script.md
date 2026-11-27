# Day 7 Demo Script & Handoff: Person-2 (ML: Drift & Retraining)

**Student Role:** Person-2 (Machine Learning Engineer — Drift & Retraining)  
**Topics Owned:** Graded Topic 4 (*Drift / Degradation Experiment*) & Topic 5 (*Retraining Trigger*)  
**Evidence Files:**
- `reports/drift/live_drift_experiment_report.md`
- `reports/drift/live_drift_experiment_report.json`
- `reports/drift/trigger_events.jsonl`
- `reports/drift/trigger_state.json`
- `reports/monitoring/trigger_events.jsonl`
- `reports/monitoring/trigger_state.json`

---

## 1. Live Demonstration Script

### Presentation Step 1: Graded Topic 4 (Drift / Degradation Experiment)
1. **Explain the Setup:**
   > *"In Phase 3, we transitioned from static batch comparisons to live sliding-window monitoring. We pre-declared our PSI and KS-test thresholds in `configs/monitoring/thresholds.yaml` on Day 2 before any experiment was conducted."*
2. **Show the Control Phase:**
   > *"During the baseline control phase, student traffic ran normally. Notice that 0 false-alarm drift alerts or triggers were raised."*
3. **Show the Shift Point & Detection Delay:**
   > *"At timestamp `T_shift = 10:30:00Z`, we injected documented synthetic distribution shifts (`response_time_seconds * 0.45`, `hint_count + 2`, plus outcome shift simulating concept degradation). As seen in `reports/drift/live_drift_experiment_report.md`, drift was detected in the very first shifted window (`w-20261126T1030Z-15m`) with an observed detection delay of 16.0 minutes."*
4. **Point Out Data Drift vs. Performance Degradation:**
   > *"We explicitly separate input data drift (`FEATURE_DRIFT:psi:hint_count`, `FEATURE_DRIFT:psi:response_time_seconds`) from downstream model performance degradation (`PERFORMANCE_DEGRADATION:f1_drop`, Mean F1 `0.7533 -> 0.3904`), ensuring clear root-cause observability."*

---

### Presentation Step 2: Graded Topic 5 (Automated Retraining Trigger)
1. **Show Trigger Policy & Autonomous Execution:**
   > *"Our declared trigger policy (`configs/monitoring/retrain_trigger.yaml`) checks for $\ge 2$ critical feature drift alerts (excluding `PREDICTION_DRIFT`) or output prediction PSI $\ge 0.20$ or F1 drop $\ge 0.05$. In our experiment, the trigger fired automatically without anyone executing training by hand."*
2. **Show Reproducibility & Lineage:**
   > *"The trigger engine (`src/monitoring/trigger.py`):*
   > *1. Created an immutable DB snapshot of the recent joined monitoring window (`data_snapshot_sha256`).*
   > *2. Automatically executes `dvc repro train_challenger`, saving challenger artifacts only to `models/track_b/challenger/` (never overwriting the `models/track_b/candidate/` champion fallback; retraining uses the documented shifted dataset `learner_interactions_shifted_v1.csv`).*
   > *3. Registers the resulting new model version in the MLflow Model Registry under the alias `challenger`.*
   > *4. Tags the version with the trigger ID, evidence window IDs, and data snapshot hash."*
3. **Show Cooldown Protection:**
   > *"To prevent compute exhaustion on our AWS Free-Tier host and stop runaway loops, a window-time cooldown (`simulated_cooldown_windows: 5` = 75 min at 15-min windows) was activated immediately (`cooldown_until`). Subsequent shifted windows correctly reported `status: cooldown_blocked`, refusing redundant runs."*

---

## 2. Section 9 Handoff Document

| Prompt | Required Answer |
| :--- | :--- |
| **Which graded topic(s) did I own?** | **Topic 4 (Drift / Degradation Experiment)** and **Topic 5 (Retraining Trigger)**. |
| **What evidence shows it works?** | • `reports/drift/live_drift_experiment_report.md` & `.json` (detection delay & false alarm metrics)<br>• `reports/drift/trigger_events.jsonl` & `trigger_state.json` (experiment trigger & cooldown audit)<br>• Unit & regression test suite (`tests/test_drift_window.py`, `tests/test_trigger.py`, `tests/test_monitoring_window.py`, `tests/test_monitoring_detection_report.py`). |
| **What remains incomplete?** | Visual display of the experiment shift point on Person 4's Streamlit Admin Dashboard panels. |
| **What could fail in production?** | If EC2 Free-Tier `t2.micro` memory limits are breached during complex retraining, the trigger's DVC execution will fall back or dispatch a GitHub Actions workflow. |
| **Who owns the next step and by when?** | **Person 3** (evaluating Champion vs. Challenger under Gate v2) and **Person 4** (displaying trigger history on dashboard), Day 5. |

---

> [!NOTE]
> **Release Language:** All models and traffic streams are explicitly documented as **SYNTHETIC** learner data. They validate the autonomous MLOps architectural loop, not clinical claims regarding real student cognition.
