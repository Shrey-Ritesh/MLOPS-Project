# Phase 3: Live Inference, Model Comparison & Registry Governance (Person 3)

**Owner:** Person 3 (ML/backend: Live inference & Model Comparison)  
**Synthetic-data notice:** All Track B models (`baseline_logreg_v1`, `candidate_hgb_shifted_v2`, and Phase 3 challengers) are trained and evaluated on **SYNTHETIC learner data only** and are not validated for real students.

---

## 1. Architecture Summary

1. **Model Gate (`docs/monitoring/model_gate_decision.md`):** Outcome C — no Phase 1 XGBoost artefact exists in the repository, git history, or DVC pipeline. The live champion is Track B `candidate_hgb_shifted_v2` (`learner-difficulty@production`), pinned strictly to **`scikit-learn==1.7.2`**.
2. **Question Topic Tagging (`src/learner/topic_tagger.py`):** Maps JEE Chemistry questions onto the 22 Track B Chemistry topics from `configs/track_b/topic_difficulty_priors.csv`. Coverage report: `reports/track_b/topic_tagging_coverage.json` (`178/215` = `82.79%` across `20/22` topics; remaining image-only questions are honestly flagged `unknown`).
3. **Versioned Inference Wrapper (`src/learner/progress_model.py`):**
   - Loads from MLflow by alias (`models:/learner-difficulty@production`) with automatic fallback to `models/track_b/candidate/model.joblib` and a deterministic rule-based fallback (`rule_fallback` / `fallback-v1`) if both are unavailable or inference exceeds `timeout_ms` (default 500 ms).
   - Verifies `model.feature_names_in_` directly against the active monitoring contract and raises `FeatureContractMismatchError` loudly on any mismatch.
   - Polls the MLflow `production` alias on a TTL (`PROGRESS_MODEL_REFRESH_SECONDS`, default 60 s) so promotion and rollback take effect **without rebuilding or restarting the app**.
4. **Live Practice Hook (`src/learner/inference_hook.py`, `app.py`):**
   - Practice tab captures `topic`, `selected_difficulty`, `hint_count` (via "Show Hint"), and `response_time_seconds`, and builds the 8 Track B features strictly from **pre-submission** state (zero future leakage).
   - Every submission logs one `PredictionRecord` (via `PredictionLogger.log`) and one `OutcomeRecord` (via `PredictionLogger.log_outcome`) with pseudonymous `learner_key`.
   - *My Progress* tab displays the latest step-down estimate, model version, short reason, and synthetic-data notice. The Chemistry Tutor tab is untouched.
5. **Champion vs Challenger Comparison & Gate v2 (`src/track_b/model_comparison.py`, `configs/track_b/promotion_gate_v2.yaml`):**
   - `promotion_gate_v2.yaml` was committed before any challenger existed (git history is the evidence) and preserves all v1 thresholds without loosening.
   - Evaluates both models on **identical rows** (verified via SHA-256 row fingerprints) across `(a) original_test`, `(b) shifted_test`, and `(c) live_window` (joined labels from `monitoring.sqlite3`).
   - Computes F1, ROC-AUC, precision, recall, calibration (Brier score), single-row p50/p95 latency, and model size (MB), and logs all metrics and JSON reports to MLflow.
   - Supports three outcomes: `PROMOTE_CHALLENGER_PENDING_APPROVAL`, `INCONCLUSIVE` (when `live_window` has < 50 labelled rows or single-class labels), and `REJECT_CHALLENGER_RETAIN_CHAMPION`.
6. **Approval-Gated Registry Promotion & Rollback (`src/track_b/registry_ops.py`):**
   - `promote` refuses to move `production` unless Gate v2 returned `PROMOTE_CHALLENGER_PENDING_APPROVAL` **and** a recorded human approval JSON (`approval_id`, `approved_by`, `approved_at`, `challenger_version`, `rationale`, optional `gate_decision_sha256`) is supplied.
   - Archives the previous champion under alias `archived` and writes an immutable entry to `reports/track_b/registry_alias_history.json`.
   - `reject` assigns `rejected_candidate` without touching `production`.
   - `rollback` moves `production` back to the target or `archived` version; `ProgressModel` picks up the rollback automatically without a rebuild.

---

## 2. Operating Commands

```bash
# 1. Regenerate Chemistry topic tags and coverage report
python -m src.learner.topic_tagger

# 2. Run 3-dataset Champion vs Challenger comparison + Gate v2 + MLflow logging
python -m src.track_b.model_comparison \
  --champion-alias production \
  --challenger-alias challenger \
  --monitoring-db data/state/monitoring.sqlite3 \
  --log-mlflow

# 3. Promote challenger ONLY after Person 4 records approval JSON
python -m src.track_b.registry_ops promote \
  --decision reports/track_b/promotion_decision_v2.json \
  --approval docs/governance/promotion_approval.json

# 4. Roll back production alias (zero rebuild; app picks up new version on next refresh)
python -m src.track_b.registry_ops rollback \
  --reason "Post-promotion degradation observed" \
  --operator "person3"
```
