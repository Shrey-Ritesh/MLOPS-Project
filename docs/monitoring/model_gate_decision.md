# Phase 3 Day 1: Model Gate Decision & Contract Note (Person 3)

**Date:** 26 Nov 2026 · **Owner:** Person 3 (with Person 2) · **Branch audited:** `origin/dev` (`e992ccd`) · **Status:** Decided (Outcome C)

---

## 1. Search Record — Phase 1 XGBoost Artefact

| Location searched | Method | Result |
|---|---|---|
| Current repository tree (`origin/dev`) | `git ls-files`, search for `xgb`, `.ubj`, `.pkl`, `.joblib`, `.ipynb` | **NOT FOUND** — only `models/track_b/{baseline,candidate}/model.joblib` exist |
| All local and remote git branches (22 refs, including `feature/track-b`, `dev`, `main`, `backend`, `shrad/dev`, `track-b-ncert`, `docker`) | `git ls-tree -r --name-only <ref>` | **NOT FOUND** — no XGBoost artefact, training script, or notebook exists on any branch |
| Full git commit history | `git log --all --grep="xgboost" -i` and `git log --all -S "xgboost"` | **NOT FOUND** — only mentioned in `docs/planning/phase3_weekly_work_split_roles_milestones.md` (`33c8976`) |
| Dependency manifests | `requirements.txt`, `requirements-app.txt`, `requirements-ci.txt` | **NOT FOUND** — `xgboost` is not listed in any requirements file |
| DVC pipeline & remote config | `dvc.yaml`, `dvc.lock`, `.dvc/config`, `data/*.dvc` | **NOT FOUND** — DVC stages cover Track B (`generate` → `baseline` → `shifted` → `drift` → `candidate` → `gate`) and data/assets only |
| Local / shared MLflow store | `src/track_b/tracking.py`, `docs/monitoring/day1_deployed_state_audit.md` §3 & §5 | **NOT FOUND** — EC2 had no MLflow store; Track B registers `learner-difficulty` (`baseline_logreg_v1`, `candidate_hgb_shifted_v2`) |

**Verdict:** Phase 1 XGBoost artefact is **NOT FOUND**.

---

## 2. Model Gate Outcome: **Outcome C**

Per `docs/planning/phase3_weekly_work_split_roles_milestones.md` §3 (Day 1 Model Gate):

- **Champion (live inference):** Track B production model **`candidate_hgb_shifted_v2`** (`HistGradientBoostingClassifier`, registered model `learner-difficulty`, alias `production`), trained on **SYNTHETIC data only — not validated for real students**.
- **Challenger family for retraining:** Retrained HistGradientBoosting on the new labelled snapshot (and optionally XGBoost trained on the exact same 8-feature contract as a second challenger, honestly labelled as a new Phase 3 challenger — never renamed as "the Phase 1 XGBoost model").
- **Governance rule:** Track B is **never** called "the Phase 1 XGBoost model". Every UI surface, report, comparison output, and model card carrying this model displays the synthetic-data limitation label.

---

## 3. Verified Champion Model Contract (Read Directly from Artefact)

| Field | Value | Source |
|---|---|---|
| Artefact path | `models/track_b/candidate/model.joblib` | Repo + `joblib.load` under `scikit-learn==1.7.2` |
| Registered model name | `learner-difficulty` | `src/track_b/tracking.py` |
| Registry alias | `production` | `src/track_b/register_mlflow.py` |
| Model ID | `candidate_hgb_shifted_v2` | `models/track_b/candidate/model_card.json` |
| Estimator pipeline | `Pipeline([('features', ColumnTransformer), ('model', HistGradientBoostingClassifier)])` | Verified from `model.joblib` |
| Feature order (`feature_names_in_`) | `['topic', 'selected_difficulty', 'recent_accuracy', 'mistake_count', 'hint_count', 'response_time_seconds', 'prior_interaction_count', 'previous_difficulty']` | Verified directly from `model.joblib` (`feature_names_in_`), matches `configs/monitoring/contracts/track_b_difficulty_v1.yaml` and `src/track_b/schema.py::FEATURE_COLUMNS` |
| Target | `switch_to_lower_difficulty` (`0` or `1`) | `src/track_b/schema.py` |
| Decision threshold | `0.70` | `models/track_b/candidate/model_card.json` |
| Post-processing guard | `apply_difficulty_boundary_guard`: forces `score = 0.0` when `selected_difficulty == "Easy"` | `src/track_b/modeling.py` |
| Training scikit-learn version | **`1.7.2`** (pinned across `requirements.txt`, `requirements-ci.txt`, `requirements-app.txt`; loading under 1.9.1 fails with `ModuleNotFoundError: No module named '_loss'`) | `model_card.json` + local verification |
| Training dataset SHA-256 | `52876f92add9058271b7d42bccb7c83e197815a19be7184642ab8e8e24fbafc0` (`data/synthetic/learner_interactions_shifted_v1.csv`, 19,737 rows, 560 synthetic learners) | `model_card.json` |
| Synthetic-data label | `"Trained and evaluated on SYNTHETIC data only; not validated for real students."` | `model_card.json` |

---

## 4. Compatibility & Runtime Actions Confirmed

1. **scikit-learn version lock (`1.7.2` only):** `requirements-ci.txt` and `requirements-app.txt` must pin `scikit-learn==1.7.2` (never `1.9.1`).
2. **MLflow registry in Compose:** Confirmed Person 1's proposal (`mlflow` service at `http://mlflow:5000` on `mlflow_data` volume). `ProgressModel` loads from MLflow by alias (`production`) with a local `models/track_b/candidate/model.joblib` fallback and a rule-based fallback if both are unavailable or time out.
3. **Inference location:** Confirmed Person 1's audit §6.3 — inference runs in the Streamlit `app` process on practice submission, logging to `/app/data/state/monitoring.sqlite3` via `PredictionLogger`.
