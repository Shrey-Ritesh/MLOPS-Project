# Track B — MLOps demonstration (synthetic learners, drift, promotion gate)

**Scope:** a synthetic MLOps experiment for Physics and Chemistry (40 chapters). It is **not** a validated model for real-student decisions. Every behavioural rule is a documented simulation assumption; public sources only motivate the direction of the effects (see `public_difficulty_themes.md`).

## Reproduce
```bash
pip install -r requirements.txt
dvc repro            # generate -> baseline -> shifted -> drift -> candidate -> promotion gate
pytest tests/test_track_b.py
```
Stages are defined in `dvc.yaml`; outputs are kept in Git (`cache: false`). Configs are in `configs/track_b/`, code in `src/track_b/`.

## What was run
1. **Evidence → priors:** `configs/track_b/topic_difficulty_priors.csv` (SATHEE chapter figures, low–medium confidence, with a cross-check note per chapter).
2. **Synthetic data:** `data/synthetic/learner_interactions_v1.csv` (800 learners, seed in `learner_scenarios.yaml`), split by learner.
3. **Baseline:** logistic regression vs a majority-class reference, on held-out learners.
4. **Controlled shift:** `learner_interactions_shifted_v1.csv` — two documented changes (more hint-reliant learners; six multi-step chapters made harder), see `drift_experiment.yaml`.
5. **Drift and performance, reported separately:** `drift_report.json`, `shifted_evaluation.json`.
6. **Candidate:** same model family retrained on shifted-population training learners; compared on the same fixed test learners (`candidate_comparison.json`).
7. **Gate:** thresholds in `promotion_gate.yaml` were committed before the candidate was trained; the decision is in `promotion_decision.json`.

## Results (synthetic data)
| | Baseline, original test | Baseline, shifted test | Candidate, shifted test | Candidate, original test |
|---|---|---|---|---|
| F1 | 0.5908 | 0.5427 | 0.5381 | 0.5725 |
| ROC-AUC | 0.8972 | 0.8793 | 0.8779 | 0.8937 |

The majority-class reference predicts "no switch" for every row (F1 = 0.0, accuracy 0.8332), so accuracy alone would be misleading here.

**Data drift (`drift_report.json`):** modest. Mean hint count rose from 0.5339 to 0.7594 (KS significant; PSI 0.0564, which is coarse for a small discrete count). No feature crossed the PSI alert threshold of 0.2; target prevalence was unchanged (0.1601 vs 0.1611).

**Performance change (`shifted_evaluation.json`):** the original baseline's F1 fell by 0.0481 on the shifted test learners. Drift and performance are reported separately and neither is assumed to imply the other.

**Promotion decision: REJECT_CANDIDATE_RETAIN_BASELINE.** Failed criteria: minimum_predictive_quality, improvement_over_baseline. The candidate did not meet the pre-committed minimum F1 and did not improve on the baseline on the shifted test learners. All other criteria passed (regression tolerance, schema/leakage/learner-split validity, reproducibility fields). The gate file was committed before the candidate was trained (`gate_integrity` in the decision record).

## Interpretation and limits
- The shift was deliberately mild and mostly changed how one feature (hints) relates to the target. Retraining on shifted data did not recover enough to clear the gate, so the baseline is retained. A rejection is a valid outcome of the gate, not a failure of the experiment.
- Thresholds were not changed after seeing the candidate's results. A stronger shift would be a new experiment (v2) with its own gate committed first.
- All numbers describe synthetic data. Real-student validity would need real interaction data.

## MLflow (local)
Runs and models are also logged to a **local** MLflow store (`mlflow.db`, not committed). Re-create it from the saved artifacts, without retraining:
```bash
python -m src.track_b.register_mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://localhost:5000
```
- Experiment `track_b_learner_difficulty` has two runs: the baseline and the candidate (parameters, metrics, dataset/config hashes, reports).
- Registered model `learner-difficulty`: v1 = baseline, alias `production`; v2 = candidate, alias `rejected_candidate`, tagged with the gate decision. The registry follows `promotion_decision.json`.
- To use a shared server later, set `MLFLOW_TRACKING_URI=http://<server>:5000`; no code changes.
- Loading a registered model needs the repo on `PYTHONPATH` (the pipeline uses helpers from `src.track_b.modeling`).
