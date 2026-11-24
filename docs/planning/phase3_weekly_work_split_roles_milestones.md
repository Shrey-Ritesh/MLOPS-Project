# Phase 3: Monitoring, Feedback & Governance — Work Split, Roles, and Daily Milestones

**Project:** Chemistry-focused AI learning assistant for JEE preparation  
**Planning horizon:** 7 working days  
**Builds on:** Phase 2 handoff (snapshot 20 Nov 2026): AWS-deployed app, MiniLM question-only retriever, MLflow retrieval comparison, Track B drift / promotion-gate / registry demo.  
**Graded Phase 3 topics:** 1. Model Monitoring · 2. Logging · 3. Performance Dashboard · 4. Drift / Degradation Experiment · 5. Retraining Trigger · 6. Model Comparison · 7. Governance & Model Card  
**Release target:** The learner-progress model runs inside the deployed app. Every prediction is logged and monitored. A dashboard shows model health. A drift experiment pushes a real retraining trigger, which produces a challenger. The challenger is compared to the champion and is promoted only after a recorded, gated, human-approved decision. All of it is documented in a model card and governance record.

> **Planning rule:** This is a working agreement. It does not claim that any of this work is done. Check the repository, the deployed revision, the MLflow store, and the model artefacts before changing or claiming anything. Phase 1 already ran an *offline* drift experiment and promotion gate (Track B). Phase 3 has to show these working as a **running system on the live prediction path**. Re-running the Phase 1 notebook-style experiment does not count.

---

## 0. Repository facts this plan is based on (checked 24 Nov 2026, branch `chore/repo-cleanup`)

| Fact | Where | Consequence for Phase 3 |
|---|---|---|
| **No XGBoost artefact, code, or dependency exists in the repo.** | repo-wide search; `requirements*.txt` | Day 1 **model gate** (§3): find the Phase 1 XGBoost model or record that it is missing. |
| Track B has a trained, registered learner model. `candidate_hgb_shifted_v2` (HistGradientBoosting, threshold 0.7) was promoted over `baseline_logreg_v1`. `register_mlflow.py` sets the `production` / `rejected_candidate` aliases. | `models/track_b/`, `reports/track_b/promotion_decision.json`, `src/track_b/register_mlflow.py` | There is already a champion, a registry, and a gate to build on. **Synthetic data only**, so the label must appear everywhere this model does. |
| Track B features: `topic, selected_difficulty, recent_accuracy, mistake_count, hint_count, response_time_seconds, prior_interaction_count, previous_difficulty`. Target: `switch_to_lower_difficulty`. | Track B model card | The live app captures **none** of hints, response time, or chosen difficulty, and questions have no `topic`. These must be captured for live inference. |
| Drift code already exists: PSI (numeric and categorical) and a KS test, with a PSI alert level of 0.2 and KS α of 0.01. The baseline vs shifted report showed baseline F1 falling 0.5908 → 0.5427. | `src/track_b/drift.py`, `evaluate_drift.py`, `configs/track_b/drift_experiment.yaml`, `reports/track_b/drift_report.json` | Reuse it for **scheduled monitoring windows** over the live prediction log. Do not rewrite it. |
| A promotion gate with criteria fixed in advance (minimum F1, gain, regression tolerance, data validity, reproducibility) exists, and its integrity is checked against git history. | `configs/track_b/promotion_gate.yaml`, `src/track_b/promotion_gate.py` | This is the base for **Model Comparison**. Extend it to v2. Do not loosen v1 after seeing results. |
| A DVC pipeline exists: generate → baseline → shifted → drift → candidate → gate. | `dvc.yaml` | The **retraining trigger** can call `dvc repro` on a new data snapshot instead of a new script. |
| There are two attempt stores (API JSONL and app SQLite), and the app hard-codes `concept_ids=("chemistry",)`. | `src/api/routes.py`, `app.py` | A single prediction/event log is the base for Logging. Questions need topic tags. |
| Deployment uses Docker Compose on EC2 (port 80) via GitHub Actions, on the AWS Free Plan with no OIDC. | `docker-compose.yml`, `.github/workflows/deploy.yml` | The monitoring and dashboard must run inside the existing stack. Use no paid managed services without a cost check. |
| The Phase 2 runtime retrieval-quality package is not in the tree. | handoff §7.2 | Record its status in the Day 1 audit. It is not part of the graded scope. |

---

## 1. Shared outcome — what a grader should be able to see

| # | Graded topic | What "done" looks like at the Day 7 demo |
|---|---|---|
| 1 | **Model Monitoring** | A scheduled monitoring job computes service health (latency, errors, fallback rate), input validity, prediction distribution, and (when labels arrive) live performance per window. Breaches raise a visible alert with a severity. |
| 2 | **Logging** | Every prediction writes one structured, versioned, privacy-safe record: features, prediction, model version, latency. Delayed ground-truth labels are joined back to it. Logs can be queried and survive redeploys. |
| 3 | **Performance Dashboard** | An admin-only dashboard shows model performance, drift, operational health, alerts, retraining runs, and registry history over time. It reads only from the logs, monitoring outputs, and MLflow. |
| 4 | **Drift / Degradation Experiment** | A controlled experiment sends shifted synthetic traffic through the **live** prediction path. Monitoring detects data drift and performance degradation. The report separates the two and states the detection delay. |
| 5 | **Retraining Trigger** | A declared trigger policy (drift and/or degradation thresholds, minimum sample size, cooldown) automatically starts a reproducible retraining run, which produces a registered challenger. |
| 6 | **Model Comparison** | Champion vs challenger is evaluated on the same fixed test sets plus the recent live window, with a v2 promotion gate declared in advance. Results and the decision are logged to MLflow. |
| 7 | **Governance & Model Card** | A model card per registered version, a promotion approval record (who, when, which gate result), an audit trail of which version served which predictions, data-retention and privacy rules, and a rollback procedure. |

### Scope boundaries

- **In scope:** live inference for the learner-progress model; prediction and label logging; a monitoring job; an admin dashboard; a drift experiment on the live path; an automated retraining trigger and pipeline; champion/challenger comparison; MLflow registry governance; model cards; minimal student-facing use of predictions.
- **Not a goal:** a new monitoring SaaS or paid AWS service without a cost check; a new recommender; retrieval or embedding changes; a frontend rewrite; real-world efficacy claims. Retraining happens **only when the trigger fires**, never by hand "to get a better number".
- **Carry-over (not graded, do only if time allows):** the full automated student journey from handoff §13.2. The minimum this week is that predictions appear in the existing *My Progress* tab with a "why" note, so the model has real usage to monitor.

---

## 2. Topic ownership and roles

Each graded topic has **one directly responsible owner**. Topic 7 has a lead and a section from every member.

| Graded topic | Owner | Reviewer |
|---|---|---|
| 1. Model Monitoring | **Person 1** | Person 2 |
| 2. Logging | **Person 1** | Person 3 |
| 3. Performance Dashboard | **Person 4** | Person 1 |
| 4. Drift / Degradation Experiment | **Person 2** | Person 3 |
| 5. Retraining Trigger | **Person 2** | Person 1 |
| 6. Model Comparison | **Person 3** | Person 2 |
| 7. Governance & Model Card | **Person 4 (lead)** + a section from each member | All |
| Foundation: live inference in the app | **Person 3** | Person 1 |

| Member | Role | Owns | Does not own | Required end-of-week evidence |
|---|---|---|---|---|
| **Person 1** | MLOps platform: Logging & Monitoring | Prediction and label log schema and storage; structured app logs; retention and backup; scheduled monitoring job (health, input validity, prediction drift, live performance); alert thresholds and alert sink; deployment of all of the above on EC2 | Model training, gate thresholds, dashboard UI | Prediction log populated from the live app; monitoring job runs on a schedule on AWS and writes window reports; at least one alert fired and recorded during the drift experiment |
| **Person 2** | ML: Drift & Retraining | Reference profile from training data; monitoring-window drift metrics (reusing `src/track_b/drift.py`); a simulated-traffic generator; the drift/degradation experiment design and report; retraining trigger policy; an automated retraining pipeline that produces a registered challenger | Live inference code, dashboard, final promotion approval | Experiment report showing the injected shift, detection window, drift vs performance separation; a trigger fired automatically and produced a challenger run in MLflow |
| **Person 3** | ML/backend: Live inference & Model Comparison | Day 1 model gate (with Person 2); topic tagging plus capture of the missing features; a `ProgressModel` wrapper with version metadata and fallback; the inference hook in the app; promotion gate v2; champion/challenger comparison job; registry alias changes after approval | Monitoring job, dashboard, drift injection | Live predictions with model version on AWS; comparison report plus gate decision in MLflow; alias change done only after a recorded approval |
| **Person 4** | Dashboard & Governance lead | Admin performance dashboard; access control on it; model card template and cards per version; governance record (approval flow, audit trail, retention, roles, rollback); assembling the final governance pack | Metric computation logic (only displays it), training, gate thresholds | Dashboard live on AWS showing every panel in §4.4; model cards for champion and challenger; signed approval record for the promotion decision |

### Team-wide rules

- Small PRs into `dev`; nobody commits directly to `main`. Release PR from `dev` to `main` on Day 6.
- Freeze the **prediction-log schema**, **monitoring-window report shape**, and **gate v2 thresholds** by the end of Day 2. Thresholds must be committed **before** any challenger exists. Git history is the evidence.
- Synthetic learners only, in tests, experiments, and screenshots. Every synthetic model or report carries the synthetic label.
- Do not touch the Chroma index or the retriever.
- Escalate if blocked for more than half a day. The model gate has a pre-agreed fallback, so nobody waits on it.

---

## 3. Daily milestone plan

## Day 1 — Audit, model gate, and contracts

**Exit condition:** the team knows which model is the champion, what the deployed revision contains, and has draft contracts for the prediction log, monitoring report, and governance record.

| Owner | Tasks | Artifact / evidence | Acceptance criteria | Coordination |
|---|---|---|---|---|
| **Person 1** | Record the deployed revision (image tag, commit), volumes, and data paths on EC2. Check where MLflow runs and whether it persists. Draft the **prediction-log schema** (§4.1) and choose storage (recommended: a SQLite table on the persisted volume, exported to Parquet per window). | Deployed-state audit; log schema draft. | Every claim is labelled *verified on host / in repo / unknown*. No host changes today. | Schema review with Persons 2 and 3. |
| **Person 2** | Draft the **reference profile**: per-feature distributions and the prediction-score distribution from the champion's training data. Read `drift.py` and `drift_experiment.yaml`, and list what needs to change to run over time windows instead of two fixed CSVs. | Reference-profile spec; reuse plan for drift code. | The reference is tied to the champion's `dataset_sha256`. | Feature list from the model gate outcome. |
| **Person 3** | **Model gate (with Person 2).** Look for the Phase 1 XGBoost model in team drives, notebooks, old branches, MLflow, and DVC remotes. Record the artefact, features, target, threshold, and versions. | Model-contract note: FOUND or NOT FOUND, with the places searched. | Feature order is read from the artefact itself. | Tell the whole team the outcome by midday. |
| **Person 4** | List the questions the dashboard must answer (for the team and for graders). Draft the model card template (extending `model_card.json`) and the governance record outline: roles, approval steps, audit fields, retention. | Dashboard requirements; card template; governance outline. | Each dashboard panel maps to a graded topic. | Gather which fields each owner will produce. |
| **All** | Record the model gate decision. Agree on the alert sink (recommended: an alerts table plus a dashboard banner, with optional email or GitHub issue via the existing Actions setup). | Interface agreement; task board. | Every task has an owner and a day. | — |

**Day 1 model gate**

| Outcome | Champion | Challenger family for retraining |
|---|---|---|
| **A. XGBoost found, features obtainable from the app** | Phase 1 XGBoost (add the pinned `xgboost` version to requirements) | Retrained XGBoost on the new snapshot, plus Track B HGB as a comparison point if features match |
| **B. XGBoost found, but features not yet captured** | Track B `production` alias for live inference this week. XGBoost is registered and compared offline. | Same as A once the features exist |
| **C. XGBoost not found** | Track B `production` alias (`candidate_hgb_shifted_v2`), **clearly labelled synthetic, not validated for real students** | Retrained HGB, plus XGBoost trained on the same features as a second challenger (this is how "XGBoost" enters the system honestly) |

Never rename Track B as "the Phase 1 XGBoost model". Whichever outcome applies, it is recorded in the governance record and on every model card.

## Day 2 — Foundations behind frozen contracts

**Exit condition:** the champion can produce versioned predictions in the app, those predictions are logged, and the thresholds for drift, the trigger, and gate v2 are committed.

| Owner | Tasks | Artifact / evidence | Acceptance criteria | Coordination |
|---|---|---|---|---|
| **Person 1** | Implement the prediction logger and label joiner. Ground truth for `switch_to_lower_difficulty` is the learner's next interaction, joined on `learner_id` + sequence. Add structured JSON app logs (request ID, latency, status). Persist on the named volume and add a nightly backup. | Logging PR with tests; backup script. | Exactly one log row per prediction. No answer text or username. A label joins to the right prediction in tests. | Hook point from Person 3. |
| **Person 2** | Commit `configs/monitoring/thresholds.yaml` (PSI and KS levels per feature, prediction-PSI, minimum window size) and `configs/monitoring/retrain_trigger.yaml` (conditions, minimum labelled samples, cooldown, max runs per day). | Threshold configs merged **before** any challenger exists. | Each threshold has a one-line reason. Git timestamp predates the Day 4 experiment. | Review with Persons 1 and 3. |
| **Person 3** | Build the `ProgressModel` wrapper: load by registry alias, `predict()` returns score, label, `model_name`, `model_version`, `feature_hash`, and latency, with a timeout and a rule-based fallback. Tag questions with a topic (Track B topic list, Chemistry chapters) and capture `selected_difficulty`, `hint_count`, and `response_time_seconds` in the practice flow. Commit **promotion gate v2** thresholds. | Inference PR; topic tags with a coverage report; `configs/track_b/promotion_gate_v2.yaml`. | Feature order is checked against the contract, and a mismatch fails loudly. The app works with the model missing. Gate v2 is committed before any challenger. | Logger interface from Person 1. |
| **Person 4** | Dashboard skeleton: a Streamlit admin page (or a separate service on the same host) with role-gated access, reading fixture monitoring outputs. | Dashboard PR. | A non-admin account cannot open it. All panel placeholders map to §4.4. | Report shapes from Persons 1 and 2. |
| **All** | Shared fixtures: synthetic traffic samples, a monitoring-window report, a registry snapshot. | `tests/fixtures/monitoring/*`. | Used in everyone's tests. | — |

## Day 3 — Integration checkpoint 1: log → monitor → dashboard

**Exit condition:** live predictions on AWS flow into the log, the monitoring job produces window reports on a schedule, and the dashboard shows them.

| Owner | Tasks | Artifact / evidence | Acceptance criteria | Coordination |
|---|---|---|---|---|
| **Person 1** | Monitoring job (`src/monitoring/run_window.py`): per window, compute health (p50/p95 latency, error rate, fallback rate), input validity (schema, ranges, unseen topics), and prediction distribution. Call Person 2's drift functions. Write a JSON report plus alert rows. Schedule it on EC2 (cron or a compose sidecar). | Monitoring PR; first scheduled reports on the host. | The job runs unattended. Breaches create alerts with a severity. Windows that are too small are reported as `insufficient_data`, not as "no drift". | Thresholds from Person 2. |
| **Person 2** | Adapt drift code for windows: reference profile vs current window, PSI and KS per feature, prediction-score PSI, and label-based F1/ROC-AUC when enough labels exist. Build the **traffic generator**: replays the baseline and shifted synthetic learners through the real inference path at a set rate. | Drift-window module with tests; traffic generator script. | Same results as `drift_report.json` when given the same two datasets (regression test). Generator traffic is tagged `source=simulated`. | Inference entry point from Person 3. |
| **Person 3** | Deploy the inference hook to AWS (with Person 1). Show predictions with the model version and a short reason in *My Progress*, labelled as an estimate. | Live predictions in the log on AWS. | Each log row has the model version. The tutor tab is unaffected. | Deploy window agreed with the team. |
| **Person 4** | Connect the dashboard to the real log, monitoring reports, and MLflow. Panels: health, prediction volume, prediction distribution, and alerts. | Dashboard reading live data. | Numbers match a hand-run query on the same window. | — |
| **All** | 30-minute integration review. Every mismatch gets an owner. | Integration checklist. | No critical mismatch left without an owner. | — |

## Day 4 — Drift / degradation experiment and retraining trigger

**Exit condition:** the injected shift is detected by monitoring, the trigger fires on its own, and a challenger is trained and registered.

| Owner | Tasks | Artifact / evidence | Acceptance criteria | Coordination |
|---|---|---|---|---|
| **Person 1** | Run monitoring at a shorter interval during the experiment. Confirm alerts fire and are recorded. Make sure the retraining job runs with bounded resources on the host (or through a manual-dispatch GitHub Actions workflow if the host is too small). | Alert records; retraining execution setup. | Alerts carry a timestamp, window ID, metric, value, and threshold. | Trigger call from Person 2. |
| **Person 2** | **Experiment:** phase 1 sends baseline traffic (control); phase 2 switches to shifted traffic (the documented shift in `drift_experiment.yaml`, plus at least one new controlled shift, e.g. faster response times or a larger share of one topic). Measure when drift is detected and when performance degrades. **Trigger:** the trigger evaluates window reports, and when conditions hold (with cooldown and minimum labels) it snapshots recent labelled data, runs `dvc repro` for the candidate stage, and logs and registers the challenger with the `challenger` alias. | Experiment run log; trigger PR; challenger run plus registered version in MLflow. | The trigger fires **without anyone running it by hand**. It does not fire in the control phase. The cooldown stops repeated firing. The challenger run records the data snapshot hash, config hash, git commit, and the trigger event ID. | Thresholds are the ones committed on Day 2. |
| **Person 3** | Build the comparison job: champion vs challenger on (a) the fixed original test set, (b) the fixed shifted test set, (c) the recent labelled live window. Metrics: F1, ROC-AUC, precision, recall, calibration (Brier), latency, model size. Evaluate gate v2. | Comparison PR; comparison report. | Both models are scored on identical rows. The decision follows gate v2 exactly. An "inconclusive" result is allowed and reported as such. | Challenger from Person 2. |
| **Person 4** | Dashboard panels: drift over time (PSI per feature), performance over time, trigger events, and retraining runs. | Dashboard PR. | The experiment's shift point is visible on the time series. | — |
| **All** | Watch the experiment together for at least one shift. Record the timeline. | Experiment timeline note. | Seen in the running system, not in screenshots. | — |

## Day 5 — Comparison decision, governance, and failure handling

**Exit condition:** the promotion decision is made through the gate plus a human approval, the registry reflects it, and governance artefacts exist for both versions.

| Owner | Tasks | Artifact / evidence | Acceptance criteria | Coordination |
|---|---|---|---|---|
| **Person 1** | Failure tests for monitoring: empty window, missing labels, log write failure, monitoring job crash (its next run catches up). Test that a backup restores. | Failure-path tests; restore record. | No silent gaps. Missed windows are reported. | — |
| **Person 2** | Write the **Drift / Degradation Experiment report**: hypothesis, injected shifts, detection delay per metric, data drift vs performance drift kept separate, false-alarm count in the control phase, and limitations (synthetic data). Log it to MLflow. | Experiment report (JSON plus Markdown). | Every number can be reproduced from the logged run plus config. | — |
| **Person 3** | Log the gate v2 decision to MLflow. If it passes and is approved, move the `production` alias to the challenger and tag the old champion `archived` (or `rejected_candidate` if it fails). Test rollback: move the alias back, and the app picks it up without a rebuild. | Gate decision JSON; registry alias history; rollback record. | The alias moves **only** after Person 4 records approval. The app's live log shows the version switch. | Approval from Person 4. |
| **Person 4** | **Governance:** a model card for the champion and the challenger (purpose, data, features, metrics per test set, gate result, limitations, synthetic label, intended use and misuse); approval record (approver, timestamp, gate result reference, rationale); audit trail (which version served each prediction, from the log); retention and privacy rules; RACI. | `docs/governance/` plus `models/**/model_card.json` updates. | Every registered version has a card. The approval record links to the gate decision and the MLflow run IDs. | A section from each owner. |
| **All** | P0/P1 defect review and explicit scope decisions. | Prioritised defect list. | Every P0 is fixed or blocks the release. | — |

## Day 6 — Full integration and release candidate

**Exit condition:** everything is merged and deployed, and the whole monitoring → drift → trigger → comparison → governance chain runs on AWS.

| Owner | Tasks | Artifact / evidence | Acceptance criteria |
|---|---|---|---|
| **Person 1** | Release PR from `dev` to `main`; deploy through CD; confirm the scheduled monitoring and backup run after the deploy; update `docs/devops_runbook.md` (monitoring, alerts, retraining, rollback). | Deploy record; runbook. | A teammate can operate the system from the runbook alone. |
| **Person 2** | Run the experiment end to end once more on the release build (short version) to confirm it reproduces. | Repro record. | Same trigger behaviour as on Day 4. |
| **Person 3** | Full test suite; verify the live model version matches the `production` alias; finish the comparison documentation. | Test report. | Live log version = registry alias. |
| **Person 4** | Dashboard final pass on AWS; governance pack assembled. | Dashboard URL (admin); governance pack. | Every panel in §4.4 shows real data. |
| **All** | Freeze scope; run the regression suite. | RC tag. | No unexplained failing checks. |

## Day 7 — Graded demo and handoff

Demo order follows the graded list, so each topic has a clear owner and evidence.

| # | Topic | Presenter | What to show live |
|---|---|---|---|
| 2 | Logging | Person 1 | A practice answer in the app → a new prediction row with version, features, and latency → its label joined later. |
| 1 | Model Monitoring | Person 1 | Latest window report, an alert from the experiment, and an `insufficient_data` window. |
| 3 | Performance Dashboard | Person 4 | Health, performance, drift, alerts, retraining runs, and registry history panels. |
| 4 | Drift / Degradation Experiment | Person 2 | Shift point → detection on the dashboard; report with detection delay and control-phase false alarms. |
| 5 | Retraining Trigger | Person 2 | Trigger event → automatic `dvc repro` run → registered challenger with lineage; cooldown preventing a second run. |
| 6 | Model Comparison | Person 3 | Champion vs challenger table on three datasets; gate v2 result; alias change and rollback. |
| 7 | Governance & Model Card | Person 4 + all | Model cards, approval record, audit trail query, retention policy, model gate outcome stated honestly. |

---

## 4. Interface contracts (freeze by end of Day 2)

### 4.1 Prediction log record — owner: Person 1

| Field | Notes |
|---|---|
| `prediction_id`, `occurred_at` (UTC), `schema_version`, `request_id` | |
| `learner_id` (pseudonymous), `question_id`, `topic` | Never a username, email, or answer text. |
| `features` (JSON, in contract order), `feature_hash` | Exactly what the model received. |
| `score`, `label`, `threshold` | |
| `model_name`, `model_version`, `registry_alias`, `fallback` (bool) | Audit trail for governance. |
| `latency_ms`, `status` (`ok` / `timeout` / `error` / `fallback`) | Monitoring inputs. |
| `source` (`live` / `simulated`) | Keeps the experiment's traffic apart from real usage. |
| `label_observed`, `label_joined_at` | Filled in later by the label joiner. |

### 4.2 Monitoring window report — owners: Person 1 (health/validity), Person 2 (drift/performance)

`window_id, start, end, n_predictions, n_labelled, health{p50,p95,error_rate,fallback_rate}, validity{schema_failures,out_of_range,unseen_topics}, drift{feature: {psi, ks_stat, ks_p, status}}, prediction_psi, performance{f1, roc_auc, brier} | insufficient_data, alerts[], reference_dataset_sha256, champion_version`.

### 4.3 Retraining trigger event — owner: Person 2

`trigger_id, fired_at, condition_met (which rule), evidence_window_ids, data_snapshot_sha256, cooldown_until, pipeline_run_id, challenger_model_version | failed(reason)`.

### 4.4 Dashboard panels — owner: Person 4

1. Operational health (latency, error rate, fallback rate, volume) · 2. Prediction distribution vs reference · 3. Feature drift heatmap (PSI per feature per window) · 4. Live performance over time (with label coverage) · 5. Alerts table · 6. Trigger events and retraining runs · 7. Champion vs challenger comparison · 8. Registry and approval history (alias changes with approver).

### 4.5 Governance record — owner: Person 4

Model gate outcome · RACI per graded topic · model card per version · approval record per alias change · retention (prediction logs N days, then aggregated) and privacy rules · rollback procedure · known limitations (synthetic data, small live volume, threshold provenance).

---

## 5. Definition of done per graded topic

| Topic | Done when | Not sufficient on its own |
|---|---|---|
| 1. Model Monitoring | Scheduled job on AWS; health, validity, drift, and performance per window; alerts recorded; small windows flagged | A notebook run once; `/health` returning 200 |
| 2. Logging | Every live prediction logged with version and features; labels joined; persistent and backed up; privacy-safe | `print` statements; logs lost on redeploy; answer text in logs |
| 3. Performance Dashboard | Admin-only, live data, all §4.4 panels, numbers match source queries | Static charts or screenshots; a student progress page |
| 4. Drift / Degradation Experiment | Shift injected through the live path; detection delay measured; data vs performance drift separated; control-phase false alarms reported | Re-running the Phase 1 offline CSV comparison |
| 5. Retraining Trigger | Thresholds committed in advance; fires automatically; cooldown works; challenger registered with full lineage | Someone runs `train_candidate.py` by hand |
| 6. Model Comparison | Same rows for both models; three evaluation sets; gate v2 committed in advance; decision logged; alias change and rollback tested | Comparing on different data; thresholds edited after the results |
| 7. Governance & Model Card | Cards per version; approval record linked to the gate and MLflow; audit trail; retention; honest model identity | One generic model card; promotion without a recorded approver |

---

## 6. Priority and fallback rules

| Priority | Items | Decision rule |
|---|---|---|
| **P0** | Model gate decision; prediction logging; scheduled monitoring; drift experiment through the live path; automatic trigger; comparison with gate v2; model cards | These are the graded topics. Fix them or block the release. |
| **P1** | Dashboard with all panels; label joining for live performance; approval and rollback; backup and restore | Ship the smallest working version. If live labels are too few, show performance on simulated labelled traffic and say so. |
| **P2** | Real XGBoost champion (if found late); external alert channel (email or GitHub issue); student-journey automation carry-over | Defer with an owner and a date. Never fake a result to fill a slot. |

**If the EC2 instance is too small** for retraining: run the trigger's pipeline through a `workflow_dispatch` GitHub Actions job started by the trigger, and record that choice in the runbook.

---

## 7. Branch and review workflow

1. Branches: `feat/prediction-logging`, `feat/monitoring-job`, `feat/drift-windows`, `feat/retrain-trigger`, `feat/model-comparison`, `feat/perf-dashboard`, `docs/governance`.
2. PRs go into `dev` with CI and review. A release PR from `dev` to `main` on Day 6.
3. Threshold configs (`thresholds.yaml`, `retrain_trigger.yaml`, `promotion_gate_v2.yaml`) get their own PRs, merged before Day 4. Git history is the grading evidence that they were declared first.
4. No AWS resources are created without a cost check (the account is on the Free Plan).

---

## 8. Final acceptance checklist

Tick an item only after you have seen the evidence for it.

- [ ] **Model gate:** the champion's identity (Phase 1 XGBoost / Track B synthetic) is recorded truthfully.
- [ ] **Logging:** live predictions on AWS have version, features, latency, and source. Labels join. Logs survive a redeploy.
- [ ] **Monitoring:** the scheduled job produces window reports on AWS. Alerts fire on breaches. Small windows show `insufficient_data`.
- [ ] **Dashboard:** admin-only, all eight panels on live data, numbers checked against source queries.
- [ ] **Drift experiment:** control and shifted phases run through the live path. Detection delay and false alarms are reported. Data vs performance drift are kept separate.
- [ ] **Retraining trigger:** fired automatically from monitoring evidence, not in the control phase. Cooldown observed. Challenger registered with lineage.
- [ ] **Model comparison:** same rows, three datasets, gate v2 committed in advance, decision logged in MLflow.
- [ ] **Governance:** a card per version, an approval record before the alias change, rollback tested, retention and privacy documented.
- [ ] Synthetic-data limitations are labelled everywhere synthetic models or traffic appear.

## 9. Day 7 handoff format

| Prompt | Required answer |
|---|---|
| Which graded topic(s) did I own? | Topic number(s) and the deliverable link. |
| What evidence shows it works? | MLflow run ID, dashboard panel, log query, CI link, or live demo step. |
| What remains incomplete? | The specific gap and its impact. |
| What could fail in production? | The concrete risk and its mitigation. |
| Who owns the next step and by when? | Named person and date. |

**Release language:** if the checklist passes, call the outcome a **monitored, governed release candidate with an automated drift-to-retraining loop demonstrated on synthetic traffic**. Do not describe synthetic-data performance as evidence about real students.
