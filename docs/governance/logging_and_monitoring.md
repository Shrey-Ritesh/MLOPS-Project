# Governance section: Logging and Monitoring (Person 1)

Input to the governance record (lead: Person 4). It covers the parts of §4.5 that come from the
logging and monitoring platform: the audit trail of which model served each prediction, data
retention, privacy rules, backups, and alert records. Everything here describes **synthetic**
learners and simulated traffic, plus a few test accounts; nothing is evidence about real students.

Sources of truth: `src/monitoring/schema.py` (tables), `configs/monitoring/retention.yaml`
(retention), `docs/monitoring/prediction_log_schema.md` (log contract), `docs/devops_runbook.md`
§11 (operations) and §11.9 (server verification).

## 1. Audit trail: which model version served each prediction

Every prediction made by the app (and by the traffic simulator) writes exactly one row to the
`predictions` table in `/app/data/state/monitoring.sqlite3` on the `app_state` volume. Each row
records the model that produced it:

| Column | What it proves |
|---|---|
| `model_name`, `model_version` | Registry name and version, e.g. `learner-difficulty` / `2`, or `rule_fallback` / `fallback-v1` |
| `model_id` | Model card id, e.g. `candidate_hgb_shifted_v2`, linking to the model card |
| `registry_alias` | The alias it was loaded by (`production`) |
| `fallback`, `status` | Whether the rule-based fallback answered instead of the model |
| `features`, `feature_hash`, `threshold`, `score`, `label` | Exactly what the model received and returned |
| `source` | `live` (real app use) or `simulated` (experiment traffic) |
| `occurred_at`, `request_id` | When, and the matching line in the JSON app log |

**Audit query** (which version served how many predictions, and when). Run on the server:

```bash
docker compose exec monitor python -c "
import sqlite3
c = sqlite3.connect('/app/data/state/monitoring.sqlite3')
q = '''SELECT model_name, model_version, registry_alias, source, COUNT(*),
              MIN(occurred_at), MAX(occurred_at)
       FROM predictions GROUP BY 1, 2, 3, 4 ORDER BY 6'''
for r in c.execute(q): print(r)"
```

After an alias change (promotion or rollback), the version column changes from the next prediction
on, without a rebuild, so this query shows the switch time. The approval that allowed the switch
is recorded in Person 4's approval record; this table shows that it actually happened.

## 2. Privacy rules

| Rule | How it is enforced |
|---|---|
| No username, email or raw learner id is stored | The logger stores `learner_key = HMAC-SHA256(LOG_PSEUDONYM_KEY, learner_id)`, first 16 hex characters. The raw id contains the username, so it is never written. |
| No question text, answer text, OCR output or images | Not part of the record; only the eight model features are stored. |
| No exception messages | Only the exception class name (`error_type`), because messages can contain input values. |
| App logs carry no personal data | Phase 3 log events are one JSON line each and pass an allowlist (`src/monitoring/app_logging.py`); learner ids, features and answer text cannot appear. |
| The pseudonym key is a secret | Kept only in the server's `.env` (mode 600), never in git or chat, and never rotated (rotating breaks label joining). A leak is handled as an incident and recorded here. |
| Experiment traffic is kept apart | Every simulated row has `source = simulated`; the simulator also uses its own learner ids. |

**Verified on the server, 28 Nov 2026:** after real practice answers from a test account, its
username was searched for in the monitoring database, its WAL files, the window snapshots and the
`app` / `monitor` logs: **0 occurrences** (runbook §11.9). The same check runs in the integration
tests.

## 3. Data retention

Applied by the monitor once a day, **only after that day's backup succeeded**
(`configs/monitoring/retention.yaml`):

| Data | Kept for | Reason |
|---|---|---|
| Raw `predictions` rows (features, learner key) | 30 days | Retraining uses the recent labelled window; labels expire after 72 h |
| `attempt_outcomes` rows | 30 days | Only needed to join labels |
| Window snapshot files (`windows/<id>/predictions.csv`) | 30 days | Hold raw rows; the report itself is kept in the database |
| Window reports (`monitoring_windows`) | Kept | Aggregates only, no learner keys; needed for trends |
| Alerts | Kept | Audit trail of what monitoring saw |
| Nightly backups | Last 7 | A deleted raw row survives at most 7 more days inside a backup |

Rows still waiting for a label are never deleted. Preview what would be deleted:
`docker compose exec monitor python -m src.monitoring.retention --dry-run`.

## 4. Backups and restore

- Nightly after 02:00 UTC, the monitor copies the database with SQLite's online backup API,
  checks its integrity and keeps the last 7 copies on the `monitoring_backups` volume.
  A failed backup logs `backup_failed` and is retried the same day.
- A restore never overwrites an existing file; it restores into a new file that is then swapped in
  (runbook §11.6).
- **Verified on the server, 28 Nov 2026:** backup, then restore into a scratch file: integrity ok,
  all predictions, labels and outcomes present.
- The MLflow registry (`/mlflow/mlflow.db`) is not in the nightly backup; it is copied by hand after
  every registry change (runbook §12). This is a known limitation.

## 5. Monitoring and alert records

- The `monitor` service evaluates one window at a time (15 minutes normally, 2 minutes during the
  drift experiment): service health (p50/p95 latency, error and fallback rate), input validity,
  prediction distribution, feature drift and, once labels have joined, live F1 / ROC-AUC.
- Windows with too few predictions are recorded as `insufficient_data`, never as "no drift".
- Every breach writes a row to the `alerts` table with `created_at`, `window_id`, `severity`
  (`warning` / `critical`), `category`, `metric`, `value`, `threshold` and a message, and an
  `alert_raised` line in the monitor log. Alerts are kept permanently.
- Missed windows are caught up on the next tick (up to a limit), so there are no silent gaps.

Alert query:

```bash
docker compose exec monitor python -c "
import sqlite3
c = sqlite3.connect('/app/data/state/monitoring.sqlite3')
for r in c.execute('SELECT created_at, window_id, severity, metric, value, threshold FROM alerts ORDER BY created_at'):
    print(r)"
```

## 6. Retraining runs

The monitor hands each window to the retraining trigger (Person 2). Retraining runs inside the
`monitor` container, capped at 1 CPU and 1 GB, and writes only to
`/app/data/state/retraining/`, never over the committed champion. Each trigger event is appended to
`trigger_events.jsonl` there, and the challenger is registered in MLflow with the `challenger`
alias and lineage tags (data snapshot hash, config hash, git commit, trigger id). The trigger is off
by default and switched on only for the experiment (runbook §11.4).

## 7. Roles

| Responsibility | Owner |
|---|---|
| Prediction log, label joining, retention, backups, monitoring job, alerts, deployment | Person 1 |
| Drift thresholds, retraining trigger policy | Person 2 |
| Model loading, promotion gate, alias changes | Person 3 |
| Approval record, dashboard, governance pack | Person 4 |

## 8. Known limitations

- Synthetic learners and simulated traffic only; live volume is a handful of test answers.
- One EC2 host: a lost disk loses both the database and its backups (no off-host copy, to stay on
  the AWS Free Plan).
- The MLflow registry is backed up by hand, not automatically.
- Retention deletes raw rows after 30 days, so the per-prediction audit trail covers 30 days;
  window reports and alerts cover the whole project.
