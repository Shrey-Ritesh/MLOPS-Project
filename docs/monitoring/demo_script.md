# Day 7 demo script and handoff: Person 1 (MLOps platform: logging and monitoring)

**Graded topics:** 2 (Logging) and 1 (Model Monitoring), deployed on EC2.
**Time:** about 8 minutes, then Person 2 continues with the drift experiment and the retraining trigger (`docs/day7_person2_demo_script.md`).
**Everything is synthetic.** The model and the simulated traffic come from synthetic learner data. The demo shows the operating loop, not claims about real students.

## 0. Before the demo (10 minutes beforehand)

1. Start the instance and note the new public IP. Update the security group's SSH and HTTP rules to the current IP if you are on a new network (college Wi-Fi blocks port 8501, so use port 80).
2. SSH in, then check everything is up:
   ```bash
   cd ~/MLOPS-Project
   docker compose ps                    # app, api, chroma, mlflow, monitor: all "Up", monitor "(healthy)"
   docker compose exec -T monitor python -m src.monitoring.status </dev/null | head -20
   ```
3. Open in Chrome (not Brave): `http://<ip>` (the app). In a second terminal, open the MLflow tunnel: `ssh -i ~/Desktop/jee-tutor-key.pem -L 5001:127.0.0.1:5001 ubuntu@<ip>`, then `http://localhost:5001`.
4. Have a demo learner account signed in on the Practice tab.

**If the server is unavailable,** run steps 2–4 locally against the fixture: `python -m src.monitoring.status --db tests/fixtures/monitoring/monitoring_sample.sqlite3` and `python -m src.monitoring.detection_report --db tests/fixtures/monitoring/monitoring_sample.sqlite3 --shift-at 2026-11-20T03:00:00Z --format md`.

## 1. One prediction, end to end (topic 2: logging), about 3 minutes

> *"Every prediction the app makes is written once to a monitoring database, with the model version, features, latency and status. The learner's identity is replaced by a keyed pseudonym."*

1. In the app, answer one practice question and submit.
2. Show the structured log line the app wrote:
   ```bash
   docker compose logs app --since 2m | grep '"event":"prediction"' | tail -1
   ```
   Point out: `request_id`, `model_name`, `model_version`, `status`, `fallback`, `latency_ms`. There is no learner ID and no feature values in the log line.
3. Show the row in the database, and the matching answer outcome:
   ```bash
   docker compose exec -T monitor python -m src.monitoring.status </dev/null | sed -n '/### Predictions/,/### Window status/p'
   ```
4. **Privacy check, live:** search the database for the demo learner's username. It is not there.
   ```bash
   docker compose exec -T monitor sh -c 'grep -c -a "<demo-username>" /app/data/state/monitoring.sqlite3 || true' </dev/null   # prints 0
   ```
5. Say what makes the log trustworthy:
   - **Schema:** `docs/monitoring/prediction_log_schema.md`. Every row records its `contract_id` and `schema_version`.
   - **Never breaks the app:** if logging fails, the learner still gets their answer, and the failure is logged as `prediction_log_write_failed`.
   - **Retention:** 30 days for raw rows, aggregates kept (`configs/monitoring/retention.yaml`). Backups nightly, 7 kept.

## 2. Windows and alerts (topic 1: monitoring), about 3 minutes

> *"A separate monitor container evaluates fixed 15-minute windows. It joins the delayed ground truth, checks health, input validity, drift and performance, and raises alerts against thresholds that were written down before any traffic."*

1. Show the latest windows and alerts:
   ```bash
   docker compose exec -T monitor python -m src.monitoring.status </dev/null | sed -n '/### Window status/,$p'
   ```
2. Open one full window report and walk through it: `health` (p95 latency, error, timeout and fallback rates), `validity` (schema failures, out-of-range values, unseen topics), `drift`, `performance`, then `alerts`.
   ```bash
   docker compose exec -T monitor sh -c 'ls -t /app/data/state/monitoring/windows | head -1 | xargs -I{} cat /app/data/state/monitoring/windows/{}/report.json' </dev/null | python3 -m json.tool | head -60
   ```
3. Point out the honesty rules:
   - **Small windows:** a window with fewer than 10 predictions is `insufficient_data`, never "healthy".
   - **Unmeasured drift:** drift that wasn't measured shows `not_configured` or `error`, never "no drift".
   - **Downtime:** after downtime the monitor catches up on up to 96 windows, and raises a `monitoring_gap` alert beyond that.
4. Show that the monitor itself is monitored: `docker compose ps monitor` shows `(healthy)`, from a heartbeat written every tick.

## 3. Operations, about 2 minutes

1. **Backup and restore:**
   ```bash
   docker compose exec -T monitor python -m src.monitoring.status </dev/null | sed -n '/### Newest backups/,$p'
   ```
   Then say: "We restored one backup into a fresh file and it passed the integrity check" (runbook §11.6 and §11.9).
2. **MLflow:** in the tunnel tab, show `learner-difficulty` with its `production` alias. The app loads that alias. Every logged prediction carries the version it came from, so a window report shows which model it measured.
3. **Hand over to Person 2:** "On top of these windows, Person 2 ran the drift experiment and the retraining trigger. The detection-delay numbers come from `python -m src.monitoring.detection_report`, which reads the same window table."

## 4. Handoff (Section 9)

| Prompt | Answer |
|---|---|
| **Which graded topics did I own?** | Topic 2 (Logging) and topic 1 (Model Monitoring), including their deployment on EC2. |
| **What evidence shows it works?** | <ul><li>**Tests:** `tests/test_monitoring_*.py`. They cover the logger, label joining, windows, failure paths, contracts, retention, the detection report and status, plus `tests/test_monitoring_integration.py`: the app's inference path through to a window report.</li><li>**Server:** runbook §11.9 verification record.</li><li>**Fixture evidence:** `tests/fixtures/monitoring/` (6 hours of windows, with an outage, a latency spike, a quiet window and an unseen topic).</li><li>**Detection report:** `docs/monitoring/monitoring_job.md` §6.</li></ul> |
| **What remains incomplete?** | <ul><li>No external alert channel (email or GitHub issue). Alerts go to the database, the JSON logs and the dashboard.</li><li>The chapter-mastery contract is a tested draft, not active.</li><li>The MLflow registry is not in the nightly backup yet. It is copied by hand (runbook §12).</li></ul> |
| **What could fail in production?** | <ul><li>**Database:** SQLite on one instance. It is fine at this traffic, but there is a single writer and no replication. If the disk is lost, data since the last nightly backup is lost.</li><li>**Key:** losing `LOG_PSEUDONYM_KEY` breaks the link between old and new rows for each learner.</li><li>**Labels:** a learner who never comes back leaves their last prediction unlabelled. It expires after 72 hours.</li></ul> |
| **Who owns the next step?** | <ul><li>**Person 4:** dashboard panels read `monitoring_windows` and `alerts`.</li><li>**Person 2:** the trigger reads the same tables.</li><li>**Person 3:** seeds the registry after the release.</li></ul> |
