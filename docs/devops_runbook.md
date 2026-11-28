# DevOps runbook and release checklist (Phase 2 + Phase 3 monitoring)

Scope: sections 1-9 describe a tested **release candidate** with documented production blockers; running the stack locally uses no cloud credentials. Section 10 describes the AWS deployment (EC2, ECR, GitHub Actions) set up afterwards, which does use AWS credentials. Section 11 covers Phase 3 prediction logging and monitoring. It is a **draft**: it is tested locally, but has not yet been run on the server. Each step there says whether it has been verified.

## 1. Services and ports

| Service | Image/command | Host port | Health check |
|---|---|---|---|
| `chroma` | `chromadb/chroma:1.5.3` | 8000 | `/api/v2/heartbeat` |
| `api` | this repo, `uvicorn src.api.app:app` | 8001 | `GET /health` (503 until questions are loaded) |
| `app` | this repo, Streamlit | 80 and 8501 | `/_stcore/health` |
| `monitor` | this repo, `python -m src.monitoring.scheduler` (0.5 CPU, 512 MB) | - | heartbeat no older than 5 min (`--check-heartbeat 300`) |
| `indexer` | this repo, `profiles: [tools]`, run on demand | - | exits 0 when both indexes are built |

Start: `docker compose up --build -d` (after the index steps in the README). Status: `docker compose ps` shows health for each service.

## 2. Configuration and secrets guide

Copy `.env.example` to `.env`. `.env` is git-ignored and excluded from the image by `.dockerignore`; secrets are injected at runtime, never baked in.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY` | yes (app) | - | The `app` container refuses to start without it and says so by name only |
| `GROQ_MODEL` | no | code default `qwen/qwen3.8-27b`; `.env.example` sets `openai/gpt-oss-120b` | chat model; the value in `.env` wins |
| `OCR_SPACE_API_KEY` | no | - | handwriting transcription is disabled with a warning if missing |
| `CHROMA_HOST`, `CHROMA_PORT` | no | from `configs/*.yaml`; compose sets `chroma:8000` | |
| `JEE_QUESTIONS_PATH` | no | `data/processed/jee_questions.jsonl`, then the checked-in fixture | the API reports 503 on `/health` if no questions load |
| `CORS_ALLOWED_ORIGINS` | no | `http://localhost:8501,http://127.0.0.1:8501` | comma-separated; never `*` by default; credentials are disabled |
| `MAX_REQUEST_BYTES` | no | `1000000` | API request body limit; larger requests get HTTP 413 |
| `MLFLOW_TRACKING_URI` | no | local sqlite | Track B only |
| `HF_HOME` | no | `/cache/huggingface` in containers | embedding model cache (named volume) |
| `USERS_CSV_PATH`, `LEARNER_DB_PATH` | no | `data/users.csv`, `data/learner_history.sqlite3`; compose sets `/app/data/state/...` | student accounts and progress; the `app_state` volume keeps them across deploys |
| `SEED_DEMO_ACCOUNT` | no | off | `1` creates a demo login (`student` / `password123`) on first start; local testing only |
| `LOG_PSEUDONYM_KEY` | yes for prediction logging | - | Secret (at least 16 characters) used to pseudonymise learner IDs in the prediction log. If it is missing, the app starts with a warning and prediction logging is off. **Never change it once set** (§11.6). |
| `MONITORING_DB_PATH` | no | `data/state/monitoring.sqlite3`; compose sets `/app/data/state/monitoring.sqlite3` | prediction log, window reports and alerts, on the `app_state` volume |
| `MONITORING_WINDOW_MINUTES` | no | `15` | monitoring window length; set to `2` during the drift experiment (§11.4) |
| `MONITORING_TRIGGER_ENABLED` | no | `0` | `1` makes the monitor pass each window to the retraining trigger (drift experiment, §11.4) |
| `MONITORING_TRIGGER_SIMULATED` | no | `0` | `1` uses the experiment cooldown (5 windows) instead of 6 hours |
| `MONITORING_TRIGGER_EXECUTE` | no | `1` | `0` = dry run: record the trigger event and snapshot, skip retraining |
| `MONITORING_TICK_SECONDS` | no | `60` | how often the monitor wakes up |
| `MONITORING_CONTRACT` | no | `configs/monitoring/contracts/track_b_difficulty_v1.yaml` | Model contract (features and label rule). Must be the same for `app` and `monitor`; switching models is described in the schema doc, §7. |

Check configuration without starting anything: `python -m src.settings` (prints variable names only, never values).

## 3. CI stages

| Workflow / job | Trigger | What it checks | Required check |
|---|---|---|---|
| `CI / checks` | every PR and push to `dev`/`main` | YAML config validation, error-only lint, full unit suite | yes |
| `CI / api-smoke` | same | real `uvicorn` server on fixture questions + `scripts/smoke_test_api.py` (8 checks); no Chroma, MLflow or secrets | no |
| `Docker build and smoke test` | PRs touching image inputs | image builds, compose file valid, missing config fails clearly, API smoke test in the container, Streamlit health | no |

Local equivalents:

```bash
python -m pytest -q
JEE_QUESTIONS_PATH=tests/fixtures/jee_questions_sample.jsonl python -m uvicorn src.api.app:app --port 8000 &
python scripts/smoke_test_api.py http://127.0.0.1:8000
```

Dependency note: `requirements.txt` is the full development freeze. `requirements-app.txt` is the runtime subset used by the image (CPU-only `torch`). `requirements-ci.txt` pins `fastapi`/`starlette` to the same versions as `requirements.txt` so CI tests what ships. The image uses Python 3.12 because `numpy==2.5.4` does not support 3.11.

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `api` unhealthy, `/health` returns 503 | no questions loaded | `dvc pull data/processed.dvc`, or set `JEE_QUESTIONS_PATH` to a valid JSONL |
| Every API request returns 500 | FastAPI/Starlette version mismatch | install the pinned versions; `tests/test_api_startup.py` guards this |
| `app` container restarts in a loop | missing/placeholder `GROQ_API_KEY` | check `docker compose logs app`; set the key in `.env` |
| Answers have no NCERT evidence | Chroma not running or indexes not built | `docker compose up -d chroma`, then `docker compose run --rm indexer`; the API degrades to "retrieval unavailable" instead of failing |
| `pip install` fails in Docker build with a numpy error | base image Python < 3.12 | keep `python:3.12-slim` |
| Docker build is killed (exit 137) | not enough memory in the Docker VM | give Docker at least 4 GB (Colima: `colima start --memory 6`) |
| `docker compose` is "unknown command" | Compose plugin not installed | use the standalone `docker-compose` binary |
| Indexer ends with `PermissionError: ... 'reports/retrieval'` | the `index_reports` volume was created root-owned by an image that lacked `/app/reports` (fixed in the `Dockerfile`) | rebuild the image, `docker volume rm <project>_index_reports`, run the indexer again (it uses upsert, so re-runs are safe) |
| Answer shows "generated without project references because the local retrieval index was unavailable" although the indexes are built | the `app` container could not read `data/processed/jee_questions.jsonl` (fixed: `app` now mounts `./data/processed`) | check `docker compose exec app ls /app/data/processed`; test with `docker compose exec app python -c "from src.retrieval.retrieve import retrieve; print(retrieve('isomerism','medium',subject='Chemistry')['status'])"` |

## 5. Logs

- All services log to stdout/stderr: `docker compose logs -f api app`.
- The API logs warnings for skipped invalid records, retrieval outages and write failures. Learner content is never logged: the retrieval warning logs only the query length and exception type, and invalid-attempt warnings log only the exception type (covered by `tests/test_api_hardening.py`).
- Attempt records (which contain learner answers) live in the `attempts_data` volume at `/app/data/attempts/attempts.jsonl`, not in logs.
- Phase 3 events are **one JSON line each** (`"event": "prediction"`, `window_evaluated`, `alert_raised`, `backup_completed`, and so on). Only fields on an allowlist are written (`src/monitoring/app_logging.py`), so learner IDs, features and answer text can never appear in them. Filter them with `docker compose logs monitor | grep '"event":"alert_raised"'`.
- Container logs are capped at 10 MB × 5 files per service (`x-logging` in `docker-compose.yml`), and they are lost when a container is recreated. The lasting record is the `predictions` table, not the container logs.

## 6. Rollback / revert

Code: `main` and `dev` only accept PRs.
1. Find the merge commit of the bad PR: `git log --merges --oneline -10 origin/main`.
2. Revert it on a branch: `git checkout -b revert/<pr> origin/main && git revert -m 1 <merge-sha>`.
3. Open a PR; CI must pass; merge.

Running stack: `git checkout <last-good-sha> && docker compose up --build -d`. Named volumes (`chroma_data`, `attempts_data`, `hf_cache`) survive `docker compose down`; only `down -v` deletes them. Indexes can always be rebuilt from `data/processed` with the `indexer` service. A bad embedding/index change is rolled back by re-running the indexer at the previous config.

## 7. Security and production-readiness checklist

Done in this repository:
- [x] No secrets in git or the image: `.env*` ignored (except `.env.example`), `.dockerignore` excludes `.env`, `.git`, `data`, `models`, `reports`, `tests`.
- [x] Containers run as a non-root user.
- [x] Multi-stage image with CPU-only torch; runtime has no build tooling.
- [x] Required configuration validated at startup with name-only errors.
- [x] API request bodies bounded (`MAX_REQUEST_BYTES`, 413); OCR images bounded to 1 MB / 25 MP by the handwriting module.
- [x] CORS restricted to configured origins; credentials disabled.
- [x] Errors return a sanitized envelope; the 500 handler never returns stack traces or config.
- [x] Learner content and secrets kept out of logs.
- [x] Dependencies in the image are pinned.

Still required before any real deployment (not done here, by design):
- [ ] Authentication/authorization on the API and Streamlit app (currently none; anyone who can reach the ports can read attempts by session id).
- [ ] TLS termination and a reverse proxy; do not expose ports 8000/8001/8501 directly.
- [ ] Rate limiting and per-request timeouts at the proxy.
- [ ] A secrets manager and short-lived credentials instead of `.env` files.
- [ ] Vulnerability scanning of the image and dependencies (e.g. `pip-audit`, Trivy) and a patch cadence.
- [ ] Retention, backup and deletion policy for `attempts.jsonl` (learner answers are personal data); consent and privacy review, since uploaded handwriting images are sent to OCR.space.
- [ ] Monitoring/alerting (metrics, uptime checks) and log shipping. *Phase 3 adds model monitoring and an alerts table (§11). There are still no uptime checks from outside the server and no external alert channel.*
- [ ] Hosted CD with protected-environment approval.

## 8. Known limitations

- The Streamlit page does not yet submit confirmed OCR text to the evaluator API (see README).
- The Phase 2 corpus was reduced to Chemistry, but the Streamlit page still offers Physics and Mathematics. Behaviour for those subjects (expected: answers without project references) has not been tested here.
- The `checks` workflow is the only required status check; `api-smoke` and the Docker job are advisory.
- Admin bypass of branch protection is allowed so the owner can recover from mistakes.
- Embedding model is downloaded from Hugging Face on first use; first start needs network access.

## 9. Clean-start test record (2026-11-08)

Method: `git clone` of the branch into a fresh directory, `docker-compose up -d --build chroma api app` with only `GROQ_API_KEY` set, Chroma published on a non-default host port because 8000 was already taken on the test machine. Tested by the author on one machine (macOS, Colima, 6 GB / 4 CPU); **not yet repeated by a teammate**.

| Step | Result |
|---|---|
| Image build from a clean clone | succeeds (about 2 GB) |
| Fresh clone, no `data/processed` (no DVC pull) | API `/health` returns **503** with a clear reason ("no questions loaded"); `chroma` and `app` healthy |
| Fixture copied to `data/processed/jee_questions.jsonl` before `up` | API healthy with 25 questions; `scripts/smoke_test_api.py` 8/8; Streamlit `/_stcore/health` ok |
| Full data (215 Chemistry questions) | API healthy; smoke 8/8 |
| Chroma stopped while running | API stays up, `/health` 200, smoke 8/8; evidence is reported as unavailable |
| `GROQ_API_KEY` empty | `app` exits 1 with a name-only message |
| Oversized request (1.5 MB) | HTTP 413 |
| Cross-origin request from an unlisted origin | no CORS headers returned |
| Containers | run as the non-root `app` user |

Setup notes found during the test:
- Without DVC access, copy `tests/fixtures/jee_questions_sample.jsonl` to `data/processed/jee_questions.jsonl` before starting the stack. This is a demo-size dataset only.
- With Colima, bind mounts only work for paths under your home directory.
- Retrieval evidence needs the indexes (`docker-compose run --rm indexer`), which needs the processed data and a first-run model download; this was not exercised in the clean-start test. It was exercised later on the EC2 server (section 10), which exposed two bugs now listed in section 4.
- The Docker image was never buildable from `dev` before this work (Python 3.11 vs `numpy==2.5.4`), and `docker compose up` could never start the API and app (the Chroma healthcheck used `curl`, which that image lacks).

## 10. AWS deployment (EC2 + ECR + GitHub Actions)

**Status:** the server, registry and credentials are set up. On 2026-11-09 the stack was run on the server by hand with real API keys and the Chemistry data, and a Chemistry question was answered in the browser with NCERT and JEE references. The deploy workflow was also run end to end the same day from a temporary branch (build, push to ECR, SSM deploy; the server came up healthy on the new image). Its first run from `main` has not happened yet.

### 10.1 Design

One EC2 server in the team AWS account (region `ap-southeast-2`) runs the same `docker-compose.yml` as local development. On every push to `main`, `.github/workflows/deploy.yml` builds the image, pushes it to ECR, and tells the server through AWS SSM Run Command (no SSH) to run `scripts/deploy.sh`, which pulls the image and restarts `chroma`, `api`, `app` and (from Phase 3) `monitor`. Compose uses the image named by `APP_IMAGE`; without it, compose builds locally as before.

| Resource | Value |
|---|---|
| Server | EC2 `m7i-flex.large` (2 vCPU, 8 GiB), Ubuntu 24.04 amd64, 30 GiB gp3, name/tag `jee-tutor` / `project=jee-tutor` |
| Server IAM role | `jee-tutor-ec2-role`: SSM core, ECR read-only, read-only access to the DVC bucket |
| Registry | ECR private repo `jee-tutor` (lifecycle rule keeps the 5 newest images) |
| CI credentials | IAM user `jee-tutor-github-deploy`, policy in `deploy/aws/github-deploy-user-policy.json` (push to one repo, run a command on one instance) |
| GitHub config | secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; variable `DEPLOY_INSTANCE_ID` |
| Budget | AWS Budgets `jee-tutor-monthly`, $50, alerts to the team |

### 10.2 Constraints of this AWS account

- It is on the AWS **Free Plan**: spending is covered by credits (about $100, expiring 2027-04-05), and **services stop when the credits run out** instead of billing a card.
- Only "free tier eligible" instance types can be launched. `t3.medium` is not allowed; the smallest with at least 4 GiB is `c7i-flex.large`; `m7i-flex.large` (8 GiB) was chosen to leave headroom for the embedding model and Chroma.
- An organization service control policy **denies creating an OIDC identity provider**, so GitHub OIDC could not be used. CI therefore uses access keys of a least-privilege IAM user stored as GitHub secrets. Rotate or delete that key when the project ends.
- **Do not activate "advanced features"** in AWS Settings to work around this. It requires upgrading off the Free Plan, **cannot be reversed**, and removes spend limits.

### 10.3 Server setup

`deploy/server/bootstrap.sh` installs Docker, the AWS CLI, DVC and clones the repo. Then, on the server:

```bash
cd ~/MLOPS-Project
cp .env.example .env        # fill in GROQ_API_KEY and OCR_SPACE_API_KEY; never commit it
~/dvcenv/bin/dvc pull data/processed.dvc
docker compose --profile tools run --rm indexer   # builds the Chroma indexes (first run downloads the embedding model)
docker compose up -d --build chroma api app
```

DVC on EC2 reads its credentials from the instance role. Without `AWS_METADATA_SERVICE_TIMEOUT=5`, `AWS_METADATA_SERVICE_NUM_ATTEMPTS=3` and `AWS_DEFAULT_REGION=ap-southeast-2` (set by the bootstrap script) the first credential lookup can time out and DVC reports misleading "missing files" errors.

**Data must be pushed before it can be pulled.** `dvc pull` fails with `Missing cache files` if the commit that updated `data/processed.dvc` was never followed by `dvc push`. This happened once: the Chemistry-only data (hash `27ad98...`) was committed on 2026-11-05 but not uploaded to the bucket. After changing a `.dvc` file, run `dvc push` before opening the PR.

### 10.4 Deploying and rolling back

- **Normal deploy:** merge a PR to `main`. The server must be running and registered with SSM, or the workflow fails at the deploy step.
- **Manual deploy or rollback** (on the server): check out the wanted commit, then `bash scripts/deploy.sh <account>.dkr.ecr.ap-southeast-2.amazonaws.com/jee-tutor:<git-sha>`. ECR keeps the last 5 images.
- **Failed deploy:** the workflow prints the script output and the last API log lines. The previous containers keep running until `docker compose up` replaces them.

### 10.5 Cost and operations

- The instance costs about $0.12 per hour while running (about $87 per month if left on); a stopped instance costs only its disk. **Stop it when it is not needed** (EC2 console: Instance state -> Stop). Never choose Terminate, which deletes the server and its disk.
- A stopped-then-started instance gets a **new public IP address**. Update any SSH rule or bookmark, or attach an Elastic IP for the demo.
- SSH (port 22) is allowed only from named IP addresses. To work from another network, add that IP to the security group, or use SSM Session Manager from the EC2 console (EC2 -> Connect -> Session Manager).
- The app is published on port 80 as well as 8501, so it opens as `http://<public-ip>`. Use port 80: some networks block 8501 (the college Wi-Fi does). Allow port 80 in the security group for the networks that need it, or Anywhere-IPv4 only for a demo, and see section 7 for the missing authentication and TLS.
- Some networks send traffic out through more than one public IP (the college uses at least two), so a rule for one IP may not cover every device there.
- There is no HTTPS yet. Browsers that upgrade to `https://` automatically (Brave, for example) fail with "connection reset"; use Chrome or Edge, or turn the upgrade off for the site.

## 11. Monitoring and prediction logging (Phase 3, owner: Person 1)

**Status: running on the server since 28 Nov 2026** (release #30, with the fixes released in #32 and #34). The live check on 28 Nov is recorded in §11.9: predictions, labels, a window, a backup and a restore. Steps still open (the window-length change and alerts from real traffic) happen in the Day 4 experiment.

Background docs: `docs/monitoring/prediction_log_schema.md` (what is logged), `docs/monitoring/monitoring_job.md` (window report format), `configs/monitoring/health.yaml` (thresholds and the reason for each).

### 11.1 How it fits together

```
practice answer -> app: model predicts -> PredictionLogger -> predictions table ──┐
                                                                                  │  monitoring.sqlite3
monitor (every 60 s): join labels -> evaluate due windows -> alerts ─────────────┤  (app_state volume)
                      nightly 02:00 UTC: backup -> monitoring_backups volume      │
dashboard (Person 4), retraining trigger (Person 2): read only ──────────────────┘
```

| Data | Where | Survives redeploy | Backed up |
|---|---|---|---|
| Predictions, window reports, alerts, scheduler state | `/app/data/state/monitoring.sqlite3` (`app_state` volume) | yes | nightly, 7 copies |
| Per-window snapshots (`report.json`, `predictions.csv`) | `/app/data/state/monitoring/windows/<window_id>/` | yes | no (can be rebuilt from the database) |
| Backups | `/app/data/backups/monitoring-<UTC timestamp>.sqlite3` (`monitoring_backups` volume) | yes | — |

### 11.2 First-time setup on the server (once, before the first Phase 3 deploy)

1. **Done 2026-11-27.** Generate the pseudonym key **once** on the server (`openssl rand -hex 32`).
2. **Done 2026-11-27.** Add it to `~/MLOPS-Project/.env` as `LOG_PSEUDONYM_KEY=<value>`. Never commit it, paste it into chat, or rotate it. If the instance is ever rebuilt, copy the old `.env` across instead of generating a new key, or old and new rows stop linking up.
3. **Done 2026-11-28 (#30).** Merge the release into `main`. `deploy.sh` pulls and starts `monitor` together with the other services.
4. **Done 2026-11-28: 0 warnings.** Check (§11.3) that `monitor` is healthy and that the app log contains **no** `LOG_PSEUDONYM_KEY is not set` warning:
   `docker compose logs app | grep -c LOG_PSEUDONYM_KEY` should print `0`.

### 11.3 Daily checks

```bash
cd ~/MLOPS-Project
docker compose ps monitor                                  # expect "healthy"
docker compose logs --since 1h monitor | grep -E 'monitor_tick_failed|alert_raised|backup_failed'
docker compose exec -T monitor python - <<'PY'
import sqlite3, json
c = sqlite3.connect("/app/data/state/monitoring.sqlite3")
print("predictions:", c.execute("select count(*) from predictions").fetchone()[0])
print("last windows:")
for row in c.execute("select window_id, status, n_predictions, n_labelled from monitoring_windows order by window_end desc limit 5"):
    print("  ", row)
print("open alerts (24h):")
for row in c.execute("select created_at, severity, metric, value, threshold from alerts where created_at > strftime('%Y-%m-%dT%H:%M:%SZ','now','-1 day') order by created_at desc limit 20"):
    print("  ", row)
print("state:", dict(c.execute("select key, value from monitor_state")))
PY
```

How to read the result:

- `insufficient_data` windows are normal when there is little traffic (fewer than 10 predictions). They are **not** a fault.
- `last_window_end` should be within about 20 minutes of now (window length plus a 5-minute label lag).
- `last_backup_date` should be today, or yesterday if it is before 02:00 UTC.

### 11.4 Drift experiment setup (Day 4): window length and the retraining trigger

The `monitor` passes every evaluated window to Person 2's retraining trigger (`src/monitoring/trigger.py`) when `MONITORING_TRIGGER_ENABLED=1`. The trigger decides on its own: conditions, cooldown, daily cap and minimum records all live in `configs/monitoring/retrain_trigger.yaml`. Nobody runs it by hand. It is **off by default**, so normal days never retrain.

**Before the experiment**, on the server:

```bash
cd ~/MLOPS-Project
cp .env .env.bak-$(date +%Y%m%d)
cat >> .env <<'ENV'
MONITORING_WINDOW_MINUTES=2
MONITORING_TRIGGER_ENABLED=1
MONITORING_TRIGGER_SIMULATED=1
ENV
docker compose up -d monitor                          # recreates only the monitor container
docker compose logs monitor | grep monitor_started    # expect "window=2m"
```

`MONITORING_TRIGGER_SIMULATED=1` uses the experiment cooldown: `simulated_cooldown_windows` × window length, i.e. 5 × 2 min. Add `MONITORING_TRIGGER_EXECUTE=0` for a dry run that records the trigger event and snapshot without training.

**During the experiment:**

```bash
docker compose logs -f monitor | grep -E 'window_evaluated|alert_raised|trigger_(fired|blocked|failed)'
docker compose exec monitor cat /app/data/state/retraining/trigger_events.jsonl   # one line per firing
```

When the trigger fires, it:
1. snapshots the window's labelled rows to `/app/data/state/retraining/snapshots/` and records the SHA-256;
2. tries `dvc repro train_challenger`. `dvc` is not in the image, so it falls back to `python -m src.track_b.train_candidate`, which writes to `/app/data/state/retraining/challenger/`;
3. registers the challenger in MLflow with alias `challenger` and tags `trigger_id`, `snapshot_sha256` and `evidence_window_ids`. `production` is not touched. The model card records `git_commit` (baked into the image at build time), `config_sha256` and `dataset_sha256`.

**Resources:** `monitor` is capped at 1 CPU and 1 GB. A retrain measured locally takes ~3 s of training (~9 s including registration) and ~300 MB peak, inside one monitor tick, well within the 5-minute heartbeat. The training data and baseline are mounted read-only from the checkout (`./data/synthetic`, `./models/track_b/baseline`). Outputs live on the `app_state` volume, never over the committed champion in `models/track_b/candidate/`.

**For Person 3's comparison**, the challenger is at `/app/data/state/retraining/challenger` in the `monitor` container, and in MLflow as `learner-difficulty@challenger`.

**After the experiment:** remove the three lines from `.env` (or restore the backup), then run `docker compose up -d monitor`. Windows go back to 15 minutes with a short bridge window, with no gap and no overlap, and the trigger is off again.

### 11.5 Alerts: what they mean and what to do

| Metric | Category | Likely cause | First action |
|---|---|---|---|
| `error_rate`, `timeout_rate` | health | The model fails or is too slow | `docker compose logs app \| grep prediction_log_write_failed`; check the model loaded (Person 3) |
| `fallback_rate` | health | The model could not load, so the rule-based fallback is serving | Check the app's startup logs for model loading; check the `production` alias / model file |
| `latency_p95_ms` | health | CPU starvation (retraining running?) or a cold model load | `docker stats`; retraining is capped, so check the caps were not removed |
| `schema_failure_rate` | validity | Code and contract disagree (feature names, schema version) | A bug: compare the inference hook with `src/monitoring/schema.py` |
| `out_of_range_rate` | validity | Impossible values captured (negative counts, accuracy above 1) | A capture bug in the practice flow (Person 3) |
| `unseen_topic_rate` | validity | Questions tagged with topics the model never saw | Fix the topic tags, or accept and note it (it also means retraining should be considered) |
| `psi.*`, performance metrics | drift / performance | Input or accuracy drift (Person 2's evaluator) | Look at the dashboard trend. The retraining trigger acts on these, not people. |
| `monitoring_gap` | monitoring | The monitor was down longer than the catch-up limit (96 windows) | Find out why it was down (`docker compose ps`, instance stopped?). The skipped windows are named in the alert message. |

### 11.6 Backup and restore

- **Automatic:** nightly after 02:00 UTC. Each backup is checked for integrity, and the last 7 are kept. A failed backup logs `backup_failed` and is retried on the next tick the same day.
- **Manual backup:** `docker compose exec monitor python -m src.monitoring.backup --dest /app/data/backups`
- **Restore** (the restore command never overwrites an existing file):

  ```bash
  docker compose exec monitor ls /app/data/backups
  docker compose stop app monitor                       # nothing may write while the live file is replaced
  docker compose run --rm --no-deps monitor python -m src.monitoring.backup \
      --restore /app/data/backups/monitoring-<timestamp>.sqlite3 --to /app/data/state/monitoring.restored.sqlite3
  docker compose run --rm --no-deps monitor sh -c 'cd /app/data/state && \
      mv monitoring.sqlite3 monitoring.broken-$(date -u +%Y%m%dT%H%M%SZ).sqlite3 && \
      rm -f monitoring.sqlite3-wal monitoring.sqlite3-shm && mv monitoring.restored.sqlite3 monitoring.sqlite3'
  docker compose up -d app monitor
  ```

  Predictions made between the backup and the restore are lost. The monitor re-evaluates any windows after its saved `last_window_end`.
- **Retention** (`configs/monitoring/retention.yaml`): once a day, and **only after that day's backup succeeded**, the monitor deletes raw `predictions` and `attempt_outcomes` rows and window snapshot folders older than 30 days. Window reports and alerts are kept. Rows still waiting for a label are never deleted. To preview what would be deleted: `docker compose exec monitor python -m src.monitoring.retention --dry-run`. A deleted row can survive up to 7 more days inside a backup.
- **Never rotate `LOG_PSEUDONYM_KEY`.** Old and new rows for the same learner would stop matching, which breaks label joining for every learner who is active at that moment. If the key leaks, treat it as an incident: decide as a team, and record it in the governance record (Person 4).

### 11.7 Rollback

- **Bad monitoring code or thresholds:** roll back like any other release (§6, §10.4). The database schema only ever gains tables with `CREATE ... IF NOT EXISTS`, so older code still runs against a newer database.
- **Stop monitoring without touching the app:** `docker compose stop monitor`. The app keeps logging predictions. When `monitor` starts again, it catches up on missed windows (up to 96) and raises `monitoring_gap` for the rest.
- **Model rollback** (alias change) is Person 3's procedure. This section only makes sure it shows up in the log: every prediction row records `model_version` and `registry_alias`, so the switch is visible in `health.by_model_version` in the next window report.

### 11.8 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `monitor` unhealthy | No tick has finished in 5 minutes | `docker compose logs --tail 50 monitor`; look for `monitor_tick_failed` and its `error_type` |
| `PermissionError` on `/app/data/backups` | The `monitoring_backups` volume was created by an image without that folder (the same problem as `index_reports` in §4) | Use an image that has the folder (current `Dockerfile`), `docker volume rm mlops-project_monitoring_backups`, `docker compose up -d monitor` |
| Every window is `insufficient_data` | No predictions are being logged | Check `LOG_PSEUDONYM_KEY` is set (§11.2), check the inference hook is deployed, look for `prediction_log_write_failed` in the app logs |
| `prediction_log_write_failed` with `PredictionRecordError` | The hook sends a record that breaks the contract (wrong features, request ID containing the learner ID, and so on) | Fix the hook. The log's rules are in the schema doc, §5 |
| Many `pending` labels never join | Learners stop after one answer, or `learner_seq` has gaps | Expected for the last answer of a session; they expire after 72 h. Gaps mean a prediction was not logged, so check the write failures. |
| `drift` shows `"error"` in reports | Person 2's evaluator raised an exception | The report has `drift_error_type`. Monitoring keeps running; tell Person 2. |
| `drift` and `performance` stay `"not_configured"` with ≥ 10 predictions | The evaluator cannot load `reports/monitoring/reference_profile.json` or `configs/monitoring/thresholds.yaml` (happened on 28 Nov: the profile was not in the image, fixed in #33) | `docker compose logs monitor \| grep drift_evaluator_unavailable` shows the `error_type`. Check that the image contains the file. CI's "Monitor image can evaluate drift" step guards this. |
| A one-off `docker compose run … monitor` (or `app`) behaves like old code | In an SSH shell `APP_IMAGE` is not set, so Compose falls back to `jee-tutor:local`, an older image, not the one `deploy.sh` started | Use `docker compose exec` on the running container, or set the tag first: `export APP_IMAGE=$(docker inspect -f '{{.Config.Image}}' $(docker compose ps -q app))` |
| `mlflow` keeps restarting | OOM kill inside its 768 MB limit (`sudo dmesg -T \| grep -i oom`) | Make sure `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` is set on the service (§12) |

### 11.9 Verification record

| Step | Locally | On the server |
|---|---|---|
| `LOG_PSEUDONYM_KEY` added to `~/MLOPS-Project/.env` (64 hex chars, generated on the server, `.env` mode 600, old file kept as `.env.bak-20261127`) | — | yes, 2026-11-27 |
| Logger, label joiner, backups, window evaluation, scheduler (unit tests) | yes (`tests/test_monitoring_*.py`) | — |
| App inference path (Person 3's `InferenceService`, PR #26) → log → labels → window (`tests/test_monitoring_integration.py`) | yes, 2026-11-27, against PR #26 head `c2e6191` | — |
| App inference path on the server: 3 live answers (Hard → Medium → Easy, one test account) → 3 predictions (`learner_seq` 0, 1, 2; latency 9–15 ms; no fallback) → 3 outcomes linked by `request_id` → labels 1, 1, pending | — | yes, 2026-11-28 |
| Privacy: the test account's username in the monitoring DB, its WAL files, window snapshots and the `app`/`monitor` logs | yes (integration test) | yes, 2026-11-28: 0 occurrences |
| Scheduler end-to-end as a process (labels joined, window evaluated, backup, heartbeat) | yes, 2026-11-25 | yes, 2026-11-28 |
| Window evaluated from live traffic (`w-20261128T0600Z-15m`: 3 predictions, 2 labelled, `insufficient_data`, no alerts, snapshot files written) | — | yes, 2026-11-28 |
| `monitor` container built and healthy | no (no Docker on the dev machine) | yes, 2026-11-28 |
| Backup restored into a working database (`backup` → `--restore` into a scratch file: integrity ok, 3 predictions, 2 labels, 3 outcomes) | yes (test) | yes, 2026-11-28 |
| Backups land on the `monitoring_backups` volume | — | yes, 2026-11-28 |
| Drift evaluator runs in the deployed image (Person 2's sample windows, 60 rows each, scored by `production` v2 from MLflow, run inside the live `monitor` container) | yes (tests) | yes, 2026-11-28, after #33/#34. Baseline window: 3 features flagged, prediction PSI 0.04, F1 0.75. Shifted window: 5 features flagged, prediction PSI 0.055, F1 0.92. **Caution for Day 4:** the unshifted baseline already flags 3 features (≥ 2 fires the trigger), because PSI is noisy on 60 rows and `topic` has 40 categories. Expect control-window false alarms unless windows are larger or `topic` is excluded from the trigger count. |
| Window length change during the experiment | yes (test) | not yet (Day 4) |
| Retraining trigger called by the monitor (fires on a shifted window, cooldown holds the next one, a crash never stops monitoring) | yes, 2026-11-28 (`tests/test_monitoring_trigger_hook.py`); full retrain path (fallback training → MLflow `challenger` alias with lineage tags, `production` untouched) run against a local MLflow, ~9 s | not yet (Day 4) |
| Alert recorded from real traffic | — | not yet (Day 4) |

## 12. MLflow registry on the server (Phase 3)

**Decision (27 Nov 2026): yes.** Person 3's app loads the champion by alias (`learner-difficulty@production`), the retraining trigger registers challengers, and the gate and rollback commands move aliases. All of that needs a registry the server can reach. Until the registry is seeded, the app falls back to the committed local model, so nothing breaks in the meantime.

**Why:** the plan has the app load the model by registry alias, the retraining trigger register challengers, and governance show alias history. Before this, MLflow existed only as `mlflow.db` on laptops (git-ignored, not in the image), so none of that could work on AWS.

**What:** the `mlflow` Compose service, using the official image `ghcr.io/mlflow/mlflow:v3.17.1`, the same version as the pinned client.

| Item | Value |
|---|---|
| Storage | SQLite `/mlflow/mlflow.db` and artifacts in `/mlflow/artifacts`, on the `mlflow_data` volume (survives deploys) |
| Reached from containers | `http://mlflow:5000` (`MLFLOW_TRACKING_URI` is set for `app` and `monitor`) |
| Reached from the server | `http://127.0.0.1:5001` only. It is not published to the internet, so no security-group change is needed. |
| Reached from a laptop | an SSH tunnel: `ssh -i <key> -L 5001:127.0.0.1:5001 ubuntu@<ip>`, then open `http://localhost:5001` |
| Resources | 1 worker (SQLite needs a single writer), 0.5 CPU, 768 MB. `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` is required: by default MLflow 3.x also starts a job runner with 7 worker processes (~1.5 GB), which was OOM-killed in a restart loop on the first server deploy (2026-11-28). With it off, the server uses ~360 MB. |
| Host check | `--allowed-hosts` includes `mlflow:5000`. Without it, MLflow 3.x answers **403** to the other containers. This was verified locally on 2026-11-26: refused by default, accepted with the flag, and `evil.com` still refused. |

**Seed the registry once** (done 2026-11-28). The simplest way runs the script in a one-off container from the app image, which already has `scikit-learn==1.7.2` and reaches `http://mlflow:5000`. The image does not contain `reports/`, so mount the checkout's reports read-only (the script only reads them):

```bash
export APP_IMAGE=$(docker inspect -f '{{.Config.Image}}' $(docker compose ps -q app))   # the deployed image, not jee-tutor:local
docker compose run --rm --no-deps -T -v "$PWD/reports:/app/reports:ro" app python -m src.track_b.register_mlflow
```

From a laptop instead: open the tunnel and run `MLFLOW_TRACKING_URI=http://localhost:5001 python -m src.track_b.register_mlflow` in an environment with `scikit-learn==1.7.2`. The champion does not load under 1.9.1.

**Backups:** the registry is not covered by the nightly monitoring backup yet. Until it is, copy it by hand after any registry change:

```bash
docker compose exec mlflow python -c "import sqlite3; s=sqlite3.connect('/mlflow/mlflow.db'); d=sqlite3.connect('/mlflow/mlflow-backup.db'); s.backup(d)"
```

**Verification record:**

- Locally (2026-11-26): the real `mlflow server` 3.17.1 process; a model was logged, registered, given an alias and loaded back by alias over HTTP.
- On the server (2026-11-28): the first deploy (#30) showed an OOM restart loop, fixed in #31/#32. Since then `mlflow` is healthy, ~350 MB, 0 restarts. The registry is seeded: `learner-difficulty` v1 (baseline) and v2 (`candidate_hgb_shifted_v2`, gate `PROMOTE_CANDIDATE`), with alias `production` → v2. The app loads it by alias (`source_backend: mlflow`, threshold 0.7, boundary guard on). A hand backup was taken right after seeding (`/mlflow/mlflow-backup.db`, integrity ok, alias present).

## 13. Automated retraining trigger and drift experiment (Person 2)

**Owner:** Person 2 (ML: Drift & Retraining)  
**Topics:** Graded Topic 4 (Drift / Degradation Experiment) and Topic 5 (Retraining Trigger)

### 13.1 Configuration and policies

- **Drift Thresholds:** `configs/monitoring/thresholds.yaml` defines PSI warning (0.10), PSI alert (0.20), KS-test $\alpha$ (0.01), prediction PSI alert (0.15), `min_rows_per_category: 5` (suppresses 40-topic false alerts on < 200-row windows), and min sample sizes (50).
- **Trigger Policy:** `configs/monitoring/retrain_trigger.yaml` defines trigger conditions:
  - $\ge 2$ features in alert status (PSI $\ge 0.20$, excluding `PREDICTION_DRIFT` and `exclude_features_from_trigger`: `topic`, `prior_interaction_count`) OR prediction PSI $\ge 0.20$ OR F1 drop $\ge 0.05$ (with `min_labeled_for_perf: 1000`, since 300-row windows have F1 sd $\approx 0.07$; performance still renders on the dashboard at `min_labeled: 20`).
  - **Guardrails:** Cooldown period (6 hours / 5 simulated windows), max 3 runs per day ceiling, `min_window_records: 50`.
  - **Live Day 4 traffic replay (on the server):** Run inside the **`app`** container (only `app` has `LOG_PSEUDONYM_KEY`; `monitor` does not, and the image excludes `scripts/` and `data/synthetic/` so both must be mounted read-only):
    ```bash
    export APP_IMAGE=$(docker inspect -f '{{.Config.Image}}' $(docker compose ps -q app))
    # 5 control windows (300 rows per 2-minute window = 1500 rows)
    docker compose run --rm --no-deps -T \
      -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/data/synthetic:/app/data/synthetic:ro" \
      app python scripts/simulate_traffic.py --mode baseline --limit 1500 --rate 2.5
    # 3 shifted windows (900 rows at 2.5 req/s)
    docker compose run --rm --no-deps -T \
      -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/data/synthetic:/app/data/synthetic:ro" \
      app python scripts/simulate_traffic.py --mode shifted --limit 900 --rate 2.5
    ```
  - **Target Stage:** `dvc repro train_challenger` (outputs only to `models/track_b/challenger/`, never overwriting the `models/track_b/candidate/` champion fallback).

### 13.2 Operating the Retraining Trigger

The trigger engine (`src/monitoring/trigger.py`) evaluates window reports generated by the scheduled monitor:

```bash
# Evaluate a window report and trigger retraining if thresholds breached
python -m src.monitoring.trigger --window-report reports/monitoring/windows/<window_id>/report.json --db data/state/monitoring.sqlite3
```

**Trigger event lifecycle:**
1. Evaluates alerts and metrics against declared policy (excluding `PREDICTION_DRIFT` and `exclude_features_from_trigger` from feature drift counts; using `baseline_f1` from `thresholds.yaml`).
2. Only if conditions hold, checks cooldown in `reports/monitoring/trigger_state.json` using window end time (`report["end"]`).
3. Creates immutable snapshot of joined monitoring rows (`data/snapshots/snapshot_<timestamp>.csv`) and computes its SHA256 for lineage (note: candidate training still uses `learner_interactions_shifted_v1.csv` to preserve `learner_key` privacy and `.meta.json` learner splits).
4. Executes `dvc repro train_challenger` (writing to `models/track_b/challenger/`).
5. Registers the new challenger model version in MLflow Model Registry under alias `challenger` (only if registration created a new version).
6. Appends audit event to `reports/monitoring/trigger_events.jsonl` with lineage (`trigger_id`, `data_snapshot_sha256`, `cooldown_until`).

### 13.3 Reproducing the Live Drift Experiment

To reproduce the Day 4 controlled drift & degradation experiment:

```bash
# Runs control phase (baseline) -> injects shift at T_shift -> dry-run trigger + cooldown -> generates report
python scripts/run_drift_experiment.py
```

Outputs (isolated under `reports/drift/`, leaving committed champion model and `reports/monitoring/trigger_*` untouched):
- Machine-readable results: `reports/drift/live_drift_experiment_report.json`
- Human-readable summary: `reports/drift/live_drift_experiment_report.md`
- Experiment trigger logs: `reports/drift/trigger_state.json`, `reports/drift/trigger_events.jsonl`

### 13.4 Troubleshooting and Cooldown Recovery

| Symptom | Cause | Remedy |
|---|---|---|
| Trigger skips with `cooldown_blocked` | Cooldown period is active to prevent runaway retraining | Wait for `cooldown_until` timestamp, or reset state for testing: `rm reports/monitoring/trigger_state.json` |
| `dvc repro train_challenger` fails on host | Insufficient CPU/RAM on t2.micro instance | Trigger logs fallback to `python -m src.track_b.train_candidate --output-dir models/track_b/challenger` |
| Challenger alias not showing in MLflow | MLflow server not reachable | Verify `MLFLOW_TRACKING_URI` or local SQLite registry |
