# Phase 3 Day 1: deployed-state audit (Person 1)

**Date:** 25 Nov 2026 · **Branch audited:** `origin/dev` at `53a61d8` · **Host checked:** 25 Nov 2026, 21:30 IST (instance `jee-tutor`, read-only) · **Status:** complete

Each line is labelled with how it was checked:

- **[repo]** read from the repository on `origin/dev`.
- **[local]** seen on a developer machine. This is not evidence about the server.
- **[host]** checked on the EC2 instance.
- **[unknown]** not checked yet.

No changes were made to the server.

---

## 1. Deployed revision

| Item | Value | Source |
|---|---|---|
| Deploy trigger | Every push to `main` runs `.github/workflows/deploy.yml`: build the image, push it to ECR `jee-tutor` with tags `<sha>` and `latest`, then SSM runs `scripts/deploy.sh <image>` on the instance. | [repo] |
| Expected deployed commit | `e9c7c3a` (merge of #22, the port-80 fix), the tip of `origin/main` | [repo] |
| `dev` vs `main` | `dev` is one merge ahead (`53a61d8`, #23 repo cleanup) and has not been released. | [repo] |
| Image tag actually running | `jee-tutor:e9c7c3a504734c4493adc974fea88841e1ebbc3e` for `api` and `app`; `chromadb/chroma:1.5.3`. All three are healthy. | [host] |
| Commit checked out on the host | `e9c7c3a` (detached HEAD). It matches the image, so the server runs exactly `main`. | [host] |
| Repo path on the host | `/home/ubuntu/MLOPS-Project` (`APP_DIR` in `deploy.yml`) | [repo] |
| Region / instance | `ap-southeast-2`; instance ID is in the repo variable `DEPLOY_INSTANCE_ID` | [repo] |

## 2. Services, ports, and volumes (`docker-compose.yml`)

| Service | Command | Host ports | Volumes |
|---|---|---|---|
| `chroma` | `chromadb/chroma:1.5.3` | 8000 | `chroma_data:/data` |
| `api` | FastAPI `src.api.app` | 8001→8000 | `./data/processed` (ro), `attempts_data:/app/data/attempts`, `hf_cache` |
| `app` | Streamlit `app.py` | 80→8501, 8501 | `./data/processed` (ro), `app_state:/app/data/state`, `hf_cache` |
| `indexer` | profile `tools` only | — | `index_reports`, `hf_cache` |

All of the above is [repo].

**Persistent data paths inside containers**

| Path | Volume | Contents | Survives a redeploy? |
|---|---|---|---|
| `/app/data/state/users.csv` | `app_state` | Student accounts | Yes. [repo], and earlier observed: the `hello` account survived the #21/#22 deploy |
| `/app/data/state/learner_history.sqlite3` | `app_state` | App attempt history (table `attempts`) | Yes (same volume) |
| `/app/data/attempts/attempts.jsonl` | `attempts_data` | API attempt records, which contain learner answers | Yes. Only `docker compose down -v` deletes named volumes |
| `/data` (chroma) | `chroma_data` | Vector indexes | Yes |

**Backups:** none exist. No backup script or schedule is in the repo [repo]. `docs/devops_runbook.md` lists retention and backup as an open item.

## 3. MLflow and the model artefacts

| Finding | Source | Consequence |
|---|---|---|
| MLflow only runs on developer machines. The tracking URI defaults to `sqlite:///mlflow.db` in the repo root (`src/track_b/tracking.py`), and `MLFLOW_TRACKING_URI` can override it. No MLflow server is in Compose. | [repo] | There is no shared registry on AWS, so "load the model by registry alias" has no registry to load from in production. |
| `mlflow.db`, `mlruns/` and `mlartifacts/` are git-ignored and listed in `.dockerignore`. | [repo] | The registry cannot be copied to the server through git or the image. |
| The image contains no model. `.dockerignore` excludes `models/` and `reports/`, and `requirements-app.txt` does not pin `mlflow`, `scikit-learn`, `pandas` or `joblib`. On the host, scikit-learn, joblib and pandas are present only as indirect dependencies, and `mlflow` is missing. | [repo] + [host] | **Blocker for Person 3's inference hook:** the runtime image cannot load `model.joblib` today. |
| The Track B models are committed to git: `models/track_b/{baseline,candidate}/model.joblib` plus `model_card.json`. | [repo] | They can reach the server through the git checkout that `deploy.sh` already does, or be added to the image. |
| scikit-learn versions disagree: `requirements.txt` pins 1.7.2 (the baseline card records 1.7.2), while `requirements-ci.txt` pins 1.9.1. | [repo] | **Confirmed on the host (§5):** the deployed image has 1.9.1, and both models warn when loaded. |
| The local `mlflow.db` (14 Oct) is out of date. Its `production` alias points to v1 `baseline_logreg_v1` and `rejected_candidate` to v2 `candidate_logreg_shifted_v1`. But `reports/track_b/promotion_decision.json` and the model files (15 Oct) say **`candidate_hgb_shifted_v2` was promoted**. | [local] + [repo] | Rebuild the registry with `python -m src.track_b.register_mlflow` before anyone trusts an alias. The HGB champion is not registered on this machine. |
| Model card `config` paths use Windows backslashes (`configs\\track_b\\baseline.yaml`). | [repo] | Small issue: normalise these when the cards are extended (Person 4). |

**Proposal for discussion, so Person 3 isn't blocked:**

1. Run a small MLflow server as a Compose service on the instance. Its backend is SQLite on a new `mlflow_data` volume, and its artifacts live on the same volume. It is free, with no managed service.
2. Bind it to `127.0.0.1` so it is reachable only from the instance itself, not from outside.
3. The app, the monitoring job, and the retraining trigger reach it through `MLFLOW_TRACKING_URI=http://mlflow:5000`.
4. Pin `scikit-learn==1.7.2` (to match the saved models), `joblib` and `pandas`, and add an `mlflow-skinny` client to `requirements-app.txt`.

The team needs to agree on this. Person 3 owns model loading.

## 4. Existing logging and attempt data

| Finding | Source |
|---|---|
| There are two separate attempt stores: the API JSONL (`AttemptStore`, `/app/data/attempts/attempts.jsonl`) and the app SQLite table `attempts(attempt_id, learner_id, question_id, concept_ids, outcome, answer_mode, attempted_at)`. | [repo] |
| The app hard-codes `concept_ids=("chemistry",)` (`app.py:592`, `app.py:655`). Questions have no topic. | [repo] |
| None of the model features `hint_count`, `response_time_seconds` or `selected_difficulty` are captured anywhere. | [repo] |
| Application logs are plain `logging.getLogger` in `src/api/app.py`. There are no request IDs, no latency measurements and no JSON format. Container logs use Docker's default `json-file` driver, which is lost when a container is recreated. | [repo] |
| **Privacy:** `learner_id` is built as `lrn-<username>-<8 hex>` (`app.py:124`), so it contains the username. | [repo] |

The privacy finding means the prediction log must **not** store `learner_id` as it is. See the schema draft, field `learner_key`.

## 5. Host findings (checked 25 Nov 2026)

| Item | Value | Consequence |
|---|---|---|
| Volumes | `mlops-project_{app_state, attempts_data, chroma_data, hf_cache, index_reports}` | The Compose project name is `mlops-project`. A new volume would be named `mlops-project_<name>`. |
| `app_state` contents | `users.csv` (1 account) and `learner_history.sqlite3` (16 KB) | `monitoring.sqlite3` will sit next to them. |
| App attempt rows | **2 rows, 1 learner** (13 Nov 2026) | There is almost no real usage. Live performance metrics will be based on `simulated` traffic, and the reports must say so (plan §6, P1 rule). |
| API attempts JSONL | Does not exist (nothing has been written through the API) | The API attempt store is unused in production. The prediction log does not need to read from it. |
| Container log driver | `json-file`, no size limit, for all three containers | Docker logs are lost when a container is recreated, and they can grow without limit. Day 2: set `max-size`/`max-file` in Compose. Docker logs are not the persistent record; the `predictions` table is. |
| Disk | 29 GB root, 7.6 GB used (27%). Images use 3.8 GB, including one old image (`694ebee`, 2.98 GB) | Enough room. `deploy.sh` still never removes old images (about 3 GB per deploy). Prune after a deploy, or add pruning to `deploy.sh`. |
| Memory / CPU | 7.8 GB RAM (about 7 GB free), **2 vCPUs** | A monitoring sidecar is cheap. Retraining a HistGradientBoosting model on about 10k synthetic rows takes seconds, so it can run on the host if CPU is capped (`cpus: 1`). The GitHub Actions fallback is probably not needed. |
| Scheduler | No user crontab. Only the system jobs `e2scrub_all` and `sysstat` | Nothing to conflict with. I recommend a Compose sidecar (the schedule is versioned with the code and deploys with CD) rather than the host crontab. |
| Server time zone | UTC | Timestamps in the log and reports are UTC everywhere. |
| MLflow on the host | Not present (no `mlflow.db`, no `mlruns/`, no `mlflow` package in the image) | Confirms §3: the server has no registry. |
| Models on the host | `~/MLOPS-Project/models/track_b/{baseline,candidate}` exist in the git checkout, but **not inside the image** (`/app/models` is missing) | Mounting `./models` read-only, as is already done for `data/processed`, would make them available without rebuilding the image. |
| ML libraries in the image | scikit-learn **1.9.1**, joblib 1.6.0, scipy 1.18.1, pandas, all pulled in by `sentence-transformers`, not pinned by us | See the next row. |
| **Model load test** | Both `model.joblib` files raise `InconsistentVersionWarning`: they were saved with scikit-learn **1.7.2** and are loaded with 1.9.1 | **Blocker for Person 3.** Either pin `scikit-learn==1.7.2` in `requirements-app.txt`, or retrain both models through DVC under 1.9.1 and record the new hashes. Do not ignore the warning: scikit-learn says results may be wrong. Method: a throwaway `docker run --rm --network none` of the deployed image with `models/` mounted read-only. |
| `.env` variable names | `GROQ_API_KEY`, `GROQ_MODEL`, `OCR_SPACE_API_KEY` (MLflow, Chroma host and port are commented out) | `LOG_PSEUDONYM_KEY` is the new variable to add (schema draft, §5). |
| Backups | None | Confirms §2. |

Commands used (read-only): `git rev-parse HEAD`, `docker compose ps`, `docker volume ls`, `docker inspect` (log config), `docker system df`, `df`, `free`, `nproc`, `crontab -l`, `timedatectl`, and `docker compose exec` to count rows and list files.

## 6. Open questions for the team

1. **Model gate (Person 3):** the outcome (A, B or C) decides the feature list the log records. The schema below uses the Track B features, which apply under outcomes B and C.
2. **MLflow on AWS:** should we accept the proposal in §3, or does someone prefer a different approach?
3. **Where the logger lives:** inference will run in the Streamlit `app` process (the practice flow is there, not in the API). The prediction log should therefore live on `app_state`. Person 3 should confirm.
