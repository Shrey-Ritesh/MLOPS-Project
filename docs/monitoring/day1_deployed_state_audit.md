# Phase 3 Day 1: deployed-state audit (Person 1)

**Date:** 25 Nov 2026 · **Branch audited:** `origin/dev` at `53a61d8` · **Status:** repo side done, host side pending

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
| Image tag actually running | — | [unknown]: run `docker compose ps --format '{{.Image}}'` on the host |
| Commit checked out on the host | — | [unknown]: run `git -C ~/MLOPS-Project rev-parse HEAD` |
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
| The image contains no model. `.dockerignore` excludes `models/` and `reports/`, and `requirements-app.txt` has no `mlflow`, `scikit-learn`, `pandas` or `joblib`. | [repo] | **Blocker for Person 3's inference hook:** the runtime image cannot load `model.joblib` today. |
| The Track B models are committed to git: `models/track_b/{baseline,candidate}/model.joblib` plus `model_card.json`. | [repo] | They can reach the server through the git checkout that `deploy.sh` already does, or be added to the image. |
| scikit-learn versions disagree: `requirements.txt` pins 1.7.2 (the baseline card records 1.7.2), while `requirements-ci.txt` pins 1.9.1. | [repo] | A pickled model may not load under a different scikit-learn version. Pin one version for the runtime image. |
| The local `mlflow.db` (14 Oct) is out of date. Its `production` alias points to v1 `baseline_logreg_v1` and `rejected_candidate` to v2 `candidate_logreg_shifted_v1`. But `reports/track_b/promotion_decision.json` and the model files (15 Oct) say **`candidate_hgb_shifted_v2` was promoted**. | [local] + [repo] | Rebuild the registry with `python -m src.track_b.register_mlflow` before anyone trusts an alias. The HGB champion is not registered on this machine. |
| Model card `config` paths use Windows backslashes (`configs\\track_b\\baseline.yaml`). | [repo] | Small issue: normalise these when the cards are extended (Person 4). |

**Proposal for discussion, so Person 3 isn't blocked:**

1. Run a small MLflow server as a Compose service on the instance. Its backend is SQLite on a new `mlflow_data` volume, and its artifacts live on the same volume. It is free, with no managed service.
2. Bind it to `127.0.0.1` so it is reachable only from the instance itself, not from outside.
3. The app, the monitoring job, and the retraining trigger reach it through `MLFLOW_TRACKING_URI=http://mlflow:5000`.
4. Add `scikit-learn` (pinned to one version), `joblib`, `pandas` and an `mlflow-skinny` client to `requirements-app.txt`.

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

## 5. Host checks to run when the instance is started (read-only)

```bash
cd ~/MLOPS-Project
git rev-parse HEAD; git status -sb | head -1
docker compose ps --format 'table {{.Service}}\t{{.Image}}\t{{.Status}}'
docker volume ls | grep -E 'app_state|attempts_data|chroma_data|hf_cache'
docker compose exec app ls -la /app/data/state
docker compose exec app python -c "import sqlite3;c=sqlite3.connect('/app/data/state/learner_history.sqlite3');print(c.execute('select count(*) from attempts').fetchone())"
df -h /; free -m; nproc          # headroom for a monitoring sidecar and retraining runs
crontab -l 2>/dev/null           # is any schedule already in place?
```

Fill these results into §1 and §2 and change the labels to [host].

## 6. Open questions for the team

1. **Model gate (Person 3):** the outcome (A, B or C) decides the feature list the log records. The schema below uses the Track B features, which apply under outcomes B and C.
2. **MLflow on AWS:** should we accept the proposal in §3, or does someone prefer a different approach?
3. **Where the logger lives:** inference will run in the Streamlit `app` process (the practice flow is there, not in the API). The prediction log should therefore live on `app_state`. Person 3 should confirm.
