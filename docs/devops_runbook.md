# DevOps runbook and release checklist (Phase 2)

Scope: a tested **release candidate** with documented production blockers. Nothing here deploys to a public environment, and no cloud credentials are used or required.

## 1. Services and ports

| Service | Image/command | Host port | Health check |
|---|---|---|---|
| `chroma` | `chromadb/chroma:1.5.3` | 8000 | `/api/v2/heartbeat` |
| `api` | this repo, `uvicorn src.api.app:app` | 8001 | `GET /health` (503 until questions are loaded) |
| `app` | this repo, Streamlit | 8501 | `/_stcore/health` |
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

## 5. Logs

- All services log to stdout/stderr: `docker compose logs -f api app`.
- The API logs warnings for skipped invalid records, retrieval outages and write failures. Learner content is never logged: the retrieval warning logs only the query length and exception type, and invalid-attempt warnings log only the exception type (covered by `tests/test_api_hardening.py`).
- Attempt records (which contain learner answers) live in the `attempts_data` volume at `/app/data/attempts/attempts.jsonl`, not in logs.

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
- [ ] Monitoring/alerting (metrics, uptime checks) and log shipping.
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
- Retrieval evidence needs the indexes (`docker-compose run --rm indexer`), which needs the processed data and a first-run model download; this was not exercised in the clean-start test.
- The Docker image was never buildable from `dev` before this work (Python 3.11 vs `numpy==2.5.4`), and `docker compose up` could never start the API and app (the Chroma healthcheck used `curl`, which that image lacks).
