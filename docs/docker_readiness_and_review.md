# Docker Readiness and Teammate Review

**Purpose:** summarize the Docker work in this branch and give a teammate with a
working Docker Engine a repeatable review checklist.

## Readiness summary

The Docker configuration and service definitions are present. The Compose file
was checked with `docker compose config --quiet` and parsed successfully.

The images have **not** been built and the containers have **not** been started
from this environment: the Docker CLI is installed, but the Docker Desktop
Linux Engine is unavailable here. Treat runtime readiness as **pending teammate
verification**.

## What is implemented

| Area | Current implementation |
| --- | --- |
| Image | Multi-stage Dockerfile based on Python 3.12 slim. It installs CPU-only PyTorch in a build stage, copies the app runtime dependencies into a smaller runtime stage, and runs as the non-root `app` user. |
| App service | Streamlit starts on container port 8501, published as `localhost:8501`. It checks required environment settings before startup and has an HTTP health check. |
| Evaluator API | FastAPI/uvicorn starts on container port 8000, published as `localhost:8001`. Its health check uses `/health`; the question JSONL must be present for the API to report healthy. |
| Chroma | ChromaDB image `chromadb/chroma:1.5.3`, published as `localhost:8000`, with a persistent `chroma_data` volume and a heartbeat health check. |
| Indexer | A Compose `tools` profile service builds the NCERT and JEE indexes in Chroma from the DVC-managed processed JSONL files. |
| Container networking | App, API, and indexer use the Compose hostname `chroma:8000`. Retrieval configuration and the evaluator's reachability check accept `CHROMA_HOST` and `CHROMA_PORT` overrides. |
| Persistent data | Named volumes hold Chroma collections, evaluator attempt records, Hugging Face model cache, and index reports. Processed DVC data is mounted read-only into the API and indexer. |
| Secrets | `.env` is excluded from the build context. Compose passes Groq and OCR.space keys as environment variables; the Dockerfile does not bake them into the image. |

The runtime dependency list is in `requirements-app.txt`; it is intentionally
smaller than the development freeze in `requirements.txt`.

## Service map

```text
Browser
  └── Streamlit app :8501 ──> Groq API
          ├──> OCR.space (only when a learner uploads an image)
          └──> Chroma :8000 (retrieval)

Browser / API client
  └── Evaluator API :8001 ──> Chroma :8000
          ├──> read-only /app/data/processed/jee_questions.jsonl
          └──> attempts_data volume

One-time indexer ──> Chroma :8000
          ├──> read-only processed JSONL files
          └──> shared hf_cache volume
```

The Streamlit tutor and evaluator are separate services. The current Streamlit
page does not submit answers to the evaluator API automatically.

## Required local setup

1. Copy `.env.example` to `.env` and set a valid `GROQ_API_KEY`. Set
   `OCR_SPACE_API_KEY` to enable image transcription; OCR is optional for app
   startup.
2. Pull the processed data from the configured DVC remote:

   ```bash
   dvc pull data/processed.dvc
   ```

3. Confirm these files exist under `data/processed/`:
   `jee_questions.jsonl` and `ncert_learning_units.jsonl`.
4. Use Docker Desktop in Linux-container mode, with Compose available and
   network access for the Python package indexes and the CPU PyTorch wheel
   index. The first build and embedding-model download can take a while.

## Reviewer run-through

From the repository root, run:

```bash
docker compose config --quiet
docker compose build
docker compose up -d chroma
docker compose ps
docker compose run --rm indexer
docker compose up -d
docker compose ps
```

The `indexer` service waits for Chroma to become healthy, then builds both
collections. Review its output for successful completion of both commands. If
the build or indexing step fails, inspect:

```bash
docker compose logs --tail=100 chroma
docker compose logs --tail=100 indexer
```

After startup, check:

- Streamlit opens at <http://localhost:8501>; its health endpoint is
  <http://localhost:8501/_stcore/health>.
- The evaluator API documentation opens at <http://localhost:8001/docs>.
- `GET http://localhost:8001/health` reports `healthy` and a nonzero
  `chemistry_questions_loaded` count.
- Chroma's heartbeat at <http://localhost:8000/api/v2/heartbeat> responds
  successfully.
- Logs show that the app and API containers started without missing-key,
  missing-file, or connection errors:

  ```bash
  docker compose logs --tail=100 app api
  ```

Then exercise the user paths:

1. Submit a typed tutoring request and confirm a response is returned.
2. Upload a permitted PNG/JPEG handwriting image, review/correct its OCR text,
   choose **Confirmed handwriting** under **Answer input**, and submit it. The
   image is sent to OCR.space; the corrected transcription is sent to Groq.
   This requires working Groq and OCR.space keys and external network access.
3. Separately call `POST http://localhost:8001/api/evaluate` with a valid
   `question_id`, `answer_text`, `input_mode`, and `session_id`. Confirm the
   response is returned, then query
   `GET http://localhost:8001/api/attempts/{session_id}` to check attempt
   persistence.
4. Restart the app/API and confirm Chroma collections and evaluator attempts
   remain. `docker compose down` retains named volumes; `docker compose down -v`
   deletes them and should only be used for an intentional clean-data check.

## Items for teammate evaluation

- Does the CPU-only PyTorch install and `requirements-app.txt` install succeed
  on the teammate's Docker host?
- Does Chroma's Bash `/dev/tcp` health-check command work in the pinned Chroma
  image?
- Does the read-only Windows bind mount expose the DVC-pulled files inside the
  API and indexer containers?
- Do the named volumes initialize with write access for the non-root app user,
  especially `attempts_data`, `hf_cache`, and `index_reports`?
- Do all three health checks become healthy after the expected startup delay?
- Does indexing complete and do both collections persist after a container
  restart?
- Do typed and handwriting tutoring requests work with the teammate's real API
  keys? Verify that no secret value appears in logs.

## Current limits

- No image build, container startup, health-check, Chroma, API, Groq, or OCR.space
  live call has been verified from this machine because its Docker Engine is
  unavailable.
- The evaluator API has its own health and attempt persistence, but the
  Streamlit tutoring flow is not currently wired to that API's evaluation and
  attempt-saving endpoints.
- OCR quality on handwritten Chemistry examples still depends on real,
  student-reviewed samples; unit tests mock OCR.space responses.
