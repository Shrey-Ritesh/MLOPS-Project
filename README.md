## JEE Tutor

This repository contains the source code and management scripts for a tutor app
designed to teach students different JEE problems. It is part of an MLOps
course project in the education domain.

Team: Shraddha PS, Shreesha DR, Shrey Ritesh, Shreyank DK

## Repository layout

| Folder / file | What is in it |
|---|---|
| `app.py` | Streamlit tutor app (sign-in, tutor, practice, progress, debug tabs) |
| `src/api/` | FastAPI answer-evaluation service |
| `src/evaluation/` | Answer grading: normalisers, attempt store, retriever adapter |
| `src/retrieval/` | Embeddings, Chroma index builders, `retrieve.py`, and the MiniLM / chEmbed benchmark tools (`benchmark_minilm.py`, `benchmark_chembed.py`, `evaluate_chembed.py`) |
| `src/retrieval_mlflow/` | Registers the retrieval benchmark runs in MLflow |
| `src/handwriting/`, `src/learner/`, `src/tutoring/` | OCR.space transcription, learner history and recommendations, tutor prompt |
| `src/data/`, `src/models/` | NCERT/JEE PDF extraction pipeline and its data models (built the corpus in Phase 1) |
| `src/track_b/` | **Phase 1** Track B: synthetic learners, drift detection and promotion gate (the DVC pipeline in `dvc.yaml`) |
| `configs/` | Retrieval, MLflow and Track B configuration |
| `data/` | DVC pointers (`processed.dvc`, `assets.dvc`), `benchmarks/` (retrieval question sets), Track B synthetic data |
| `reports/` | All results: `retrieval/` (index manifests, benchmark runs, analysis, evaluations), `track_b/`, `data_quality/` |
| `models/track_b/` | Phase 1 Track B models |
| `tests/` | pytest suite (run by CI) and its fixtures |
| `scripts/` | Utility scripts: `deploy.sh` (server deploy), `smoke_test_api.py` (CI), retrieval analysis tools, `phase1/` (Phase 1 retrieval evaluation) |
| `deploy/` | AWS IAM policy and the one-time server bootstrap script |
| `.github/workflows/` | CI (`ci.yml`, `docker.yml`) and CD (`deploy.yml`) |
| `docs/` | Runbook, CI/branch workflow, design notes; `planning/` holds the roadmap and phase plans |
| `Dockerfile`, `docker-compose.yml` | Container image and the stack (Chroma, API, app, indexer) |

## Run the project with Docker

Prerequisites: Docker Desktop with Compose, access to the configured DVC remote,
and API keys for Groq and OCR.space. Copy `.env.example` to `.env` and set
`GROQ_API_KEY` and `OCR_SPACE_API_KEY`.

Pull the processed question and learning-unit data, start Chroma, and build
both retrieval indexes:

```bash
dvc pull data/processed.dvc
docker compose up -d chroma
docker compose run --build --rm indexer
docker compose up --build -d
```

The indexer downloads the sentence-transformer embedding model on its first
run. Docker keeps its Hugging Face cache, Chroma collections, and evaluator
attempt records in named volumes.

Open the Streamlit tutor at <http://localhost:8501>. The evaluator API is at
<http://localhost:8001>, with interactive API documentation at
<http://localhost:8001/docs>. Chroma is available on port 8000. Stop the stack
with `docker compose down`; named data volumes remain available for the next
start. To remove those persisted volumes, run `docker compose down -v`.

## Streamlit tutor frontend

To run the frontend directly on the host, install dependencies with
`pip install -r requirements.txt`, copy `.env.example` to `.env`, and run:

```bash
streamlit run app.py
```

The frontend retrieves subject-matched NCERT excerpts and JEE questions from
the project's Chroma indexes and supplies them to Groq as reference material.
If retrieval is unavailable, the tutor still responds without project context.

### Handwritten answer transcription

The handwritten-answer panel uses OCR.space Engine 3 for handwriting
recognition. Set `OCR_SPACE_API_KEY` in `.env` using the key from your OCR.space
account. The app accepts PNG or JPEG images up to 1 MB and validates the image
content before upload. OCR text is shown in an editable review field; correct
Chemistry notation before confirming it. Uploaded images are sent to OCR.space.

The evaluator API and append-only attempt store run as a separate service from
the Streamlit page. To get tutor feedback on an image, transcribe it in the
handwriting panel, review and confirm the text, choose **Confirmed handwriting**
under **Answer input**, optionally add the original question/topic, and submit.
The confirmed OCR text then enters the same Groq tutoring request as a typed
answer. The Streamlit page does not automatically submit it to the evaluator
API; that API remains a separate question-ID-based workflow.

## Prepare DVC data and retrieval indexes on the host

The processed source records are tracked with DVC. Pull them first (access to
the configured DVC remote is required), then start Chroma and build both indexes:

```bash
dvc pull data/processed.dvc
docker compose up -d chroma
python -m src.retrieval.index_builder
python -m src.retrieval.jee_index_builder_v1
```

DVC manages the source JSONL files; the indexes are stored in the Chroma
service. The default host endpoint is `localhost:8000`, configured through
`configs/retrieval.yaml` and `configs/jee_retrieval_v1.yaml`. Docker overrides
the endpoint with the internal `chroma` service name.
