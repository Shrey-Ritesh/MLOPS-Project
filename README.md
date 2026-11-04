## JEE Tutor

This repository contains the source code and management scripts for a tutor app
designed to teach students different JEE problems. It is part of an MLOps
course project in the education domain.

Team: Shraddha PS, Shreesha DR, Shrey Ritesh, Shreyank DK

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
the Streamlit page. OCR text is editable and confirmed in the frontend, but the
current Streamlit page does not automatically submit it to the evaluator.

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
