## JEE-Tutor app for students 
This repository contains the source code and management scripts for a Tutor app, designed to teach students different JEE problems.

SDGs covered: 4, 10

This project is as a part of the MLOps course, and aims to implement a relevant solution in the education domain while also covering the complete MLOps cycle.

Team: Shraddha PS, Shreesha DR, Shrey Ritesh, Shreyank DK

## Streamlit tutor frontend

Install dependencies with `pip install -r requirements.txt`. Copy
`.env.example` to `.env` and replace the Groq API key placeholder. Start the
frontend from the repository root with:

```bash
streamlit run app.py
```

The frontend retrieves subject-matched NCERT excerpts and JEE questions from
the project's Chroma indexes and supplies them to Groq as reference material.
If retrieval is unavailable, the tutor still responds without project context.

### Prepare DVC data and retrieval indexes

The processed source records are tracked with DVC. Pull them first (access to
the configured DVC remote is required), then start Chroma and build both indexes:

```bash
dvc pull data/processed.dvc
docker compose up -d chroma
python -m src.retrieval.index_builder
python -m src.retrieval.jee_index_builder_v1
```

DVC manages the source JSONL files; the indexes are stored in the Chroma
service. Keep Chroma running while using the frontend. The default endpoint is
`localhost:8000`, configured through `configs/retrieval.yaml` and
`configs/jee_retrieval_v1.yaml`.
