# Local startup

## Configuration

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Check it without starting the app:

```bash
python -m src.settings      # prints names of missing/invalid variables, never values
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq chat API key |
| `GROQ_MODEL` | no | override default chat model |
| `CHROMA_HOST`, `CHROMA_PORT` | no | override the Chroma endpoint in `configs/*.yaml` (needed in containers) |
| `MLFLOW_TRACKING_URI` | no | MLflow server (default: local sqlite) |

## Run without Docker

```bash
pip install -r requirements.txt        # or requirements-app.txt for just the app
docker compose up -d chroma            # optional: retrieval degrades gracefully without it
streamlit run app.py                   # http://localhost:8501
```

## Run in Docker

```bash
docker build -t jee-tutor .
# Chroma on the host (docker compose up -d chroma); reach it via host.docker.internal
docker run --rm -p 8501:8501 --env-file .env \
  -e CHROMA_HOST=host.docker.internal jee-tutor
```

- Startup runs `python -m src.settings` first and exits with a clear message if configuration is missing.
- Health check: `curl http://localhost:8501/_stcore/health` (Streamlit built-in; also the image `HEALTHCHECK`).
- The container runs as a non-root user. The sentence-transformers model is downloaded on first query; mount a volume on `/home/app/.cache/huggingface` to keep it between runs.
- `data/`, `models/`, `reports/` and `.env` are excluded from the image. Retrieval needs a populated Chroma index (see the README for the DVC + index steps).
