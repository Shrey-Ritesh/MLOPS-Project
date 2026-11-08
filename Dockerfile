# ---- builder: install runtime dependencies into a venv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch: the default Linux wheel pulls multi-GB CUDA libraries we never use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt ./requirements-app.txt
RUN pip install -r requirements-app.txt

# ---- runtime: venv + app code only, non-root ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/data/attempts /cache/huggingface \
    && chown -R app:app /app /cache

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app.py ./app.py
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs

USER app

EXPOSE 8501 8000

# Default process is the Streamlit app; compose overrides the command for the API/indexer
# and defines per-service health checks.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
