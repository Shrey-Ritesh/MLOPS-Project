# syntax=docker/dockerfile:1

# ---- builder: resolve and install dependencies into a venv ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# CPU-only torch keeps the image far smaller than the default CUDA wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# ---- runtime: venv + app code only, non-root ----
FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

# Fail fast with a clear message (names only, never values) if config is missing.
CMD ["sh", "-c", "python -m src.settings && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
