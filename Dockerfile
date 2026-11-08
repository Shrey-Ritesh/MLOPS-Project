FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/data/attempts /cache/huggingface \
    && chown -R app:app /app /cache

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY --chown=app:app app.py ./app.py
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs

USER app

EXPOSE 8501 8000
