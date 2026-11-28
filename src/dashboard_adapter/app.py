from fastapi import FastAPI, Response
from src.dashboard_adapter.metrics import get_metrics

app = FastAPI(title="JEE Tutor Dashboard Adapter")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "Dashboard adapter running"}

@app.get("/metrics")
def metrics():
    payload, content_type = get_metrics()
    return Response(
        content=payload,
        media_type=content_type,
    )