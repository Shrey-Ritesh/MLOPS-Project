"""FastAPI application factory for the JEE Tutor API."""
from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="JEE Chemistry Tutor Evaluation API",
        description="Backend evaluation service for Chemistry JEE preparation.",
        version="0.2.0",
    )

    # Enable CORS for local Streamlit / frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        field_errors = [
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', 'Invalid value')}"
            for err in errors
        ]
        summary_msg = "; ".join(field_errors) if field_errors else "Validation error"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": errors,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": summary_msg,
                    "status_code": 422,
                    "details": errors,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "status_code": exc.status_code,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing the request.",
                    "status_code": 500,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

    app.include_router(router)
    return app


# Root application instance for uvicorn
app = create_app()

