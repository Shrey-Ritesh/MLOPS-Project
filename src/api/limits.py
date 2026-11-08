"""Request-size limiting and CORS configuration for the evaluation API."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping

DEFAULT_MAX_REQUEST_BYTES = 1_000_000
DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")


def max_request_bytes(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    raw = env.get("MAX_REQUEST_BYTES", "")
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_MAX_REQUEST_BYTES


def cors_origins(env: Mapping[str, str] | None = None) -> list[str]:
    """Comma-separated CORS_ALLOWED_ORIGINS; defaults to the local Streamlit origin."""
    env = os.environ if env is None else env
    raw = env.get("CORS_ALLOWED_ORIGINS", "")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)


class MaxBodySizeMiddleware:
    """Reject request bodies larger than ``max_bytes`` with a 413 JSON error.

    Checks the declared Content-Length first, then counts streamed bytes so that
    chunked requests without a length cannot bypass the limit.
    """

    def __init__(self, app, max_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes or max_request_bytes()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)

    async def _reject(self, send) -> None:
        body = json.dumps(
            {
                "detail": "Request body too large.",
                "error": {
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": f"Request body exceeds {self.max_bytes} bytes.",
                    "status_code": 413,
                },
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass
