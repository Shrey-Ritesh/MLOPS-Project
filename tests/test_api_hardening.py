"""Request limits, CORS configuration and log redaction."""
import asyncio
import json
import logging

from src.api.limits import MaxBodySizeMiddleware, cors_origins, max_request_bytes
from src.evaluation.attempts import AttemptStore
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter


async def _echo_app(scope, receive, send):
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(len(body)).encode()})


def _call(app, chunks, headers=()):
    sent = []
    queue = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive():
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": list(headers)}
    asyncio.run(app(scope, receive, send))
    return sent


def test_small_body_passes():
    sent = _call(MaxBodySizeMiddleware(_echo_app, max_bytes=10), [b"12345"])
    assert sent[0]["status"] == 200


def test_declared_content_length_over_limit_is_rejected():
    sent = _call(
        MaxBodySizeMiddleware(_echo_app, max_bytes=10),
        [b"x"],
        headers=[(b"content-length", b"500")],
    )
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_streamed_body_without_length_is_rejected():
    sent = _call(MaxBodySizeMiddleware(_echo_app, max_bytes=10), [b"123456", b"789012"])
    assert sent[0]["status"] == 413


def test_max_request_bytes_env():
    assert max_request_bytes({"MAX_REQUEST_BYTES": "2048"}) == 2048
    assert max_request_bytes({"MAX_REQUEST_BYTES": "junk"}) == 1_000_000
    assert max_request_bytes({}) == 1_000_000


def test_cors_defaults_are_not_wildcard():
    assert "*" not in cors_origins({})
    assert cors_origins({"CORS_ALLOWED_ORIGINS": "https://a.example, https://b.example"}) == [
        "https://a.example",
        "https://b.example",
    ]


def test_retrieval_failure_log_does_not_contain_query(caplog):
    secret = "my private answer about benzene"

    def boom(*args, **kwargs):
        raise RuntimeError(f"failed for {secret}")

    adapter = NCERTRetrieverAdapter(retrieve_fn=boom)
    with caplog.at_level(logging.WARNING):
        adapter.fetch_evidence(secret)
    assert secret not in caplog.text


def test_invalid_attempt_log_does_not_contain_stored_answer(tmp_path, caplog):
    secret = "learner typed this secret answer"
    path = tmp_path / "attempts.jsonl"
    path.write_text(json.dumps({"submitted_answer": secret, "bogus": True}) + "\n")
    store = AttemptStore(data_path=path)
    with caplog.at_level(logging.WARNING):
        store.count()
    assert secret not in caplog.text
