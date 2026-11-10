#!/usr/bin/env python3
"""Standalone Docker and local service smoke-test script (Day 6).

Requires zero external packages (pure standard library urllib.request).
Can be run inside a Docker container, in CI pipelines, or during local development
to verify the API is up, ready, and grading answers correctly.

Usage:
    python scripts/smoke_test_api.py [BASE_URL]
    # e.g.:
    python scripts/smoke_test_api.py http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def _http_request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute an HTTP request and return (status_code, response_json)."""
    headers = {"Accept": "application/json"}
    body_bytes = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body_bytes = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status_code = resp.status
            content = resp.read().decode("utf-8")
            return status_code, json.loads(content) if content else {}
    except urllib.error.HTTPError as err:
        content = err.read().decode("utf-8")
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}
        return err.code, parsed


def run_smoke_tests(base_url: str) -> bool:
    """Execute complete smoke-test suite against the target API."""
    base = base_url.rstrip("/")
    session_id = f"smoke-test-{os.getpid()}"
    print(f"==================================================")
    print(f"JEE CHEMISTRY EVALUATION API — SMOKE TEST SUITE")
    print(f"Target Base URL: {base}")
    print(f"Session ID:      {session_id}")
    print(f"==================================================\n")

    passed_count = 0
    total_count = 8

    # 1. Health and Readiness Probe
    print("[1/8] Probing GET /health ...", end=" ")
    try:
        code, body = _http_request(f"{base}/health")
        count = body.get("chemistry_questions_loaded", 0)
        if code == 200 and body.get("status") == "healthy" and count > 0:
            print(f"PASS (200 OK, {count} Chemistry questions ready)")
            passed_count += 1
        else:
            print(f"FAIL (Code: {code}, Body: {body})")
            return False
    except Exception as exc:
        print(f"CONNECTION ERROR: Could not connect to {base}/health: {exc}")
        return False

    # 2. List Questions
    print("[2/8] Testing GET /api/questions?subject=Chemistry&limit=5 ...", end=" ")
    code, body = _http_request(f"{base}/api/questions?subject=Chemistry&limit=5")
    if code == 200 and body.get("count") == 5 and len(body.get("questions", [])) == 5:
        print("PASS (5 questions retrieved)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 3. Get Question Details
    test_q_id = "jee-2025-22jan-s1-q52"
    print(f"[3/8] Testing GET /api/questions/{test_q_id} ...", end=" ")
    code, body = _http_request(f"{base}/api/questions/{test_q_id}")
    if code == 200 and body.get("question_id") == test_q_id and body.get("is_mcq") is True:
        print("PASS (MCQ metadata verified)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 4. Evaluate Correct MCQ Submission
    print("[4/8] Testing POST /api/evaluate (Correct MCQ: '2') ...", end=" ")
    code, body = _http_request(
        f"{base}/api/evaluate",
        method="POST",
        data={
            "question_id": test_q_id,
            "answer_text": "2",
            "input_mode": "typed",
            "session_id": session_id,
        },
    )
    if code == 200 and body.get("outcome") == "correct" and body.get("score") == 1.0 and body.get("is_correct") is True:
        print("PASS (Scored 1.0, Correct)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 5. Evaluate Incorrect MCQ Submission
    print("[5/8] Testing POST /api/evaluate (Incorrect MCQ: 'Option 1') ...", end=" ")
    code, body = _http_request(
        f"{base}/api/evaluate",
        method="POST",
        data={
            "question_id": test_q_id,
            "answer_text": "Option 1",
            "input_mode": "typed",
            "session_id": session_id,
        },
    )
    if code == 200 and body.get("outcome") == "incorrect" and body.get("score") == 0.0 and body.get("is_correct") is False:
        print("PASS (Scored 0.0, Grounded Feedback Generated)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 6. Evaluate Numerical Calculation with Units
    test_num_id = "jee-2025-22jan-s1-q71"
    print(f"[6/8] Testing POST /api/evaluate (Numerical: '6') ...", end=" ")
    code, body = _http_request(
        f"{base}/api/evaluate",
        method="POST",
        data={
            "question_id": test_num_id,
            "answer_text": "6",
            "input_mode": "typed",
            "session_id": session_id,
        },
    )
    if code == 200 and body.get("outcome") == "correct" and body.get("score") == 1.0:
        print("PASS (Numerical calculation verified)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 7. Query Learner Attempt History
    print(f"[7/8] Testing GET /api/attempts/{session_id} ...", end=" ")
    code, body = _http_request(f"{base}/api/attempts/{session_id}")
    if code == 200 and isinstance(body, list) and len(body) >= 3:
        print(f"PASS ({len(body)} attempts persisted and retrieved)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    # 8. Error Envelope Handling (404 Not Found)
    print("[8/8] Testing HTTP 404 Error Envelope ...", end=" ")
    code, body = _http_request(f"{base}/api/questions/jee-ghost-non-existent-999")
    if code == 404 and "error" in body and body["error"].get("code") == "HTTP_404":
        print("PASS (Clean structured error envelope)")
        passed_count += 1
    else:
        print(f"FAIL (Code: {code}, Body: {body})")
        return False

    print("\n--------------------------------------------------")
    print(f"SMOKE TEST SUMMARY: {passed_count}/{total_count} CHECKS PASSED (100%)")
    print("--------------------------------------------------")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    success = run_smoke_tests(target)
    sys.exit(0 if success else 1)

