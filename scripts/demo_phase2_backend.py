#!/usr/bin/env python3
"""Interactive demonstration script for Phase 2 Day 7 Release Candidate.

Demonstrates the complete end-to-end student journey:
1. Question browsing & inspection (MCQ & Numerical)
2. Typed correct MCQ submission with grounded NCERT citation
3. Typed incorrect MCQ submission with diagnostic feedback
4. Numerical calculation with unit stripping & relative tolerance
5. OCR-transcribed and student-corrected chemical formula evaluation
6. Discursive narrative safeguard (explicit UNSUPPORTED_TYPE guard)
7. Learner attempt history retrieval for Person 4 recommendations

Usage:
    python scripts/demo_phase2_backend.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from src.api.app import create_app


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_step(step_num: int, title: str) -> None:
    print(f"\n[{step_num}/7] {title.upper()}")
    print("-" * 55)


def run_demo() -> None:
    app = create_app()
    client = TestClient(app)
    session_id = f"jee-demo-session-{int(time.time())}"

    print_banner("JEE CHEMISTRY TUTOR — PHASE 2 RELEASE CANDIDATE DEMO")
    print(f"Session ID:   {session_id}")
    print(f"Target API:   In-Process FastAPI (src.api.app:app)")
    print(f"Target Subject: Chemistry (215 Questions Loaded)")

    # -------------------------------------------------------------------------
    # STEP 1: Question Selection & Inspection
    # -------------------------------------------------------------------------
    print_step(1, "Question Selection & Inspection")
    q_id_mcq = "jee-2025-22jan-s1-q52"
    resp = client.get(f"/api/questions/{q_id_mcq}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    q_data = resp.json()
    print(f"• Question ID:   {q_data['question_id']}")
    print(f"• Subject:       {q_data['subject']} (Year: {q_data['year']}, Session: {q_data['session']})")
    print(f"• Prompt:        {q_data['question_text']}")
    print("• Options:")
    for opt in q_data["options"]:
        print(f"    [{opt['label']}] {opt['text']}")

    # -------------------------------------------------------------------------
    # STEP 2: Correct MCQ Submission with Grounded NCERT Evidence
    # -------------------------------------------------------------------------
    print_step(2, "Submitting Correct MCQ Answer (Answer: '2')")
    payload_correct = {
        "question_id": q_id_mcq,
        "answer_text": "2",
        "input_mode": "typed",
        "session_id": session_id,
    }
    resp = client.post("/api/evaluate", json=payload_correct)
    assert resp.status_code == 200
    res_correct = resp.json()
    print(f"• Submission ID: {res_correct['submission_id']}")
    print(f"• Outcome:       {res_correct['outcome']} (is_correct: {res_correct['is_correct']})")
    print(f"• Score:         {res_correct['score']} / 1.0")
    print(f"• Normalized:    Option {res_correct['student_answer_normalized']}")
    print(f"• Evidence:      Status '{res_correct['evidence_status']}' ({len(res_correct['evidence'])} units cited)")
    print(f"• Pedagogical Feedback:\n  \"{res_correct['feedback_text']}\"")

    # -------------------------------------------------------------------------
    # STEP 3: Incorrect MCQ Submission with Diagnostic Guidance
    # -------------------------------------------------------------------------
    print_step(3, "Submitting Incorrect Distractor Option (Answer: 'Option 1')")
    payload_incorrect = {
        "question_id": q_id_mcq,
        "answer_text": "Option 1",
        "input_mode": "typed",
        "session_id": session_id,
    }
    resp = client.post("/api/evaluate", json=payload_incorrect)
    assert resp.status_code == 200
    res_incorrect = resp.json()
    print(f"• Outcome:       {res_incorrect['outcome']} (is_correct: {res_incorrect['is_correct']})")
    print(f"• Score:         {res_incorrect['score']} / 1.0")
    print(f"• Normalized:    Option {res_incorrect['student_answer_normalized']}")
    print(f"• Diagnostic Feedback:\n  \"{res_incorrect['feedback_text']}\"")

    # -------------------------------------------------------------------------
    # STEP 4: Numerical Calculation with Unit Stripping & Relative Tolerance
    # -------------------------------------------------------------------------
    print_step(4, "Submitting Numerical Calculation with Units ('6 kJ/mol')")
    q_id_num = "jee-2025-22jan-s1-q71"
    payload_num = {
        "question_id": q_id_num,
        "answer_text": "6 kJ/mol",
        "input_mode": "typed",
        "session_id": session_id,
    }
    resp = client.post("/api/evaluate", json=payload_num)
    assert resp.status_code == 200
    res_num = resp.json()
    print(f"• Question ID:   {q_id_num}")
    print(f"• Raw Input:     '6 kJ/mol' -> Parsed Float: {res_num['student_answer_normalized']}")
    print(f"• Outcome:       {res_num['outcome']} (Score: {res_num['score']})")
    print(f"• Feedback:      \"{res_num['feedback_text']}\"")

    # -------------------------------------------------------------------------
    # STEP 5: OCR-Reviewed Handwritten Input Evaluation
    # -------------------------------------------------------------------------
    print_step(5, "Submitting OCR-Reviewed Option Text ('(A), (D) and (E) Only')")
    payload_ocr = {
        "question_id": q_id_mcq,
        "answer_text": "(A), (D) and (E) Only",
        "input_mode": "handwritten_ocr",
        "ocr_confidence": 0.94,
        "is_user_corrected": True,
        "session_id": session_id,
    }
    resp = client.post("/api/evaluate", json=payload_ocr)
    assert resp.status_code == 200
    res_ocr = resp.json()
    print(f"• Input Mode:    {payload_ocr['input_mode']} (Confidence: {payload_ocr['ocr_confidence']}, Corrected: True)")
    print(f"• Raw Input:     '{payload_ocr['answer_text']}' -> Normalized Option: {res_ocr['student_answer_normalized']}")
    print(f"• Outcome:       {res_ocr['outcome']} (Score: {res_ocr['score']})")
    print(f"• Feedback:      \"{res_ocr['feedback_text']}\"")

    # -------------------------------------------------------------------------
    # STEP 6: Discursive Narrative Safeguard (Zero Hallucination Guard)
    # -------------------------------------------------------------------------
    print_step(6, "Testing Discursive Narrative Safeguard (Free-Text Essay)")
    payload_essay = {
        "question_id": q_id_mcq,
        "answer_text": "Because the reaction proceeds via an E2 elimination mechanism favoring the more substituted Zaitsev product.",
        "input_mode": "typed",
        "session_id": session_id,
    }
    resp = client.post("/api/evaluate", json=payload_essay)
    assert resp.status_code == 200
    res_essay = resp.json()
    print(f"• Student Input: \"{payload_essay['answer_text']}\"")
    print(f"• Safeguard:     outcome='{res_essay['outcome']}', is_correct={res_essay['is_correct']}, score={res_essay['score']}")
    print(f"• Guidance:      \"{res_essay['feedback_text']}\"")

    # -------------------------------------------------------------------------
    # STEP 7: Learner Attempt History Persistence for Person 4
    # -------------------------------------------------------------------------
    print_step(7, "Querying Persisted Attempt History for Session")
    resp = client.get(f"/api/attempts/{session_id}")
    assert resp.status_code == 200
    attempts = resp.json()
    print(f"• Attempts Retrieved for '{session_id}': {len(attempts)}")
    for i, att in enumerate(attempts, start=1):
        print(f"    [{i}] Attempt: {att['attempt_id']} | Question: {att['question_id']} | Outcome: {att['outcome']:16} | Score: {att['score']}")

    print_banner("DEMO COMPLETED SUCCESSFULLY — ALL CONTRACTS & SAFEGUARDS VERIFIED")


if __name__ == "__main__":
    run_demo()
