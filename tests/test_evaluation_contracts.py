"""Acceptance tests for evaluation schemas, contracts, and normalizers.

Verifies:
- Pydantic contract validation (rejection of malformed / extra fields)
- Generalised MCQ normalization (labels, letters, formatted text, verbatim options)
- Generalised Numerical parsing (floats, scientific notation, units, tolerances)
- Edge case handling (empty strings, out-of-bounds options, non-numeric values)
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.evaluation.normalizer import (
    compare_numerical,
    detect_question_type,
    normalize_mcq_answer,
    parse_numerical_answer,
)
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    AttemptRecord,
    EvaluationOutcome,
    EvaluationResponse,
    EvidenceItem,
    EvidenceStatus,
    InputMode,
    QuestionType,
)
from src.models.jee import JEEOption


# --- 1. Schema Validation & Contract Tests ---

def test_valid_answer_submission_request():
    req = AnswerSubmissionRequest(
        question_id="jee-2025-22jan-s1-q52",
        answer_text="2",
        input_mode=InputMode.TYPED,
        session_id="session-user-123",
    )
    assert req.question_id == "jee-2025-22jan-s1-q52"
    assert req.answer_text == "2"
    assert req.input_mode == InputMode.TYPED
    assert req.session_id == "session-user-123"
    assert req.ocr_confidence is None
    assert req.is_user_corrected is False


def test_answer_submission_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AnswerSubmissionRequest(
            question_id="jee-2025-22jan-s1-q52",
            answer_text="2",
            session_id="session-123",
            unexpected_field="hack",  # extra field forbidden
        )


def test_answer_submission_requires_non_empty_ids():
    with pytest.raises(ValidationError):
        AnswerSubmissionRequest(
            question_id="",  # min_length=1
            answer_text="2",
            session_id="session-123",
        )


def test_valid_evaluation_response():
    resp = EvaluationResponse(
        submission_id="sub-001",
        question_id="jee-2025-22jan-s1-q52",
        question_type=QuestionType.MCQ,
        outcome=EvaluationOutcome.CORRECT,
        is_correct=True,
        score=1.0,
        student_answer_normalized="2",
        feedback_text="Correct! Propene and trans-isomers were identified properly.",
        evidence_status=EvidenceStatus.GROUNDED,
        evidence=[
            EvidenceItem(
                unit_id="ncert-chem-ch12-p01",
                chapter_title="Organic Chemistry",
                page=345,
                excerpt="Propene cannot show geometrical isomerism because one carbon has two identical H atoms.",
                relevance_score=0.92,
            )
        ],
    )
    assert resp.is_correct is True
    assert resp.score == 1.0
    assert len(resp.evidence) == 1
    assert resp.evidence[0].page == 345


def test_valid_attempt_record():
    attempt = AttemptRecord(
        attempt_id="att-999",
        session_id="user-42",
        question_id="jee-2025-22jan-s1-q52",
        subject="Chemistry",
        concept_tags=["organic_chemistry", "geometrical_isomerism"],
        input_mode=InputMode.HANDWRITTEN_OCR,
        outcome=EvaluationOutcome.INCORRECT,
        score=0.0,
    )
    assert attempt.subject == "Chemistry"
    assert attempt.outcome == EvaluationOutcome.INCORRECT
    assert attempt.score == 0.0


# --- 2. MCQ Normalizer Tests ---

@pytest.fixture
def sample_mcq_options():
    return [
        JEEOption(label="1", text="(A) and (E) Only"),
        JEEOption(label="2", text="(A), (D) and (E) Only"),
        JEEOption(label="3", text="(B) and (C) Only"),
        JEEOption(label="4", text="(C), (D) and (E) Only"),
    ]


@pytest.mark.parametrize(
    "raw_input, expected_label",
    [
        ("1", "1"),
        ("2", "2"),
        (" 3 ", "3"),
        ("Option 4", "4"),
        ("option 2", "2"),
        ("(1)", "1"),
        ("[3]", "3"),
        ("2.", "2"),
        ("a", "1"),
        ("A", "1"),
        ("b", "2"),
        ("B", "2"),
        ("c", "3"),
        ("d", "4"),
        ("Option B", "2"),
        ("(B)", "2"),
        ("(A), (D) and (E) Only", "2"),  # verbatim option text match
        ("(a) and (e) only", "1"),        # case-insensitive option text match
        # Edge cases: Roman numerals
        ("II", "2"),
        ("ii", "2"),
        ("(iv)", "4"),
        ("IV", "4"),
        # Edge cases: Quoted inputs
        ('"2"', "2"),
        ("'B'", "2"),
        # Edge cases: Compound responses (Label + Text)
        ("2. (A), (D) and (E) Only", "2"),
        ("B) (A), (D) and (E) Only", "2"),
        ("Option B: (A), (D) and (E) Only", "2"),
        ("2\n(A), (D) and (E) Only", "2"),
        ("2: But-2-ene", "2"),
    ],
)
def test_mcq_normalizer_valid_variants(sample_mcq_options, raw_input, expected_label):
    label, err = normalize_mcq_answer(raw_input, sample_mcq_options)
    assert err is None
    assert label == expected_label


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "5",
        "E",
        "Option 9",
        "random guessing text",
        "None of the above",
    ],
)
def test_mcq_normalizer_invalid_inputs(sample_mcq_options, invalid_input):
    label, err = normalize_mcq_answer(invalid_input, sample_mcq_options)
    assert label is None
    assert err is not None


# --- 3. Numerical Parser & Tolerance Tests ---

@pytest.mark.parametrize(
    "raw_input, expected_val",
    [
        ("275", 275.0),
        (" 275 ", 275.0),
        ("275.0", 275.0),
        ("0275", 275.0),
        ("2.75e2", 275.0),
        ("2.75E+2", 275.0),
        ("2.75 x 10^2", 275.0),
        ("2.75 * 10^2", 275.0),
        ("2.75 × 10^2", 275.0),
        ("275 %", 275.0),
        ("6 kJ/mol", 6.0),
        ("60 s", 60.0),
        ("45 K", 45.0),
        ("-40", -40.0),
        ("\u221240", -40.0),  # unicode minus
        ("0", 0.0),
        # Edge cases: Thousands separator commas
        ("2,850", 2850.0),
        ("1,000,000", 1000000.0),
        # Edge cases: Fractions / Rational expressions
        ("1/2", 0.5),
        ("3/4", 0.75),
        ("5 / 2", 2.5),
        ("-1/4", -0.25),
        # Edge cases: Unicode dashes / minus signs
        ("\u201345", -45.0),  # en-dash
        ("\u201445", -45.0),  # em-dash
        # Edge cases: Quoted numbers
        ('"275"', 275.0),
        ("'275'", 275.0),
        # Edge cases: Variations in scientific powers
        ("2.75 x 10 ^ 2", 275.0),
        ("2.75 * 10**2", 275.0),
        ("+275", 275.0),
        ("275.", 275.0),
    ],
)
def test_numerical_parser_valid_variants(raw_input, expected_val):
    val, err = parse_numerical_answer(raw_input)
    assert err is None
    assert val == pytest.approx(expected_val)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "abc",
        "two hundred",
        "NaN",
        "Infinity",
        "--40",
    ],
)
def test_numerical_parser_invalid_inputs(invalid_input):
    val, err = parse_numerical_answer(invalid_input)
    assert val is None
    assert err is not None


def test_compare_numerical_tolerance():
    # 274.5 vs 275.0 is within 1% relative tolerance (|274.5 - 275| / 275 = 0.0018 < 0.01)
    assert compare_numerical(274.5, 275.0, rel_tol=0.01) is True
    # Exact match
    assert compare_numerical(275.0, 275.0) is True
    # Out of tolerance (270 vs 275 is ~1.8% difference)
    assert compare_numerical(270.0, 275.0, rel_tol=0.01) is False


# --- 4. Question Type Detection Tests ---

def test_detect_question_type():
    mcq_options = [JEEOption(label="1", text="Opt 1")]
    assert detect_question_type(mcq_options) == QuestionType.MCQ
    assert detect_question_type([]) == QuestionType.NUMERICAL
    assert detect_question_type(None) == QuestionType.NUMERICAL

