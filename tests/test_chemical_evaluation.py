"""Tests for chemical formula normalization, discursive free-text safeguards, and synthesized feedback (Day 4).

Verifies:
- Formula subscript and superscript translation (H₂O -> H2O, Fe³⁺ -> Fe^{3+})
- Ion charge representations (Fe^3+, Fe3+, Fe^{3+})
- Organic structural formulas (CH3-CH2-OH vs CH3CH2OH)
- Explicit UNCERTAIN/UNSUPPORTED_TYPE safeguard for narrative discursive explanations
- Synthesized pedagogical feedback referencing NCERT principles
"""
from __future__ import annotations

import pytest

from src.evaluation.chemical_normalizer import (
    compare_chemical_formulas,
    is_discursive_free_text,
    normalize_chemical_formula,
)
from src.evaluation.repository import QuestionRepository
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    EvaluationOutcome,
    EvidenceItem,
    EvidenceStatus,
    InputMode,
    QuestionType,
)
from src.evaluation.service import EvaluationService
from src.models.jee import ExtractionStatus, JEEOption, JEEQuestionRecord


# --- 1. Chemical Normalizer Unit Tests ---

@pytest.mark.parametrize(
    "raw_input, expected_normalized",
    [
        ("H\u2082O", "H2O"),                         # H₂O
        ("CO\u2082", "CO2"),                         # CO₂
        ("Fe\u00b3\u207a", "Fe^{3+}"),               # Fe³⁺
        ("Fe^3+", "Fe^{3+}"),                        # Fe^3+
        ("Fe3+", "Fe^{3+}"),                         # Fe3+
        ("SO\u2084\u00b2\u207b", "SO4^{2-}"),        # SO₄²⁻
        ("SO4^2-", "SO4^{2-}"),                      # SO4^2-
        ("NO\u2082\u207b", "NO2^{-} "),              # NO₂⁻
        ("CO2 (g)", "CO2"),                          # Gas state symbol
        ("H2O (l)", "H2O"),                          # Liquid state symbol
        ("NaCl (aq)", "NaCl"),                       # Aqueous state symbol
    ],
)
def test_normalize_chemical_formula(raw_input, expected_normalized):
    cleaned_expected = expected_normalized.strip()
    assert normalize_chemical_formula(raw_input) == cleaned_expected


def test_compare_chemical_formulas_organic_bonding():
    # Condensed formula vs single-bonded formula
    assert compare_chemical_formulas("CH3-CH2-OH", "CH3CH2OH") is True
    assert compare_chemical_formulas("CH3CH2OH", "CH3-CH2-OH") is True
    assert compare_chemical_formulas("CH3-COOH", "CH3COOH") is True


def test_compare_chemical_formulas_unicode_variants():
    assert compare_chemical_formulas("H\u2082O", "H2O") is True
    assert compare_chemical_formulas("Fe\u00b3\u207a", "Fe^3+") is True
    assert compare_chemical_formulas("SO\u2084\u00b2\u207b", "SO4^2-") is True


# --- 2. Free-Text Safeguard Tests ---

@pytest.mark.parametrize(
    "discursive_text",
    [
        "Because entropy increases and delta G is negative at higher temperatures",
        "Due to the presence of lone pairs on the nitrogen atom",
        "I think option 2 is correct because the trans isomer has zero dipole moment",
        "As per the molecular orbital theory the bond order is two",
        "The reaction mechanism proceeds via carbocation intermediate",
    ],
)
def test_detect_discursive_free_text_positive(discursive_text):
    assert is_discursive_free_text(discursive_text) is True


@pytest.mark.parametrize(
    "concise_input",
    [
        "2",
        "Option 2",
        "(B)",
        "275",
        "2.75 x 10^2",
        "Propene",
        "H2O",
        "Fe^3+",
    ],
)
def test_detect_discursive_free_text_negative(concise_input):
    assert is_discursive_free_text(concise_input) is False


# --- 3. Service Integration with Formula & Discursive Safeguards ---

@pytest.fixture
def formula_mcq_question():
    return JEEQuestionRecord(
        question_id="test-formula-q01",
        source_id="src-formula-01",
        year=2025,
        session="22 Jan",
        shift="1",
        subject="Chemistry",
        question_text="Which molecule has a bent molecular geometry?",
        options=[
            JEEOption(label="1", text="CO2"),
            JEEOption(label="2", text="H2O"),
            JEEOption(label="3", text="CH4"),
            JEEOption(label="4", text="BF3"),
        ],
        correct_answer="2",
        solution_text=None,
        source_page=14,
        asset_ids=[],
        extraction_status=ExtractionStatus.SUCCESS,
        difficulty=None,
    )


class MockFormulaQuestionRepo(QuestionRepository):
    def __init__(self, question: JEEQuestionRecord):
        super().__init__(data_path="dummy.jsonl")
        self._questions_by_id = {question.question_id: question}
        self._loaded = True


def test_mcq_chemical_formula_option_match(formula_mcq_question):
    # Student submits unicode formula "H₂O" instead of option label "2"
    repo = MockFormulaQuestionRepo(formula_mcq_question)
    service = EvaluationService(
        repository=repo,
        retriever_adapter=NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: {"status": "ok", "ncert": []}),
    )

    sub = AnswerSubmissionRequest(
        question_id="test-formula-q01",
        answer_text="H\u2082O",  # H₂O
        input_mode=InputMode.TYPED,
        session_id="chem-student-01",
    )
    resp = service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.score == 1.0
    assert resp.student_answer_normalized == "2"


def test_discursive_narrative_safeguard_triggers_unsupported(formula_mcq_question):
    repo = MockFormulaQuestionRepo(formula_mcq_question)
    service = EvaluationService(
        repository=repo,
        retriever_adapter=NCERTRetrieverAdapter(retrieve_fn=lambda *a, **k: {"status": "ok", "ncert": []}),
    )

    sub = AnswerSubmissionRequest(
        question_id="test-formula-q01",
        answer_text="Because water has two lone pairs on oxygen causing repulsion and bent shape",
        input_mode=InputMode.TYPED,
        session_id="chem-student-02",
    )
    resp = service.evaluate_answer(sub)

    assert resp.outcome == EvaluationOutcome.UNSUPPORTED_TYPE
    assert resp.is_correct is None
    assert resp.score == 0.0
    assert "narrative explanation" in resp.feedback_text.lower()
    assert "definitive answer" in resp.feedback_text.lower()


def test_synthesized_feedback_distinction(formula_mcq_question):
    # Mock evidence item
    mock_item = EvidenceItem(
        unit_id="ncert-bonding-bent",
        chapter_title="Chemical Bonding",
        page=112,
        excerpt="Water has two bonding pairs and two lone pairs resulting in a bent geometry.",
        relevance_score=0.94,
    )

    repo = MockFormulaQuestionRepo(formula_mcq_question)
    retriever = NCERTRetrieverAdapter(
        retrieve_fn=lambda *a, **k: {
            "status": "ok",
            "ncert": [
                {
                    "id": mock_item.unit_id,
                    "document": mock_item.excerpt,
                    "metadata": {"chapter_title": mock_item.chapter_title, "page": mock_item.page},
                    "score": mock_item.relevance_score,
                }
            ],
        }
    )
    service = EvaluationService(repository=repo, retriever_adapter=retriever)

    # 1. Correct submission
    resp_correct = service.evaluate_answer(
        AnswerSubmissionRequest(
            question_id="test-formula-q01",
            answer_text="2",
            session_id="test-sess",
        )
    )
    assert "Reference: NCERT Chemical Bonding (p. 112)" in resp_correct.feedback_text
    assert "Correct!" in resp_correct.feedback_text

    # 2. Incorrect submission
    resp_incorrect = service.evaluate_answer(
        AnswerSubmissionRequest(
            question_id="test-formula-q01",
            answer_text="1",
            session_id="test-sess",
        )
    )
    assert "To understand the underlying principle, review NCERT Chemical Bonding (p. 112)" in resp_incorrect.feedback_text
    assert "Incorrect" in resp_incorrect.feedback_text


def test_assertion_reason_option_text_not_flagged_as_discursive():
    """Verify that when a student enters the verbatim sentence of an assertion-reason option,
    it is correctly parsed as an MCQ choice rather than triggering discursive UNSUPPORTED_TYPE."""
    from src.models.jee import JEEOption
    question = JEEQuestionRecord(
        question_id="test-stmt-q01",
        subject="Chemistry",
        year=2025,
        session="jan",
        shift="s1",
        source_id="src-test-01",
        source_page=1,
        asset_ids=[],
        extraction_status="success",
        question_text="Given below are two statements: Statement I and Statement II.",
        options=[
            JEEOption(label="1", text="Statement I is incorrect but Statement II is correct"),
            JEEOption(label="2", text="Both Statement I and Statement II are correct"),
            JEEOption(label="3", text="Statement I is correct but Statement II is incorrect"),
            JEEOption(label="4", text="Both Statement I and Statement II are incorrect"),
        ],
        correct_answer="2",
        difficulty=None,
        solution_text=None,
    )
    repo = MockFormulaQuestionRepo(question)
    service = EvaluationService(repository=repo)

    sub = AnswerSubmissionRequest(
        question_id="test-stmt-q01",
        answer_text="Both Statement I and Statement II are correct",
        session_id="test-sess",
    )
    resp = service.evaluate_answer(sub)
    assert resp.outcome == EvaluationOutcome.CORRECT
    assert resp.is_correct is True
    assert resp.student_answer_normalized == "2"


def test_multistatement_prefix_option_matching():
    """Verify options starting with parenthesized letters like (A) and (B) only match their true option,
    not prematurely chopped to Option 1 (A)."""
    from src.models.jee import JEEOption
    from src.evaluation.normalizer import normalize_mcq_answer
    options = [
        JEEOption(label="1", text="(A) and (C) only"),
        JEEOption(label="2", text="(B) and (D) only"),
        JEEOption(label="3", text="(A) and (B) only"),
        JEEOption(label="4", text="(C) and (D) only"),
    ]
    lbl, err = normalize_mcq_answer("(A) and (B) only", options)
    assert err is None
    assert lbl == "3"


def test_chemical_and_math_operators_preserved_in_option_matching():
    """Verify comparison operators (<, >, =, +, -) in thermodynamic options distinguish distinct states."""
    from src.models.jee import JEEOption
    from src.evaluation.normalizer import normalize_mcq_answer
    options = [
        JEEOption(label="1", text="ΔU < 0, q = 0, w > 0"),
        JEEOption(label="2", text="ΔU = 0, q = 0, w = 0"),
        JEEOption(label="3", text="ΔU > 0, q = 0, w > 0"),
        JEEOption(label="4", text="ΔU = 0, q < 0, w > 0"),
    ]
    lbl, err = normalize_mcq_answer("ΔU > 0, q = 0, w > 0", options)
    assert err is None
    assert lbl == "3"


