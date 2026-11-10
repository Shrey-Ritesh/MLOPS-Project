"""Integration tests for Person 3 (Backend Evaluator) and Person 4 (Handwriting & Personalization).

Verifies the complete vertical integration agreed in phase2_weekly_work_split_roles_milestones.md:
1. Handwritten OCR confirmed input -> Person 3 Evaluator -> Person 4 SQLite Attempt Store -> Person 4 Practice Recommendation
2. Typed answer input -> Person 3 Evaluator -> Person 4 SQLite Attempt Store -> Person 4 Practice Recommendation
3. Discursive narrative essay -> Person 3 UNSUPPORTED_TYPE safeguard -> Person 4 history as 'uncertain'
4. Concept mastery tracking across multiple attempts generating pedagogical practice recommendations
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.evaluation.repository import QuestionRepository
from src.evaluation.retriever_adapter import NCERTRetrieverAdapter
from src.evaluation.schemas import (
    AnswerSubmissionRequest,
    EvaluationOutcome,
    EvidenceItem,
    EvidenceStatus,
    InputMode,
)
from src.evaluation.service import EvaluationService
from src.handwriting.ocr_space import TranscriptionResult
from src.learner.history import (
    AttemptRecord as P4AttemptRecord,
    Outcome as P4Outcome,
    PracticeCandidate,
    SQLiteAttemptStore,
    recommend_practice,
)
from src.models.jee import ExtractionStatus, JEEOption, JEEQuestionRecord


# --- Fixtures ---

@pytest.fixture
def mock_chemistry_questions() -> list[JEEQuestionRecord]:
    return [
        JEEQuestionRecord(
            question_id="jee-chem-isom-01",
            source_id="src-01",
            year=2025,
            session="jan",
            shift="s1",
            subject="Chemistry",
            question_text="Which of the following compounds exhibits geometrical isomerism?",
            options=[
                JEEOption(label="1", text="Propene"),
                JEEOption(label="2", text="But-2-ene"),
                JEEOption(label="3", text="2-Methylpropene"),
                JEEOption(label="4", text="Ethene"),
            ],
            correct_answer="2",
            source_page=10,
            asset_ids=[],
            extraction_status=ExtractionStatus.SUCCESS,
            difficulty=None,
            solution_text=None,
        ),
        JEEQuestionRecord(
            question_id="jee-chem-isom-02",
            source_id="src-01",
            year=2025,
            session="jan",
            shift="s2",
            subject="Chemistry",
            question_text="Identify the molecule with zero dipole moment among the following isomers:",
            options=[
                JEEOption(label="1", text="cis-But-2-ene"),
                JEEOption(label="2", text="trans-But-2-ene"),
                JEEOption(label="3", text="cis-1,2-Dichloroethene"),
                JEEOption(label="4", text="1,1-Dichloroethene"),
            ],
            correct_answer="2",
            source_page=12,
            asset_ids=[],
            extraction_status=ExtractionStatus.SUCCESS,
            difficulty=None,
            solution_text=None,
        ),
        JEEQuestionRecord(
            question_id="jee-chem-thermo-01",
            source_id="src-02",
            year=2025,
            session="jan",
            shift="s1",
            subject="Chemistry",
            question_text="For a cyclic process, calculate the net change in internal energy (ΔU) in Joules:",
            options=[],
            correct_answer="0",
            source_page=45,
            asset_ids=[],
            extraction_status=ExtractionStatus.SUCCESS,
            difficulty=None,
            solution_text=None,
        ),
    ]


class MockRepo(QuestionRepository):
    def __init__(self, questions: list[JEEQuestionRecord]):
        super().__init__(data_path="dummy.jsonl")
        self._questions_by_id = {q.question_id: q for q in questions}
        self._loaded = True


def _map_outcome_to_p4(outcome: EvaluationOutcome) -> P4Outcome:
    """Helper mapping Person 3 EvaluationOutcome to Person 4 Outcome enum."""
    if outcome == EvaluationOutcome.CORRECT:
        return "correct"
    elif outcome == EvaluationOutcome.INCORRECT:
        return "incorrect"
    elif outcome == EvaluationOutcome.PARTIAL:
        return "partial"
    else:
        return "uncertain"


# --- Integration Tests ---

def test_handwritten_ocr_confirmed_evaluation_to_recommendation(mock_chemistry_questions):
    """Scenario 1: Student uses OCR, confirms transcription, evaluates, and updates mastery."""
    repo = MockRepo(mock_chemistry_questions)
    retriever = NCERTRetrieverAdapter(
        retrieve_fn=lambda *a, **k: {
            "status": "ok",
            "ncert": [
                {
                    "id": "ncert-hydrocarbons-378",
                    "document": "But-2-ene exhibits geometrical isomerism due to restricted rotation about the C=C bond.",
                    "metadata": {"chapter_title": "Hydrocarbons", "page": 378},
                    "score": 0.95,
                }
            ],
        }
    )
    eval_service = EvaluationService(repository=repo, retriever_adapter=retriever)
    p4_store = SQLiteAttemptStore(database_path=":memory:")

    learner_id = "student-jee-2025"
    target_q = mock_chemistry_questions[0]

    # 1. Simulate Person 4 OCR transcription & user confirmation
    ocr_result = TranscriptionResult(text="2. But-2-ene\n", recognizer="OCR.space Engine 3")
    confirmed_text = ocr_result.text.strip()

    # 2. Submit to Person 3 Evaluator via common contract
    submission = AnswerSubmissionRequest(
        question_id=target_q.question_id,
        answer_text=confirmed_text,
        input_mode=InputMode.HANDWRITTEN_OCR,
        session_id=learner_id,
        is_user_corrected=True,
    )
    eval_resp = eval_service.evaluate_answer(submission)

    # 3. Verify Person 3 Evaluator correctly handled OCR compound text
    assert eval_resp.outcome == EvaluationOutcome.CORRECT
    assert eval_resp.is_correct is True
    assert eval_resp.score == 1.0
    assert eval_resp.student_answer_normalized == "2"
    assert "NCERT Hydrocarbons (p. 378)" in eval_resp.feedback_text

    # 4. Map and persist to Person 4 SQLiteAttemptStore
    p4_record = P4AttemptRecord(
        learner_id=learner_id,
        question_id=target_q.question_id,
        concept_ids=("isomerism", "organic_chemistry"),
        outcome=_map_outcome_to_p4(eval_resp.outcome),
        answer_mode="handwritten",
        attempted_at=datetime.now(timezone.utc),
    )
    p4_store.save(p4_record)

    # 5. Verify learner history in Person 4 store
    history = p4_store.list_for_learner(learner_id)
    assert len(history) == 1
    assert history[0].outcome == "correct"
    assert history[0].answer_mode == "handwritten"

    # 6. Build practice candidates and get next recommendation
    candidates = [
        PracticeCandidate(question_id=q.question_id, concept_ids=("isomerism",), title=q.question_text[:30])
        for q in mock_chemistry_questions if q.options
    ]
    rec = recommend_practice(history, candidates, minimum_evidence=2)
    assert rec is not None
    # Next question recommended to build evidence
    assert rec.question_id == "jee-chem-isom-02"
    assert "build evidence" in rec.reason.lower()


def test_typed_numerical_error_triggers_targeted_practice_recommendation(mock_chemistry_questions):
    """Scenario 2: Student repeatedly fails thermodynamics, triggering weak-area recommendation."""
    repo = MockRepo(mock_chemistry_questions)
    eval_service = EvaluationService(repository=repo)
    p4_store = SQLiteAttemptStore(database_path=":memory:")

    learner_id = "student-thermo-weak"
    thermo_q = mock_chemistry_questions[2]  # correct_answer is "0"

    # Student makes 3 incorrect attempts submitting wrong numerical values (e.g. "100", "50", "25")
    for wrong_ans in ["100 J", "50 J", "25 J"]:
        sub = AnswerSubmissionRequest(
            question_id=thermo_q.question_id,
            answer_text=wrong_ans,
            input_mode=InputMode.TYPED,
            session_id=learner_id,
        )
        resp = eval_service.evaluate_answer(sub)
        assert resp.outcome == EvaluationOutcome.INCORRECT

        record = P4AttemptRecord(
            learner_id=learner_id,
            question_id=thermo_q.question_id,
            concept_ids=("thermodynamics",),
            outcome=_map_outcome_to_p4(resp.outcome),
            answer_mode="typed",
            attempted_at=datetime.now(timezone.utc),
        )
        p4_store.save(record)

    # Verify history
    history = p4_store.list_for_learner(learner_id)
    assert len(history) == 3

    # Generate practice recommendation
    candidates = [
        PracticeCandidate(question_id=thermo_q.question_id, concept_ids=("thermodynamics",), title="Cyclic Process ΔU"),
        PracticeCandidate(question_id="jee-chem-isom-01", concept_ids=("isomerism",), title="Geometrical Isomerism"),
    ]
    rec = recommend_practice(history, candidates, minimum_evidence=3, mastery_threshold=0.7)
    assert rec is not None
    assert rec.concept_id == "thermodynamics"
    assert "below the 70% practice threshold" in rec.reason.lower()
    assert rec.accuracy == 0.0


def test_discursive_essay_safeguard_in_integrated_pipeline(mock_chemistry_questions):
    """Scenario 3: Student types an open-ended narrative paragraph into objective question."""
    repo = MockRepo(mock_chemistry_questions)
    eval_service = EvaluationService(repository=repo)
    p4_store = SQLiteAttemptStore(database_path=":memory:")

    learner_id = "student-essayist"
    target_q = mock_chemistry_questions[0]

    sub = AnswerSubmissionRequest(
        question_id=target_q.question_id,
        answer_text="Because double bonds prevent free rotation and cis-trans isomers have different boiling points.",
        input_mode=InputMode.TYPED,
        session_id=learner_id,
    )
    resp = eval_service.evaluate_answer(sub)

    # Person 3 safeguard flags UNSUPPORTED_TYPE
    assert resp.outcome == EvaluationOutcome.UNSUPPORTED_TYPE
    assert resp.is_correct is None
    assert "narrative explanation" in resp.feedback_text.lower()

    # Mapped to Person 4 'uncertain'
    record = P4AttemptRecord(
        learner_id=learner_id,
        question_id=target_q.question_id,
        concept_ids=("isomerism",),
        outcome=_map_outcome_to_p4(resp.outcome),
        answer_mode="typed",
        attempted_at=datetime.now(timezone.utc),
    )
    p4_store.save(record)

    # Person 4 recommend_practice ignores 'uncertain' outcomes from accuracy calculations
    history = p4_store.list_for_learner(learner_id)
    candidates = [PracticeCandidate(question_id=target_q.question_id, concept_ids=("isomerism",), title="Isomerism")]
    rec = recommend_practice(history, candidates, minimum_evidence=1)
    assert rec is not None
    # Still considered 0 evidence attempts for accuracy scoring
    assert rec.evidence_attempts == 0
