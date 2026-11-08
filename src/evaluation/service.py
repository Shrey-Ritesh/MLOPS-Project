"""Core business logic service for student answer evaluation.

Evaluates typed answers and OCR-reviewed text against canonical JEE questions.
Orchestrates:
- Deterministic answer grading (MCQ options, numerical tolerances, chemical formulas)
- Explicit uncertainty safeguards for open-ended free-text responses
- NCERT evidence retrieval and pedagogical feedback synthesis
- Attempt persistence for personalization and recommendations
"""
from __future__ import annotations

import uuid
from typing import Any

from src.models.jee import JEEQuestionRecord
from .attempts import AttemptStore
from .chemical_normalizer import (
    compare_chemical_formulas,
    is_discursive_free_text,
    normalize_chemical_formula,
)
from .normalizer import (
    compare_numerical,
    detect_question_type,
    normalize_mcq_answer,
    parse_numerical_answer,
)
from .repository import QuestionRepository
from .retriever_adapter import NCERTRetrieverAdapter
from .schemas import (
    AnswerSubmissionRequest,
    AttemptRecord,
    EvaluationOutcome,
    EvaluationResponse,
    EvidenceItem,
    EvidenceStatus,
    QuestionType,
)


class QuestionNotFoundError(Exception):
    """Raised when a question cannot be found in the repository."""


class EvaluationService:
    """Service orchestrating question evaluation, evidence grounding, and attempt logging."""

    def __init__(
        self,
        repository: QuestionRepository | None = None,
        retriever_adapter: NCERTRetrieverAdapter | None = None,
        attempt_store: AttemptStore | None = None,
    ) -> None:
        self.repository = repository or QuestionRepository()
        self.retriever = retriever_adapter or NCERTRetrieverAdapter()
        self.attempt_store = attempt_store or AttemptStore()

    def evaluate_answer(self, submission: AnswerSubmissionRequest) -> EvaluationResponse:
        """Evaluate a student's answer submission against the canonical question record."""
        question_id = submission.question_id.strip()
        question = self.repository.get_question(question_id)
        if question is None:
            raise QuestionNotFoundError(f"Question with ID '{question_id}' not found.")

        submission_id = f"sub-{uuid.uuid4().hex[:12]}"
        q_type = detect_question_type(question.options)

        # Edge case: Question is missing an answer key in the dataset
        if not question.correct_answer or not question.correct_answer.strip():
            response = EvaluationResponse(
                submission_id=submission_id,
                question_id=question_id,
                question_type=q_type,
                outcome=EvaluationOutcome.KEY_UNAVAILABLE,
                is_correct=None,
                score=0.0,
                student_answer_normalized=None,
                feedback_text="Answer key is unavailable for this question in the repository.",
                evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                evidence=[],
            )
            self._record_attempt_safely(submission, question, response)
            return response

        # Safeguard: Explicit uncertainty for unsupported discursive free-text narratives
        # Note: In MCQs, verify the input does not match an option's exact text before flagging as discursive narrative.
        is_mcq_match = False
        if q_type == QuestionType.MCQ and question.options:
            norm_lbl, _ = normalize_mcq_answer(submission.answer_text, question.options)
            if norm_lbl is not None:
                is_mcq_match = True
            else:
                raw_text = submission.answer_text.strip()
                for opt in question.options:
                    if compare_chemical_formulas(raw_text, opt.text):
                        is_mcq_match = True
                        break

        if not is_mcq_match and is_discursive_free_text(submission.answer_text):
            response = EvaluationResponse(
                submission_id=submission_id,
                question_id=question_id,
                question_type=q_type,
                outcome=EvaluationOutcome.UNSUPPORTED_TYPE,
                is_correct=None,
                score=0.0,
                student_answer_normalized=None,
                feedback_text=(
                    "Your response appears to be an open-ended narrative explanation. "
                    "JEE evaluation requires a specific option choice (1–4) or a numerical calculation. "
                    "Please submit a definitive answer."
                ),
                evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                evidence=[],
            )
            self._record_attempt_safely(submission, question, response)
            return response

        # Fetch NCERT evidence for feedback grounding (resilient to offline Chroma)
        evidence, evidence_status = self.retriever.fetch_evidence(
            query=question.question_text,
            subject=question.subject,
        )

        # Route by question type
        if q_type == QuestionType.MCQ:
            response = self._evaluate_mcq(submission, question, submission_id, evidence, evidence_status)
        elif q_type == QuestionType.NUMERICAL:
            response = self._evaluate_numerical(submission, question, submission_id, evidence, evidence_status)
        else:
            response = EvaluationResponse(
                submission_id=submission_id,
                question_id=question_id,
                question_type=q_type,
                outcome=EvaluationOutcome.UNSUPPORTED_TYPE,
                is_correct=None,
                score=0.0,
                student_answer_normalized=None,
                feedback_text=f"Question type '{q_type.value}' is currently unsupported.",
                evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                evidence=[],
            )

        # Persist attempt record for Person 4
        self._record_attempt_safely(submission, question, response)
        return response

    def _record_attempt_safely(
        self,
        submission: AnswerSubmissionRequest,
        question: JEEQuestionRecord,
        response: EvaluationResponse,
    ) -> None:
        """Record attempt in store without failing the evaluation response on persistence errors."""
        try:
            attempt = AttemptRecord(
                attempt_id=f"att-{uuid.uuid4().hex[:12]}",
                session_id=submission.session_id,
                question_id=question.question_id,
                subject=question.subject,
                concept_tags=[question.subject.casefold()],
                input_mode=submission.input_mode,
                outcome=response.outcome,
                score=response.score,
            )
            self.attempt_store.record_attempt(attempt)
        except Exception:
            # Persistence failure should never crash the learner's response
            pass

    def _format_evidence_citation(self, evidence: list[EvidenceItem], is_correct: bool) -> str:
        """Format a clear, pedagogically synthesized citation from retrieved NCERT units."""
        if not evidence:
            return ""
        top_item = evidence[0]
        page_info = f" (p. {top_item.page})" if top_item.page else ""
        if is_correct:
            return f" Reference: NCERT {top_item.chapter_title}{page_info} — \"{top_item.excerpt[:180]}...\""
        else:
            return f" To understand the underlying principle, review NCERT {top_item.chapter_title}{page_info}: \"{top_item.excerpt[:180]}...\""

    def _evaluate_mcq(
        self,
        submission: AnswerSubmissionRequest,
        question: JEEQuestionRecord,
        submission_id: str,
        evidence: list[EvidenceItem],
        evidence_status: EvidenceStatus,
    ) -> EvaluationResponse:
        normalized_label, err = normalize_mcq_answer(submission.answer_text, question.options)

        # Fallback: check if student typed a chemical formula matching an option text
        if err is not None:
            raw_text = submission.answer_text.strip()
            for opt in question.options:
                if compare_chemical_formulas(raw_text, opt.text):
                    normalized_label = opt.label
                    err = None
                    break

        if err is not None:
            return EvaluationResponse(
                submission_id=submission_id,
                question_id=question.question_id,
                question_type=QuestionType.MCQ,
                outcome=EvaluationOutcome.INVALID_INPUT,
                is_correct=None,
                score=0.0,
                student_answer_normalized=None,
                feedback_text=f"Invalid answer: {err}",
                evidence_status=evidence_status,
                evidence=evidence,
            )

        target_key, key_err = normalize_mcq_answer(question.correct_answer, question.options)
        if key_err is not None or target_key is None:
            return EvaluationResponse(
                submission_id=submission_id,
                question_id=question.question_id,
                question_type=QuestionType.MCQ,
                outcome=EvaluationOutcome.KEY_UNAVAILABLE,
                is_correct=None,
                score=0.0,
                student_answer_normalized=normalized_label,
                feedback_text=(
                    f"The official answer key for this question is marked as '{question.correct_answer}' "
                    "or is disputed/dropped in the repository. Answer could not be graded."
                ),
                evidence_status=evidence_status,
                evidence=evidence,
            )

        is_correct = (normalized_label == target_key)
        outcome = EvaluationOutcome.CORRECT if is_correct else EvaluationOutcome.INCORRECT
        score = 1.0 if is_correct else 0.0

        citation = self._format_evidence_citation(evidence, is_correct)
        if is_correct:
            feedback = f"Correct! Option {normalized_label} is the right answer.{citation}"
        else:
            feedback = (
                f"Incorrect. You selected option {normalized_label}, "
                f"but the recorded correct answer is option {target_key}.{citation}"
            )

        return EvaluationResponse(
            submission_id=submission_id,
            question_id=question.question_id,
            question_type=QuestionType.MCQ,
            outcome=outcome,
            is_correct=is_correct,
            score=score,
            student_answer_normalized=normalized_label,
            feedback_text=feedback,
            evidence_status=evidence_status,
            evidence=evidence,
        )

    def _evaluate_numerical(
        self,
        submission: AnswerSubmissionRequest,
        question: JEEQuestionRecord,
        submission_id: str,
        evidence: list[EvidenceItem],
        evidence_status: EvidenceStatus,
    ) -> EvaluationResponse:
        parsed_val, err = parse_numerical_answer(submission.answer_text)

        if err is not None:
            return EvaluationResponse(
                submission_id=submission_id,
                question_id=question.question_id,
                question_type=QuestionType.NUMERICAL,
                outcome=EvaluationOutcome.INVALID_INPUT,
                is_correct=None,
                score=0.0,
                student_answer_normalized=None,
                feedback_text=f"Invalid numerical answer: {err}",
                evidence_status=evidence_status,
                evidence=evidence,
            )

        key_val, key_err = parse_numerical_answer(question.correct_answer)
        if key_err is not None or key_val is None:
            return EvaluationResponse(
                submission_id=submission_id,
                question_id=question.question_id,
                question_type=QuestionType.NUMERICAL,
                outcome=EvaluationOutcome.KEY_UNAVAILABLE,
                is_correct=None,
                score=0.0,
                student_answer_normalized=str(parsed_val),
                feedback_text="Answer key in repository could not be parsed as a valid numerical value.",
                evidence_status=evidence_status,
                evidence=evidence,
            )

        is_correct = compare_numerical(parsed_val, key_val, rel_tol=0.01, abs_tol=1e-3)
        outcome = EvaluationOutcome.CORRECT if is_correct else EvaluationOutcome.INCORRECT
        score = 1.0 if is_correct else 0.0

        citation = self._format_evidence_citation(evidence, is_correct)
        if is_correct:
            feedback = f"Correct! Calculated value {parsed_val} matches the expected answer {key_val}.{citation}"
        else:
            feedback = (
                f"Incorrect. Your calculated value was {parsed_val}, "
                f"but the recorded correct answer is {key_val}.{citation}"
            )

        return EvaluationResponse(
            submission_id=submission_id,
            question_id=question.question_id,
            question_type=QuestionType.NUMERICAL,
            outcome=outcome,
            is_correct=is_correct,
            score=score,
            student_answer_normalized=str(parsed_val),
            feedback_text=feedback,
            evidence_status=evidence_status,
            evidence=evidence,
        )
