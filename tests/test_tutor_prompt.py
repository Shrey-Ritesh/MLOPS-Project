"""Tests that typed and confirmed OCR work reach the same tutor prompt."""
from __future__ import annotations

import pytest

from src.tutoring.prompt import build_tutor_user_message


def test_question_only_prompt_preserves_existing_typed_tutor_flow():
    message = build_tutor_user_message(
        question_or_topic="Explain Le Chatelier's principle.",
        answer_text="",
        answer_input_mode="typed",
    )

    assert message == "Explain Le Chatelier's principle."


def test_typed_answer_and_question_are_combined_for_tutor():
    message = build_tutor_user_message(
        question_or_topic="Find the oxidation state of sulfur in H2SO4.",
        answer_text="+4",
        answer_input_mode="typed",
    )

    assert "Find the oxidation state of sulfur in H2SO4." in message
    assert "Student's answer or work (typed by the student):\n+4" in message
    assert "show the relevant reasoning" in message


def test_confirmed_handwriting_is_included_as_answer_text():
    message = build_tutor_user_message(
        question_or_topic="Find the oxidation state of sulfur in H2SO4.",
        answer_text="+6",
        answer_input_mode="handwriting OCR",
    )

    assert "student-confirmed OCR transcription" in message
    assert "\n+6" in message
    assert "Find the oxidation state" in message


def test_handwriting_without_question_does_not_invite_guessing():
    message = build_tutor_user_message(
        question_or_topic="  ",
        answer_text="x = 4",
        answer_input_mode="handwriting OCR",
    )

    assert "No original question was supplied" in message
    assert "ask the student for it instead of guessing" in message
    assert "x = 4" in message


@pytest.mark.parametrize(
    ("question_or_topic", "answer_text", "answer_input_mode"),
    [
        ("", "", "typed"),
        ("Question", "Answer", "unsupported"),
    ],
)
def test_invalid_tutor_prompt_inputs_raise(question_or_topic, answer_text, answer_input_mode):
    with pytest.raises(ValueError):
        build_tutor_user_message(
            question_or_topic=question_or_topic,
            answer_text=answer_text,
            answer_input_mode=answer_input_mode,
        )
