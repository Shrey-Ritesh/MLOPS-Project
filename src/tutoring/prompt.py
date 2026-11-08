"""Build user messages for typed and OCR-transcribed tutoring inputs."""
from __future__ import annotations


def build_tutor_user_message(
    *,
    question_or_topic: str,
    answer_text: str,
    answer_input_mode: str,
) -> str:
    """Combine the student's question/context and answer into one LLM request.

    The uploaded image itself is not sent to Groq: OCR.space transcribes it and
    the student confirms/corrects that text before it enters this prompt.
    """
    question = question_or_topic.strip()
    answer = answer_text.strip()
    if not question and not answer:
        raise ValueError("Enter a question/topic or an answer/work sample.")
    if not answer:
        return question

    if answer_input_mode not in {"typed", "handwriting OCR"}:
        raise ValueError("answer_input_mode must be 'typed' or 'handwriting OCR'.")

    sections = []
    if question:
        sections.append(f"Question or topic:\n{question}")
    else:
        sections.append(
            "No original question was supplied. If it is needed to review the work, "
            "ask the student for it instead of guessing."
        )
    source_note = (
        "student-confirmed OCR transcription; notation may still be uncertain"
        if answer_input_mode == "handwriting OCR"
        else "typed by the student"
    )
    sections.append(f"Student's answer or work ({source_note}):\n{answer}")
    sections.append(
        "Please review the answer or work, explain what is correct or needs fixing, "
        "and show the relevant reasoning."
    )
    return "\n\n".join(sections)
