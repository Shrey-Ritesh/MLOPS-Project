"""Pure session-state helpers for Streamlit navigation, practice selection, and sign-out."""
from __future__ import annotations

import time
from typing import Any, MutableMapping

APP_VIEWS: tuple[str, ...] = (
    "Chemistry Tutor",
    "Practice Problems",
    "My Progress",
    "Debug & Inspection",
)

USER_SESSION_KEYS: tuple[str, ...] = (
    "user",
    "active_view",
    "pending_active_view",
    "selected_question_id",
    "pending_practice_question_id",
    "practice_question_picker",
    "practice_loaded_notice",
    "practice_question_started_at",
    "practice_hint_count",
    "practice_evaluation_result",
    "practice_last_img_id",
    "practice_ocr_text",
    "practice_ocr_meta",
    "last_selected_difficulty",
    "last_prediction_event",
    "last_uploaded_file_id",
    "extracted_ocr_text",
    "raw_ocr_text",
    "ocr_meta",
    "debug_info",
)


def queue_practice_question(
    state: MutableMapping[str, Any],
    question_id: str,
    *,
    switch_view: bool = False,
    notice: str | None = None,
    started_at: float | None = None,
) -> None:
    """Queue a question selection for the next rerun before widgets instantiate."""
    state["pending_practice_question_id"] = question_id
    state["practice_question_started_at"] = time.monotonic() if started_at is None else started_at
    state["practice_hint_count"] = 0
    state.pop("practice_evaluation_result", None)
    if notice:
        state["practice_loaded_notice"] = notice
    if switch_view:
        state["pending_active_view"] = "Practice Problems"


def sync_navigation_and_practice_state(state: MutableMapping[str, Any]) -> None:
    """Apply pending navigation or question changes before widgets bind their keys."""
    if state.get("active_view") not in APP_VIEWS:
        state["active_view"] = APP_VIEWS[0]

    pending_view = state.pop("pending_active_view", None)
    if pending_view in APP_VIEWS:
        state["active_view"] = pending_view

    pending_qid = state.pop("pending_practice_question_id", None)
    if pending_qid:
        state["selected_question_id"] = pending_qid
        state["practice_question_picker"] = pending_qid


def clear_user_session_state(state: MutableMapping[str, Any]) -> None:
    """Reset authentication and all user-scoped session keys on sign-out."""
    state["authenticated"] = False
    for key in USER_SESSION_KEYS:
        state.pop(key, None)
