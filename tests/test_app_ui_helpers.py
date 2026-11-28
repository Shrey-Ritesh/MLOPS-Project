"""Unit tests for question asset resolution used by the Streamlit GUI."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.assets import AssetRepository
from src.evaluation.ui_state import (
    APP_VIEWS,
    clear_user_session_state,
    queue_practice_question,
    sync_navigation_and_practice_state,
)


def test_resolve_for_question_replaces_tags_and_resolves_paths(tmp_path: Path) -> None:
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    img1 = crops_dir / "ast-1111aaaa.png"
    img1.write_bytes(b"png-bytes-1")

    manifest_path = tmp_path / "assets_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "asset_id": "ast-1111aaaa",
                "file_path": str(img1),
                "asset_type": "diagram",
                "source_page": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    repo = AssetRepository(manifest_path=manifest_path, assets_dir=crops_dir)
    cleaned, resolved = repo.resolve_for_question(
        "Identify the product: [IMAGE: ast-1111aaaa]",
        asset_ids=["ast-1111aaaa"],
    )

    assert cleaned == "Identify the product: (Figure 1)"
    assert len(resolved) == 1
    assert resolved[0].asset_id == "ast-1111aaaa"
    assert resolved[0].figure_number == 1
    assert resolved[0].file_path == img1
    assert resolved[0].exists is True
    assert resolved[0].source_page == 3


def test_resolve_for_question_deduplicates_and_reports_missing_files(tmp_path: Path) -> None:
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    img2 = crops_dir / "ast-inline99.png"
    img2.write_bytes(b"png-bytes-2")

    repo = AssetRepository(
        manifest_path=tmp_path / "missing_manifest.jsonl",
        assets_dir=crops_dir,
    )
    cleaned, resolved = repo.resolve_for_question(
        "Compare [IMAGE: ast-missing1] with [IMAGE: ast-inline99] and [IMAGE: ast-missing1].",
        asset_ids=["ast-missing1"],
    )

    assert cleaned == "Compare (Figure 1) with (Figure 2) and (Figure 1)."
    assert [r.asset_id for r in resolved] == ["ast-missing1", "ast-inline99"]
    assert resolved[0].exists is False
    assert resolved[1].exists is True
    assert resolved[1].file_path == img2


def test_resolve_falls_back_to_assets_dir_when_manifest_path_moved(tmp_path: Path) -> None:
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    relocated = crops_dir / "ast-reloc123.png"
    relocated.write_bytes(b"png")

    manifest_path = tmp_path / "assets_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "asset_id": "ast-reloc123",
                "file_path": "/nonexistent/other/dir/ast-reloc123.png",
                "asset_type": "diagram",
                "source_page": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    repo = AssetRepository(manifest_path=manifest_path, assets_dir=crops_dir)
    _, resolved = repo.resolve_for_question("Question stem", asset_ids=["ast-reloc123"])

    assert len(resolved) == 1
    assert resolved[0].exists is True
    assert resolved[0].file_path == relocated


def test_repo_chemistry_sample_question_asset_resolves() -> None:
    manifest = Path("data/processed/assets_manifest.jsonl")
    crops_dir = Path("data/assets/crops")
    if not manifest.is_file() or not crops_dir.is_dir():
        return

    repo = AssetRepository(manifest_path=manifest, assets_dir=crops_dir)
    cleaned, resolved = repo.resolve_for_question(
        "Choose the correct answer from the options given below : [IMAGE: ast-ba245e9a3b3243f5]",
        asset_ids=["ast-ba245e9a3b3243f5"],
    )
    assert "[IMAGE:" not in cleaned
    assert "(Figure 1)" in cleaned
    assert len(resolved) == 1
    assert resolved[0].exists is True



def test_queue_practice_question_switches_view_and_selects_picker() -> None:
    state: dict[str, object] = {
        "active_view": "My Progress",
        "selected_question_id": "jee-old-01",
        "practice_question_picker": "jee-old-01",
        "practice_evaluation_result": {"is_correct": True},
        "practice_hint_count": 2,
    }

    queue_practice_question(
        state,
        "jee-rec-99",
        switch_view=True,
        notice="Loaded recommended question `jee-rec-99`.",
        started_at=123.5,
    )
    sync_navigation_and_practice_state(state)

    assert state["active_view"] == "Practice Problems"
    assert state["selected_question_id"] == "jee-rec-99"
    assert state["practice_question_picker"] == "jee-rec-99"
    assert state["practice_loaded_notice"] == "Loaded recommended question `jee-rec-99`."
    assert state["practice_hint_count"] == 0
    assert state["practice_question_started_at"] == 123.5
    assert "practice_evaluation_result" not in state
    assert "pending_active_view" not in state
    assert "pending_practice_question_id" not in state


def test_sync_navigation_defaults_invalid_view_and_preserves_question() -> None:
    state: dict[str, object] = {"active_view": "Unknown View"}

    sync_navigation_and_practice_state(state)

    assert state["active_view"] == APP_VIEWS[0]


def test_clear_user_session_state_resets_user_and_practice_keys() -> None:
    state: dict[str, object] = {
        "authenticated": True,
        "user": {"username": "student"},
        "active_view": "Practice Problems",
        "selected_question_id": "jee-rec-99",
        "practice_question_picker": "jee-rec-99",
        "last_prediction_event": object(),
        "debug_info": {"status": "ok"},
    }

    clear_user_session_state(state)

    assert state["authenticated"] is False
    for key in (
        "user",
        "active_view",
        "selected_question_id",
        "practice_question_picker",
        "last_prediction_event",
        "debug_info",
    ):
        assert key not in state
