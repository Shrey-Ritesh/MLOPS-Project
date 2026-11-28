"""Resolve question image assets from the processed asset manifest and crop directory."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_ASSETS_MANIFEST_PATH = Path("data/processed/assets_manifest.jsonl")
DEFAULT_ASSETS_DIR = Path("data/assets/crops")
IMAGE_TAG_PATTERN = re.compile(r"\[IMAGE:\s*([A-Za-z0-9_-]+)\]")


@dataclass(frozen=True)
class AssetRecord:
    """One asset entry loaded from assets_manifest.jsonl."""

    asset_id: str
    file_path: str
    asset_type: str = "diagram"
    source_page: int | None = None


@dataclass(frozen=True)
class ResolvedAsset:
    """A resolved figure reference ready for UI rendering."""

    asset_id: str
    figure_number: int
    file_path: Path
    exists: bool
    asset_type: str = "diagram"
    source_page: int | None = None


class AssetRepository:
    """Lazy, read-only index mapping question asset_ids to local image files."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        assets_dir: str | Path | None = None,
    ) -> None:
        env_manifest = os.environ.get("ASSETS_MANIFEST_PATH", "").strip()
        env_dir = os.environ.get("ASSETS_DIR", "").strip()
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else (Path(env_manifest) if env_manifest else DEFAULT_ASSETS_MANIFEST_PATH)
        )
        self.assets_dir = (
            Path(assets_dir)
            if assets_dir is not None
            else (Path(env_dir) if env_dir else DEFAULT_ASSETS_DIR)
        )
        self._assets_by_id: dict[str, AssetRecord] = {}
        self._loaded = False

    def load(self) -> None:
        """Load asset metadata from assets_manifest.jsonl if present."""
        self._assets_by_id.clear()
        if not self.manifest_path.is_file():
            self._loaded = True
            return

        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    item = json.loads(raw_line)
                    asset_id = str(item.get("asset_id") or "").strip()
                    file_path = str(item.get("file_path") or "").strip()
                    if not asset_id or not file_path:
                        continue
                    self._assets_by_id[asset_id] = AssetRecord(
                        asset_id=asset_id,
                        file_path=file_path,
                        asset_type=str(item.get("asset_type") or "diagram"),
                        source_page=item.get("source_page"),
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid asset manifest line %s:%d: %s",
                        self.manifest_path,
                        line_no,
                        exc,
                    )
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        """Return manifest metadata for a given asset_id, if indexed."""
        self._ensure_loaded()
        return self._assets_by_id.get(asset_id.strip())

    def resolve_asset_path(self, asset_id: str) -> tuple[Path, AssetRecord | None]:
        """Resolve an asset_id to a local image path, checking manifest then crops dir."""
        clean_id = asset_id.strip()
        record = self.get_asset(clean_id)
        if record is not None:
            candidate = Path(record.file_path)
            if candidate.is_file():
                return candidate, record
            fallback = self.assets_dir / candidate.name
            if fallback.is_file():
                return fallback, record
            return candidate, record

        fallback = self.assets_dir / f"{clean_id}.png"
        return fallback, None

    def resolve_for_question(
        self,
        question_text: str,
        asset_ids: Sequence[str] = (),
    ) -> tuple[str, list[ResolvedAsset]]:
        """Format inline [IMAGE: ...] tags and resolve ordered figure assets.

        Returns ``(cleaned_question_text, resolved_assets)`` where inline
        ``[IMAGE: <id>]`` markers are replaced with ``(Figure N)`` matching the
        rendered figures below the stem.
        """
        ordered_ids: list[str] = []
        seen: set[str] = set()

        def _add_id(raw_id: str) -> None:
            clean = raw_id.strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered_ids.append(clean)

        for aid in asset_ids:
            _add_id(aid)
        for match in IMAGE_TAG_PATTERN.finditer(question_text):
            _add_id(match.group(1))

        fig_by_id = {aid: idx + 1 for idx, aid in enumerate(ordered_ids)}

        def _replace_tag(match: re.Match[str]) -> str:
            fig_num = fig_by_id.get(match.group(1).strip())
            return f"(Figure {fig_num})" if fig_num else ""

        cleaned_text = IMAGE_TAG_PATTERN.sub(_replace_tag, question_text)
        cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text).strip()

        resolved: list[ResolvedAsset] = []
        for idx, aid in enumerate(ordered_ids, start=1):
            path, record = self.resolve_asset_path(aid)
            resolved.append(
                ResolvedAsset(
                    asset_id=aid,
                    figure_number=idx,
                    file_path=path,
                    exists=path.is_file(),
                    asset_type=record.asset_type if record else "diagram",
                    source_page=record.source_page if record else None,
                )
            )
        return cleaned_text, resolved
