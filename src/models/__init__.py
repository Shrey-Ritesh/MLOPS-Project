
# src/models/__init__.py

from src.models.source import SourceRecord, SourceType, InventoryStatus
from src.models.ncert import NCERTChapterRecord, ContentBlock, BlockType
from src.models.assets import AssetRecord, BoundingBox, AssetType
from src.models.jee import (
    JEEQuestionRecord,
    JEEOption,
    Difficulty,
    DifficultySource,
    DifficultyStatus,
    ExtractionStatus,
    ReviewStatus,
)

RECORD_MODELS = {
    "source": SourceRecord,
    "ncert_chapter": NCERTChapterRecord,
    "asset": AssetRecord,
    "jee_question": JEEQuestionRecord,
}

__all__ = [
    "SourceRecord",
    "SourceType",
    "InventoryStatus",
    "NCERTChapterRecord",
    "ContentBlock",
    "BlockType",
    "AssetRecord",
    "BoundingBox",
    "AssetType",
    "JEEQuestionRecord",
    "JEEOption",
    "Difficulty",
    "DifficultySource",
    "DifficultyStatus",
    "ExtractionStatus",
    "ReviewStatus",
    "RECORD_MODELS",
]
