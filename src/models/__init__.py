
# src/models/__init__.py

from src.models.source import SourceRecord
from src.models.ncert import NCERTChapterRecord, ContentBlock
from src.models.assets import AssetRecord, BoundingBox
from src.models.jee import JEEQuestionRecord, JEEOption

RECORD_MODELS = {
    "source": SourceRecord,
    "ncert_chapter": NCERTChapterRecord,
    "asset": AssetRecord,
    "jee_question": JEEQuestionRecord,
}

__all__ = [
    "SourceRecord",
    "NCERTChapterRecord",
    "ContentBlock",
    "AssetRecord",
    "BoundingBox",
    "JEEQuestionRecord",
    "JEEOption",
    "RECORD_MODELS",
]
