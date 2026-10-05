from __future__ import annotations

import re
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    LIST = "list"
    OTHER = "other"

class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

class ContentBlock(ContractModel):
    """Ordered content block nested inside an NCERT chapter record."""

    block_id: NonEmptyString
    block_type: BlockType
    text_raw: str | None
    text_cleaned: str | None
    page: PositiveInt
    asset_ids: list[NonEmptyString]
    order_index: NonNegativeInt
    original_label: str | None


class NCERTChapterRecord(ContractModel):
    """One chapter from one source-document version/language."""

    chapter_id: NonEmptyString
    source_id: NonEmptyString
    subject: NonEmptyString
    class_level: PositiveInt | None
    language: NonEmptyString
    chapter_number: str | None
    chapter_title: str | None
    page_start: PositiveInt
    page_end: PositiveInt
    text_raw: str
    text_cleaned: str
    content_blocks: list[ContentBlock]
    asset_ids: list[NonEmptyString]
    extraction_status: ExtractionStatus

    @model_validator(mode="after")
    def validate_chapter_record(self) -> "NCERTChapterRecord":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")

        if not self.content_blocks and self.extraction_status != ExtractionStatus.FAILED:
            raise ValueError(
                "content_blocks may be empty only when extraction_status='failed'"
            )

        # Each block must lie within the chapter's PDF page range.
        outside = [
            block.block_id
            for block in self.content_blocks
            if not self.page_start <= block.page <= self.page_end
        ]
        if outside:
            raise ValueError(
                "Every content block page must fall within page_start/page_end; "
                f"out-of-range block IDs: {outside}"
            )

        # Local order checks. Uniqueness/contiguity is also checked at dataset level.
        indices = [block.order_index for block in self.content_blocks]
        if len(indices) != len(set(indices)):
            raise ValueError("content block order_index values must be unique within a chapter")

        block_ids = [block.block_id for block in self.content_blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_id values must be unique within a chapter")

        return self
