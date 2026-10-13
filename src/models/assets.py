from __future__ import annotations

import math
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from src.models.base import ContractModel


NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AssetType(str, Enum):
    IMAGE = "image"
    DIAGRAM = "diagram"
    PAGE_CROP = "page_crop"
    OTHER = "other"

class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"

class BoundingBox(ContractModel):
    """Source-page coordinates in PyMuPDF page points, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_box_order(self) -> "BoundingBox":
        if not all(math.isfinite(value) for value in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("bbox coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bbox must have positive width and height")
        return self


class AssetRecord(ContractModel):
    """One extracted binary asset; serialized to assets_manifest.jsonl."""

    asset_id: NonEmptyString
    source_id: NonEmptyString
    sha256: Sha256Hex
    asset_type: AssetType
    file_path: NonEmptyString
    source_page: PositiveInt
    bbox: BoundingBox | None
    associated_record_ids: list[NonEmptyString]
    review_status: ReviewStatus

    @field_validator("file_path")
    @classmethod
    def validate_relative_file_path(cls, value: str) -> str:
        # Contract paths use forward slashes and are relative to the documented data root.
        if "\\" in value:
            raise ValueError("file_path must use forward slashes, not backslashes")
        path = PurePosixPath(value)
        if path.is_absolute() or value.startswith("/"):
            raise ValueError("file_path must be relative, not an absolute path")
        if any(part == ".." for part in path.parts):
            raise ValueError("file_path must not traverse outside the data root ('..')")
        if value in {"", "."}:
            raise ValueError("file_path must identify a file")
        return value
