from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.base import ContractModel


NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

class SourceType(str, Enum):
    NCERT = "ncert"
    JEE_MAIN = "jee_main"


class InventoryStatus(str, Enum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

class SourceRecord(ContractModel):
    """One registered source-document version; serialized to source_manifest.jsonl."""

    source_id: NonEmptyString
    logical_source_id: NonEmptyString | None
    source_type: SourceType
    original_filename: NonEmptyString
    sha256: Sha256Hex
    file_size_bytes: Annotated[int, Field(gt=0)]
    subject: str | None
    language: str | None
    publication_year: PositiveInt | None
    exam_year: PositiveInt | None
    edition: str | None
    class_level: PositiveInt | None
    session: str | None
    shift: str | None
    source_locator: NonEmptyString
    inventory_status: InventoryStatus