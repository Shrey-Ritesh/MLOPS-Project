"""Handwriting input adapters for the JEE Tutor app."""

from src.handwriting.ocr_space import (
    ImageValidationError,
    OCRSpaceError,
    OCRSpaceRecognizer,
    TranscriptionResult,
    validate_image,
)

__all__ = [
    "ImageValidationError",
    "OCRSpaceError",
    "OCRSpaceRecognizer",
    "TranscriptionResult",
    "validate_image",
]
