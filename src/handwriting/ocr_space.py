"""Validated image upload and OCR.space handwriting adapter."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import warnings
from typing import Any

from PIL import Image, UnidentifiedImageError
import requests


MAX_IMAGE_BYTES = 1_000_000
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}
OCR_SPACE_ENDPOINT = "https://api.ocr.space/parse/image"


class ImageValidationError(ValueError):
    """Raised when an uploaded answer image is invalid or unsupported."""


class OCRSpaceError(RuntimeError):
    """Raised when OCR.space cannot return a usable transcription."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    recognizer: str = "OCR.space Engine 3"
    exit_code: int | None = None


def validate_image(image_bytes: bytes) -> tuple[str, str]:
    """Validate image content, size, and dimensions; return format and MIME."""
    if not image_bytes:
        raise ImageValidationError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image is too large. The free API limit is 1 MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = (image.format or "").upper()
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise ImageValidationError("This file is not a valid supported image.") from exc

    if image_format not in SUPPORTED_FORMATS:
        raise ImageValidationError("Use a PNG or JPEG image.")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("Image dimensions are invalid or exceed the 25 megapixel limit.")
    return image_format, SUPPORTED_FORMATS[image_format]


class OCRSpaceRecognizer:
    """Recognize handwritten text using OCR.space Engine 3."""

    def __init__(self, api_key: str, timeout_seconds: float = 45.0) -> None:
        if not api_key.strip():
            raise OCRSpaceError("Set OCR_SPACE_API_KEY before using OCR.")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds

    def transcribe(
        self,
        image_bytes: bytes,
        *,
        mime_type: str | None = None,
        image_format: str | None = None,
    ) -> TranscriptionResult:
        detected_format, detected_mime = validate_image(image_bytes)
        image_format = (image_format or detected_format).upper()
        mime_type = mime_type or detected_mime
        if image_format != detected_format or mime_type != detected_mime:
            raise ImageValidationError("Image format metadata does not match the uploaded content.")

        extension = ".png" if image_format == "PNG" else ".jpg"
        safe_filename = f"answer{extension}"
        try:
            response = requests.post(
                OCR_SPACE_ENDPOINT,
                files={"file": (safe_filename, BytesIO(image_bytes), mime_type)},
                data={
                    "language": "auto",
                    "isOverlayRequired": "false",
                    "OCREngine": "3",
                    "scale": "true",
                },
                headers={"apikey": self._api_key},
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise OCRSpaceError("The OCR request timed out. Retry or enter the answer manually.") from exc
        except requests.RequestException as exc:
            raise OCRSpaceError("Could not reach the OCR.space service.") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise OCRSpaceError("OCR.space returned an unreadable response.") from exc

        if response.status_code >= 400:
            raise OCRSpaceError(f"OCR.space request failed (HTTP {response.status_code}).")
        if not isinstance(payload, dict):
            raise OCRSpaceError("OCR.space returned an unexpected response format.")

        exit_code = _optional_int(payload.get("OCRExitCode"))
        parsed_results = payload.get("ParsedResults") or []
        texts = [
            str(result.get("ParsedText") or "").strip()
            for result in parsed_results
            if isinstance(result, dict) and str(result.get("FileParseExitCode")) == "1"
        ]
        text = "\n".join(part for part in texts if part)
        if not text:
            detail = payload.get("ErrorMessage") or payload.get("ErrorDetails")
            message = str(detail).strip() if detail else "No readable text was found."
            raise OCRSpaceError(f"{message} Retry or use typed entry.")

        return TranscriptionResult(text=text, exit_code=exit_code)


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
