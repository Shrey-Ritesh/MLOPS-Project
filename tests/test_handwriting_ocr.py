"""Unit tests for Person 4 handwriting OCR adapter (src/handwriting/ocr_space.py).

Verifies:
- Image validation (PNG, JPEG, size limits, dimensions, invalid bytes)
- Error handling (empty images, oversized images, unsupported formats)
- OCR.space recognizer initialization and API response handling (mocked)
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
import requests

from src.handwriting.ocr_space import (
    ImageValidationError,
    MAX_IMAGE_BYTES,
    OCRSpaceError,
    OCRSpaceRecognizer,
    validate_image,
)


def _create_test_image(format_name: str = "PNG", size: tuple[int, int] = (100, 100)) -> bytes:
    """Helper to create small valid image bytes in memory."""
    buf = BytesIO()
    img = Image.new("RGB", size, color="white")
    img.save(buf, format=format_name)
    return buf.getvalue()


# --- 1. Image Validation Tests ---

def test_validate_image_valid_png():
    img_bytes = _create_test_image("PNG")
    fmt, mime = validate_image(img_bytes)
    assert fmt == "PNG"
    assert mime == "image/png"


def test_validate_image_valid_jpeg():
    img_bytes = _create_test_image("JPEG")
    fmt, mime = validate_image(img_bytes)
    assert fmt == "JPEG"
    assert mime == "image/jpeg"


def test_validate_image_empty_bytes():
    with pytest.raises(ImageValidationError) as exc:
        validate_image(b"")
    assert "empty" in str(exc.value).lower()


def test_validate_image_oversized():
    fake_large_bytes = b"0" * (MAX_IMAGE_BYTES + 10)
    with pytest.raises(ImageValidationError) as exc:
        validate_image(fake_large_bytes)
    assert "too large" in str(exc.value).lower()


def test_validate_image_corrupted_data():
    with pytest.raises(ImageValidationError) as exc:
        validate_image(b"not an image binary payload")
    assert "not a valid supported image" in str(exc.value).lower()


def test_validate_image_unsupported_format():
    buf = BytesIO()
    img = Image.new("RGB", (50, 50), color="blue")
    img.save(buf, format="GIF")
    with pytest.raises(ImageValidationError) as exc:
        validate_image(buf.getvalue())
    assert "use a png or jpeg" in str(exc.value).lower()


# --- 2. Recognizer Tests ---

def test_recognizer_requires_api_key():
    with pytest.raises(OCRSpaceError) as exc:
        OCRSpaceRecognizer(api_key="   ")
    assert "ocr_space_api_key" in str(exc.value).lower()


def test_recognizer_successful_transcription():
    img_bytes = _create_test_image("PNG")
    recognizer = OCRSpaceRecognizer(api_key="test-api-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "OCRExitCode": 1,
        "ParsedResults": [
            {
                "FileParseExitCode": "1",
                "ParsedText": "2. But-2-ene\r\n",
            }
        ],
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = recognizer.transcribe(img_bytes)
        assert result.text == "2. But-2-ene"
        assert result.exit_code == 1
        assert "OCR.space" in result.recognizer
        mock_post.assert_called_once()


def test_recognizer_timeout_error():
    img_bytes = _create_test_image("PNG")
    recognizer = OCRSpaceRecognizer(api_key="test-api-key")

    with patch("requests.post", side_effect=requests.Timeout("Connection timed out")):
        with pytest.raises(OCRSpaceError) as exc:
            recognizer.transcribe(img_bytes)
        assert "timed out" in str(exc.value).lower()


def test_recognizer_service_error():
    img_bytes = _create_test_image("PNG")
    recognizer = OCRSpaceRecognizer(api_key="test-api-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"ErrorMessage": ["Service unavailable"]}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OCRSpaceError) as exc:
            recognizer.transcribe(img_bytes)
        assert "503" in str(exc.value) or "failed" in str(exc.value).lower()


def test_recognizer_no_readable_text_found():
    img_bytes = _create_test_image("PNG")
    recognizer = OCRSpaceRecognizer(api_key="test-api-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "OCRExitCode": 2,
        "ParsedResults": [{"FileParseExitCode": "2", "ParsedText": ""}],
        "ErrorMessage": ["No text detected on image"],
    }

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OCRSpaceError) as exc:
            recognizer.transcribe(img_bytes)
        assert "no readable text" in str(exc.value).lower() or "no text detected" in str(exc.value).lower()

