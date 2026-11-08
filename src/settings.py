"""Environment configuration validation for the tutor app.

Run ``python -m src.settings`` to check the environment before starting the app.
Problems are reported by variable name only; values are never printed.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping

PLACEHOLDER_VALUES = {"", "your_groq_api_key_here", "changeme"}


def validate_environment(env: Mapping[str, str] | None = None) -> list[str]:
    """Return blocking problems; empty means the app can start."""
    env = os.environ if env is None else env
    problems: list[str] = []

    if env.get("GROQ_API_KEY", "").strip().lower() in PLACEHOLDER_VALUES:
        problems.append("GROQ_API_KEY is not set (or still has the placeholder value)")

    port = env.get("CHROMA_PORT")
    if port is not None and port != "":
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            problems.append("CHROMA_PORT must be an integer between 1 and 65535")

    return problems


def optional_warnings(env: Mapping[str, str] | None = None) -> list[str]:
    """Return non-blocking notices about features that will be unavailable."""
    env = os.environ if env is None else env
    warnings: list[str] = []
    if env.get("OCR_SPACE_API_KEY", "").strip().lower() in PLACEHOLDER_VALUES | {"your_ocr_space_api_key_here"}:
        warnings.append("OCR_SPACE_API_KEY is not set: handwritten-answer transcription is disabled")
    return warnings


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # same behaviour as app.py; real env vars still win
    except ImportError:
        pass
    problems = validate_environment()
    for warning in optional_warnings():
        print(f"warning: {warning}", file=sys.stderr)
    if not problems:
        print("Configuration OK")
        return 0
    print("Invalid configuration:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print("See .env.example for the expected variables.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
