import re

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def make_source_id(sha256: str) -> str:
    """Create a stable source-document-version ID from its exact byte hash."""
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("sha256 must be a lowercase 64-character hex digest")
    return f"src-{sha256}"


def make_logical_source_id(identifier: str) -> str:
    """Normalize a human/source-system identifier used to group versions."""
    value = identifier.strip().lower()
    if not value:
        raise ValueError("logical source identifier cannot be empty")
    return value
