import re

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def make_source_id(sha256: str) -> str:
    """Create a stable source-document-version ID from its exact byte hash."""
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("sha256 must be a lowercase 64-character hex digest")
    return f"src-{sha256}"


def make_chapter_id(source_id: str, chapter_key: str) -> str:
    """Create a stable chapter ID from its source version and a chapter key.

    The key is the chapter number when known, otherwise a page-based key; it
    never depends on absolute paths or processing order.
    """
    key = re.sub(r"[^a-z0-9]+", "-", chapter_key.strip().lower()).strip("-")
    if not source_id or not key:
        raise ValueError("source_id and chapter_key must be non-empty")
    return f"{source_id}-ch-{key}"


def make_block_id(chapter_id: str, order_index: int) -> str:
    """Block ID, unique within a chapter and stable across deterministic reruns."""
    return f"{chapter_id}-b{order_index:05d}"


def make_logical_source_id(identifier: str) -> str:
    """Normalize a human/source-system identifier used to group versions."""
    value = identifier.strip().lower()
    if not value:
        raise ValueError("logical source identifier cannot be empty")
    return value
