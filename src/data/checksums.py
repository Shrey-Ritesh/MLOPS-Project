from pathlib import Path
import hashlib

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: str | Path) -> int:
    """Return file size in bytes."""
    return Path(path).stat().st_size
