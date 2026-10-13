"""Shared configuration, JSONL, text formatting, and ChromaDB helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
import yaml


def load_config(path: str | Path = "configs/retrieval.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Retrieval config not found: {config_path.resolve()}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for section in ("embedding", "document", "vector_store"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config must contain a '{section}' mapping")
    return config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Learning-unit JSONL not found: {jsonl_path.resolve()}")
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_no}: expected a JSON object")
            records.append(record)
    return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_text(record: dict[str, Any], template: str) -> str:
    """Build exactly the same document text for indexing and query provenance."""
    values = {
        "subject": record.get("subject") or "",
        "class_level": record.get("class_level") if record.get("class_level") is not None else "",
        "chapter_title": record.get("chapter_title") or "",
        "unit_title": record.get("unit_title") or "",
        "text": record.get("text") or "",
    }
    try:
        return template.format(**values).strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid document.text_template {template!r}: {exc}") from exc


def connect_chroma(config: dict[str, Any]):
    store = config["vector_store"]
    client = chromadb.HttpClient(
        host=str(store.get("host", "localhost")),
        port=int(store.get("port", 8000)),
    )
    client.heartbeat()
    return client


def get_or_create_collection(client, config: dict[str, Any]):
    store = config["vector_store"]
    metric = str(store.get("distance_metric", "cosine")).lower()
    if metric not in {"cosine", "l2", "ip"}:
        raise ValueError("vector_store.distance_metric must be cosine, l2, or ip")
    name = str(store.get("collection_name", "ncert_learning_units_v1"))

    try:
        collection = client.get_collection(name=name)
    except Exception as exc:
        # Only create when the collection genuinely does not exist. Avoid hiding
        # unrelated server/client errors as a collection-not-found condition.
        try:
            names = {c.name for c in client.list_collections()}
        except Exception:
            raise exc
        if name in names:
            raise exc
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": metric},
        )

    actual_metric = (collection.metadata or {}).get("hnsw:space", "l2")
    if actual_metric != metric:
        raise ValueError(
            f"Collection '{name}' uses metric '{actual_metric}', but config requests '{metric}'. "
            "Choose a new collection name or rebuild it with --reset."
        )
    return collection


def chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalars; serialize list fields as JSON."""
    metadata: dict[str, Any] = {}
    scalar_fields = (
        "chapter_id", "source_id", "subject", "language", "chapter_number",
        "chapter_title", "unit_title", "unit_type", "unit_order",
        "page_start", "page_end", "source_extraction_status", "generation_status",
    )
    for field in scalar_fields:
        value = record.get(field)
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value
    for field in ("block_ids", "asset_ids", "review_flags"):
        value = record.get(field, [])
        metadata[field] = json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
    return metadata
