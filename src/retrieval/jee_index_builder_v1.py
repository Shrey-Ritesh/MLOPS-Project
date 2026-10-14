"""Build the v1 Chroma index for canonical JEE Main questions.

Run from repository root:
  python -m src.retrieval.jee_index_builder_v1 --limit 10
  python -m src.retrieval.jee_index_builder_v1
  python -m src.retrieval.jee_index_builder_v1 --reset
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import sentence_transformers
import re

from .common import connect_chroma, get_or_create_collection, load_config, read_jsonl, sha256_file
from .embedder import encode_texts, load_embedder


def question_document(record: dict[str, Any], template: str) -> str:
    options = record.get("options") or []
    rendered_options = []
    for option in options:
        if isinstance(option, dict):
            label = str(option.get("label") or "").strip()
            text = str(option.get("text") or "").strip()
            if label or text:
                rendered_options.append(f"{label}. {text}".strip(". "))
        elif isinstance(option, str) and option.strip():
            rendered_options.append(option.strip())

    q = str(record.get("question_text") or "").strip()
    question = re.sub(r"\[IMAGE:\s*[^]]+\]", "", q)
    fields = {
        # "subject": str(record.get("subject") or "").strip(),
        "question_text": question,
        # "options": " ".join(rendered_options),
    }
    try:
        return template.format(**fields).strip()
    except KeyError as exc:
        raise ValueError(f"Unsupported template placeholder {exc}; allowed: subject, year, question_text, options") from exc


def question_metadata(record: dict[str, Any]) -> dict[str, Any]:
    # Chroma metadata values must be scalar. Missing/unknown values stay absent.
    fields = (
        "source_id", "subject", "year", "session", "shift", "source_page",
        "difficulty", "difficulty_source", "difficulty_status", "extraction_status",
    )
    metadata: dict[str, Any] = {"record_type": "jee_question", "embedder_version": "v1"}
    for field in fields:
        value = record.get(field)
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value
    metadata["asset_ids"] = json.dumps(record.get("asset_ids") if isinstance(record.get("asset_ids"), list) else [], ensure_ascii=False)
    return metadata


def run(config_path: Path, reset: bool = False, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    doc_cfg = config["document"]
    input_path = Path(doc_cfg["input_path"])
    if not input_path.is_file():
        raise FileNotFoundError(f"JEE question JSONL not found: {input_path}")
    records = read_jsonl(input_path)
    records_read = len(records)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        records = records[:limit]

    id_field = doc_cfg.get("id_field", "question_id")
    template = doc_cfg.get("text_template", "{question_text}")
    skip_empty = bool(doc_cfg.get("skip_empty_text", True))
    prepared: list[tuple[dict[str, Any], str, str]] = []
    seen_ids: set[str] = set()
    skipped_empty = 0
    for line_no, record in enumerate(records, 1):
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Input record {line_no} is missing non-empty '{id_field}'")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate {id_field}: {record_id}")
        seen_ids.add(record_id)
        question_text = record.get("question_text")
        if not isinstance(question_text, str) or not question_text.strip():
            if skip_empty:
                skipped_empty += 1
                continue
            raise ValueError(f"Question {record_id!r} has empty question_text")
        rendered = question_document(record, template)
        if rendered:
            prepared.append((record, record_id, rendered))
        else:
            skipped_empty += 1
    if not prepared:
        raise ValueError("No non-empty JEE questions to index")

    client = connect_chroma(config)
    collection_name = config["vector_store"].get("collection_name", "jee_questions_v1")
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = get_or_create_collection(client, config)
    model = load_embedder(config)
    batch_size = int(config["vector_store"].get("upsert_batch_size", 64))
    if batch_size < 1:
        raise ValueError("vector_store.upsert_batch_size must be >= 1")

    indexed = 0
    expected_dim = None
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        texts = [item[2] for item in batch]
        for i, text in enumerate(texts):
            tokens = len(model.tokenizer.encode(text))
            if tokens > 256:
                print(f"\nQuestion ID: {batch[i][1]}")
                print(f"Token count: {tokens}")
                print(f"Text: {text}\n")
        vectors = encode_texts(model, texts, config)
        # vectors = encode_texts(model, [item[2] for item in batch], config)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values")
        dim = int(vectors.shape[1])
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            raise ValueError("Embedding dimensions changed within one build")
        collection.upsert(
            ids=[item[1] for item in batch],
            embeddings=vectors.tolist(),
            documents=[item[2] for item in batch],
            metadatas=[question_metadata(item[0]) for item in batch],
        )
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(prepared)} JEE questions")

    manifest = {
        "embedder_version": "v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "records_read_before_limit": records_read,
        "records_considered": len(records),
        "records_indexed": indexed,
        "records_skipped_empty": skipped_empty,
        "collection_name": collection.name,
        "distance_metric": config["vector_store"].get("distance_metric", "cosine"),
        "embedding_model": config["embedding"]["model_name"],
        "embedding_model_revision": config["embedding"].get("model_revision"),
        "embedding_dimension": expected_dim,
        "normalize_embeddings": bool(config["embedding"].get("normalize_embeddings", True)),
        "text_template": template,
        "python_version": platform.python_version(),
        "sentence_transformers_version": sentence_transformers.__version__,
        "chromadb_client_version": chromadb.__version__,
        "collection_count_after_build": collection.count(),
        "note": "Derived index; rebuild from JSONL/config. chapter_id is not required for embedding.",
    }
    manifest_path = Path("reports/retrieval/jee_index_build_manifest_v1.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jee_retrieval_v1.yaml"))
    parser.add_argument("--reset", action="store_true", help="recreate only the configured JEE collection")
    parser.add_argument("--limit", type=int, default=None, help="index first N records for a smoke test")
    args = parser.parse_args()
    run(args.config, args.reset, args.limit)


if __name__ == "__main__":
    main()
