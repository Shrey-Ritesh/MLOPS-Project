"""Build a ChromaDB index from final NCERT learning units.

Run from repository root:
    python -m src.retrieval.index_builder --limit 10
    python -m src.retrieval.index_builder
    python -m src.retrieval.index_builder --reset
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import sentence_transformers

from .common import (
    chroma_metadata,
    connect_chroma,
    document_text,
    get_or_create_collection,
    load_config,
    read_jsonl,
    sha256_file,
)
from .embedder import encode_texts, load_embedder


def run(config_path: str | Path, *, reset: bool = False, limit: int | None = None) -> dict:
    config = load_config(config_path)
    input_path = Path(config["document"].get("input_path", "data/processed/ncert_learning_units.jsonl"))
    records = read_jsonl(input_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        records = records[:limit]
    if not records:
        raise ValueError(f"No learning units found in {input_path}")

    id_field = str(config["document"].get("id_field", "unit_id"))
    template = str(config["document"].get(
        "text_template",
        "Subject: {subject}\nChapter: {chapter_title}\nTopic: {unit_title}\n\n{text}",
    ))
    skip_empty = bool(config["document"].get("skip_empty_text", True))

    prepared: list[tuple[dict, str, str]] = []
    seen_ids: set[str] = set()
    skipped_empty = 0
    for line_no, record in enumerate(records, start=1):
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Record {line_no} is missing a non-empty '{id_field}'")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate {id_field}: {record_id}")
        seen_ids.add(record_id)
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            if skip_empty:
                skipped_empty += 1
                continue
            raise ValueError(f"Learning unit '{record_id}' has empty text")
        formatted = document_text(record, template)
        if not formatted:
            skipped_empty += 1
            continue
        prepared.append((record, record_id, formatted))

    if not prepared:
        raise ValueError("No non-empty learning units remain after validation")

    print(f"Read {len(records)} records; prepared {len(prepared)}; skipped empty {skipped_empty}.", flush=True)
    client = connect_chroma(config)
    collection_name = str(config["vector_store"].get("collection_name", "ncert_learning_units_v1"))
    if reset:
        print(f"Reset requested: deleting collection '{collection_name}' if it exists.", flush=True)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            # Safe for a missing collection; get_or_create_collection below verifies state.
            pass

    collection = get_or_create_collection(client, config)
    model = load_embedder(config)
    batch_size = int(config["vector_store"].get("upsert_batch_size", 32))
    if batch_size < 1:
        raise ValueError("vector_store.upsert_batch_size must be at least 1")

    indexed = 0
    embedding_dimension = None
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        vectors = encode_texts(model, [item[2] for item in batch], config, identifiers=[item[1] for item in batch])
        dimension = int(vectors.shape[1])
        if embedding_dimension is None:
            embedding_dimension = dimension
        elif dimension != embedding_dimension:
            raise ValueError("Embedding dimension changed within the same index build")

        collection.upsert(
            ids=[item[1] for item in batch],
            embeddings=vectors.tolist(),
            documents=[item[2] for item in batch],
            metadatas=[chroma_metadata(item[0]) for item in batch],
        )
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(prepared)} learning units.", flush=True)

    count_after = collection.count()
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "records_read": len(records),
        "records_indexed_this_run": indexed,
        "records_skipped_empty": skipped_empty,
        "collection_name": collection.name,
        "collection_count_after_build": count_after,
        "distance_metric": config["vector_store"].get("distance_metric", "cosine"),
        "embedding_model": config["embedding"]["model_name"],
        "embedding_model_revision": config["embedding"].get("model_revision"),
        "embedding_dimension": embedding_dimension,
        "normalize_embeddings": bool(config["embedding"].get("normalize_embeddings", True)),
        "text_template": template,
        "python_version": platform.python_version(),
        "sentence_transformers_version": sentence_transformers.__version__,
        "chromadb_client_version": chromadb.__version__,
    }
    manifest_path = Path(config.get("outputs", {}).get(
        "manifest_path", "reports/retrieval/index_build_manifest.json"
    ))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIndex build complete. Collection count: {count_after}", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--reset", action="store_true", help="delete and recreate the configured collection")
    parser.add_argument("--limit", type=int, default=None, help="index only the first N records")
    args = parser.parse_args()
    run(args.config, reset=args.reset, limit=args.limit)


if __name__ == "__main__":
    main()
