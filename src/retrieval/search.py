"""Semantic search over indexed NCERT learning units.

Examples:
    python -m src.retrieval.search "What is dimensional analysis?"
    python -m src.retrieval.search "Explain Newton's laws" --top-k 3
"""
from __future__ import annotations

import argparse
import json

from .common import connect_chroma, get_or_create_collection, load_config
from .embedder import encode_texts, load_embedder


def search(query: str, config_path: str = "configs/retrieval.yaml", top_k: int | None = None) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    config = load_config(config_path)
    client = connect_chroma(config)
    collection = get_or_create_collection(client, config)
    count = collection.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{collection.name}' is empty. Run "
            "'python -m src.retrieval.index_builder --limit 10' first."
        )

    model = load_embedder(config)
    vector = encode_texts(model, [query.strip()], config)[0].tolist()
    requested_k = top_k if top_k is not None else config.get("evaluation", {}).get("default_top_k", 5)
    k = max(1, min(int(requested_k), count))
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for record_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
        metadata = metadata or {}
        hits.append({
            "unit_id": record_id,
            "distance": distance,
            "unit_title": metadata.get("unit_title"),
            "chapter_title": metadata.get("chapter_title"),
            "subject": metadata.get("subject"),
            "chapter_id": metadata.get("chapter_id"),
            "source_id": metadata.get("source_id"),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "block_ids": json.loads(metadata.get("block_ids", "[]")),
            "asset_ids": json.loads(metadata.get("asset_ids", "[]")),
            "document": doc,
        })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="natural-language student query")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(search(args.query, args.config, args.top_k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
