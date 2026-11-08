"""Run raw top-K retrieval against the ChEmbed NCERT collection (for v1, JEE is not retrieved).

Run from the repository root:
    python -m src.retrieval.benchmark_chembed \
        --benchmark chemistry_retrieval_benchmark_v1.json

This is deliberately separate from retrieve.py (production heuristic reranking)
and from test_retrieval.py (the original MiniLM two-corpus benchmark).
It performs raw Chroma vector search only. It does not modify any collection
or calculate relevance metrics before ChEmbed-only hits are annotated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import connect_chroma, load_config
from .embedder import encode_texts, load_embedder


ROOT = Path(__file__).resolve().parents[2]


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_benchmark(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError("Benchmark JSON must contain a 'queries' list.")
    if not data["queries"]:
        raise ValueError("Benchmark contains no queries.")
    return data


def make_query(item: dict[str, Any]) -> str:
    """Use exactly the original benchmark protocol: question + student response."""
    question = str(item.get("question") or "").strip()
    response = str(item.get("student_response") or "").strip()
    if not question:
        raise ValueError(f"{item.get('query_id', '<unknown>')} has no question.")
    if not response:
        raise ValueError(
            f"{item.get('query_id', '<unknown>')} has no student_response."
        )
    return (
        f"Student question: {question}\n"
        f"Student response: {response}"
    ).strip()


def get_existing_collection(client: Any, name: str) -> Any:
    """Open only; never create or reset a collection during benchmarking."""
    try:
        collection = client.get_collection(name=name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open Chroma collection {name!r}. "
            "Confirm the index exists and Chroma is running."
        ) from exc
    count = collection.count()
    if count < 1:
        raise RuntimeError(f"Collection {name!r} is empty.")
    return collection


def retrieve_one(
    collection: Any,
    model: Any,
    query_text: str,
    config: dict[str, Any],
    top_k: int,
    query_task: str,
) -> tuple[list[dict[str, Any]], float, float]:
    """Return hits, embedding latency and Chroma query latency separately."""
    encode_start = time.perf_counter()
    vector = encode_texts(
        model,
        [query_text],
        config,
        task=query_task,
    )[0].tolist()
    embedding_ms = (time.perf_counter() - encode_start) * 1000.0

    k = min(top_k, collection.count())
    query_start = time.perf_counter()
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chroma_ms = (time.perf_counter() - query_start) * 1000.0

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for rank, (record_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}
        hit = {
            "rank": rank,
            "id": record_id,
            "distance": float(distance) if distance is not None else None,
            "document": document or "",
            "metadata": metadata,
        }
        hits.append(hit)

    return hits, embedding_ms, chroma_ms


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 10000):
        candidate = directory / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique output path in {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="chemistry_retrieval_benchmark_v1.json",
    )
    parser.add_argument(
        "--config",
        default="configs/retrieval_chembed.yaml",
        help="ChEmbed config; must point to chembed_index.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="artifacts/retrieval_benchmark/chembed",
    )
    parser.add_argument(
        "--query-task",
        choices=["query", "document"],
        default="query",
        help="Embedding task/prefix for query texts. Verify against model documentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional query limit for smoke testing; omit to run all queries.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    benchmark_path = root_path(args.benchmark)
    config_path = root_path(args.config)
    benchmark = read_benchmark(benchmark_path)
    config = load_config(config_path)

    collection_name = config.get("vector_store", {}).get("collection_name")
    if not collection_name:
        raise ValueError("ChEmbed config has no vector_store.collection_name.")
    if collection_name != "chembed_index":
        raise ValueError(
            f"Expected collection 'chembed_index', got {collection_name!r}. "
            "Refusing to benchmark the wrong collection."
        )

    # This config must be the exact config used to build the ChEmbed collection.
    client = connect_chroma(config)
    collection = get_existing_collection(client, collection_name)
    model = load_embedder(config)

    queries = benchmark["queries"]
    if args.limit is not None:
        queries = queries[:args.limit]

    results: list[dict[str, Any]] = []
    total_latencies: list[float] = []
    embedding_latencies: list[float] = []
    chroma_latencies: list[float] = []

    for index, item in enumerate(queries, start=1):
        query_id = str(item.get("query_id") or f"Q{index:02d}")
        query_text = make_query(item)

        hits, embedding_ms, chroma_ms = retrieve_one(
            collection=collection,
            model=model,
            query_text=query_text,
            config=config,
            top_k=args.top_k,
            query_task=args.query_task,
        )
        total_ms = embedding_ms + chroma_ms
        embedding_latencies.append(embedding_ms)
        chroma_latencies.append(chroma_ms)
        total_latencies.append(total_ms)

        results.append(
            {
                "query_id": query_id,
                "chapter": item.get("chapter"),
                "query_text": query_text,
                "student_response": item.get("student_response"),
                "target_concepts": item.get("target_concepts", []),
                "corpus": "ncert",
                "collection_name": collection_name,
                "latency_ms": {
                    "embedding": round(embedding_ms, 3),
                    "chroma_query": round(chroma_ms, 3),
                    "total": round(total_ms, 3),
                },
                "annotation_status": "unannotated_candidate_hits",
                "hits": hits,
            }
        )
        print(
            f"[{index}/{len(queries)}] {query_id}: "
            f"{len(hits)} hits | total={total_ms:.2f} ms "
            f"(embed={embedding_ms:.2f}, chroma={chroma_ms:.2f})"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root_path(args.output_dir)

    raw_path = unique_path(
        output_dir,
        f"chembed_retrieval_results_{timestamp}.json",
    )
    summary_path = unique_path(
        output_dir,
        f"chembed_retrieval_summary_{timestamp}.json",
    )

    embedding_settings = config.get("embedding", {})
    summary = {
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_version": benchmark.get("version"),
        "benchmark_path": str(benchmark_path),
        "run_timestamp_utc": timestamp,
        "model": {
            "provider": embedding_settings.get("provider"),
            "model_name": embedding_settings.get("model_name"),
            "model_revision": embedding_settings.get("model_revision"),
            "dimension": embedding_settings.get("embedding_dimension"),
            "max_seq_length": embedding_settings.get("max_seq_length"),
            "pooling": embedding_settings.get("pooling"),
            "normalize_embeddings": embedding_settings.get(
                "normalize_embeddings", True
            ),
            "document_prefix": embedding_settings.get("document_prefix", ""),
            "query_prefix": embedding_settings.get("query_prefix", ""),
            "query_task": args.query_task,
        },
        "collection": {
            "name": collection_name,
            "count": collection.count(),
            "distance_metric": config.get("vector_store", {}).get(
                "distance_metric", "cosine"
            ),
        },
        "query_protocol": {
            "composition": "Student question + Student response",
            "query_count": len(results),
            "top_k": args.top_k,
            "reranking": False,
            "distance_cutoff": None,
        },
        "latency_ms": {
            "total_p50": percentile(total_latencies, 0.50),
            "total_p95": percentile(total_latencies, 0.95),
            "embedding_p50": percentile(embedding_latencies, 0.50),
            "embedding_p95": percentile(embedding_latencies, 0.95),
            "chroma_query_p50": percentile(chroma_latencies, 0.50),
            "chroma_query_p95": percentile(chroma_latencies, 0.95),
            "total_mean": statistics.mean(total_latencies) if total_latencies else None,
        },
        "evaluation_note": (
            "Raw vector retrieval only. No relevance metrics are calculated. "
            "Annotate newly retrieved ChEmbed chunk IDs in the pooled annotation "
            "dataset before computing ranking metrics."
        ),
    }

    raw_path.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark.get("benchmark_id"),
                "run_timestamp_utc": timestamp,
                "model": summary["model"],
                "collection": summary["collection"],
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nRaw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    print("No collection data was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
