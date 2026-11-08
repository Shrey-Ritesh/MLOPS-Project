"""
Benchmark semantic retrieval against the project's existing NCERT and JEE Chroma
collections. Run from the repository root:

    python -m src.retrieval.test_retrieval \
      --benchmark chemistry_retrieval_benchmark_v1.json

By default, this evaluates raw vector-search rankings (before retrieve.py's
heuristic reranking) and writes per-query hits plus a summary. It does not
modify collections or apply a distance threshold.

Expected project modules:
  src.retrieval.common: load_config, connect_chroma
  src.retrieval.embedder: load_embedder, encode_texts

Relevance annotations are optional. Ranking metrics are emitted only where
the benchmark contains annotations; unlabelled queries still produce retrieval
results for inspection and annotation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import connect_chroma, load_config
from .embedder import encode_texts, load_embedder


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError(f"Benchmark must be an object containing a 'queries' list: {path}")
    return data


def _query_text(item: dict[str, Any], mode: str = "question_and_response") -> str:
    """Build query text without including expected answer/explanation (avoids leakage)."""
    question = str(item.get("question") or "").strip()
    response = str(item.get("student_response") or "").strip()
    if mode == "question":
        text = question
    elif mode == "question_and_response":
        text = "\nStudent question: " + question + "\nStudent response: " + response
    else:
        raise ValueError(f"Unknown query mode: {mode}")
    if not text.strip():
        raise ValueError(f"{item.get('query_id', '<unknown>')} has an empty retrieval query")
    return text.strip()


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if key in {"block_ids", "asset_ids"} and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _get_collection(client: Any, config: dict[str, Any], corpus: str) -> Any:
    name = config.get("vector_store", {}).get("collection_name")
    if not name:
        raise ValueError(f"{corpus} config has no vector_store.collection_name")
    # Do not create collections during evaluation; a missing index should fail clearly.
    try:
        collection = client.get_collection(name=name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {corpus.upper()} collection {name!r}. "
            "Build/index that collection first and verify its config."
        ) from exc
    count = collection.count()
    if count < 1:
        raise RuntimeError(f"{corpus.upper()} collection {name!r} is empty.")
    return collection


def _retrieve_one(
    collection: Any,
    model: Any,
    query: str,
    config: dict[str, Any],
    corpus: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    vector = encode_texts(model, [query], config)[0].tolist()
    embed_and_query_started = time.perf_counter()
    count = collection.count()
    k = min(max(1, int(top_k)), count)
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[dict[str, Any]] = []
    for rank, (record_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        metadata = metadata or {}
        hit = {
            "rank": rank,
            "id": record_id,
            "distance": float(distance) if distance is not None else None,
            "corpus": corpus,
            "document": document or "",
        }
        # Preserve useful metadata without assuming both collections share a schema.
        for key in (
            "subject", "chapter_id", "chapter_title", "unit_title",
            "question_id", "source_id", "page_start", "page_end",
            "block_ids", "asset_ids",
        ):
            if key in metadata:
                hit[key] = _metadata_value(metadata, key)
        hits.append(hit)
    return hits, elapsed_ms


def _grade_for_id(record_id: str, item: dict[str, Any]) -> int:
    grades = item.get("relevance_grades") or {}
    value = grades.get(record_id, 0)
    try:
        grade = int(value)
    except (TypeError, ValueError):
        grade = 0
    return max(0, min(2, grade))


def _metric_values(hits: list[dict[str, Any]], item: dict[str, Any], k: int) -> dict[str, Any]:
    """
    Supports either:
      - relevant_unit_ids: list of all known relevant IDs (binary relevance)
      - relevance_grades: {id: 0|1|2} (graded relevance; grade > 0 is relevant)
    Recall is meaningful only if the annotated relevant-ID set is reasonably complete.
    """
    relevant_ids = {str(x) for x in (item.get("relevant_unit_ids") or [])}
    grades = item.get("relevance_grades") or {}
    has_labels = bool(relevant_ids) or bool(grades)
    if not has_labels:
        return {"label_status": "unlabelled"}

    # If graded labels exist, grade > 0 defines binary relevance. Include explicitly
    # listed relevant IDs even if they have not been assigned a graded value.
    relevant_ids |= {str(key) for key, value in grades.items() if _safe_int(value) > 0}
    top = hits[:k]
    ranked_relevant = [
        1 if str(hit["id"]) in relevant_ids or _grade_for_id(str(hit["id"]), item) > 0 else 0
        for hit in top
    ]
    first_rank = next((i for i, rel in enumerate(ranked_relevant, start=1) if rel), None)
    hit_rate = 1.0 if first_rank is not None else 0.0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    precision = sum(ranked_relevant) / len(top) if top else 0.0
    recall = sum(ranked_relevant) / len(relevant_ids) if relevant_ids else None

    # nDCG with graded gain 2^rel - 1 and log2(rank+1) discount.
    gains = []
    for hit in top:
        grade = _grade_for_id(str(hit["id"]), item)
        if str(hit["id"]) in relevant_ids and grade == 0:
            grade = 1
        gains.append((2**grade - 1) / math.log2(hit["rank"] + 1) if grade > 0 else 0.0)
    dcg = sum(gains)
    ideal_grades = sorted(
        [max(0, min(2, _safe_int(v))) for v in grades.values()],
        reverse=True,
    )
    # Include binary relevant IDs missing a grade as grade 1 for ideal ranking.
    graded_keys = {str(key) for key in grades}
    ideal_grades.extend([1] * len(relevant_ids - graded_keys))
    ideal_grades = sorted(ideal_grades, reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0
    )
    ndcg = dcg / idcg if idcg > 0 else None

    return {
        "label_status": "labelled",
        f"Hit@{k}": hit_rate,
        f"MRR@{k}": mrr,
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"nDCG@{k}": ndcg,
        "first_relevant_rank": first_rank,
        "known_relevant_count": len(relevant_ids),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        row[metric] for row in rows
        if isinstance(row.get(metric), (int, float)) and row[metric] is not None
    ]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="chemistry_retrieval_benchmark_v1.json",
        help="Path to the benchmark JSON file (default: ./chemistry_retrieval_benchmark_v1.json)",
    )
    parser.add_argument("--ncert-config", default="configs/retrieval.yaml")
    parser.add_argument("--jee-config", default="configs/jee_retrieval_v1.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--query-mode",
        choices=["question_and_response", "question"],
        default="question_and_response",
        help="Whether retrieval uses the question plus student response or only the question.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/retrieval_benchmark",
        help="Directory for raw per-query results and summary JSON.",
    )
    parser.add_argument(
        "--include-corpora",
        nargs="+",
        choices=["ncert", "jee"],
        default=["ncert", "jee"],
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {benchmark_path.resolve()}. Pass --benchmark with its path."
        )
    benchmark = _read_json(benchmark_path)
    ncert_cfg = load_config(args.ncert_config)
    jee_cfg = load_config(args.jee_config)
    configs = {"ncert": ncert_cfg, "jee": jee_cfg}

    # Query vectors are shared across the two corpora. Enforce matching model settings
    # so one vector is valid for both collections.
    embedding_fields = ("model_name", "model_revision", "normalize_embeddings")
    mismatches = {
        field: (ncert_cfg.get("embedding", {}).get(field), jee_cfg.get("embedding", {}).get(field))
        for field in embedding_fields
        if ncert_cfg.get("embedding", {}).get(field) != jee_cfg.get("embedding", {}).get(field)
    }
    if mismatches:
        raise ValueError(
            "NCERT/JEE embedding configs differ. A single query vector cannot safely be "
            f"used for both collections. Differences: {mismatches}"
        )

    client = connect_chroma(ncert_cfg)
    collections = {
        corpus: _get_collection(client, configs[corpus], corpus)
        for corpus in args.include_corpora
    }
    model = load_embedder(ncert_cfg)

    results: list[dict[str, Any]] = []
    for item in benchmark["queries"]:
        query_id = str(item.get("query_id") or f"query_{len(results)+1}")
        query_text = _query_text(item, args.query_mode)
        row: dict[str, Any] = {
            "query_id": query_id,
            "chapter": item.get("chapter"),
            "query_text": query_text,
            "student_response": item.get("student_response"),
            "target_concepts": item.get("target_concepts", []),
            "corpora": {},
        }
        for corpus, collection in collections.items():
            hits, latency_ms = _retrieve_one(
                collection, model, query_text, configs[corpus], corpus, args.top_k
            )
            # IDs are scoped to a corpus; retain separate annotations per corpus when
            # supplied, otherwise use generic annotations for backwards compatibility.
            annotation = item.get("annotations", {}).get(corpus, item)
            metrics = _metric_values(hits, annotation, min(5, args.top_k))
            row["corpora"][corpus] = {
                "collection_name": configs[corpus]["vector_store"]["collection_name"],
                "latency_ms": round(latency_ms, 3),
                "metrics": metrics,
                "hits": hits,
            }
        results.append(row)
        print(f"Retrieved {query_id}: " + ", ".join(
            f"{corpus}={len(row['corpora'][corpus]['hits'])} hits"
            for corpus in row["corpora"]
        ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_dir / f"retrieval_results_{stamp}.json"
    summary_path = output_dir / f"retrieval_summary_{stamp}.json"

    # Per-corpus summary includes only labelled queries for relevance metrics.
    summary: dict[str, Any] = {
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_version": benchmark.get("version"),
        "benchmark_path": str(benchmark_path),
        "run_timestamp_utc": stamp,
        "query_mode": args.query_mode,
        "top_k": args.top_k,
        "embedding": ncert_cfg.get("embedding", {}),
        "collections": {},
        "note": (
            "Raw Chroma vector-search ranking; no heuristic reranking and no distance cutoff. "
            "Relevance metrics are omitted for unlabelled queries."
        ),
    }
    for corpus in collections:
        corpus_rows = [r["corpora"][corpus] for r in results]
        labelled_rows = [
            r for r in corpus_rows
            if r["metrics"].get("label_status") == "labelled"
        ]
        latencies = [float(r["latency_ms"]) for r in corpus_rows]
        summary["collections"][corpus] = {
            "collection_name": configs[corpus]["vector_store"]["collection_name"],
            "count": collections[corpus].count(),
            "query_count": len(corpus_rows),
            "labelled_query_count": len(labelled_rows),
            "mean_metrics_on_labelled_queries": {
                metric: _mean_metric(
                    [r["metrics"] for r in labelled_rows], metric
                )
                for metric in ("Hit@5", "MRR@5", "Precision@5", "Recall@5", "nDCG@5")
            },
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 3) if latencies else None,
                "p50": round(_percentile(latencies, 0.50), 3) if latencies else None,
                "p95": round(_percentile(latencies, 0.95), 3) if latencies else None,
            },
        }

    raw_path.write_text(json.dumps({
        "benchmark_id": benchmark.get("benchmark_id"),
        "run_timestamp_utc": stamp,
        "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRaw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    print("No collection data was modified.")


if __name__ == "__main__":
    main()
