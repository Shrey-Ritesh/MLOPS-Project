#!/usr/bin/env python3
"""
Evaluate an already-computed ChEmbed retrieval run.

This script deliberately does NOT load MiniLM, ChromaDB, or any embedding model.
It scores the ranked hits stored in benchmark_chembed.py's output JSON.

Expected input:
  {
    "benchmark_id": "...",
    "results": [
      {
        "query_id": "Q01",
        "corpus": "ncert",
        "hits": [
          {"rank": 1, "id": "...", "relevance_score": 2, ...}
        ]
      }
    ]
  }

Every hit being evaluated must have an explicit relevance_score in {0, 1, 2}.
Unknown labels are never silently treated as irrelevant.

Important:
- Metrics use the same Hit, MRR, Precision, Recall and graded nDCG definitions
  as src/retrieval/test_retrieval.py.
- Recall is pool-based: its denominator is the number of positively labelled
  records in THIS run's retrieved/annotated pool. It is not exhaustive corpus
  recall. For a fair MiniLM-vs-ChEmbed comparison, annotate the UNION of both
  systems' retrieved IDs and score both runs against that shared pool.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Input JSON root must be an object.")
    return data


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def validate_and_normalize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        raise ValueError("No query results found in input.")

    seen_queries: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for row in results:
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("A result row is missing query_id.")
        if query_id in seen_queries:
            raise ValueError(f"Duplicate query_id: {query_id}")
        seen_queries.add(query_id)

        corpus = str(row.get("corpus") or "").lower()
        if corpus != "ncert":
            raise ValueError(
                f"{query_id}: expected ChEmbed corpus='ncert', got {corpus!r}."
            )

        hits = row.get("hits")
        if not isinstance(hits, list) or not hits:
            raise ValueError(f"{query_id}: hits must be a non-empty list.")

        seen_ids: set[str] = set()
        clean_hits = []
        for position, hit in enumerate(hits, start=1):
            record_id = str(hit.get("id") or "").strip()
            if not record_id:
                raise ValueError(f"{query_id}: hit at position {position} has no id.")
            if record_id in seen_ids:
                raise ValueError(f"{query_id}: duplicate hit id {record_id!r}.")
            seen_ids.add(record_id)

            if "relevance_score" not in hit or hit["relevance_score"] is None:
                raise ValueError(
                    f"{query_id}: hit {record_id!r} has no relevance_score. "
                    "Annotate every hit before evaluation."
                )
            try:
                grade = int(hit["relevance_score"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{query_id}: invalid relevance_score for {record_id!r}: "
                    f"{hit['relevance_score']!r}"
                )
            if grade not in (0, 1, 2):
                raise ValueError(
                    f"{query_id}: relevance_score must be 0, 1, or 2; "
                    f"got {grade} for {record_id!r}."
                )

            normalized_hit = dict(hit)
            normalized_hit["id"] = record_id
            normalized_hit["rank"] = safe_int(hit.get("rank")) or position
            normalized_hit["relevance_score"] = grade
            clean_hits.append(normalized_hit)

        clean_hits.sort(key=lambda item: item["rank"])
        normalized.append({**row, "query_id": query_id, "hits": clean_hits})

    return normalized


def calculate_metrics(hits: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top_hits = hits[:k]
    grades_by_id = {str(hit["id"]): int(hit["relevance_score"]) for hit in hits}
    relevant_ids = {
        record_id for record_id, grade in grades_by_id.items() if grade > 0
    }

    ranked_relevant = [
        1 if int(hit["relevance_score"]) > 0 else 0 for hit in top_hits
    ]
    first_rank = next(
        (index for index, relevant in enumerate(ranked_relevant, start=1) if relevant),
        None,
    )

    hit_rate = 1.0 if first_rank is not None else 0.0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    precision = sum(ranked_relevant) / len(top_hits) if top_hits else 0.0
    recall = (
        sum(ranked_relevant) / len(relevant_ids) if relevant_ids else None
    )

    # Match test_retrieval.py: graded gain = 2^grade - 1.
    dcg = sum(
        (2 ** int(hit["relevance_score"]) - 1) / math.log2(hit["rank"] + 1)
        for hit in top_hits
        if int(hit["relevance_score"]) > 0
    )
    ideal_grades = sorted(grades_by_id.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0
    )
    ndcg = dcg / idcg if idcg > 0 else None

    return {
        f"Hit@{k}": hit_rate,
        f"MRR@{k}": mrr,
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"nDCG@{k}": ndcg,
        "first_relevant_rank": first_rank,
        "known_relevant_count_in_annotated_pool": len(relevant_ids),
        "annotated_hit_count": len(hits),
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        row[key]
        for row in rows
        if isinstance(row.get(key), (int, float)) and row[key] is not None
    ]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate graded relevance annotations in a ChEmbed retrieval run."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="ChEmbed retrieval results JSON produced by benchmark_chembed.py.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cutoff for ranking metrics (default: 5, matching test_retrieval.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/retrieval_benchmark/chembed/evaluation",
        help="Directory for timestamped evaluation JSON files.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    input_path = Path(args.results)
    if not input_path.is_file():
        raise FileNotFoundError(f"Results file not found: {input_path.resolve()}")

    source = read_json(input_path)
    raw_results = source.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Input JSON must contain a 'results' list.")

    results = validate_and_normalize(raw_results)
    per_query = []
    latencies = []

    for row in results:
        metrics = calculate_metrics(row["hits"], args.top_k)
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        per_query.append({
            "query_id": row["query_id"],
            "chapter": row.get("chapter"),
            "corpus": row.get("corpus"),
            "collection_name": row.get("collection_name"),
            "latency_ms": latency,
            "metrics": metrics,
            "hits": [
                {
                    "rank": hit["rank"],
                    "id": hit["id"],
                    "relevance_score": hit["relevance_score"],
                    "distance": hit.get("distance"),
                    "document": hit.get("document"),
                    "metadata": hit.get("metadata"),
                }
                for hit in row["hits"]
            ],
        })

        print(
            f"{row['query_id']}: "
            + ", ".join(
                f"{name}={value:.4f}" if isinstance(value, (int, float))
                else f"{name}={value}"
                for name, value in metrics.items()
                if name in {
                    f"Hit@{args.top_k}", f"MRR@{args.top_k}",
                    f"Precision@{args.top_k}", f"Recall@{args.top_k}",
                    f"nDCG@{args.top_k}",
                }
            )
        )

    metric_names = [
        f"Hit@{args.top_k}", f"MRR@{args.top_k}",
        f"Precision@{args.top_k}", f"Recall@{args.top_k}",
        f"nDCG@{args.top_k}",
    ]
    summary = {
        "benchmark_id": source.get("benchmark_id"),
        "model": source.get("model"),
        "collection": source.get("collection"),
        "source_results_file": str(input_path),
        "query_count": len(per_query),
        "top_k": args.top_k,
        "metrics": {
            name: mean_metric([item["metrics"] for item in per_query], name)
            for name in metric_names
        },
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "recall_interpretation": (
            "Pool-based recall: denominator is the number of grade>0 records "
            "within this run's annotated retrieved hits, not the full NCERT corpus."
        ),
        "comparison_warning": (
            "Do not compare these metrics directly with MiniLM unless both runs "
            "are evaluated against the same pooled set of candidate IDs and labels."
        ),
        "per_query": per_query,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detail_path = output_dir / f"chembed_evaluation_{stamp}.json"
    summary_path = output_dir / f"chembed_evaluation_summary_{stamp}.json"

    # Never overwrite prior evaluation artifacts.
    for path in (detail_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    with detail_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark_id": source.get("benchmark_id"),
                "source_results_file": str(input_path),
                "top_k": args.top_k,
                "results": per_query,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nAggregate metrics")
    for name, value in summary["metrics"].items():
        print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: N/A")
    print(f"\nDetailed results: {detail_path.resolve()}")
    print(f"Summary:          {summary_path.resolve()}")


if __name__ == "__main__":
    main()
