#!/usr/bin/env python3
"""Build a shared NCERT relevance pool and evaluate MiniLM vs ChEmbed.

Inputs are annotated benchmark JSON files with:
  queries[].query_id
  queries[].annotations.ncert.relevance_grades

The order of relevance_grades keys is treated as retrieval rank order, matching
the order in which the annotation maps were serialized. All IDs from both
models are pooled; conflicting grades are rejected rather than silently
resolved. Metrics are calculated against the union of positively judged IDs.

This script intentionally stops if the two input files contain identical
NCERT ranked IDs and grades for every query: such files do not contain two
independent retrieval runs to compare.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return data


def load_queries(data: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{label}: expected a non-empty 'queries' list")

    result = {}
    for query in queries:
        qid = str(query.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"{label}: query missing query_id")
        if qid in result:
            raise ValueError(f"{label}: duplicate query_id {qid}")

        grades = (
            query.get("annotations", {})
                 .get("ncert", {})
                 .get("relevance_grades")
        )
        if not isinstance(grades, dict) or not grades:
            raise ValueError(f"{label}/{qid}: missing annotations.ncert.relevance_grades")

        normalized = {}
        for record_id, grade in grades.items():
            record_id = str(record_id)
            if grade not in (0, 1, 2):
                raise ValueError(
                    f"{label}/{qid}: invalid grade {grade!r} for {record_id}"
                )
            normalized[record_id] = int(grade)

        result[qid] = {
            "query": query,
            "grades": normalized,
            # JSON object insertion order preserves the retrieved ranking.
            "ranked_ids": list(normalized.keys()),
        }
    return result


def metrics(ranked_ids: list[str], pooled_grades: dict[str, int], k: int) -> dict[str, Any]:
    top_ids = ranked_ids[:k]
    top_grades = [pooled_grades[rid] for rid in top_ids]
    relevant_pool = sum(grade > 0 for grade in pooled_grades.values())
    relevant_top = sum(grade > 0 for grade in top_grades)

    first_relevant = next(
        (rank for rank, grade in enumerate(top_grades, 1) if grade > 0),
        None,
    )
    dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(top_grades, 1)
    )
    ideal = sorted(pooled_grades.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, 1)
    )

    return {
        f"Hit@{k}": 1.0 if first_relevant is not None else 0.0,
        f"MRR@{k}": 1.0 / first_relevant if first_relevant else 0.0,
        f"Precision@{k}": relevant_top / k,
        f"Recall@{k}": relevant_top / relevant_pool if relevant_pool else None,
        f"nDCG@{k}": dcg / idcg if idcg else None,
        "relevant_in_top_k": relevant_top,
        "relevant_in_pooled_judgements": relevant_pool,
        "candidate_count": len(ranked_ids),
    }


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minilm",
        default="src/retrieval/chemistry_retrieval_benchmark_v1_annotated_minilm.json",
    )
    parser.add_argument(
        "--chembed",
        default="src/retrieval/chemistry_retrieval_benchmark_v1_annotated_chembed.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="artifacts/retrieval_benchmark/pooled_evaluation",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    minilm_path, chembed_path = rooted(args.minilm), rooted(args.chembed)
    minilm_data, chembed_data = read_json(minilm_path), read_json(chembed_path)

    if minilm_data.get("benchmark_id") != chembed_data.get("benchmark_id"):
        raise ValueError("MiniLM and ChEmbed benchmark_id values differ")

    minilm = load_queries(minilm_data, "MiniLM")
    chembed = load_queries(chembed_data, "ChEmbed")
    if set(minilm) != set(chembed):
        raise ValueError("MiniLM and ChEmbed query IDs differ")

    identical_runs = True
    per_query = []

    for qid in sorted(minilm):
        m_ids = minilm[qid]["ranked_ids"]
        c_ids = chembed[qid]["ranked_ids"]
        m_grades = minilm[qid]["grades"]
        c_grades = chembed[qid]["grades"]

        if m_ids != c_ids or m_grades != c_grades:
            identical_runs = False

        # Merge labels over the union of retrieved IDs. Conflicting human
        # judgements must be resolved manually; never choose one implicitly.
        pooled = dict(m_grades)
        for record_id, grade in c_grades.items():
            if record_id in pooled and pooled[record_id] != grade:
                raise ValueError(
                    f"{qid}: conflicting relevance grades for {record_id}: "
                    f"MiniLM={pooled[record_id]}, ChEmbed={grade}"
                )
            pooled[record_id] = grade

        m_metrics = metrics(m_ids, pooled, args.top_k)
        c_metrics = metrics(c_ids, pooled, args.top_k)
        per_query.append({
            "query_id": qid,
            "pooled_candidate_count": len(pooled),
            "pooled_relevant_count": sum(g > 0 for g in pooled.values()),
            "minilm": m_metrics,
            "chembed": c_metrics,
        })

    if identical_runs:
        raise ValueError(
            "The supplied MiniLM and ChEmbed files contain identical NCERT "
            "ranked IDs and identical relevance grades for all queries. "
            "They do not encode two independent retrieval runs, so pooled "
            "model-comparison metrics cannot be calculated. Check that the "
            "ChEmbed file contains ChEmbed's own retrieved chunk IDs and rank order."
        )

    metric_names = [
        f"Hit@{args.top_k}", f"MRR@{args.top_k}",
        f"Precision@{args.top_k}", f"Recall@{args.top_k}",
        f"nDCG@{args.top_k}",
    ]
    summary = {
        "benchmark_id": minilm_data.get("benchmark_id"),
        "top_k": args.top_k,
        "inputs": {"minilm": str(minilm_path), "chembed": str(chembed_path)},
        "evaluation_note": (
            "Both rankings are scored against the union of the two systems' "
            "annotated NCERT candidate IDs. Recall is pool-based, not exhaustive "
            "corpus recall."
        ),
        "aggregate_metrics": {
            model: {
                name: average([row[model] for row in per_query], name)
                for name in metric_names
            }
            for model in ("minilm", "chembed")
        },
        "per_query": per_query,
    }

    out_dir = rooted(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = out_dir / f"pooled_ncert_evaluation_{stamp}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
