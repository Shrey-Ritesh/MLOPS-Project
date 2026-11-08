#!/usr/bin/env python3
"""
Offline analysis of an existing annotated Chroma retrieval run.

Run from the repository root:
    python scripts/analyze_retrieval_distances.py

Defaults:
    input:  src/retrieval/annotated_chunks.json
    output: artifacts/retrieval_analysis/

No embedding model, Chroma connection, or retrieval operation is performed.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORPORA = ("ncert", "jee")
GRADES = (0, 1, 2)


def percentile(values: list[float], p: float) -> float | None:
    """Linearly interpolated percentile, with p in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "mean": None, "median": None, "std_dev": None,
            "min": None, "q1": None, "q3": None, "max": None, "iqr": None,
        }
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q1": q1,
        "q3": q3,
        "max": max(values),
        "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
    }


def average_ranks(values: list[float]) -> list[float]:
    """Ranks with average ranks for ties; smallest value receives rank 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_run(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"Input not found: {path.resolve()}\n"
            "Expected src/retrieval/annotated_chunks.json when run from repository root. "
            "Pass --input if your filename differs."
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SystemExit("Expected a JSON object containing a 'results' array.")
    return data


def collect_rows(data: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    rows = {corpus: [] for corpus in CORPORA}
    warnings: list[str] = []
    seen_queries: set[str] = set()

    for result_index, result in enumerate(data["results"]):
        if not isinstance(result, dict):
            warnings.append(f"results[{result_index}] is not an object; skipped")
            continue
        query_id = str(result.get("query_id") or f"index_{result_index}").strip()
        if query_id in seen_queries:
            warnings.append(f"Duplicate query_id {query_id!r} encountered.")
        seen_queries.add(query_id)

        corpora = result.get("corpora") or {}
        for corpus in CORPORA:
            corpus_result = corpora.get(corpus) or {}
            hits = corpus_result.get("hits") or []
            for hit_index, hit in enumerate(hits):
                if not isinstance(hit, dict):
                    warnings.append(f"{query_id}/{corpus}/hits[{hit_index}] is not an object; skipped")
                    continue

                grade_raw = hit.get("relevance_score")
                if isinstance(grade_raw, bool) or grade_raw not in GRADES:
                    warnings.append(
                        f"{query_id}/{corpus}/hit[{hit_index}] has invalid/missing "
                        f"relevance_score {grade_raw!r}; skipped"
                    )
                    continue

                rank_raw = hit.get("rank")
                try:
                    rank = int(rank_raw)
                    if rank < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    warnings.append(f"{query_id}/{corpus}/hit[{hit_index}] has invalid rank; skipped")
                    continue

                distance = safe_float(hit.get("distance"))
                if distance is None:
                    warnings.append(
                        f"{query_id}/{corpus}/rank_{rank} has missing/invalid distance; "
                        "retained for rank analysis only"
                    )

                rows[corpus].append({
                    "query_id": query_id,
                    "rank": rank,
                    "id": hit.get("id"),
                    "grade": int(grade_raw),
                    "distance": distance,
                    "chapter": result.get("chapter"),
                })

    for corpus in CORPORA:
        rows[corpus].sort(key=lambda row: (row["query_id"], row["rank"]))
    return rows, warnings


def analyse_corpus(rows: list[dict[str, Any]], cutoff: int) -> dict[str, Any]:
    by_grade: dict[int, list[dict[str, Any]]] = {g: [] for g in GRADES}
    by_rank: dict[int, Counter] = defaultdict(Counter)
    query_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_grade[row["grade"]].append(row)
        by_rank[row["rank"]][row["grade"]] += 1
        query_groups[row["query_id"]].append(row)

    distance_by_grade = {}
    rank_by_grade = {}
    for grade in GRADES:
        grade_rows = by_grade[grade]
        distances = [r["distance"] for r in grade_rows if r["distance"] is not None]
        ranks = [float(r["rank"]) for r in grade_rows]
        distance_by_grade[str(grade)] = describe(distances)
        rank_by_grade[str(grade)] = describe(ranks)

    valid_distance_rows = [r for r in rows if r["distance"] is not None]
    relevance = [float(r["grade"]) for r in valid_distance_rows]
    # Lower Chroma distance should correspond to higher relevance, hence negate distance.
    negative_distances = [-r["distance"] for r in valid_distance_rows]
    overall_spearman = spearman(relevance, negative_distances)

    # Query-level correlation is useful only when a query has at least two hits and
    # both relevance grades and distances vary.
    per_query_correlations = []
    for query_id, query_rows in sorted(query_groups.items()):
        usable = [r for r in query_rows if r["distance"] is not None]
        corr = spearman(
            [float(r["grade"]) for r in usable],
            [-r["distance"] for r in usable],
        )
        if corr is not None:
            per_query_correlations.append({"query_id": query_id, "spearman": corr})

    rank_distribution = []
    max_rank = max((r["rank"] for r in rows), default=0)
    for rank in range(1, max_rank + 1):
        counts = by_rank.get(rank, Counter())
        total = sum(counts.values())
        rank_distribution.append({
            "rank": rank,
            "grade_counts": {str(g): counts.get(g, 0) for g in GRADES},
            "total": total,
            "grade_2_rate": counts.get(2, 0) / total if total else None,
            "grade_1_or_2_rate": (
                (counts.get(1, 0) + counts.get(2, 0)) / total if total else None
            ),
        })

    query_summaries = []
    for query_id, query_rows in sorted(query_groups.items()):
        ordered = sorted(query_rows, key=lambda r: r["rank"])
        grade2 = [r for r in ordered if r["grade"] == 2]
        grade12 = [r for r in ordered if r["grade"] >= 1]
        query_summaries.append({
            "query_id": query_id,
            "hit_count": len(ordered),
            "grade_counts": {str(g): sum(r["grade"] == g for r in ordered) for g in GRADES},
            "first_grade_2_rank": min((r["rank"] for r in grade2), default=None),
            "first_relevant_rank": min((r["rank"] for r in grade12), default=None),
            "grade_2_in_top_k": any(r["rank"] <= cutoff and r["grade"] == 2 for r in ordered),
            "grade_2_in_ranks_6_to_10": any(
                6 <= r["rank"] <= 10 and r["grade"] == 2 for r in ordered
            ),
            "top_k_grade_counts": {
                str(g): sum(r["rank"] <= cutoff and r["grade"] == g for r in ordered)
                for g in GRADES
            },
        })

    top_k_rows = [r for r in rows if r["rank"] <= cutoff]
    tail_rows = [r for r in rows if cutoff < r["rank"] <= max(cutoff + 5, 10)]

    return {
        "hit_count": len(rows),
        "query_count": len(query_groups),
        "grade_counts": {str(g): len(by_grade[g]) for g in GRADES},
        "grade_proportions": {
            str(g): (len(by_grade[g]) / len(rows) if rows else None) for g in GRADES
        },
        "distance_distribution_by_grade": distance_by_grade,
        "rank_distribution_by_grade": rank_by_grade,
        "distance_relevance_relationship": {
            "spearman_relevance_vs_negative_distance": overall_spearman,
            "interpretation": (
                "Positive values mean higher relevance tends to have lower distance. "
                "This is an association across the annotated retrieved pool, not proof "
                "that distance is a calibrated relevance score."
            ),
            "valid_hit_count": len(valid_distance_rows),
            "queries_with_defined_spearman": len(per_query_correlations),
            "mean_query_spearman": (
                statistics.fmean(x["spearman"] for x in per_query_correlations)
                if per_query_correlations else None
            ),
            "per_query": per_query_correlations,
        },
        "rank_position_analysis": {
            "cutoff": cutoff,
            "top_k_grade_counts": {
                str(g): sum(r["rank"] <= cutoff and r["grade"] == g for r in rows)
                for g in GRADES
            },
            "ranks_after_cutoff_to_max_10_grade_counts": {
                str(g): sum(cutoff < r["rank"] <= max(cutoff + 5, 10) and r["grade"] == g
                            for r in rows)
                for g in GRADES
            },
            "by_rank": rank_distribution,
            "queries_with_grade_2_first_appearing_after_cutoff": [
                q["query_id"] for q in query_summaries
                if q["first_grade_2_rank"] is not None and q["first_grade_2_rank"] > cutoff
            ],
            "queries_with_grade_2_at_ranks_6_to_10": [
                q["query_id"] for q in query_summaries if q["grade_2_in_ranks_6_to_10"]
            ],
        },
        "per_query": query_summaries,
        "top_k_distance_summary": describe(
            [r["distance"] for r in top_k_rows if r["distance"] is not None]
        ),
        "post_cutoff_distance_summary": describe(
            [r["distance"] for r in tail_rows if r["distance"] is not None]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="src/retrieval/annotated_chunks.json",
        help="Annotated retrieval run JSON (relative to repository root).",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/retrieval_analysis",
        help="Directory for JSON output (relative to repository root).",
    )
    parser.add_argument("--cutoff", type=int, default=5, help="Top-k boundary; default: 5.")
    args = parser.parse_args()

    if args.cutoff < 1:
        parser.error("--cutoff must be >= 1")

    input_path = Path(args.input)
    data = load_run(input_path)
    rows, warnings = collect_rows(data)

    report = {
        "analysis_type": "offline_retrieval_distance_and_rank_analysis",
        "source": {
            "input_path": str(input_path),
            "benchmark_id": data.get("benchmark_id"),
            "run_timestamp_utc": data.get("run_timestamp_utc"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": args.cutoff,
        "methodology": {
            "retrieval_reexecuted": False,
            "embedding_model_loaded": False,
            "chroma_accessed": False,
            "distance_assumption": (
                "Distances are analysed in their stored numeric form. Lower is treated "
                "as closer because these are Chroma nearest-neighbour result distances. "
                "The script does not assume a particular metric or convert distance to similarity."
            ),
            "spearman": (
                "Computed between relevance grade (0/1/2) and negative distance, so a "
                "positive coefficient indicates that more relevant hits tend to be closer."
            ),
            "scope": (
                "All analyses describe only the retrieved and annotated hits in this run; "
                "they do not describe the full corpus distribution."
            ),
        },
        "corpora": {
            corpus: analyse_corpus(rows[corpus], args.cutoff)
            for corpus in CORPORA
        },
        "warnings": warnings,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"retrieval_distance_analysis_{stamp}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Analysed {sum(x['hit_count'] for x in report['corpora'].values())} hits.")
    for corpus in CORPORA:
        result = report["corpora"][corpus]
        corr = result["distance_relevance_relationship"]["spearman_relevance_vs_negative_distance"]
        print(
            f"{corpus.upper()}: {result['hit_count']} hits, "
            f"grade counts={result['grade_counts']}, "
            f"Spearman(relevance, -distance)={corr}"
        )
    print(f"JSON report: {output_path.resolve()}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see JSON report)")


if __name__ == "__main__":
    main()
