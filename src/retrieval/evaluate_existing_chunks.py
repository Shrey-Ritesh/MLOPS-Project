#!/usr/bin/env python3
"""Evaluate relevance labels on already-retrieved chunks in annotated_chunks.json.

This script does NOT embed queries, connect to Chroma, or perform retrieval.
It reads each query's stored ranked hits and each hit's `relevance_score`.

Expected input shape:
  results[].corpora.{ncert,jee}.hits[]
  each hit: {rank, id, relevance_score, ...}

Grades: 0 = not relevant, 1 = partially relevant, 2 = highly relevant.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Input must be an object containing a 'results' list")
    return data


def validate_hits(query_id: str, corpus: str, hits: Any) -> list[dict[str, Any]]:
    if not isinstance(hits, list):
        raise ValueError(f"{query_id}/{corpus}: hits must be a list")
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    cleaned = []
    for i, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise ValueError(f"{query_id}/{corpus}/hits[{i}] must be an object")
        rid = str(hit.get("id", "")).strip()
        if not rid:
            raise ValueError(f"{query_id}/{corpus}/hits[{i}] has no id")
        if rid in seen_ids:
            raise ValueError(f"{query_id}/{corpus}: duplicate chunk id {rid!r}")
        seen_ids.add(rid)
        try:
            rank = int(hit["rank"])
            raw_grade = hit["relevance_score"]
            if isinstance(raw_grade, bool):
                raise ValueError
            grade = int(raw_grade)
            if float(raw_grade) != grade or grade not in (0, 1, 2):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"{query_id}/{corpus}/{rid}: expected integer rank and relevance_score 0, 1, or 2"
            )
        if rank < 1 or rank in seen_ranks:
            raise ValueError(f"{query_id}/{corpus}: invalid/duplicate rank {rank}")
        seen_ranks.add(rank)
        row = dict(hit)
        row["rank"] = rank
        row["relevance_score"] = grade
        cleaned.append(row)
    return sorted(cleaned, key=lambda x: x["rank"])


def calculate_metrics(hits: list[dict[str, Any]], cutoff: int) -> dict[str, Any]:
    ranked = hits[:cutoff]
    relevant_all = [h for h in hits if h["relevance_score"] > 0]
    relevant_top = [h for h in ranked if h["relevance_score"] > 0]
    first_rank = relevant_top[0]["rank"] if relevant_top else None

    hit_rate = 1.0 if relevant_top else 0.0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    precision = len(relevant_top) / len(ranked) if ranked else 0.0
    # Denominator is the relevant set among the annotated retrieved pool, not the full corpus.
    recall = len(relevant_top) / len(relevant_all) if relevant_all else None

    def gain(grade: int, rank: int) -> float:
        return (2 ** grade - 1) / math.log2(rank + 1) if grade > 0 else 0.0

    dcg = sum(gain(h["relevance_score"], h["rank"]) for h in ranked)
    ideal = sorted((h["relevance_score"] for h in hits), reverse=True)[:cutoff]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, start=1))
    ndcg = dcg / idcg if idcg else None

    return {
        "label_status": "labelled" if hits else "no_hits",
        f"Hit@{cutoff}": hit_rate,
        f"MRR@{cutoff}": mrr,
        f"Precision@{cutoff}": precision,
        f"Recall@{cutoff}": recall,
        f"nDCG@{cutoff}": ndcg,
        "first_relevant_rank": first_rank,
        "known_relevant_in_annotated_pool": len(relevant_all),
        "relevant_in_cutoff": len(relevant_top),
    }


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    vals = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
    return statistics.mean(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="annotated_chunks.json", help="Existing retrieval results with manual grades")
    ap.add_argument("--output-dir", default="artifacts/retrieval_evaluation", help="Where evaluation JSON files are written")
    ap.add_argument("--cutoff", type=int, default=5, help="Ranking metric cutoff (default: 5)")
    ap.add_argument("--corpora", nargs="+", choices=["ncert", "jee"], default=["ncert", "jee"])
    args = ap.parse_args()
    if args.cutoff < 1:
        ap.error("--cutoff must be >= 1")

    source = read_input(Path(args.input))
    per_query = []
    corpus_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in args.corpora}
    grade_counts: dict[str, Counter] = {c: Counter() for c in args.corpora}

    seen_queries: set[str] = set()
    for q in source["results"]:
        qid = str(q.get("query_id", "")).strip()
        if not qid:
            raise ValueError("A result entry is missing query_id")
        if qid in seen_queries:
            raise ValueError(f"Duplicate query_id: {qid}")
        seen_queries.add(qid)
        row = {"query_id": qid, "chapter": q.get("chapter"), "corpora": {}}
        for corpus in args.corpora:
            cdata = q.get("corpora", {}).get(corpus)
            if not isinstance(cdata, dict):
                raise ValueError(f"{qid}: missing corpora.{corpus}")
            hits = validate_hits(qid, corpus, cdata.get("hits"))
            grade_counts[corpus].update(h["relevance_score"] for h in hits)
            metrics = calculate_metrics(hits, args.cutoff)
            # Keep existing retrieval details and attach evaluation metrics to each stored hit.
            evaluated_hits = []
            for hit in hits:
                h = dict(hit)
                h["is_relevant"] = h["relevance_score"] > 0
                evaluated_hits.append(h)
            row["corpora"][corpus] = {
                "collection_name": cdata.get("collection_name"),
                "retrieval_latency_ms": cdata.get("latency_ms"),
                "metrics": metrics,
                "grade_counts": dict(sorted(Counter(str(h["relevance_score"]) for h in hits).items())),
                "hits": evaluated_hits,
            }
            corpus_rows[corpus].append(metrics)
        per_query.append(row)

    summary = {
        "benchmark_id": source.get("benchmark_id"),
        "source_run_timestamp_utc": source.get("run_timestamp_utc"),
        "evaluation_timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_file": str(Path(args.input)),
        "evaluation_mode": "offline_evaluation_of_existing_ranked_hits",
        "retrieval_performed": False,
        "embedding_model_loaded": False,
        "chroma_accessed": False,
        "cutoff": args.cutoff,
        "query_count": len(per_query),
        "collections": {},
        "interpretation_note": (
            "Metrics are computed from the already-retrieved and manually graded hit lists in the input file. "
            "Recall denominator includes only relevant chunks present in that annotated retrieved pool; "
            "it is not exhaustive full-collection recall unless the relevance pool is exhaustive."
        ),
    }
    for corpus in args.corpora:
        rows = corpus_rows[corpus]
        metric_names = [f"Hit@{args.cutoff}", f"MRR@{args.cutoff}", f"Precision@{args.cutoff}", f"Recall@{args.cutoff}", f"nDCG@{args.cutoff}"]
        summary["collections"][corpus] = {
            "query_count": len(rows),
            "mean_metrics": {m: mean_metric(rows, m) for m in metric_names},
            "total_annotated_hits": sum(grade_counts[corpus].values()),
            "relevance_grade_counts": {str(k): v for k, v in sorted(grade_counts[corpus].items())},
        }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = summary["evaluation_timestamp_utc"]
    detail_path = outdir / f"offline_evaluation_{stamp}.json"
    summary_path = outdir / f"offline_evaluation_summary_{stamp}.json"
    detail_path.write_text(json.dumps({"summary": summary, "results": per_query}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Evaluated {len(per_query)} queries from existing hits; no retrieval was performed.")
    for corpus in args.corpora:
        print(f"\n{corpus.upper()} (grades: {dict(grade_counts[corpus])})")
        for metric, value in summary["collections"][corpus]["mean_metrics"].items():
            print(f"  {metric}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"\nDetailed evaluation: {detail_path}")
    print(f"Summary:             {summary_path}")

if __name__ == "__main__":
    main()
