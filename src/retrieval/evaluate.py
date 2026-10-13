"""Run a small, reproducible evaluation of the existing Chroma retrieval pipeline.

Run from the repository root after copying this file to src/retrieval/evaluate.py:
    python -m src.retrieval.evaluate --queries configs/retrieval_eval_queries.jsonl --top-k 5

Each query JSONL row requires query_id and query. Optional fields:
    query_type: definition / formula / conceptual / procedure / etc.
    relevant_unit_ids: manually judged relevant unit IDs (initially [] is fine).

Metrics are computed only for queries with at least one manually labelled relevant unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .search import search


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_queries(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Query set not found: {path}")
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not row.get("query_id") or not row.get("query", "").strip():
                raise ValueError(f"{path}:{line_no}: each row needs non-empty query_id and query")
            if row["query_id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate query_id {row['query_id']!r}")
            seen.add(row["query_id"])
            relevant = row.get("relevant_unit_ids", [])
            if not isinstance(relevant, list) or not all(isinstance(x, str) for x in relevant):
                raise ValueError(f"{path}:{line_no}: relevant_unit_ids must be a list of strings")
            rows.append(row)
    if not rows:
        raise ValueError(f"No queries found in {path}")
    return rows


def query_metrics(hits: list[dict], relevant_ids: set[str], k: int) -> dict:
    ranked_ids = [h["unit_id"] for h in hits[:k]]
    relevant_ranks = [i for i, unit_id in enumerate(ranked_ids, 1) if unit_id in relevant_ids]
    return {
        f"hit_at_{k}": int(bool(relevant_ranks)),
        f"reciprocal_rank_at_{k}": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        f"precision_at_{k}": len(relevant_ranks) / k,
        f"recall_at_{k}": len(relevant_ranks) / len(relevant_ids) if relevant_ids else None,
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
    }


def run(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    query_path = Path(args.queries)
    config_path = Path(args.config)
    queries = read_queries(query_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"retrieval_eval_{timestamp}.jsonl"
    summary_path = out_dir / f"retrieval_eval_{timestamp}_summary.json"

    rows = []
    for index, item in enumerate(queries, 1):
        hits = search(item["query"], config_path=str(config_path), top_k=args.top_k)
        relevant_ids = set(item.get("relevant_unit_ids", []))
        metrics = query_metrics(hits, relevant_ids, args.top_k) if relevant_ids else None
        row = {
            "query_id": item["query_id"],
            "query": item["query"],
            "query_type": item.get("query_type", "unspecified"),
            "relevant_unit_ids": sorted(relevant_ids),
            "label_status": "labelled" if relevant_ids else "unlabelled",
            "metrics": metrics,
            "hits": hits,
        }
        rows.append(row)
        print(f"[{index}/{len(queries)}] {item['query_id']}: {len(hits)} hits; "
              f"{'labelled' if relevant_ids else 'needs relevance labels'}")

    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labelled = [r for r in rows if r["metrics"] is not None]
    metric_names = [f"hit_at_{args.top_k}", f"reciprocal_rank_at_{args.top_k}",
                    f"precision_at_{args.top_k}", f"recall_at_{args.top_k}"]
    aggregate = {}
    for name in metric_names:
        values = [r["metrics"][name] for r in labelled if r["metrics"].get(name) is not None]
        aggregate[name] = round(mean(values), 4) if values else None

    summary = {
        "experiment_type": "retrieval_quality_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "query_set_path": str(query_path),
        "query_set_sha256": sha256_file(query_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "top_k": args.top_k,
        "total_queries": len(rows),
        "labelled_queries": len(labelled),
        "unlabelled_queries": len(rows) - len(labelled),
        "metrics_on_labelled_queries_only": aggregate,
        "results_file": str(results_path),
        "note": "Retrieval metrics require human-judged relevant_unit_ids. Distances are ranking scores, not confidence probabilities.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return results_path, summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="configs/retrieval_eval_queries.jsonl")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/retrieval/evaluations")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
