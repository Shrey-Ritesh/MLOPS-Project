"""Evaluate retrieval against manually judged relevant IDs.

The script never infers semantic relevance. It compares ranked result IDs against
the manually curated relevant IDs, computes metrics per query/corpus, then
aggregates only applicable queries whose label status is ``complete``.

Label statuses in retrieval_eval.jsonl:
  - pending: not yet fully judged; excluded from aggregate metrics
  - complete: fully judged; an empty relevant-ID list is a genuine zero-relevance
              judgement and scores Hit@k=0, MRR@k=0
  - not_applicable: retrieval from this corpus was not intended for this query
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

from src.retrieval.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = Path(__file__).with_name("retrieval_eval.jsonl")
DEFAULT_NCERT_CONFIG = ROOT / "configs" / "retrieval.yaml"
DEFAULT_JEE_CONFIG = ROOT / "configs" / "jee_retrieval_v1.yaml"
DEFAULT_RESULTS_FILE = Path(__file__).with_name("retrieval_eval_results.jsonl")
DEFAULT_REPORT_FILE = Path(__file__).with_name("retrieval_metrics_report.json")
K = 4
VALID_STATUSES = {"pending", "complete", "not_applicable"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def result_id(item: dict[str, Any], corpus: str) -> str | None:
    if corpus == "ncert":
        value = item.get("unit_id") or item.get("id")
    else:
        value = item.get("question_id") or item.get("id")
        if value is None and isinstance(item.get("record"), dict):
            value = item["record"].get("question_id")
    return str(value) if value is not None else None


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = K) -> float:
    return float(any(item_id in relevant_ids for item_id in ranked_ids[:k]))


def reciprocal_rank_at_k(
    ranked_ids: list[str], relevant_ids: set[str], k: int = K
) -> float:
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def targets_for_intent(intent: str | None) -> set[str]:
    """Mirror the intended gateway routing; do not infer applicability from result count."""
    if intent == "PRACTICE":
        return {"jee"}
    if intent == "DEFINE":
        return {"ncert"}
    if intent in {None, "UNCLEAR"}:
        return set()
    return {"ncert", "jee"}


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.eval_file)
    collected = []
    for row in rows:
        result = retrieve(
            row["query"],
            row.get("difficulty", "medium"),
            str(args.ncert_config),
            str(args.jee_config),
        )
        ncert_items = result.get("ncert", [])
        jee_items = result.get("jee_questions", [])
        output = dict(row)
        output["retrieval_status"] = result.get("status", "unknown")
        request = result.get("query_request")
        output["retrieval_intent"] = request.get("intent") if isinstance(request, dict) else row.get("expected_intent")
        output["ranked_ncert_ids"] = [result_id(item, "ncert") for item in ncert_items]
        output["ranked_jee_ids"] = [result_id(item, "jee") for item in jee_items]
        output["ncert_results_for_labeling"] = ncert_items
        output["jee_results_for_labeling"] = jee_items
        collected.append(output)
        print(f"{row['query_id']} [{output['retrieval_status']}] {row['query']}")
        print("  NCERT:", ", ".join(f"{i + 1}. {x}" for i, x in enumerate(output["ranked_ncert_ids"]) if x) or "(no results)")
        print("  JEE:  ", ", ".join(f"{i + 1}. {x}" for i, x in enumerate(output["ranked_jee_ids"]) if x) or "(no results)")
    return collected


def evaluate(rows: list[dict[str, Any]], k: int = K) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []

    for corpus in ("ncert", "jee"):
        label_key = f"relevant_{corpus}_ids"
        status_key = f"{corpus}_label_status"
        ranked_key = f"ranked_{corpus}_ids"
        applicable_targets = 0
        complete_count = 0
        pending_count = 0
        hits: list[float] = []
        rrs: list[float] = []

        for row in rows:
            intent = row.get("expected_intent") or row.get("retrieval_intent")
            intended_targets = targets_for_intent(intent)
            status = row.get(status_key)

            # Safe fallback for older eval files without explicit statuses.
            if status is None:
                if corpus not in intended_targets:
                    status = "not_applicable"
                elif row.get(label_key):
                    status = "complete"
                else:
                    status = "pending"

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"{row.get('query_id')} has invalid {status_key}={status!r}; "
                    f"use one of {sorted(VALID_STATUSES)}"
                )
            if status == "not_applicable":
                # Guard against accidentally excluding a corpus that the intent routes to.
                if corpus in intended_targets:
                    raise ValueError(
                        f"{row.get('query_id')}: {status_key} is not_applicable, "
                        f"but intent {intent!r} normally targets {corpus}. "
                        "Review routing/applicability explicitly."
                    )
                continue
            if corpus not in intended_targets:
                raise ValueError(
                    f"{row.get('query_id')}: {corpus} is not an intended target for "
                    f"{intent!r}; set {status_key}='not_applicable'."
                )

            applicable_targets += 1
            if status == "pending":
                pending_count += 1
                per_query.append({
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "corpus": corpus,
                    "label_status": "pending",
                    "retrieved_count": len(row.get(ranked_key, [])),
                    "relevant_count": len(row.get(label_key, [])),
                    "hit_at_k": None,
                    "reciprocal_rank_at_k": None,
                    "first_relevant_rank": None,
                    "note": "Excluded from aggregate until this query/corpus is fully judged.",
                })
                continue

            relevant = {str(x) for x in row.get(label_key, [])}
            ranked = [str(x) for x in row.get(ranked_key, []) if x is not None]
            hit = hit_at_k(ranked, relevant, k)
            rr = reciprocal_rank_at_k(ranked, relevant, k)
            first_rank = next(
                (i for i, item_id in enumerate(ranked[:k], start=1) if item_id in relevant),
                None,
            )
            complete_count += 1
            hits.append(hit)
            rrs.append(rr)
            per_query.append({
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "corpus": corpus,
                "label_status": "complete",
                "retrieved_count": len(ranked),
                "relevant_count": len(relevant),
                "relevant_retrieved_at_k": int(hit),
                "hit_at_k": hit,
                "first_relevant_rank": first_rank,
                "reciprocal_rank_at_k": rr,
                "note": (
                    "No relevant IDs were judged in the corpus; counted as zero."
                    if not relevant else
                    "No chunks retrieved." if not ranked else
                    None
                ),
            })

        aggregate.append({
            "corpus": corpus,
            "k": k,
            "applicable_queries": applicable_targets,
            "complete_queries": complete_count,
            "pending_queries": pending_count,
            "Hit@k": round(sum(hits) / len(hits), 4) if hits else None,
            "MRR@k": round(sum(rrs) / len(rrs), 4) if rrs else None,
            "aggregation": "macro-average across complete, applicable query-corpus pairs",
            "note": None if hits else "No complete, applicable query-corpus pairs to score.",
        })

    return per_query, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--ncert-config", type=Path, default=DEFAULT_NCERT_CONFIG)
    parser.add_argument("--jee-config", type=Path, default=DEFAULT_JEE_CONFIG)
    parser.add_argument("--collect", action="store_true", help="Run retrieval and save ranked results")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")

    if args.collect:
        collected = collect(args)
        write_jsonl(args.results_file, collected)
        print(f"\nSaved ranked results to: {args.results_file}")
        print("Label results in the evaluation file; do not edit the collected results file.")
        return

    eval_rows = read_jsonl(args.eval_file)
    if not args.results_file.exists():
        raise FileNotFoundError(
            f"No collected rankings found at {args.results_file}. Run with --collect first."
        )
    result_by_id = {row["query_id"]: row for row in read_jsonl(args.results_file)}
    merged = []
    for row in eval_rows:
        prior = result_by_id.get(row["query_id"])
        if prior is None:
            raise ValueError(f"No collected retrieval results for {row['query_id']}")
        combined = dict(prior)
        # Current evaluation file is the source of truth for human labels/statuses.
        combined.update(row)
        combined["ranked_ncert_ids"] = prior.get("ranked_ncert_ids", [])
        combined["ranked_jee_ids"] = prior.get("ranked_jee_ids", [])
        merged.append(combined)

    per_query, aggregate = evaluate(merged, args.k)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "k": args.k,
        "per_query": per_query,
        "aggregate": aggregate,
        "interpretation": (
            "Hit@k is 1 if at least one manually relevant ID appears in the first k results; "
            "MRR@k is 1/rank of the first relevant result, or 0 if none appears. "
            "Only complete and applicable labels are aggregated."
        ),
    }
    args.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for metric in aggregate:
        print(json.dumps(metric, indent=2))
    print(f"\nPer-query metrics and aggregate report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
