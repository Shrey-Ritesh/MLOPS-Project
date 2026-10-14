
"""Intent-aware retrieval and heuristic reranking for the JEE tutor.

Run from the repository root:
    python -m src.retrieval.retrieve "Explain Newton's laws of motion" --level easy

This module:
1. Uses the query gateway to determine intent and permitted corpora.
2. Fetches a larger candidate pool from each selected Chroma collection.
3. Reranks candidates using semantic distance, lexical relevance, and intent fit.
4. Selects final results using the existing difficulty-level allocation.

Reranking scores are heuristic ranking signals, not probabilities.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import connect_chroma, load_config, read_jsonl
from .embedder import encode_texts, load_embedder
from .query_gateway import build_query_request


LEVEL_ALLOCATION = {
    "easy": {"ncert": 6, "jee": 2},
    "medium": {"ncert": 4, "jee": 4},
    "hard": {"ncert": 2, "jee": 6},
}

# Retrieve more than the final quota so reranking has alternatives.
FETCH_MULTIPLIER = 5
MIN_CANDIDATES = 20

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "how", "what", "why", "when",
    "where", "which", "who", "whom", "can", "could", "would",
    "should", "will", "to", "of", "in", "on", "at", "for", "from",
    "with", "by", "and", "or", "but", "it", "its", "this", "that",
    "these", "those", "me", "my", "we", "you", "your", "explain",
    "describe", "tell", "show", "give", "make", "much", "many",
    "please", "does", "differ", "different", "between", "compare",
    "comparison", "versus", "vs",
}

_CAUSAL_CUES = (
    "because", "therefore", "hence", "due to", "caused by",
    "results in", "leads to", "reason", "mechanism",
    "as a result", "this is because",
)

_SUMMARY_CUES = (
    "summary", "key points", "in brief", "in conclusion",
    "points to ponder", "revision",
)

_SUBJECT_ALIASES = {
    "physics": {"physics"},
    "chemistry": {"chemistry"},
    "mathematics": {"mathematics", "maths", "math"},
}


def _tokens(text: str) -> set[str]:
    """Simple deterministic tokenization; no extra NLP dependency."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", text.casefold())
    return {
        word for word in words
        if word not in _STOPWORDS and len(word) > 1
    }


def _lexical_relevance(query: str, document: str) -> float:
    """Fraction of meaningful query terms present in the candidate."""
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0

    document_terms = _tokens(document)
    return len(query_terms & document_terms) / len(query_terms)


def _semantic_score(distance: float | int | None) -> float:
    """Convert cosine distance to a bounded ranking signal.

    Chroma cosine distance is typically 1 - cosine similarity.
    This is only a ranking transformation, not a calibrated probability.
    """
    if distance is None:
        return 0.0

    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0

    if value != value:  # NaN
        return 0.0

    return max(0.0, min(1.0, 1.0 - value / 2.0))


def _explicit_subjects(query: str) -> set[str]:
    """Find subjects explicitly named in a query."""
    q = query.casefold()
    found = set()

    for subject, aliases in _SUBJECT_ALIASES.items():
        if any(
            re.search(rf"\b{re.escape(alias)}\b", q)
            for alias in aliases
        ):
            found.add(subject)

    return found


def _intent_fit(
    query_request: dict[str, Any],
    corpus: str,
    document: str,
    subject: str | None,
) -> float:
    """Return a small, explainable intent-fit score in [0, 1]."""
    primary = query_request["intent"]
    secondary = set(query_request.get("secondary_intents", []))
    constraints = set(query_request.get("constraints", []))
    text = document.casefold()
    score = 0.0

    if primary == "PRACTICE" and corpus == "jee":
        score += 1.0

    if primary == "DEFINE" and corpus == "ncert":
        score += 1.0

    if primary == "SOLVE" and corpus == "jee":
        score += 0.5

    if "EXPLAIN_WHY" in {primary, *secondary}:
        if any(cue in text for cue in _CAUSAL_CUES):
            score += 0.7

    if "SUMMARIZE" in {primary, *secondary}:
        if any(cue in text for cue in _SUMMARY_CUES):
            score += 0.7

    # A subject match is supporting evidence, never a hard filter.
    expected_subject = query_request.get("subject")
    if expected_subject and subject:
        if expected_subject.casefold() == subject.casefold():
            score += 0.25

    # Comparison coverage is handled across the selected result set.
    if "cover_both_concepts" in constraints:
        score += 0.1

    return min(score, 1.0)


def _rank_candidate(
    candidate: dict[str, Any],
    query_request: dict[str, Any],
    corpus: str,
) -> dict[str, Any]:
    """Attach transparent component scores and a combined ranking score."""
    semantic = _semantic_score(candidate.get("distance"))
    lexical = _lexical_relevance(
        query_request["normalized_query"],
        candidate.get("_ranking_text", ""),
    )
    intent = _intent_fit(
        query_request,
        corpus,
        candidate.get("_ranking_text", ""),
        candidate.get("subject"),
    )

    # Semantic relevance remains the dominant signal.
    # Intent affects ordering without overpowering topical relevance.
    ranking_score = (
        0.65 * semantic
        + 0.25 * lexical
        + 0.10 * intent
    )

    candidate["ranking"] = {
        "semantic": round(semantic, 4),
        "lexical": round(lexical, 4),
        "intent_fit": round(intent, 4),
        "score": round(ranking_score, 4),
    }
    return candidate


def _select_results(
    candidates: list[dict[str, Any]],
    quota: int,
    query_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select top-ranked candidates, encouraging explicit subject coverage.

    For cross-subject comparisons, try to include both named subjects when
    the candidate pool contains suitable results. Otherwise use ranked order.
    """
    if quota <= 0 or not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["ranking"]["score"],
            -float(item.get("distance", float("inf"))),
        ),
        reverse=True,
    )

    if (
        "cover_both_concepts" not in query_request.get("constraints", [])
    ):
        return ranked[:quota]

    subjects = _explicit_subjects(query_request["normalized_query"])
    if len(subjects) < 2:
        return ranked[:quota]

    selected = []
    selected_ids = set()
    covered_subjects = set()

    # First try to cover each explicitly named subject.
    for target_subject in sorted(subjects):
        match = next(
            (
                item for item in ranked
                if item.get("subject")
                and item["subject"].casefold() == target_subject
                and item.get("unit_id", item.get("question_id"))
                not in selected_ids
            ),
            None,
        )
        if match is not None and len(selected) < quota:
            selected.append(match)
            selected_ids.add(
                match.get("unit_id", match.get("question_id"))
            )
            covered_subjects.add(target_subject)

    # Fill remaining slots in reranked order.
    for item in ranked:
        item_id = item.get("unit_id", item.get("question_id"))
        if item_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item_id)
        if len(selected) >= quota:
            break

    # Report coverage for diagnostics, without claiming that every
    # comparison question can be satisfied by the available corpus.
    for item in selected:
        item.setdefault("coverage", {})
        item["coverage"]["explicit_subjects_requested"] = sorted(subjects)
        item["coverage"]["explicit_subjects_available"] = sorted(
            covered_subjects
        )

    return selected[:quota]


@lru_cache(maxsize=4)
def load_question_lookup(
    path: str,
    id_field: str = "question_id",
) -> dict[str, dict[str, Any]]:
    """Resolve Chroma question IDs to their canonical JSONL records."""
    records = read_jsonl(path)
    lookup = {}

    for record in records:
        question_id = record.get(id_field)
        if isinstance(question_id, str) and question_id:
            lookup[question_id] = record

    return lookup


def _validate_compatible_configs(
    ncert_cfg: dict[str, Any],
    jee_cfg: dict[str, Any],
) -> None:
    """Ensure the same query vector is valid for both collections."""
    fields = ("model_name", "model_revision", "normalize_embeddings")

    for field in fields:
        ncert_value = ncert_cfg["embedding"].get(field)
        jee_value = jee_cfg["embedding"].get(field)
        if ncert_value != jee_value:
            raise ValueError(
                f"NCERT and JEE embedding configs disagree on {field!r}: "
                f"{ncert_value!r} != {jee_value!r}"
            )

    ncert_store = ncert_cfg["vector_store"]
    jee_store = jee_cfg["vector_store"]

    ncert_endpoint = (
        str(ncert_store.get("host", "localhost")),
        int(ncert_store.get("port", 8000)),
    )
    jee_endpoint = (
        str(jee_store.get("host", "localhost")),
        int(jee_store.get("port", 8000)),
    )

    if ncert_endpoint != jee_endpoint:
        raise ValueError(
            "NCERT and JEE configs point to different Chroma endpoints. "
            "This retriever currently uses one shared Chroma client."
        )


def retrieve(
    query: str,
    level: str,
    ncert_config_path: str = "configs/retrieval.yaml",
    jee_config_path: str = "configs/jee_retrieval_v1.yaml",
    fetch_multiplier: int = FETCH_MULTIPLIER,
    subject: str | None = None,
) -> dict[str, Any]:
    query = query.strip()
    level = level.casefold().strip()

    if not query:
        raise ValueError("Query must not be empty.")
    if level not in LEVEL_ALLOCATION:
        raise ValueError("Level must be easy, medium, or hard.")
    if fetch_multiplier < 1:
        raise ValueError("fetch_multiplier must be >= 1")
    if subject is not None and subject.casefold().strip() not in {
        "physics", "chemistry", "mathematics", "math"
    }:
        raise ValueError("Subject must be Physics, Chemistry, or Mathematics.")

    query_request = build_query_request(query, difficulty=level)
    if subject:
        query_request["subject"] = (
            "mathematics" if subject.casefold().strip() == "math"
            else subject.casefold().strip()
        )

    # UNCLEAR should not trigger arbitrary vector searches.
    if query_request["needs_clarification"] or not query_request["retrieval_targets"]:
        return {
            "query": query,
            "normalized_query": query_request["normalized_query"],
            "level": level,
            "query_request": query_request,
            "allocation": {"ncert": 0, "jee": 0},
            "ncert": [],
            "jee_questions": [],
            "missing_question_records": [],
            "status": "needs_clarification",
        }

    ncert_cfg = load_config(ncert_config_path)
    jee_cfg = load_config(jee_config_path)
    _validate_compatible_configs(ncert_cfg, jee_cfg)

    client = connect_chroma(ncert_cfg)
    targets = set(query_request["retrieval_targets"])

    collections = {}
    configs = {"ncert": ncert_cfg, "jee": jee_cfg}

    # Fetch existing collections only; never create collections at query time.
    for corpus in ("ncert", "jee"):
        if corpus not in targets:
            continue

        config = configs[corpus]
        collection_name = config["vector_store"]["collection_name"]
        collection = client.get_collection(name=collection_name)

        if collection.count() == 0:
            raise RuntimeError(
                f"{corpus.upper()} collection {collection_name!r} is empty."
            )
        collections[corpus] = collection

    model = load_embedder(ncert_cfg)
    query_vector = encode_texts(
        model, [query_request["normalized_query"]], ncert_cfg
    )[0].tolist()

    base_allocation = LEVEL_ALLOCATION[level]
    final_allocation = {
        corpus: (
            base_allocation[corpus] if corpus in targets else 0
        )
        for corpus in ("ncert", "jee")
    }

    candidates: dict[str, list[dict[str, Any]]] = {
        "ncert": [],
        "jee": [],
    }

    # Stage 1: fetch a larger pool from each permitted corpus.
    for corpus, collection in collections.items():
        final_k = final_allocation[corpus]
        if final_k == 0:
            continue

        count = collection.count()
        candidate_k = min(
            count,
            max(final_k * fetch_multiplier, MIN_CANDIDATES),
        )

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for record_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            if subject and str(metadata.get("subject") or "").casefold() != query_request["subject"]:
                continue
            document = document or ""

            if corpus == "ncert":
                candidate = {
                    "unit_id": record_id,
                    "distance": distance,
                    "unit_title": metadata.get("unit_title"),
                    "chapter_title": metadata.get("chapter_title"),
                    "subject": metadata.get("subject"),
                    "chapter_id": metadata.get("chapter_id"),
                    "source_id": metadata.get("source_id"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "block_ids": json.loads(
                        metadata.get("block_ids", "[]")
                    ),
                    "asset_ids": json.loads(
                        metadata.get("asset_ids", "[]")
                    ),
                    "document": document,
                    "_ranking_text": document,
                }
            else:
                candidate = {
                    "question_id": record_id,
                    "distance": distance,
                    "subject": metadata.get("subject"),
                    "_ranking_text": document,
                }

            candidates[corpus].append(
                _rank_candidate(candidate, query_request, corpus)
            )

    # Stage 2: rerank and select final per-corpus quotas.
    ncert_hits = _select_results(
        candidates["ncert"], final_allocation["ncert"], query_request
    )
    jee_candidates = _select_results(
        candidates["jee"], final_allocation["jee"], query_request
    )

    # Resolve selected JEE IDs to canonical records.
    jee_hits = []
    missing_ids = []

    if jee_candidates:
        document_config = jee_cfg["document"]
        question_path = str(Path(document_config["input_path"]))
        id_field = document_config.get("id_field", "question_id")
        question_lookup = load_question_lookup(question_path, id_field)

        for candidate in jee_candidates:
            question_id = candidate["question_id"]
            record = question_lookup.get(question_id)

            if record is None:
                missing_ids.append(question_id)
                continue

            jee_hits.append({
                "question_id": question_id,
                "distance": candidate["distance"],
                "ranking": candidate["ranking"],
                "coverage": candidate.get("coverage", {}),
                "record": record,
            })

    # Internal ranking text is not part of the public output.
    for hit in ncert_hits:
        hit.pop("_ranking_text", None)

    return {
        "query": query,
        "normalized_query": query_request["normalized_query"],
        "level": level,
        "query_request": query_request,
        "allocation": final_allocation,
        "candidate_counts": {
            "ncert": len(candidates["ncert"]),
            "jee": len(candidates["jee"]),
        },
        "ncert": ncert_hits,
        "jee_questions": jee_hits,
        "missing_question_records": missing_ids,
        "status": "ok",
        "ranking_method": (
            "heuristic_v1: semantic 0.65, lexical 0.25, intent_fit 0.10"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Student's question or learning goal")
    parser.add_argument(
        "--level", required=True, choices=["easy", "medium", "hard"]
    )
    parser.add_argument(
        "--ncert-config", default="configs/retrieval.yaml"
    )
    parser.add_argument(
        "--jee-config", default="configs/jee_retrieval_v1.yaml"
    )
    parser.add_argument(
        "--fetch-multiplier", type=int, default=FETCH_MULTIPLIER
    )
    args = parser.parse_args()

    result = retrieve(
        args.query,
        args.level,
        ncert_config_path=args.ncert_config,
        jee_config_path=args.jee_config,
        fetch_multiplier=args.fetch_multiplier,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
