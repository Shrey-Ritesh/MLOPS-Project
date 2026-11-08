"""Question-focused retrieval over NCERT learning units and optional JEE questions.

Run from the repository root:
    python -m src.retrieval.retrieve "Explain why reaction rate increases with temperature"
    python -m src.retrieval.retrieve "Explain why reaction rate increases with temperature" --include-jee

The embedding query is the question ONLY. The student's answer should be
provided separately to the evaluator, not to this retriever.

For each enabled corpus, the retriever:
1. Fetches up to ten nearest candidates from Chroma.
2. Discards candidates whose cosine distance is greater than 0.8.
3. Reranks the remaining candidates using semantic distance and lexical overlap.
4. Returns up to five NCERT chunks and, optionally, up to five JEE questions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import connect_chroma, load_config, read_jsonl
from .embedder import encode_texts, load_embedder


# Retrieval policy for the current MiniLM baseline.
# This threshold assumes Chroma cosine distance (not cosine similarity).
DISTANCE_THRESHOLD = 0.8
CANDIDATE_K = 10
NCERT_TOP_K = 5
JEE_TOP_K = 5

# Preserve the relative weighting of the previous semantic/lexical components
# (0.65 : 0.25), renormalized after removing the intent-fit component.
SEMANTIC_WEIGHT = 0.7222222222
LEXICAL_WEIGHT = 0.2777777778

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "how", "what", "why", "when",
    "where", "which", "who", "whom", "can", "could", "would",
    "should", "will", "to", "of", "in", "on", "at", "for", "from",
    "with", "by", "and", "or", "but", "it", "its", "this", "that",
    "these", "those", "me", "my", "we", "you", "your", "explain",
    "describe", "tell", "show", "give", "make", "much", "many",
    "please",
}

_VALID_SUBJECTS = {"physics", "chemistry", "mathematics", "math", "maths"}


def _tokens(text: str) -> set[str]:
    """Small deterministic tokenizer; no additional NLP dependency."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", text.casefold())
    return {
        word for word in words
        if word not in _STOPWORDS and len(word) > 1
    }


def _lexical_relevance(question: str, document: str) -> float:
    """Fraction of meaningful question terms found in a candidate document."""
    question_terms = _tokens(question)
    if not question_terms:
        return 0.0
    document_terms = _tokens(document)
    return len(question_terms & document_terms) / len(question_terms)


def _valid_distance(distance: Any) -> float | None:
    """Return a finite Chroma distance, or None for invalid values."""
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _within_distance_threshold(distance: Any) -> bool:
    """Keep distances <= 0.8; reject missing, invalid, NaN and larger values."""
    value = _valid_distance(distance)
    return value is not None and value <= DISTANCE_THRESHOLD


def _semantic_score(distance: Any) -> float:
    """Convert cosine distance to a bounded ranking signal (not a probability)."""
    value = _valid_distance(distance)
    if value is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / 2.0))


def _rank_candidate(
    candidate: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Attach transparent semantic and lexical scores to one candidate."""
    semantic = _semantic_score(candidate.get("distance"))
    lexical = _lexical_relevance(question, candidate.get("_ranking_text", ""))
    score = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical
    candidate["ranking"] = {
        "semantic": round(semantic, 4),
        "lexical": round(lexical, 4),
        "score": round(score, 4),
    }
    return candidate


def _select_results(
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the highest-ranked candidates, using distance as a tie-breaker."""
    return sorted(
        candidates,
        key=lambda item: (
            item["ranking"]["score"],
            -float(item["distance"]),
        ),
        reverse=True,
    )[:top_k]


def _metadata_list(value: Any) -> list[Any]:
    """Decode Chroma JSON-encoded list metadata without failing retrieval."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []


@lru_cache(maxsize=4)
def load_question_lookup(
    path: str,
    id_field: str = "question_id",
) -> dict[str, dict[str, Any]]:
    """Resolve Chroma question IDs to canonical JEE question JSONL records."""
    records = read_jsonl(path)
    return {
        record[id_field]: record
        for record in records
        if isinstance(record.get(id_field), str) and record[id_field]
    }


def _validate_cosine_config(config: dict[str, Any], corpus: str) -> None:
    metric = str(config.get("vector_store", {}).get("distance_metric", "cosine")).casefold()
    if metric != "cosine":
        raise ValueError(
            f"{corpus.upper()} is configured with distance metric {metric!r}; "
            "DISTANCE_THRESHOLD = 0.8 is defined for cosine distance."
        )


def _validate_embedding_compatibility(
    ncert_cfg: dict[str, Any],
    jee_cfg: dict[str, Any],
) -> None:
    """Ensure both indexes use the same embedding space and distance metric."""
    ncert_emb = ncert_cfg.get("embedding", {})
    jee_emb = jee_cfg.get("embedding", {})

    if ncert_emb.get("model_name") != jee_emb.get("model_name"):
        raise ValueError(
            f"Embedding model mismatch: {ncert_emb.get('model_name')!r} and "
            f"{jee_emb.get('model_name')!r} are incompatible."
        )

    for field in ("model_revision", "normalize_embeddings"):
        if ncert_emb.get(field) != jee_emb.get(field):
            raise ValueError(
                f"Embedding {field} mismatch: {ncert_emb.get(field)!r} != "
                f"{jee_emb.get(field)!r}"
            )

    _validate_cosine_config(ncert_cfg, "NCERT")
    _validate_cosine_config(jee_cfg, "JEE")


def _validate_shared_chroma_endpoint(
    ncert_cfg: dict[str, Any],
    jee_cfg: dict[str, Any],
) -> None:
    """Ensure both collections can be queried through the same Chroma client."""
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


def _get_existing_collection(client: Any, config: dict[str, Any], corpus: str) -> Any:
    """Open an existing collection without creating or mutating it."""
    collection_name = config.get("vector_store", {}).get("collection_name")
    if not collection_name:
        raise ValueError(f"{corpus.upper()} config has no vector_store.collection_name")
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {corpus.upper()} collection {collection_name!r}. "
            "Build the index first and verify the collection configuration."
        ) from exc

    count = collection.count()
    if count < 1:
        raise RuntimeError(f"{corpus.upper()} collection {collection_name!r} is empty.")

    # When Chroma exposes the collection metadata, verify that its actual space
    # agrees with the cosine-distance threshold used by this module.
    actual_metric = (getattr(collection, "metadata", None) or {}).get("hnsw:space")
    if actual_metric and str(actual_metric).casefold() != "cosine":
        raise ValueError(
            f"{corpus.upper()} collection {collection_name!r} uses "
            f"{actual_metric!r}, but this retriever expects cosine distance."
        )
    return collection


def _empty_result(
    question: str,
    *,
    include_jee: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "question": question,
        # Keep these aliases for callers that currently consume the older schema.
        "query": question,
        "normalized_query": question,
        "retrieval_input_type": "question_only",
        "requested_corpora": ["ncert"] + (["jee"] if include_jee else []),
        "candidate_counts": {"ncert": 0, "jee": 0},
        "eligible_counts": {"ncert": 0, "jee": 0},
        "distance_threshold": DISTANCE_THRESHOLD,
        "retrieval_limits": {
            "candidate_k_per_corpus": CANDIDATE_K,
            "ncert_max": NCERT_TOP_K,
            "jee_max": JEE_TOP_K if include_jee else 0,
        },
        "ncert": [],
        "jee_questions": [],
        "missing_question_records": [],
        "ranking_method": "semantic 0.7222, lexical 0.2778; no intent gateway",
        "status": status,
    }


def retrieve(
    question: str,
    ncert_config_path: str = "configs/retrieval.yaml",
    jee_config_path: str = "configs/jee_retrieval_v1.yaml",
    subject: str | None = None,
    include_jee: bool = False,
) -> dict[str, Any]:
    """Retrieve evidence for a question; never include the student's answer.

    NCERT is always searched. JEE retrieval is opt-in through ``include_jee``.
    Each enabled collection contributes at most ten candidates and returns at
    most five after cosine-distance filtering and heuristic reranking.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    normalized_subject = None
    if subject is not None:
        subject_alias = subject.casefold().strip()
        if subject_alias not in _VALID_SUBJECTS:
            raise ValueError("Subject must be Physics, Chemistry, or Mathematics.")
        normalized_subject = (
            "mathematics" if subject_alias in {"math", "maths"} else subject_alias
        )

    # The embedding input is deliberately just the original question.
    ncert_cfg = load_config(ncert_config_path)
    _validate_cosine_config(ncert_cfg, "NCERT")

    jee_cfg = None
    requested_corpora = ["ncert"]
    if include_jee:
        jee_cfg = load_config(jee_config_path)
        _validate_embedding_compatibility(ncert_cfg, jee_cfg)
        _validate_shared_chroma_endpoint(ncert_cfg, jee_cfg)
        requested_corpora.append("jee")

    client = connect_chroma(ncert_cfg)
    collections: dict[str, Any] = {
        "ncert": _get_existing_collection(client, ncert_cfg, "ncert")
    }
    if include_jee:
        assert jee_cfg is not None
        collections["jee"] = _get_existing_collection(client, jee_cfg, "jee")

    model = load_embedder(ncert_cfg)
    embedding = encode_texts(model, [question], ncert_cfg)[0]
    query_vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    candidates: dict[str, list[dict[str, Any]]] = {"ncert": [], "jee": []}
    candidate_counts = {"ncert": 0, "jee": 0}
    eligible_counts = {"ncert": 0, "jee": 0}

    # Fetch ten nearest candidates independently from every enabled collection.
    for corpus, collection in collections.items():
        candidate_k = min(CANDIDATE_K, collection.count())
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        raw_documents = (result.get("documents") or [[]])[0]
        raw_metadatas = (result.get("metadatas") or [[]])[0]
        raw_distances = (result.get("distances") or [[]])[0]

        documents = raw_documents if len(raw_documents) == len(ids) else [""] * len(ids)
        metadatas = raw_metadatas if len(raw_metadatas) == len(ids) else [{}] * len(ids)
        distances = raw_distances if len(raw_distances) == len(ids) else [None] * len(ids)
        candidate_counts[corpus] = len(ids)

        for record_id, document, metadata, raw_distance in zip(
            ids, documents, metadatas, distances
        ):
            # Apply the hard distance gate before any secondary ranking.
            if not _within_distance_threshold(raw_distance):
                continue

            metadata = metadata or {}
            candidate_subject = metadata.get("subject")
            if normalized_subject and str(candidate_subject or "").casefold() != normalized_subject:
                continue

            document = document or ""
            candidate = {
                "distance": float(raw_distance),
                "subject": candidate_subject,
                "_ranking_text": document,
            }

            if corpus == "ncert":
                candidate.update({
                    "unit_id": record_id,
                    "unit_title": metadata.get("unit_title"),
                    "chapter_title": metadata.get("chapter_title"),
                    "chapter_id": metadata.get("chapter_id"),
                    "source_id": metadata.get("source_id"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "block_ids": _metadata_list(metadata.get("block_ids")),
                    "asset_ids": _metadata_list(metadata.get("asset_ids")),
                    "document": document,
                })
            else:
                candidate["question_id"] = record_id

            candidates[corpus].append(_rank_candidate(candidate, question))

        eligible_counts[corpus] = len(candidates[corpus])

    # Rank each corpus independently: always up to five NCERT, optionally five JEE.
    ncert_hits = _select_results(candidates["ncert"], NCERT_TOP_K)
    jee_candidates = _select_results(candidates["jee"], JEE_TOP_K)

    # Resolve the selected JEE vector IDs to canonical source question records.
    jee_hits: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    if jee_candidates:
        assert jee_cfg is not None
        document_config = jee_cfg.get("document", {})
        question_path = str(document_config.get("input_path", ""))
        if not question_path:
            raise ValueError("JEE config has no document.input_path")
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
                "record": record,
            })

    for hit in ncert_hits:
        hit.pop("_ranking_text", None)

    status = "ok" if ncert_hits or jee_hits else "no_results_within_distance_threshold"
    return {
        "question": question,
        # Backward-compatible aliases: these values now contain the question only.
        "query": question,
        "normalized_query": question,
        "retrieval_input_type": "question_only",
        "requested_corpora": requested_corpora,
        "candidate_counts": candidate_counts,
        "eligible_counts": eligible_counts,
        "distance_threshold": DISTANCE_THRESHOLD,
        "retrieval_limits": {
            "candidate_k_per_corpus": CANDIDATE_K,
            "ncert_max": NCERT_TOP_K,
            "jee_max": JEE_TOP_K if include_jee else 0,
        },
        "ncert": ncert_hits,
        "jee_questions": jee_hits,
        "missing_question_records": missing_ids,
        "ranking_method": "semantic 0.7222, lexical 0.2778; no intent gateway",
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The exam question; the student's answer is not passed here")
    parser.add_argument("--ncert-config", default="configs/retrieval.yaml")
    parser.add_argument("--jee-config", default="configs/jee_retrieval_v1.yaml")
    parser.add_argument(
        "--include-jee",
        action="store_true",
        help="Also retrieve up to five JEE questions (NCERT retrieval is always enabled).",
    )
    parser.add_argument("--subject", choices=["physics", "chemistry", "mathematics"])
    args = parser.parse_args()

    result = retrieve(
        args.question,
        ncert_config_path=args.ncert_config,
        jee_config_path=args.jee_config,
        subject=args.subject,
        include_jee=args.include_jee,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
