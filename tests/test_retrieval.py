"""Unit tests for the question-focused MiniLM retriever.

Tests mock Chroma and the embedding model. They do not require a running
Chroma server, model downloads, or the full corpus.
"""
from __future__ import annotations

import math

import pytest

from src.retrieval import retrieve as retrieve_module
from src.retrieval.retrieve import (
    CANDIDATE_K,
    DISTANCE_THRESHOLD,
    JEE_TOP_K,
    NCERT_TOP_K,
    _validate_embedding_compatibility,
    _validate_shared_chroma_endpoint,
)


def test_retrieval_constants_match_current_policy():
    assert DISTANCE_THRESHOLD == 0.8
    assert CANDIDATE_K == 10
    assert NCERT_TOP_K == 5
    assert JEE_TOP_K == 5


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question must not be empty"):
        retrieve_module.retrieve("   ")


def test_non_string_question_is_rejected():
    with pytest.raises(TypeError, match="question must be a string"):
        retrieve_module.retrieve(None)


def test_invalid_subject_is_rejected():
    with pytest.raises(ValueError, match="Subject must be Physics, Chemistry, or Mathematics"):
        retrieve_module.retrieve("Explain Newton's laws", subject="biology")


def test_embedding_model_mismatch_is_rejected():
    ncert = _config("ncert_collection", model_name="model-a")
    jee = _config("jee_collection", model_name="model-b")

    with pytest.raises(ValueError, match="incompatible"):
        _validate_embedding_compatibility(ncert, jee)


def test_embedding_revision_mismatch_is_rejected():
    ncert = _config("ncert_collection", model_revision="revision-a")
    jee = _config("jee_collection", model_revision="revision-b")

    with pytest.raises(ValueError, match="model_revision"):
        _validate_embedding_compatibility(ncert, jee)


def test_embedding_normalization_mismatch_is_rejected():
    ncert = _config("ncert_collection", normalize_embeddings=True)
    jee = _config("jee_collection", normalize_embeddings=False)

    with pytest.raises(ValueError, match="normalize_embeddings"):
        _validate_embedding_compatibility(ncert, jee)


def test_incompatible_distance_metric_is_rejected():
    ncert = _config("ncert_collection")
    jee = _config("jee_collection")
    jee["vector_store"]["distance_metric"] = "l2"

    with pytest.raises(ValueError, match="cosine"):
        _validate_embedding_compatibility(ncert, jee)


def test_different_chroma_endpoints_are_rejected():
    ncert = _config("ncert_collection")
    jee = _config("jee_collection")
    jee["vector_store"]["port"] = 9000

    with pytest.raises(ValueError, match="different Chroma endpoints"):
        _validate_shared_chroma_endpoint(ncert, jee)


def test_compatible_embedding_configs_are_accepted():
    config = _config(
        "ncert_collection",
        model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )

    _validate_embedding_compatibility(config, config)


class FakeCollection:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.query_calls = []
        self.metadata = {"hnsw:space": "cosine"}

    def count(self):
        return len(self.rows["ids"])

    def query(self, *, query_embeddings, n_results, include):
        self.query_calls.append({
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        })

        n = min(n_results, self.count())
        return {
            field: [self.rows.get(field, [])[:n]]
            for field in ("ids", "documents", "metadatas", "distances")
        }


class FakeClient:
    def __init__(self, collections):
        self.collections = collections
        self.requested = []

    def get_collection(self, *, name):
        self.requested.append(name)
        return self.collections[name]


def _config(
    collection_name,
    *,
    model_name="all-MiniLM-L6-v2",
    model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    normalize_embeddings=True,
    host="localhost",
    port=8000,
):
    return {
        "embedding": {
            "model_name": model_name,
            "model_revision": model_revision,
            "normalize_embeddings": normalize_embeddings,
        },
        "vector_store": {
            "host": host,
            "port": port,
            "collection_name": collection_name,
            "distance_metric": "cosine",
        },
        "document": {
            "input_path": "questions.jsonl",
            "id_field": "question_id",
        },
    }


def _fake_rows(
    prefix,
    n,
    *,
    distances=None,
    subjects=None,
    include_documents=True,
):
    if distances is None:
        distances = [0.1 + i * 0.01 for i in range(n)]

    if subjects is None:
        subjects = ["Physics"] * n

    rows = {
        "ids": [f"{prefix}-{i}" for i in range(n)],
        "documents": [f"document {i} about Newton's laws and motion" for i in range(n)],
        "metadatas": [
            {
                "subject": subjects[i],
                "unit_title": f"unit-{i}",
                "chapter_title": f"chapter-{i}",
                "chapter_id": f"chapter-id-{i}",
                "source_id": f"source-{i}",
                "page_start": i + 1,
                "page_end": i + 2,
                "block_ids": '["block-1"]',
                "asset_ids": '["asset-1"]',
            }
            for i in range(n)
        ],
        "distances": distances,
    }

    if not include_documents:
        rows.pop("documents")

    return rows


def _patch_retrieval_dependencies(
    monkeypatch,
    *,
    ncert_count=10,
    jee_count=10,
    ncert_distances=None,
    jee_distances=None,
    ncert_subjects=None,
    jee_subjects=None,
    jee_records=None,
):
    ncert_cfg = _config("ncert_collection")
    jee_cfg = _config("jee_collection")

    collections = {
        "ncert_collection": FakeCollection(
            "ncert_collection",
            _fake_rows(
                "unit",
                ncert_count,
                distances=ncert_distances,
                subjects=ncert_subjects,
            ),
        ),
        "jee_collection": FakeCollection(
            "jee_collection",
            _fake_rows(
                "question",
                jee_count,
                distances=jee_distances,
                subjects=jee_subjects,
                include_documents=False,
            ),
        ),
    }

    client = FakeClient(collections)
    configs = {
        "ncert.yaml": ncert_cfg,
        "jee.yaml": jee_cfg,
    }
    encoded_texts = []

    monkeypatch.setattr(
        retrieve_module,
        "load_config",
        lambda path: configs[str(path)],
    )
    monkeypatch.setattr(
        retrieve_module,
        "connect_chroma",
        lambda config: client,
    )
    monkeypatch.setattr(
        retrieve_module,
        "load_embedder",
        lambda config: object(),
    )

    def fake_encode(model, texts, config):
        encoded_texts.append(list(texts))
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(retrieve_module, "encode_texts", fake_encode)

    records = jee_records
    if records is None:
        records = {
            f"question-{i}": {
                "question_id": f"question-{i}",
                "question_text": f"JEE question {i}",
            }
            for i in range(jee_count)
        }

    monkeypatch.setattr(
        retrieve_module,
        "load_question_lookup",
        lambda path, id_field="question_id": records,
    )

    return client, collections, encoded_texts


def test_retrieve_uses_question_only_as_embedding_input(monkeypatch):
    _, _, encoded_texts = _patch_retrieval_dependencies(monkeypatch)

    question = "Explain why reaction rate increases with temperature"
    result = retrieve_module.retrieve(question, "ncert.yaml")

    assert result["status"] == "ok"
    assert encoded_texts == [[question]]
    assert result["question"] == question
    assert result["retrieval_input_type"] == "question_only"


def test_retrieve_defaults_to_ncert_only(monkeypatch):
    client, collections, _ = _patch_retrieval_dependencies(monkeypatch)

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
    )

    assert result["status"] == "ok"
    assert result["requested_corpora"] == ["ncert"]
    assert len(result["ncert"]) == NCERT_TOP_K
    assert result["jee_questions"] == []
    assert client.requested == ["ncert_collection"]
    assert len(collections["ncert_collection"].query_calls) == 1
    assert collections["ncert_collection"].query_calls[0]["n_results"] == CANDIDATE_K
    assert collections["jee_collection"].query_calls == []


def test_retrieve_optionally_includes_jee(monkeypatch):
    _, collections, _ = _patch_retrieval_dependencies(monkeypatch)

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
        include_jee=True,
    )

    assert result["status"] == "ok"
    assert result["requested_corpora"] == ["ncert", "jee"]
    assert len(result["ncert"]) == NCERT_TOP_K
    assert len(result["jee_questions"]) == JEE_TOP_K
    assert result["missing_question_records"] == []

    assert collections["ncert_collection"].query_calls[0]["n_results"] == CANDIDATE_K
    assert collections["jee_collection"].query_calls[0]["n_results"] == CANDIDATE_K


def test_retrieve_caps_candidate_count_at_collection_size(monkeypatch):
    _, collections, _ = _patch_retrieval_dependencies(
        monkeypatch,
        ncert_count=2,
        jee_count=3,
    )

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
        include_jee=True,
    )

    assert len(result["ncert"]) == 2
    assert len(result["jee_questions"]) == 3
    assert collections["ncert_collection"].query_calls[0]["n_results"] == 2
    assert collections["jee_collection"].query_calls[0]["n_results"] == 3


def test_distance_threshold_is_inclusive_and_filters_invalid_values(monkeypatch):
    distances = [
        0.1,
        0.2,
        0.3,
        0.4,
        DISTANCE_THRESHOLD,
        DISTANCE_THRESHOLD + 0.01,
        float("nan"),
        None,
        "invalid",
        0.9,
    ]

    _, _, _ = _patch_retrieval_dependencies(
        monkeypatch,
        ncert_count=10,
        ncert_distances=distances,
    )

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
    )

    assert result["candidate_counts"]["ncert"] == 10
    assert result["eligible_counts"]["ncert"] == 5
    assert len(result["ncert"]) == 5
    assert all(hit["distance"] <= DISTANCE_THRESHOLD for hit in result["ncert"])
    assert any(hit["distance"] == DISTANCE_THRESHOLD for hit in result["ncert"])


def test_retrieve_returns_no_results_when_all_candidates_exceed_threshold(monkeypatch):
    distances = [0.81 + i * 0.01 for i in range(10)]
    _patch_retrieval_dependencies(
        monkeypatch,
        ncert_count=10,
        ncert_distances=distances,
    )

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
    )

    assert result["status"] == "no_results_within_distance_threshold"
    assert result["ncert"] == []
    assert result["eligible_counts"]["ncert"] == 0


def test_retrieve_returns_at_most_five_results_per_corpus(monkeypatch):
    _patch_retrieval_dependencies(monkeypatch, ncert_count=10, jee_count=10)

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
        include_jee=True,
    )

    assert len(result["ncert"]) <= 5
    assert len(result["jee_questions"]) <= 5


def test_subject_filter_is_applied(monkeypatch):
    subjects = ["Physics", "Chemistry", "Physics", "Chemistry", "Physics",
                "Chemistry", "Physics", "Chemistry", "Physics", "Chemistry"]

    _patch_retrieval_dependencies(
        monkeypatch,
        ncert_count=10,
        ncert_subjects=subjects,
    )

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        subject="physics",
    )

    assert result["status"] == "ok"
    assert all(hit["subject"].casefold() == "physics" for hit in result["ncert"])


def test_retrieve_reports_missing_canonical_jee_records(monkeypatch):
    _patch_retrieval_dependencies(
        monkeypatch,
        jee_count=2,
        jee_records={
            "question-0": {
                "question_id": "question-0",
                "question_text": "JEE question 0",
            }
        },
    )

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
        include_jee=True,
    )

    assert [hit["question_id"] for hit in result["jee_questions"]] == ["question-0"]
    assert result["missing_question_records"] == ["question-1"]


def test_returned_ncert_results_have_traceable_fields(monkeypatch):
    _patch_retrieval_dependencies(monkeypatch)

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
    )

    hit = result["ncert"][0]
    assert {
        "unit_id",
        "distance",
        "subject",
        "unit_title",
        "chapter_title",
        "source_id",
        "page_start",
        "page_end",
        "block_ids",
        "asset_ids",
        "document",
        "ranking",
    } <= hit.keys()
    assert "_ranking_text" not in hit


def test_returned_jee_results_have_canonical_records(monkeypatch):
    _patch_retrieval_dependencies(monkeypatch)

    result = retrieve_module.retrieve(
        "Explain Newton's laws of motion",
        "ncert.yaml",
        "jee.yaml",
        include_jee=True,
    )

    hit = result["jee_questions"][0]
    assert {"question_id", "distance", "ranking", "record"} <= hit.keys()
    assert hit["record"]["question_id"] == hit["question_id"]


def test_empty_embedding_result_does_not_pass_distance_gate():
    assert not retrieve_module._within_distance_threshold(None)
    assert not retrieve_module._within_distance_threshold(float("nan"))
    assert not retrieve_module._within_distance_threshold(math.inf)
    assert not retrieve_module._within_distance_threshold("not-a-distance")