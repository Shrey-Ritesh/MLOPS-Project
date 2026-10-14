"""Unit tests for the Phase 1 baseline retriever.

These tests mock Chroma and the embedding model: they do not require a running
Chroma server, model downloads, or the full corpus.
"""
from __future__ import annotations

import pytest

from src.retrieval import retrieve as retrieve_module
from src.retrieval.query_gateway import build_query_request
from src.retrieval.retrieve import (
    LEVEL_ALLOCATION,
    _validate_embedding_compatibility,
    retrieval_plan,
)


def test_level_allocations_are_the_phase_1_baseline():
    assert LEVEL_ALLOCATION == {
        "easy": {"ncert": 6, "jee": 2},
        "medium": {"ncert": 4, "jee": 4},
        "hard": {"ncert": 2, "jee": 6},
    }


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_practice_query_routes_only_to_jee(level):
    request = build_query_request("Give me questions on thermodynamics", difficulty=level)
    allocation = retrieval_plan(request, level)

    assert request["intent"] == "PRACTICE"
    assert allocation == {"ncert": 0, "jee": LEVEL_ALLOCATION[level]["jee"]}


def test_definition_query_routes_only_to_ncert():
    request = build_query_request("What is a carboxylic acid?", difficulty="medium")
    assert request["intent"] == "DEFINE"
    assert retrieval_plan(request, "medium") == {"ncert": 4, "jee": 0}


def test_unclear_query_has_no_retrieval_targets():
    request = build_query_request("Thermodynamics", difficulty="medium")
    assert request["retrieval_targets"] == []
    assert retrieval_plan(request, "medium") == {"ncert": 0, "jee": 0}


def test_compare_query_routes_to_both_corpora():
    request = build_query_request(
        "How does thermodynamics in physics differ from thermodynamics in chemistry",
        difficulty="medium",
    )
    assert request["intent"] == "COMPARE"
    assert request["constraints"] == ["cover_both_concepts"]
    assert set(request["retrieval_targets"]) == {"ncert", "jee"}
    assert retrieval_plan(request, "medium") == {"ncert": 4, "jee": 4}


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query must not be empty"):
        retrieve_module.retrieve("   ", "easy")


@pytest.mark.parametrize("level", ["", "beginner", "MEDIUM_PLUS"])
def test_invalid_level_is_rejected(level):
    with pytest.raises(ValueError, match="Level must be easy, medium, or hard"):
        retrieve_module.retrieve("Explain thermodynamics", level)


def test_embedding_model_mismatch_is_rejected():
    ncert = {
        "embedding": {
            "model_name": "model-a",
            "model_revision": None,
            "normalize_embeddings": True,
        }
    }
    jee = {
        "embedding": {
            "model_name": "model-b",
            "model_revision": None,
            "normalize_embeddings": True,
        }
    }
    with pytest.raises(ValueError, match="incompatible"):
        _validate_embedding_compatibility(ncert, jee)


def test_embedding_revision_mismatch_is_rejected():
    ncert = {
        "embedding": {
            "model_name": "same-model",
            "model_revision": "rev-a",
            "normalize_embeddings": True,
        }
    }
    jee = {
        "embedding": {
            "model_name": "same-model",
            "model_revision": "rev-b",
            "normalize_embeddings": True,
        }
    }
    with pytest.raises(ValueError, match="model_revision"):
        _validate_embedding_compatibility(ncert, jee)


def test_compatible_embedding_configs_are_accepted():
    config = {
        "embedding": {
            "model_name": "same-model",
            "model_revision": None,
            "normalize_embeddings": True,
        }
    }
    _validate_embedding_compatibility(config, config)


class FakeCollection:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.query_calls = []

    def count(self):
        return len(self.rows["ids"])

    def query(self, *, query_embeddings, n_results, include):
        self.query_calls.append(
            {"query_embeddings": query_embeddings, "n_results": n_results, "include": include}
        )
        n = min(n_results, self.count())
        result = {}
        for field in ("ids", "documents", "metadatas", "distances"):
            result[field] = [self.rows.get(field, [])[:n]]
        return result


class FakeClient:
    def __init__(self, collections):
        self.collections = collections
        self.requested = []

    def get_collection(self, *, name):
        self.requested.append(name)
        return self.collections[name]


def _config(collection_name, input_path="questions.jsonl"):
    return {
        "embedding": {
            "model_name": "same-model",
            "model_revision": None,
            "normalize_embeddings": True,
        },
        "vector_store": {
            "host": "localhost",
            "port": 8000,
            "collection_name": collection_name,
        },
        "document": {"input_path": input_path, "id_field": "question_id"},
    }


def _fake_rows(prefix, n, *, include_documents):
    rows = {
        "ids": [f"{prefix}-{i}" for i in range(n)],
        "metadatas": [{"subject": "Physics", "unit_title": f"unit-{i}"} for i in range(n)],
        "distances": [0.1 + i * 0.01 for i in range(n)],
    }
    if include_documents:
        rows["documents"] = [f"document-{i}" for i in range(n)]
    return rows


def _patch_retrieval_dependencies(monkeypatch, *, ncert_count=10, jee_count=10, jee_records=None):
    ncert_cfg = _config("ncert_collection")
    jee_cfg = _config("jee_collection")
    collections = {
        "ncert_collection": FakeCollection(
            "ncert_collection", _fake_rows("unit", ncert_count, include_documents=True)
        ),
        "jee_collection": FakeCollection(
            "jee_collection", _fake_rows("question", jee_count, include_documents=False)
        ),
    }
    client = FakeClient(collections)

    configs = {"ncert.yaml": ncert_cfg, "jee.yaml": jee_cfg}
    monkeypatch.setattr(retrieve_module, "load_config", lambda path: configs[str(path)])
    monkeypatch.setattr(retrieve_module, "connect_chroma", lambda config: client)
    monkeypatch.setattr(retrieve_module, "load_embedder", lambda config: object())
    monkeypatch.setattr(
        retrieve_module,
        "encode_texts",
        lambda model, texts, config: [[0.1, 0.2, 0.3]],
    )
    monkeypatch.setattr(
        retrieve_module,
        "load_question_lookup",
        lambda path, id_field="question_id": jee_records or {
            f"question-{i}": {"question_id": f"question-{i}", "question_text": f"Q{i}"}
            for i in range(jee_count)
        },
    )
    return client, collections


def test_retrieve_returns_allocated_results_and_traceable_fields(monkeypatch):
    _, collections = _patch_retrieval_dependencies(monkeypatch)
    result = retrieve_module.retrieve("Explain Newton's laws of motion", "medium",
                                     "ncert.yaml", "jee.yaml")

    assert result["status"] == "ok"
    assert result["allocation"] == {"ncert": 4, "jee": 4}
    assert len(result["ncert"]) == 4
    assert len(result["jee_questions"]) == 4
    assert result["missing_question_records"] == []
    assert {"unit_id", "distance", "subject", "document"} <= result["ncert"][0].keys()
    assert {"question_id", "distance", "record"} <= result["jee_questions"][0].keys()
    assert collections["ncert_collection"].query_calls[0]["n_results"] == 4
    assert collections["jee_collection"].query_calls[0]["n_results"] == 4


def test_retrieve_caps_k_at_collection_size(monkeypatch):
    _, collections = _patch_retrieval_dependencies(monkeypatch, ncert_count=2, jee_count=3)
    result = retrieve_module.retrieve("Explain Newton's laws of motion", "medium",
                                     "ncert.yaml", "jee.yaml")

    assert len(result["ncert"]) == 2
    assert len(result["jee_questions"]) == 3
    assert collections["ncert_collection"].query_calls[0]["n_results"] == 2
    assert collections["jee_collection"].query_calls[0]["n_results"] == 3


def test_retrieve_does_not_connect_for_unclear_query(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Retrieval dependencies should not be called")

    monkeypatch.setattr(retrieve_module, "load_config", fail_if_called)
    monkeypatch.setattr(retrieve_module, "connect_chroma", fail_if_called)

    result = retrieve_module.retrieve("Thermodynamics", "medium")
    assert result["status"] == "needs_clarification"
    assert result["ncert"] == []
    assert result["jee_questions"] == []


def test_retrieve_reports_missing_canonical_jee_records(monkeypatch):
    _, _ = _patch_retrieval_dependencies(
        monkeypatch,
        jee_count=2,
        jee_records={"question-0": {"question_id": "question-0", "question_text": "Q0"}},
    )
    result = retrieve_module.retrieve("Explain Newton's laws of motion", "easy",
                                     "ncert.yaml", "jee.yaml")

    assert [hit["question_id"] for hit in result["jee_questions"]] == ["question-0"]
    assert result["missing_question_records"] == ["question-1"]
