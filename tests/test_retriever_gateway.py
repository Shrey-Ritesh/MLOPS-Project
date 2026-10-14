from src.retrieval.query_gateway import build_query_request
from src.retrieval.retrieve import LEVEL_ALLOCATION, retrieval_plan


def test_easy_newton_routes_to_both_and_preserves_allocation():
    request = build_query_request("Explain Newton's laws of motion", difficulty="easy")
    assert request["intent"] == "EXPLAIN"
    assert retrieval_plan(request, "easy") == {"ncert": 6, "jee": 2}


def test_medium_ideal_real_gas_routes_to_both():
    request = build_query_request(
        "What is the Ideal gas and Real gas equations", difficulty="medium"
    )
    assert request["intent"] == "COMPARE"
    assert retrieval_plan(request, "medium") == {"ncert": 4, "jee": 4}


def test_hard_phenol_alcohol_routes_to_both_and_keeps_secondary_intent():
    request = build_query_request(
        "How are phenols different from alcohols, and why are they more stable",
        difficulty="hard",
    )
    assert request["intent"] == "COMPARE"
    assert "EXPLAIN_WHY" in request["secondary_intents"]
    assert retrieval_plan(request, "hard") == {"ncert": 2, "jee": 6}


def test_practice_only_routes_to_jee_without_reallocating_ncert_quota():
    request = build_query_request("Give me questions on thermodynamics", difficulty="easy")
    assert retrieval_plan(request, "easy") == {"ncert": 0, "jee": 2}


def test_definition_only_routes_to_ncert():
    request = build_query_request("What is a carboxylic acid?", difficulty="medium")
    assert retrieval_plan(request, "medium") == {"ncert": 4, "jee": 0}


def test_unclear_query_does_not_retrieve():
    request = build_query_request("Thermodynamics", difficulty="easy")
    assert retrieval_plan(request, "easy") == {"ncert": 0, "jee": 0}


def test_allocation_baseline_unchanged():
    assert LEVEL_ALLOCATION == {
        "easy": {"ncert": 6, "jee": 2},
        "medium": {"ncert": 4, "jee": 4},
        "hard": {"ncert": 2, "jee": 6},
    }
