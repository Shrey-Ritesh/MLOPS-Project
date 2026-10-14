from src.retrieval.query_gateway import build_query_request, classify_intent


def test_easy_newton_query():
    result = build_query_request("Explain Newton's laws of motion", difficulty="easy")
    assert result["intent"] == "EXPLAIN"
    assert result["subject"] == "physics"
    assert result["retrieval_targets"] == ["ncert", "jee"]
    assert result["difficulty"] == "easy"


def test_medium_ideal_real_gas_query():
    result = build_query_request("What is the Ideal gas and Real gas equations", difficulty="medium")
    assert result["intent"] == "COMPARE"
    assert "cover_both_concepts" in result["constraints"]
    assert result["subject"] == "chemistry"


def test_hard_phenol_alcohol_query():
    result = build_query_request("How are phenols different from alcohols, and why are they more stable", difficulty="hard")
    assert result["intent"] == "COMPARE"
    assert "EXPLAIN_WHY" in result["secondary_intents"]
    assert "prioritize_causal_explanation" in result["constraints"]
    assert result["subject"] == "chemistry"


def test_practice_routes_to_jee():
    result = build_query_request("Give me questions on thermodynamics")
    assert result["intent"] == "PRACTICE"
    assert result["retrieval_targets"] == ["jee"]

def test_definition_queries():
    result = build_query_request("What is a carboxylic acid?")
    assert result["intent"] == "DEFINE"

def test_summary_queries():
    result = build_query_request("Can you revise the laws of thermodynamics in Chemistry")
    assert result["intent"] == "SUMMARIZE"



def test_blank_is_unclear():
    result = build_query_request("   ")
    assert result["intent"] == "UNCLEAR"
    assert result["needs_clarification"] is True


def test_multiple_intents_preserved():
    primary, secondary = classify_intent("Compare alcohols and phenols and explain why phenols are more acidic")
    assert primary == "COMPARE"
    assert "EXPLAIN_WHY" in secondary
