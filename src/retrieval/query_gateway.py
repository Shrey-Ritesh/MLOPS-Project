"""Deterministic student-query gateway for the JEE tutor; no external dependencies."""
from __future__ import annotations
import argparse
import json
import re
from typing import Any

_PATTERNS = {
    "SOLVE": re.compile(r"\b(solve|work out|calculate|find the value|answer this|solution to|solve this|step by step solution)\b", re.I),
    "PRACTICE": re.compile(r"\b(practice|quiz me|test me|give me questions|questions on|problems on|mcqs?|mock test|worksheet)\b", re.I),
    "COMPARE": re.compile(r"\b(compare|comparison|difference(?:s)? between|differentiate|distinguish between|similarities between|versus|vs\.?)\b", re.I),
    "EXPLAIN_WHY": re.compile(r"\b(why|how does|how do|how is|how are|mechanism|reason for|explain the reason|cause of)\b", re.I),
    "SUMMARIZE": re.compile(r"\b(summar(?:y|ise|ize)|revise|revision|quick notes|key points|recap|in brief)\b", re.I),
    "DEFINE": re.compile(r"\b(define|definition of|what is|what are|meaning of)\b", re.I),
    "EXPLAIN": re.compile(r"\b(explain|teach me|help me understand|describe|learn about)\b", re.I),
}
_PRIORITY = ("SOLVE", "PRACTICE", "COMPARE", "EXPLAIN_WHY", "SUMMARIZE", "DEFINE", "EXPLAIN")
_COMPARISON_PAIRS = (("ideal gas", "real gas"), ("phenol", "alcohol"), ("mitosis", "meiosis"), ("series", "parallel"), ("acid", "base"), ("speed", "velocity"), ("mass", "weight"), ("mitochondria", "chloroplast"))


def _clean_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return re.sub(r"\s+([,.;?!])", r"\1", re.sub(r"\s+", " ", query).strip())


def _has_implicit_comparison(query: str) -> bool:
    q = query.casefold()
    return bool(re.search(r"\band\b", q) and any(a in q and b in q for a, b in _COMPARISON_PAIRS))



def classify_intent(query: str) -> tuple[str, list[str]]:
    """Return primary and secondary intents, preserving genuine multi-intent queries."""
    q = _clean_query(query)
    if not q:
        return "UNCLEAR", []

    found = {
        name
        for name, pattern in _PATTERNS.items()
        if pattern.search(q)
    }

    # Explicit comparison wording takes precedence over generic "how does".
    explicit_comparison = bool(
        re.search(
            r"\b(compare|comparison|difference(?:s)?\s+between|"
            r"differ(?:s|ed|ing)?\s+from|different\s+from|"
            r"differentiate|distinguish\s+between|"
            r"similarities\s+between|versus|vs\.?)\b",
            q,
            re.I,
        )
    )

    if explicit_comparison or _has_implicit_comparison(q):
        found.add("COMPARE")

    # "How does A differ from B?" is a comparison, not a why-question.
    # Preserve EXPLAIN_WHY when the user independently asks why or asks
    # for a cause/mechanism in addition to the comparison.
    asks_cause_explicitly = bool(
        re.search(
            r"\b(why|mechanism|reason for|cause of|"
            r"explain the reason)\b",
            q,
            re.I,
        )
    )
    if explicit_comparison and not asks_cause_explicitly:
        found.discard("EXPLAIN_WHY")

    primary = next(
        (name for name in _PRIORITY if name in found),
        "UNCLEAR",
    )
    secondary = [
        name for name in _PRIORITY
        if name in found and name != primary
    ]
    return primary, secondary

def _subject_hints(query: str) -> str | None:
    q = query.casefold()
    vocab = {
        "chemistry": ("chemistry", "phenol", "alcohol", "ideal gas", "real gas", "chemical", "molecule", "acid", "base", "thermodynamic", "organic chemistry", "bonding"),
        "physics": ("physics", "newton", "force", "motion", "velocity", "acceleration", "work energy", "kinematics", "electric", "magnetic", "wave", "optics"),
        "mathematics": ("maths", "integral", "derivative", "differentiation", "probability", "matrix", "determinant", "limit", "permutation", "combination"),
    }

    return [
        subject
        for subject, terms in vocab.items()
        if any(re.search(rf"\b{re.escape(term)}\b", q) for term in terms)
    ]


def _topic_hint(query: str) -> str | None:
    topic = query.strip(" .?!")
    prefixes = (
        r"^(please\s+)?(can you\s+)?(help me understand|teach me|explain|describe)\s+",
        r"^(please\s+)?(why|how(?:\s+(?:does|do|is|are))?)\s+",
        r"^(please\s+)?(what is|what are|define|definition of|meaning of)\s+",
        r"^(please\s+)?(compare|differentiate|distinguish between|difference between|differences between)\s+",
        r"^(please\s+)?(give me|show me)\s+(some\s+)?(practice\s+)?(questions|problems)\s+(on|about)\s+",
        r"^(please\s+)?(summarise|summarize|revise|recap|quick revision of)\s+",
    )
    for pattern in prefixes:
        topic = re.sub(pattern, "", topic, flags=re.I).strip()
    topic = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.I)
    return topic.strip(" ,:;.") or None


def build_query_request(query: str, *, difficulty: str | None = None) -> dict[str, Any]:
    """Create the structured gateway result. Difficulty is passed through, never inferred."""
    cleaned = _clean_query(query)
    intent, secondary = classify_intent(cleaned)
    if intent == "UNCLEAR":
        targets = []
    elif intent == "PRACTICE":
        targets = ["jee"]
    elif intent == "DEFINE":
        targets = ["ncert"]
    elif intent == "SOLVE":
        targets = ["ncert", "jee"]
    else:
        targets = ["ncert", "jee"]

    all_intents = {intent, *secondary}
    constraints = []
    if "COMPARE" in all_intents:
        constraints.append("cover_both_concepts")
    if "EXPLAIN_WHY" in all_intents:
        constraints.append("prioritize_causal_explanation")
    if intent == "PRACTICE":
        constraints.append("prefer_jee_questions")

    return {
        "original_query": query,
        "normalized_query": cleaned,
        "intent": intent,
        "secondary_intents": secondary,
        "subjects": _subject_hints(cleaned),
        "topic": _topic_hint(cleaned),
        "constraints": constraints,
        "retrieval_targets": targets,
        "difficulty": difficulty,
        "needs_clarification": intent == "UNCLEAR" or len(cleaned.split()) <= 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and normalize a student query.")
    parser.add_argument("query", help="natural-language student query")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    args = parser.parse_args()
    print(json.dumps(build_query_request(args.query, difficulty=args.difficulty), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
