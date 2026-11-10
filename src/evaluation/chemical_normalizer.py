"""Chemical formula and notation normalizer for JEE Chemistry.

Handles:
- Unicode subscripts: H₂O -> H2O, SO₄²⁻ -> SO4^{2-}
- Unicode superscripts & charges: Fe³⁺ -> Fe^{3+}, NO₂⁻ -> NO2^{-}
- Standardizes charges only when explicitly marked with ^ or at end of species
- Single bonds in organic chains: CH3-CH2-OH vs CH3CH2OH
- State symbols: (s), (l), (g), (aq)
- Free-text narrative detection to safeguard against guessing on discursive responses
"""
from __future__ import annotations

import re

# Unicode translation table for subscripts (stoichiometry numbers)
_SUB_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Unicode superscripts
_SUPER_CHARS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
}

_STATE_SYMBOLS_REGEX = re.compile(r"\s*\((?:s|l|g|aq|solid|liquid|gas|aqueous)\)\s*", re.IGNORECASE)

# Discursive keywords indicating the student typed an open-ended explanation
_DISCURSIVE_WORDS = {
    "because", "therefore", "since", "due to", "increases", "decreases",
    "depends", "causes", "reason", "mechanism", "entropy", "enthalpy",
    "energy", "orbital", "hybridization", "geometry", "reaction", "structure",
    "greater", "lesser", "higher", "lower", "stable", "stability", "explanation",
}


def is_discursive_free_text(text: str) -> bool:
    """Detect if an input is an open-ended narrative sentence rather than a formula or option."""
    raw = text.strip()
    words = re.findall(r"[a-zA-Z]+", raw.casefold())

    if len(words) >= 5:
        discursive_matches = set(words) & _DISCURSIVE_WORDS
        if len(discursive_matches) >= 1 or len(words) >= 7:
            return True

    low = raw.casefold()
    if low.startswith(("i think", "the answer is because", "it is because", "due to", "as per", "we know that")):
        return True

    return False


def _convert_unicode_super_and_sub(text: str) -> str:
    """Convert unicode subscripts to numbers and superscripts to explicit ^{...} markers."""
    # Convert subscripts first (these are stoichiometry numbers)
    result = text.translate(_SUB_TRANS)

    # Convert consecutive superscripts to ^{...}
    def _super_repl(match: re.Match[str]) -> str:
        s = match.group(0)
        translated = "".join(_SUPER_CHARS.get(c, c) for c in s)
        return f"^{{{translated}}}"

    result = re.sub(r"[\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u207a\u207b]+", _super_repl, result)
    return result


def normalize_chemical_formula(raw_formula: str, strip_state: bool = True) -> str:
    """Normalize a chemical formula string to a canonical representation.

    Examples:
        "H₂O" -> "H2O"
        "Fe³⁺" -> "Fe^{3+}"
        "Fe^3+" -> "Fe^{3+}"
        "Fe3+" -> "Fe^{3+}"
        "SO₄²⁻" -> "SO4^{2-}"
        "CH3-CH2-OH" -> "CH3CH2OH"
        "CO2 (g)" -> "CO2"
    """
    cleaned = raw_formula.strip()
    if not cleaned:
        return ""

    # Convert unicode superscripts and subscripts
    cleaned = _convert_unicode_super_and_sub(cleaned)

    # Strip state symbols if requested
    if strip_state:
        cleaned = _STATE_SYMBOLS_REGEX.sub("", cleaned).strip()

    # Standardize reaction arrows
    cleaned = cleaned.replace(r"\rightarrow", "->").replace(r"\to", "->").replace("→", "->")

    # Standardize explicit charges: e.g. Fe^3+, Fe^+3, Fe^{3+}, Fe^{3-}
    def _standardize_explicit_charge(m: re.Match[str]) -> str:
        content = m.group(1).strip()
        # content could be 3+, +3, 2-, -, +
        sign_match = re.search(r"[\+\-]", content)
        if not sign_match:
            return f"^{{{content}}}"
        sign = sign_match.group(0)
        num = re.sub(r"[\+\-]", "", content).strip()
        mag = num if num else "1"
        return f"^{{{mag}{sign}}}"

    cleaned = re.sub(r"\^\{?([0-9]*[\+\-][0-9]*)\}?", _standardize_explicit_charge, cleaned)

    # Handle repeated signs at the end: Fe+++ -> Fe^{3+}, Fe++ -> Fe^{2+}, O-- -> O^{2-}
    cleaned = re.sub(r"([A-Za-z0-9\]\)])\+{3}$", r"\1^{3+}", cleaned)
    cleaned = re.sub(r"([A-Za-z0-9\]\)])\+{2}$", r"\1^{2+}", cleaned)
    cleaned = re.sub(r"([A-Za-z0-9\]\)])\-{3}$", r"\1^{3-}", cleaned)
    cleaned = re.sub(r"([A-Za-z0-9\]\)])\-{2}$", r"\1^{2-}", cleaned)

    # Handle trailing charges without caret ONLY at the very end of species: Fe3+ -> Fe^{3+}
    cleaned = re.sub(r"([A-Za-z\]\)])(\d+)?([\+\-])$", r"\1^{\2\3}", cleaned)

    # Normalize ^{1+} -> ^{+} and ^{1-} -> ^{-}
    cleaned = cleaned.replace("^{1+}", "^{+}").replace("^{1-}", "^{-}")

    # Remove internal whitespace
    cleaned = re.sub(r"\s+", "", cleaned)

    return cleaned


def compare_chemical_formulas(student_formula: str, target_formula: str) -> bool:
    """Compare student chemical formula with expected formula with structural normalization."""
    norm_student = normalize_chemical_formula(student_formula)
    norm_target = normalize_chemical_formula(target_formula)

    if norm_student == norm_target:
        return True

    # Check condensed vs bonded organic formula: CH3-CH2-OH vs CH3CH2OH
    unbonded_student = norm_student.replace("-", "")
    unbonded_target = norm_target.replace("-", "")
    if unbonded_student == unbonded_target:
        return True

    # Case-insensitive comparison fallback
    if norm_student.casefold() == norm_target.casefold():
        return True

    return False
