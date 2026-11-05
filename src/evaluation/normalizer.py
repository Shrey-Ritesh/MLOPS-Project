"""Deterministic normalizers and parsers for student answers.

Designed to be generalizable across JEE Chemistry question formats:
- MCQs (option indices, option letters, Roman numerals, compound label+text, verbatim options)
- Numerical values (integers, floats, fractions, scientific notation, thousands separators, units, tolerance)
"""
from __future__ import annotations

import math
import re
from typing import Any

from src.models.jee import JEEOption
from .schemas import QuestionType

_LETTER_TO_INDEX: dict[str, str] = {
    "a": "1",
    "b": "2",
    "c": "3",
    "d": "4",
    "e": "5",
    "f": "6",
}

_ROMAN_TO_INDEX: dict[str, str] = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
}

# Common scientific units in JEE Chemistry to safely strip during numerical parsing
_CHEM_UNITS_REGEX = re.compile(
    r"""(?ix)
    \s*
    (?:
        % |
        \b(?:
            kj\s*/\s*mol | j\s*/\s*mol | kj | j | kcal\s*/\s*mol | kcal |
            g\s*/\s*mol | g\s*/\s*cm\^?3 | kg\s*/\s*m\^?3 |
            g | kg | mg |
            mol\s*/\s*l | mol\s*/\s*dm\^?3 | m | mol | mm | cm | m\^?3 | l | ml |
            k | °\s*c | \^?\\circ\s*c | °\s*f | \^?\\circ\s*f |
            atm | bar | pa | kpa | torr | mm\s*hg |
            s | sec | min | hr | hours? |
            b\.?m\.? | d | pm | nm | \u00c5 | a |
            v | ev | amu
        )\b
    )\s*$
    """
)

# Scientific notation variants like 2.75 x 10^2, 2.75 * 10^-3, 2.75 × 10 ^ 2, 2.75 * 10**2
_SCIENTIFIC_TIMES_REGEX = re.compile(
    r"(?i)([-+]?\d*\.?\d+)\s*(?:[x*×]|\\times)\s*10(?:\^|\*\*|\s*\^\s*)([-+]?\d+)"
)

# Fraction regex: e.g. "1/2", "-3/4", "5 / 2"
_FRACTION_REGEX = re.compile(
    r"^\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*$"
)

# Unicode quote and dash characters to normalize
_QUOTE_CHARS = "'\"`«»“”‘’"
_DASH_TRANSLATION = str.maketrans({
    "\u2212": "-",  # Minus sign
    "\u2013": "-",  # En-dash
    "\u2014": "-",  # Em-dash
    "\u2010": "-",  # Hyphen
    "\u2011": "-",  # Non-breaking hyphen
})


def detect_question_type(options: list[Any] | None) -> QuestionType:
    """Detect whether a question is an MCQ or Numerical based on options."""
    if options and len(options) > 0:
        return QuestionType.MCQ
    return QuestionType.NUMERICAL


def _normalize_text_for_comparison(text: str) -> str:
    """Strip extraneous punctuation, whitespace, and case for fuzzy option matching,
    while strictly preserving critical mathematical and chemical operators (+, -, =, <, >, /)."""
    cleaned = re.sub(r"[^\w\s\+\-\=\<\>\/]", " ", text.casefold())
    return " ".join(cleaned.split())


def _strip_quotes(text: str) -> str:
    """Strip surrounding quotation marks commonly added by frontends or OCR."""
    stripped = text.strip()
    while len(stripped) >= 2 and stripped[0] in _QUOTE_CHARS and stripped[-1] in _QUOTE_CHARS:
        stripped = stripped[1:-1].strip()
    return stripped


def normalize_mcq_answer(
    student_answer: str,
    options: list[dict[str, Any]] | list[JEEOption],
) -> tuple[str | None, str | None]:
    """Normalize student input for a Multiple Choice Question.

    Supports:
    - Pure labels: "1", "2", "3", "4"
    - Letter labels: "A", "B", "c", "d" (mapped to option positions)
    - Roman numerals: "I", "II", "iii", "IV" (mapped to option positions)
    - Formatted labels: "(1)", "Option 2", "1.", "[3]"
    - Compound responses: "2. But-2-ene", "B) Propene", "Option B: But-2-ene", "2\nBut-2-ene"
    - Verbatim option text matching (when student enters the option content)

    Returns:
        (normalized_label, error_message)
    """
    raw = _strip_quotes(student_answer)
    # Normalize internal newlines to spaces for multiline OCR outputs
    raw = " ".join(raw.split())
    if not raw:
        return None, "Answer cannot be empty."

    # Extract available option labels
    available_labels: list[str] = []
    option_dict_by_label: dict[str, str] = {}
    for opt in options:
        label = opt.label if isinstance(opt, JEEOption) else str(opt.get("label", ""))
        text = opt.text if isinstance(opt, JEEOption) else str(opt.get("text", ""))
        available_labels.append(label)
        option_dict_by_label[label] = text

    # Direct match on existing option label (e.g. "1", "2", "3", "4")
    if raw in available_labels:
        return raw, None

    # Case-insensitive direct label match
    for lbl in available_labels:
        if raw.casefold() == lbl.casefold():
            return lbl, None

    # Check for standalone formatted label patterns: "Option 1", "(1)", "1.", "[1]", "Option B"
    label_pattern = re.match(
        r"^(?:option|choice)?\s*[\(\[]?\s*([a-zA-Z0-9]+)\s*[\)\]\.]?$",
        raw,
        re.IGNORECASE,
    )
    if label_pattern:
        extracted = label_pattern.group(1).casefold()
        # Direct match with extracted
        for lbl in available_labels:
            if extracted == lbl.casefold():
                return lbl, None

        # Check letter alias (A, B, C, D)
        if extracted in _LETTER_TO_INDEX:
            mapped_num = _LETTER_TO_INDEX[extracted]
            if mapped_num in available_labels:
                return mapped_num, None
            for lbl in available_labels:
                if lbl.casefold() == extracted:
                    return lbl, None

        # Check Roman numeral alias (I, II, III, IV)
        if extracted in _ROMAN_TO_INDEX:
            mapped_num = _ROMAN_TO_INDEX[extracted]
            if mapped_num in available_labels:
                return mapped_num, None

    # If student typed the full or partial option text, attempt verbatim match first
    normalized_input_text = _normalize_text_for_comparison(raw)
    matching_labels: list[str] = []
    for lbl, opt_text in option_dict_by_label.items():
        if _normalize_text_for_comparison(opt_text) == normalized_input_text:
            matching_labels.append(lbl)

    if len(matching_labels) == 1:
        return matching_labels[0], None
    elif len(matching_labels) > 1:
        return None, "Input is ambiguous as it matches multiple option texts."

    # Check for compound responses: "2. But-2-ene", "Option B: But-2-ene", "B) Propene", "2 But-2-ene"
    compound_pattern = re.match(
        r"^(?:option|choice)?\s*[\(\[]?\s*([a-zA-Z0-9]+)\s*[\)\]\.\:\-]?\s+(.+)$",
        raw,
        re.IGNORECASE,
    )
    if compound_pattern:
        prefix_token = compound_pattern.group(1).casefold()
        candidate_label: str | None = None
        for lbl in available_labels:
            if prefix_token == lbl.casefold():
                candidate_label = lbl
                break
        if candidate_label is None and prefix_token in _LETTER_TO_INDEX:
            mapped_num = _LETTER_TO_INDEX[prefix_token]
            if mapped_num in available_labels:
                candidate_label = mapped_num
        if candidate_label is None and prefix_token in _ROMAN_TO_INDEX:
            mapped_num = _ROMAN_TO_INDEX[prefix_token]
            if mapped_num in available_labels:
                candidate_label = mapped_num

        if candidate_label is not None:
            # Check if the remainder text matches the option text for this label
            remainder = compound_pattern.group(2).strip()
            opt_text = option_dict_by_label.get(candidate_label, "")
            if _normalize_text_for_comparison(remainder) == _normalize_text_for_comparison(opt_text):
                return candidate_label, None
            # Even if slightly different, if delimiter was explicit (., :, -, )), trust the prefix
            if re.search(r"[\)\]\.\:\-]", raw[:len(prefix_token) + 3]):
                return candidate_label, None

    return None, f"Input '{raw}' does not match any valid option in {available_labels}."


def parse_numerical_answer(student_answer: str) -> tuple[float | None, str | None]:
    """Parse a numerical answer string into a float.

    Handles:
    - Standard integers and floats: "275", "-4.5", "0.001"
    - Scientific notation: "2.75e2", "2.75E-3", "2.75 x 10^2", "2.75 * 10**2"
    - Thousands separator commas: "2,850" -> 2850.0
    - Fractions: "1/2" -> 0.5, "5 / 2" -> 2.5
    - Stripping trailing measurement units: "275 %", "6 kJ/mol", "60 s"
    - Unicode dash/minus signs: "−40", "–40" -> -40.0
    - Enclosing quotation marks: '"275"' -> 275.0

    Returns:
        (parsed_float, error_message)
    """
    raw = _strip_quotes(student_answer)
    if not raw:
        return None, "Numerical answer cannot be empty."

    # Normalize unicode dashes/minuses
    cleaned = raw.translate(_DASH_TRANSLATION)

    # Strip measurement units if present
    cleaned = _CHEM_UNITS_REGEX.sub("", cleaned).strip()

    # Check for simple fraction format: "1/2", "3/4", "-5 / 2"
    fraction_match = _FRACTION_REGEX.match(cleaned)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator == 0:
            return None, "Division by zero in fraction."
        return float(numerator) / float(denominator), None

    # Standardize scientific times: 2.75 x 10^2, 2.75 * 10**2, 2.75 x 10 ^ 2 -> 2.75e2
    cleaned = _SCIENTIFIC_TIMES_REGEX.sub(r"\1e\2", cleaned).strip()

    # Strip thousands separator commas (e.g. 2,850 -> 2850)
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)

    try:
        val = float(cleaned)
        if math.isnan(val) or math.isinf(val):
            return None, "Numerical value must be finite."
        return val, None
    except ValueError:
        return None, f"Could not parse '{raw}' as a valid numerical value."


def compare_numerical(
    student_val: float,
    key_val: float,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare student numerical answer against key with relative and absolute tolerances.

    Uses 1% relative tolerance by default (standard for JEE numerical rounding).
    """
    return math.isclose(student_val, key_val, rel_tol=rel_tol, abs_tol=abs_tol)
