"""Deterministic, conservative text cleaning for extracted NCERT text.

Cleaning never rewrites mathematical/chemical notation, signs or labels. The raw
text is always kept alongside, so any cleaning decision can be revisited.
"""

import re

_SPACES_RE = re.compile(r"[ \t]{2,}")
_HYPHEN_BREAK_RE = re.compile(r"-\n(?=[a-z])")
_PUA_RE = re.compile("[-�]")

# Block types whose internal line breaks carry meaning and are kept.
_KEEP_LINES = {"equation", "table", "list"}


def clean_block_text(raw: str | None, block_type: str) -> str | None:
    """Clean one block's text; return None when there is no text content."""
    if raw is None:
        return None

    text = raw.replace("\xad", "").replace("\xa0", " ").replace("\r", "")
    if block_type in _KEEP_LINES:
        lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
    else:
        # Keep the hyphen of a hyphen-broken word ("well-\nknown" -> "well-known")
        # rather than guess whether it was a real compound.
        text = _HYPHEN_BREAK_RE.sub("-", text)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = _SPACES_RE.sub(" ", text).strip()

    return text or None


def private_use_ratio(text: str) -> float:
    """Fraction of characters that are private-use/replacement glyphs (likely garbled)."""
    if not text:
        return 0.0
    return len(_PUA_RE.findall(text)) / len(text)


def boilerplate_key(text: str) -> str:
    """Normalise a header/footer string so page numbers collapse to one key."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


# Symbol-font byte -> Unicode, for the Greek letters and operators NCERT physics uses.
_SYMBOL_MAP = {
    **{0x41 + i: c for i, c in enumerate("ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ")},
    **{0x61 + i: c for i, c in enumerate("αβχδεφγηιϕκλμνοπθρστυϖωξψζ")},
    0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓",
    0xB0: "°", 0xB1: "±", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•",
    0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xD0: "∠", 0xD1: "∇", 0xD6: "√",
    0xD7: "⋅", 0xB2: "″", 0xA2: "′", 0xE5: "∑", 0xF2: "∫", 0x2D: "−", 0x2A: "∗",
}
_PUA_BYTE_RE = re.compile("[\uf020-\uf0ff]")


def decode_private_use(text: str, symbol_font: bool = False) -> str:
    """Decode U+F0xx "symbol-encoded" characters back to the byte they stand for.

    Some PDFs store text as 0xF000 + byte. Text fonts decode as cp1252; Symbol-style
    fonts decode through the Greek/operator table (unknown codes are left as is).
    """
    def repl(m: re.Match) -> str:
        byte = ord(m.group()) - 0xF000
        if symbol_font and byte in _SYMBOL_MAP:
            return _SYMBOL_MAP[byte]
        if symbol_font and byte >= 0x80:
            return m.group()
        try:
            return bytes([byte]).decode("cp1252")
        except UnicodeDecodeError:
            return m.group()

    return _PUA_BYTE_RE.sub(repl, text)
