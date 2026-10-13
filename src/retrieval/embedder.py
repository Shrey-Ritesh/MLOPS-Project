"""Load a Sentence Transformers model and generate validated embeddings.

This module intentionally refuses to embed over-length inputs. Sentence
Transformers otherwise truncates inputs to the model's maximum sequence length,
which can silently discard part of an NCERT learning unit.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedder(config: dict[str, Any]) -> SentenceTransformer:
    """Load the configured embedding model and report its actual input limit."""
    settings = config.get("embedding")
    if not isinstance(settings, dict):
        raise ValueError("Config must contain an 'embedding' mapping")

    model_name = settings.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Config embedding.model_name is required")

    kwargs: dict[str, Any] = {}
    revision = settings.get("model_revision")
    if revision:
        if not isinstance(revision, str) or revision.strip().lower() in {
            "<commit-hash>", "your-commit-hash", "commit-hash"
        }:
            raise ValueError(
                "embedding.model_revision must be a real Hugging Face revision, "
                "or null for a smoke test. Do not leave a placeholder value."
            )
        kwargs["revision"] = revision
    else:
        warnings.warn(
            "embedding.model_revision is unset; the model revision is not pinned. "
            "Pin an exact Hugging Face commit before the team baseline is frozen.",
            RuntimeWarning,
            stacklevel=2,
        )

    device = settings.get("device")
    if device:
        kwargs["device"] = str(device)

    print(f"Loading embedding model: {model_name}", flush=True)
    model = SentenceTransformer(model_name, **kwargs)

    dimension = model.get_sentence_embedding_dimension()
    max_length = getattr(model, "max_seq_length", None)
    if not dimension:
        raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
    if not isinstance(max_length, int) or max_length < 1:
        raise RuntimeError("Loaded model does not expose a valid max_seq_length")

    print(
        f"Embedding model loaded (dimension={dimension}, max_seq_length={max_length})",
        flush=True,
    )
    return model


def count_tokens(model: SentenceTransformer, text: str) -> int:
    """Count tokenizer tokens including special tokens, without truncation."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The loaded SentenceTransformer model has no accessible tokenizer")
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def validate_text_lengths(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    identifiers: Sequence[str] | None = None,
) -> list[int]:
    """Validate inputs against the model limit and return token counts.

    Raises before encoding if any text exceeds max_seq_length, preventing silent
    truncation. `identifiers`, when supplied, should contain unit IDs.
    """
    if not texts:
        raise ValueError("No texts were provided for embedding")
    if identifiers is not None and len(identifiers) != len(texts):
        raise ValueError("identifiers and texts must have the same length")

    max_length = int(model.max_seq_length)
    counts: list[int] = []
    over_limit: list[tuple[int, str, int]] = []

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            label = identifiers[index] if identifiers is not None else f"input index {index}"
            raise ValueError(f"Empty/non-string embedding text for {label}")
        token_count = count_tokens(model, text)
        counts.append(token_count)
        if token_count > max_length:
            label = identifiers[index] if identifiers is not None else f"input index {index}"
            over_limit.append((index, label, token_count))

    if over_limit:
        examples = "; ".join(
            f"{label}: {count} tokens" for _, label, count in over_limit[:10]
        )
        extra = len(over_limit) - 10
        suffix = f"; plus {extra} more" if extra > 0 else ""
        raise ValueError(
            f"Refusing to embed {len(over_limit)}/{len(texts)} input(s): "
            f"they exceed the model limit of {max_length} tokens. "
            f"Examples: {examples}{suffix}. Reduce/revise these learning units; "
            "do not rely on silent model truncation."
        )

    return counts


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    config: dict[str, Any],
    *,
    identifiers: Sequence[str] | None = None,
) -> np.ndarray:
    """Encode complete texts after validating token lengths and output vectors."""
    if not texts:
        raise ValueError("encode_texts received an empty list")

    settings = config.get("embedding", {})
    token_counts = validate_text_lengths(model, texts, identifiers=identifiers)
    max_length = int(model.max_seq_length)
    print(
        f"Validated {len(texts)} texts (token range: {min(token_counts)}–{max(token_counts)}; "
        f"model limit: {max_length}).",
        flush=True,
    )

    batch_size = int(settings.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("embedding.batch_size must be at least 1")

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=bool(settings.get("show_progress_bar", True)),
        convert_to_numpy=True,
        normalize_embeddings=bool(settings.get("normalize_embeddings", True)),
        # Explicitly disable truncation at the model/tokenizer level where supported.
        # Length validation above should mean no input needs truncation.
    )
    vectors = np.asarray(vectors)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {vectors.shape}")
    if vectors.shape[0] != len(texts):
        raise ValueError(f"Expected {len(texts)} vectors, got {vectors.shape[0]}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")

    expected_dimension = model.get_sentence_embedding_dimension()
    if expected_dimension is not None and vectors.shape[1] != int(expected_dimension):
        raise ValueError(
            f"Expected embedding dimension {expected_dimension}, got {vectors.shape[1]}"
        )

    return vectors
