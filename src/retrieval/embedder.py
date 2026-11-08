"""Embedding providers with a stable interface for retrieval indexing.

Supported providers
-------------------
- sentence-transformers: preserves the existing SentenceTransformer workflow.
- huggingface-transformers: loads encoder models through AutoTokenizer/AutoModel,
  including gated repositories that provide safetensors checkpoints.

Public API retained for the retrieval code:
    load_embedder(config)
    validate_texts(model, texts, ids=None)
    encode_texts(model, texts, config, *, identifiers=None, task="document")
    model_metadata(model)

The HF_TOKEN is read from the repository-root .env (without overriding an
already-set environment variable). Inputs are validated before encoding;
truncation is never silently enabled.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class EmbeddingModel:
    """Provider-neutral container returned by load_embedder."""

    provider: str
    model: Any
    tokenizer: Any
    dimension: int
    max_seq_length: int
    settings: dict[str, Any]


def _repo_root() -> Path:
    # This file is expected at <repository>/src/retrieval/embedder.py.
    return Path(__file__).resolve().parents[2]


def _load_hf_token() -> str | None:
    """Load HF_TOKEN from the repository-root .env, if available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(_repo_root() / ".env", override=False)

    token = os.getenv("HF_TOKEN")
    return token.strip() if token and token.strip() else None


def _effective_device(settings: dict[str, Any]) -> str | None:
    device = settings.get("device")
    return str(device) if device else None


def load_embedder(config: dict[str, Any]) -> EmbeddingModel:
    """Load the configured embedding provider."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    settings = config.get("embedding")
    if not isinstance(settings, dict):
        raise ValueError("Config must contain an 'embedding' mapping")

    model_name = settings.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("embedding.model_name is required")

    revision = settings.get("model_revision")
    if revision:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("embedding.model_revision must be a non-empty string")
    else:
        warnings.warn(
            "embedding.model_revision is unset; results may not be reproducible. "
            "Pin an immutable model revision before final benchmarking.",
            RuntimeWarning,
            stacklevel=2,
        )

    provider = str(settings.get("provider", "sentence-transformers")).lower()
    device = _effective_device(settings)

    if provider in {"sentence-transformers", "sentence_transformers"}:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use this embedding provider."
            ) from exc

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if device:
            kwargs["device"] = device

        st_model = SentenceTransformer(model_name, **kwargs)
        tokenizer = st_model.tokenizer

        configured_dimension = settings.get("embedding_dimension")
        actual_dimension = st_model.get_sentence_embedding_dimension()
        dimension = int(configured_dimension or actual_dimension or 0)
        if not dimension:
            raise RuntimeError(
                f"Could not determine embedding dimension for {model_name}"
            )
        if actual_dimension and dimension != int(actual_dimension):
            raise ValueError(
                f"Configured dimension {dimension} does not match "
                f"SentenceTransformer dimension {actual_dimension}."
            )

        configured_length = settings.get("max_seq_length")
        actual_length = getattr(st_model, "max_seq_length", None)
        max_seq_length = int(configured_length or actual_length or 0)
        if max_seq_length < 1:
            raise RuntimeError(
                "Could not determine max_seq_length for SentenceTransformer."
            )
        st_model.max_seq_length = max_seq_length
        loaded_model: Any = st_model

    elif provider in {
        "huggingface-transformers",
        "huggingface_transformers",
        "transformers",
        "hf",
    }:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install torch and transformers to use the Hugging Face provider."
            ) from exc

        token = _load_hf_token()
        trust_remote_code = bool(settings.get("trust_remote_code", False))

        hf_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if revision:
            hf_kwargs["revision"] = revision
        if token:
            hf_kwargs["token"] = token

        tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)

        # ChEmbed's repository contains model.safetensors rather than
        # pytorch_model.bin. Explicitly select safetensors.
        hf_model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            **hf_kwargs,
        )

        if device:
            target_device = torch.device(device)
        else:
            target_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        hf_model.to(target_device)
        hf_model.eval()

        config_dimension = getattr(hf_model.config, "hidden_size", None)
        if config_dimension is None:
            config_dimension = getattr(hf_model.config, "n_embd", None)

        configured_dimension = settings.get("embedding_dimension")
        dimension = int(configured_dimension or config_dimension or 0)
        if dimension < 1:
            raise ValueError(
                "Set embedding.embedding_dimension; the model config did "
                "not expose a usable hidden size."
            )
        if config_dimension is not None and dimension != int(config_dimension):
            raise ValueError(
                f"Configured embedding_dimension ({dimension}) does not match "
                f"model hidden size ({config_dimension})."
            )

        configured_length = settings.get("max_seq_length")
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        model_limit = getattr(hf_model.config, "max_position_embeddings", None)

        # Ignore Hugging Face's very large sentinel values for unknown limits.
        valid_limits = [
            int(value)
            for value in (configured_length, tokenizer_limit, model_limit)
            if isinstance(value, (int, float))
            and 0 < int(value) < 1_000_000
        ]
        max_seq_length = min(valid_limits) if valid_limits else 256

        pooling = str(settings.get("pooling", "mean")).lower()
        if pooling not in {"mean", "cls"}:
            raise ValueError("embedding.pooling must be 'mean' or 'cls'.")

        # Keep the raw HF model here; encode_texts handles its forward pass.
        loaded_model = hf_model

    else:
        raise ValueError(
            f"Unsupported embedding.provider {provider!r}. Expected "
            "'sentence-transformers' or 'huggingface-transformers'."
        )

    return EmbeddingModel(
        provider=provider,
        model=loaded_model,
        tokenizer=tokenizer,
        dimension=dimension,
        max_seq_length=max_seq_length,
        settings=dict(settings),
    )


def validate_texts(
    model: EmbeddingModel,
    texts: Sequence[str],
    ids: Sequence[str] | None = None,
) -> list[int]:
    """Validate inputs and return token counts without truncation."""
    if not texts:
        raise ValueError("No texts supplied for embedding.")
    if ids is not None and len(ids) != len(texts):
        raise ValueError("IDs/texts length mismatch.")

    counts: list[int] = []
    over_limit: list[tuple[str, int]] = []

    for index, text in enumerate(texts):
        label = str(ids[index]) if ids is not None else f"input index {index}"
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or non-string text for {label}.")

        encoded = model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_count = len(encoded["input_ids"])
        counts.append(token_count)

        if token_count > model.max_seq_length:
            over_limit.append((label, token_count))

    if over_limit:
        examples = "; ".join(
            f"{label}: {count} tokens"
            for label, count in over_limit[:10]
        )
        remaining = len(over_limit) - 10
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        raise ValueError(
            f"{len(over_limit)}/{len(texts)} input(s) exceed the configured "
            f"maximum sequence length of {model.max_seq_length}. "
            f"Examples: {examples}{suffix}. No truncation was performed."
        )

    return counts


def _encode_huggingface(
    model: EmbeddingModel,
    texts: list[str],
    settings: dict[str, Any],
    batch_size: int,
    normalize_embeddings: bool,
    show_progress_bar: bool,
    task: str,
    identifiers: Sequence[str] | None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    # Prefixes are applied only to the HF provider. Empty prefixes are valid.
    if task == "document":
        task_prefix_key = "document_prefix"
    elif task == "query":
        task_prefix_key = "query_prefix"
    else:
        raise ValueError("task must be either 'document' or 'query'.")

    prefix = str(settings.get(task_prefix_key, "") or "")
    prepared_texts = [prefix + text for text in texts]

    # Validate the actual text sent to the model, including any prefix.
    validate_texts(model, prepared_texts, identifiers)

    pooling = str(settings.get("pooling", "mean")).lower()
    outputs: list[np.ndarray] = []
    device = next(model.model.parameters()).device

    for start in range(0, len(prepared_texts), batch_size):
        batch = prepared_texts[start : start + batch_size]
        encoded = model.tokenizer(
            batch,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        try:
            with torch.inference_mode():
                output = model.model(**encoded, return_dict=True)
        except TypeError:
            # Some custom model implementations do not accept return_dict.
            with torch.inference_mode():
                output = model.model(**encoded)

        token_embeddings = getattr(output, "last_hidden_state", None)
        if token_embeddings is None:
            try:
                token_embeddings = output[0]
            except (TypeError, IndexError, KeyError) as exc:
                raise RuntimeError(
                    "The Hugging Face model output does not expose "
                    "last_hidden_state or output[0]."
                ) from exc

        if pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1).to(
                token_embeddings.dtype
            )
            pooled = (token_embeddings * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        elif pooling == "cls":
            pooled = token_embeddings[:, 0, :]
        else:
            raise ValueError("embedding.pooling must be 'mean' or 'cls'.")

        if normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)

        outputs.append(pooled.float().cpu().numpy())

        if show_progress_bar:
            completed = min(start + len(batch), len(prepared_texts))
            print(f"\rEncoded {completed}/{len(prepared_texts)}", end="", flush=True)

    if show_progress_bar:
        print(flush=True)

    return np.concatenate(outputs, axis=0)


def encode_texts(
    model: EmbeddingModel,
    texts: list[str],
    config: dict[str, Any],
    *,
    identifiers: Sequence[str] | None = None,
    task: str = "document",
) -> np.ndarray:
    """Encode texts, preserving the old API and supporting document/query tasks."""
    if not isinstance(texts, list) or not texts:
        raise ValueError("encode_texts expects a non-empty list of texts.")
    if task not in {"document", "query"}:
        raise ValueError("task must be either 'document' or 'query'.")

    settings = config.get("embedding", {})
    if not isinstance(settings, dict):
        raise ValueError("Config must contain an 'embedding' mapping.")

    batch_size = int(settings.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("embedding.batch_size must be at least 1.")

    normalize = bool(settings.get("normalize_embeddings", True))
    show_progress = bool(settings.get("show_progress_bar", True))

    if model.provider in {"sentence-transformers", "sentence_transformers"}:
        validate_texts(model, texts, identifiers)
        vectors = model.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    else:
        # Do not mutate the caller's configuration while selecting a prefix.
        hf_settings = dict(settings)
        prefix_key = "document_prefix" if task == "document" else "query_prefix"
        prefix = str(hf_settings.get(prefix_key, "") or "")
        prepared_texts = [prefix + text for text in texts]

        vectors = _encode_huggingface(
            model=model,
            texts=texts,
            settings=hf_settings,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            task=task,
            identifiers=identifiers,
        )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2-D embedding matrix; received shape {vectors.shape}."
        )
    if vectors.shape != (len(texts), model.dimension):
        raise ValueError(
            f"Expected embedding matrix shape {(len(texts), model.dimension)}, "
            f"received {vectors.shape}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")

    return vectors


def model_metadata(model: EmbeddingModel) -> dict[str, Any]:
    """Return JSON-serializable metadata for the index manifest."""
    settings = model.settings
    return {
        "provider": model.provider,
        "model_name": settings.get("model_name"),
        "model_revision": settings.get("model_revision"),
        "dimension": model.dimension,
        "max_seq_length": model.max_seq_length,
        "normalize_embeddings": bool(
            settings.get("normalize_embeddings", True)
        ),
        "pooling": (
            settings.get("pooling", "sentence-transformers-native")
            if model.provider not in {
                "sentence-transformers",
                "sentence_transformers",
            }
            else "sentence-transformers-native"
        ),
        "document_prefix": settings.get("document_prefix", ""),
        "query_prefix": settings.get("query_prefix", ""),
    }
