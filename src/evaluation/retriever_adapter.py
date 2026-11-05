"""Adapter for fetching grounded NCERT evidence from the Chroma retrieval system.

Provides robust, safe retrieval integration that:
- Queries the project's NCERT Chroma index via src.retrieval.retrieve
- Formats excerpts and provenance into EvidenceItem contract models
- Gracefully handles offline vector stores, missing connections, or empty hits
  without crashing or blocking the evaluation workflow.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import Any, Callable

from .schemas import EvidenceItem, EvidenceStatus

logger = logging.getLogger(__name__)


def _is_chroma_reachable(host: str = "localhost", port: int = 8000, timeout: float = 0.2) -> bool:
    """Fast non-blocking probe to check if Chroma service is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class NCERTRetrieverAdapter:
    """Adapter decoupling the evaluation engine from the vector store implementation."""

    def __init__(
        self,
        retrieve_fn: Callable[..., dict[str, Any]] | None = None,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._chroma_host = chroma_host or os.environ.get("CHROMA_HOST", "localhost")
        self._chroma_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))

    def _get_retrieve_fn(self) -> Callable[..., dict[str, Any]]:
        if self._retrieve_fn is not None:
            return self._retrieve_fn
        from src.retrieval.retrieve import retrieve
        return retrieve

    def fetch_evidence(
        self,
        query: str,
        subject: str = "Chemistry",
        level: str = "medium",
        top_k: int = 3,
    ) -> tuple[list[EvidenceItem], EvidenceStatus]:
        """Fetch NCERT learning units relevant to the query topic.

        Returns:
            (evidence_items, evidence_status)
        """
        clean_query = query.strip()
        if not clean_query:
            return [], EvidenceStatus.INSUFFICIENT_EVIDENCE

        # If using real retrieve_fn (not a mock), fast probe vector store first
        if self._retrieve_fn is None and not _is_chroma_reachable(self._chroma_host, self._chroma_port):
            logger.info("Chroma server at %s:%d is offline; skipping retrieval.", self._chroma_host, self._chroma_port)
            return [], EvidenceStatus.RETRIEVAL_UNAVAILABLE

        try:
            retrieve_fn = self._get_retrieve_fn()
            result = retrieve_fn(clean_query, level=level, subject=subject)
        except Exception as exc:
            logger.warning(
                "Retrieval failed or vector database unreachable for query %r: %s",
                clean_query[:50],
                exc,
            )
            return [], EvidenceStatus.RETRIEVAL_UNAVAILABLE

        if not isinstance(result, dict) or result.get("status") not in {"ok", "success"}:
            # If query gateway marked query as needs_clarification or status not ok
            return [], EvidenceStatus.INSUFFICIENT_EVIDENCE

        raw_ncert_hits = result.get("ncert", [])
        if not isinstance(raw_ncert_hits, list) or not raw_ncert_hits:
            return [], EvidenceStatus.INSUFFICIENT_EVIDENCE

        evidence_items: list[EvidenceItem] = []
        for index, hit in enumerate(raw_ncert_hits[:top_k], start=1):
            if not isinstance(hit, dict):
                continue

            metadata = hit.get("metadata") or {}
            unit_id = str(hit.get("id") or metadata.get("unit_id") or f"ncert-unit-{index}").strip()
            chapter_title = (
                metadata.get("chapter_title")
                or hit.get("chapter_title")
                or hit.get("unit_title")
                or "NCERT Chemistry"
            )
            
            # Extract and validate page
            raw_page = metadata.get("page") or hit.get("page")
            page: int | None = None
            if raw_page is not None:
                try:
                    parsed_page = int(raw_page)
                    if parsed_page >= 1:
                        page = parsed_page
                except (ValueError, TypeError):
                    page = None

            doc_text = str(hit.get("document") or hit.get("text") or "").strip()
            if not doc_text:
                continue

            # Safely bound excerpt length
            excerpt = doc_text[:400].strip()

            # Bound relevance score if present
            raw_score = hit.get("score")
            score: float | None = None
            if raw_score is not None:
                try:
                    val = float(raw_score)
                    score = max(0.0, min(1.0, val))
                except (ValueError, TypeError):
                    score = None

            try:
                item = EvidenceItem(
                    unit_id=unit_id,
                    chapter_title=str(chapter_title),
                    page=page,
                    excerpt=excerpt,
                    relevance_score=score,
                )
                evidence_items.append(item)
            except Exception as exc:
                logger.warning("Failed to create EvidenceItem from hit: %s", exc)

        status = EvidenceStatus.GROUNDED if evidence_items else EvidenceStatus.INSUFFICIENT_EVIDENCE
        return evidence_items, status
