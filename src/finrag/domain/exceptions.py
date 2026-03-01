"""
finrag.domain.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~
Domain-specific exception hierarchy.

Having typed exceptions means services can catch exactly what they handle,
and the API layer can map each exception to the right HTTP status code.

Hierarchy:
    FinRAGError (base)
    ├── RetrievalError       — vector search / BM25 failures
    ├── GenerationError      — LLM call failures
    ├── CitationError        — answer returned without required citations
    ├── GuardrailTriggered   — safety policy blocked the request
    ├── InsufficientEvidence — not enough context chunks to answer safely
    ├── IndexNotFound        — vector index hasn't been built yet
    └── ConfigurationError   — missing or invalid config at startup
"""

from __future__ import annotations


class FinRAGError(Exception):
    """Base exception for all FinRAG errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class RetrievalError(FinRAGError):
    """Raised when the retrieval layer fails (DB unavailable, timeout, etc.)."""


class GenerationError(FinRAGError):
    """Raised when the LLM fails to produce a response."""


class CitationError(FinRAGError):
    """
    Raised when the LLM returns an answer without the required citations.
    The answer service will attempt one re-generation before raising this.
    """


class GuardrailTriggered(FinRAGError):
    """
    Raised when a safety policy blocks a request.
    Contains the specific rule that was violated.

    Attributes:
        rule: Human-readable name of the triggered rule
              e.g., "investment_advice", "forward_looking_statement"
    """

    def __init__(self, message: str, *, rule: str, details: dict | None = None) -> None:
        super().__init__(message, details=details)
        self.rule = rule


class InsufficientEvidence(FinRAGError):
    """
    Raised when retrieval returns too few or too low-scored chunks
    to generate a trustworthy answer.
    """

    def __init__(
        self,
        message: str = "Insufficient evidence in the corpus to answer this question.",
        *,
        chunks_found: int = 0,
        threshold: float = 0.0,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.chunks_found = chunks_found
        self.threshold = threshold


class IndexNotFound(FinRAGError):
    """Raised when the vector index directory doesn't exist or is empty."""


class ConfigurationError(FinRAGError):
    """Raised for missing or invalid configuration at startup."""
