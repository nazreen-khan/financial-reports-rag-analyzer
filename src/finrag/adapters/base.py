"""
finrag.adapters.base
~~~~~~~~~~~~~~~~~~~~~
Abstract interfaces (contracts) for all external adapters.

Design principle: Services depend on these abstractions, NOT on concrete
implementations. This means:
  - Swapping Chroma → Weaviate = write one new class, touch zero service code
  - Testing services = pass a mock that implements the interface
  - Adding OpenAI → local LLM = implement LLMClientBase with new backend

All concrete adapters live in sibling modules and inherit from these bases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from finrag.domain.models import RetrievedChunk


# ── Vector Store Interface ────────────────────────────────────────────────────

class VectorStoreBase(ABC):
    """Interface for all vector store backends (Chroma, Weaviate, FAISS…)."""

    @abstractmethod
    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """
        Insert or update chunks in the store.

        Args:
            chunks: List of dicts with keys:
                    chunk_id, doc_id, section_id, text, embedding, metadata
        Returns:
            Number of chunks successfully upserted.
        """

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top_k most similar chunks.

        Args:
            embedding: Query embedding vector
            top_k:     Number of results to return
            filters:   Metadata filters (e.g., {"doc_id": "AAPL-2023"})
        Returns:
            List of RetrievedChunk sorted by relevance descending.
        """

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to doc_id.
        Returns number of chunks deleted.
        """

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks in the store."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the store is reachable and responsive."""


# ── LLM Client Interface ──────────────────────────────────────────────────────

class LLMClientBase(ABC):
    """Interface for all LLM backends (OpenAI, local Ollama, llama.cpp…)."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Single-turn completion.

        Args:
            prompt:        User-facing prompt text
            system_prompt: Optional system message
            temperature:   Sampling temperature (0.0 = deterministic)
            max_tokens:    Maximum tokens in the response
        Returns:
            Raw completion text from the model.
        Raises:
            GenerationError on failure.
        """

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Async variant of complete()."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the LLM backend is reachable."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name (for logging)."""


# ── Embedding Interface ───────────────────────────────────────────────────────

class EmbeddingModelBase(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed
        Returns:
            List of embedding vectors (same order as input)
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (may use different pooling)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name (for logging + cache keys)."""


# ── Document Parser Interface ─────────────────────────────────────────────────

class DocumentParserBase(ABC):
    """Interface for document parsers (HTML, PDF, LlamaParse…)."""

    @abstractmethod
    def parse(self, file_path: Path, doc_id: str) -> list[dict[str, Any]]:
        """
        Parse a document into sections.

        Args:
            file_path: Path to the raw document file
            doc_id:    Stable document identifier

        Returns:
            List of section dicts with keys:
            doc_id, section_id, section_title, text, source_url,
            char_start, char_end
        """

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """e.g., 'html_local', 'llamaparse'"""


# ── EDGAR Downloader Interface ────────────────────────────────────────────────

class EDGARDownloaderBase(ABC):
    """Interface for fetching filings from SEC EDGAR."""

    @abstractmethod
    def download_10k(
        self,
        ticker: str,
        year: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Download a 10-K filing for a given ticker and fiscal year.

        Returns:
            Manifest dict: {doc_id, ticker, year, cik, accession,
                           file_path, source_url, sha256}
        Raises:
            httpx.HTTPError on download failure.
        """

    @abstractmethod
    def list_available(self, ticker: str, form: str = "10-K") -> list[dict[str, Any]]:
        """List available filings metadata without downloading."""
