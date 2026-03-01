"""
finrag.services.retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~
RetrievalService — coordinates hybrid retrieval (BM25 + vector) over the
indexed financial document corpus.

Today (Day 1): Stub with correct interface + logging.
Day 5: Wired to Chroma vector store + SentenceTransformers.
Day 6: Extended with BM25 fusion + metadata filters + hierarchical indexing.
"""

from __future__ import annotations

from finrag.adapters.base import EmbeddingModelBase, VectorStoreBase
from finrag.core.config import settings
from finrag.core.logging import get_logger
from finrag.core.tracing import get_current_trace
from finrag.domain.exceptions import IndexNotFound, RetrievalError
from finrag.domain.models import RetrievedChunk

log = get_logger(__name__)


class RetrievalService:
    """
    Retrieves relevant chunks for a query using hybrid search.

    Hybrid strategy:
      1. Embed query with the configured embedding model
      2. Vector search (cosine similarity) in Chroma
      3. BM25 keyword search over same corpus
      4. Reciprocal Rank Fusion (RRF) to merge results
      5. Apply metadata filters (ticker, year, section)

    Args:
        vector_store:    A VectorStoreBase implementation (injected)
        embedding_model: An EmbeddingModelBase implementation (injected)
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedding_model: EmbeddingModelBase | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._top_k = settings.retrieval_top_k
        self._score_threshold = settings.retrieval_score_threshold
        self._hybrid_alpha = settings.retrieval_hybrid_alpha

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query:   Natural language question
            top_k:   Override default top_k from settings
            filters: Metadata filters e.g. {"doc_id": "AAPL-2023"}

        Returns:
            List of RetrievedChunk sorted by relevance descending.

        Raises:
            IndexNotFound:  Vector index not built yet
            RetrievalError: Backend failure during search
        """
        effective_top_k = top_k or self._top_k
        trace = get_current_trace()

        log.info(
            "retrieval.start",
            query_preview=query[:100],
            top_k=effective_top_k,
            filters=filters,
            hybrid_alpha=self._hybrid_alpha,
        )

        # ── STUB: will be replaced on Day 5/6 ────────────────────────────────
        # Day 5: wire up self._embedding_model.embed_query(query)
        #         + self._vector_store.query(embedding, top_k, filters)
        # Day 6: add BM25 retriever + RRF fusion + hierarchical routing

        log.warning(
            "retrieval.stub",
            message="RetrievalService is a stub — returning empty results until Day 5",
        )

        chunks: list[RetrievedChunk] = []

        log.info(
            "retrieval.complete",
            chunks_found=len(chunks),
            top_score=chunks[0].score if chunks else 0.0,
        )

        if trace:
            trace.top_k = effective_top_k

        return chunks
