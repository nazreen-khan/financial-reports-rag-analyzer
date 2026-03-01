"""
finrag.services.answer
~~~~~~~~~~~~~~~~~~~~~~~
AnswerService — generates grounded answers from retrieved chunks,
enforcing the citation contract and safety policy.

Today (Day 1): Stub with correct interface + guardrail hook.
Day 7: Full implementation with prompting, citation enforcement, re-generation.
Day 9: Full guardrails integration.
Day 10: Multi-agent routing via LangGraph.
"""

from __future__ import annotations

import time

from finrag.adapters.base import LLMClientBase
from finrag.core.config import settings
from finrag.core.logging import get_logger
from finrag.core.tracing import get_current_trace, get_request_id
from finrag.domain.exceptions import GuardrailTriggered, InsufficientEvidence
from finrag.domain.models import RAGResponse, RefusalReason, RetrievedChunk
from finrag.domain.policy import FinancialQAPolicy
from finrag.services.retrieval import RetrievalService

log = get_logger(__name__)


class AnswerService:
    """
    Orchestrates the full RAG pipeline for a single query:
      1. Safety policy check (pre-retrieval)
      2. Retrieval
      3. Evidence gating (enough chunks? above score threshold?)
      4. Generation with citation enforcement
      5. Post-generation safety check

    Args:
        retrieval_service: Injected RetrievalService
        llm_client:        Injected LLMClientBase
        policy:            Financial QA safety policy (default: production rules)
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_client: LLMClientBase | None = None,
        policy: FinancialQAPolicy | None = None,
    ) -> None:
        self._retrieval = retrieval_service or RetrievalService()
        self._llm = llm_client
        self._policy = policy or FinancialQAPolicy()
        self._min_chunks_required = 1   # Day 9 will tune this

    def answer(
        self,
        query: str,
        *,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: policy → retrieval → generation → response.

        Args:
            query:   User's natural language question
            filters: Optional metadata filters forwarded to retrieval
            top_k:   Override retrieval top_k

        Returns:
            RAGResponse — always returns (never raises to the caller).
            Errors are wrapped as refusals or logged.
        """
        start = time.monotonic()
        request_id = get_request_id()
        trace = get_current_trace()

        log.info("answer.start", query_preview=query[:100])

        # ── Step 1: Pre-retrieval safety check ────────────────────────────────
        if settings.guardrails_enabled:
            evaluation = self._policy.evaluate(query)

            if evaluation.blocked:
                rule = evaluation.first_block
                assert rule is not None  # mypy

                log.warning(
                    "guardrail.blocked",
                    rule_id=rule.rule_id,
                    query_preview=query[:100],
                )

                if trace:
                    trace.refusal = True
                    trace.safety_triggered = True

                return RAGResponse.make_refusal(
                    request_id=request_id,
                    reason=rule.refusal_reason,
                    safety_notes=rule.safe_response,
                    latency_ms=(time.monotonic() - start) * 1000,
                )

            if evaluation.warnings:
                log.warning(
                    "guardrail.warn",
                    warning_rules=evaluation.warning_ids,
                    query_preview=query[:100],
                )

        # ── Step 2: Retrieval ─────────────────────────────────────────────────
        chunks: list[RetrievedChunk] = self._retrieval.retrieve(
            query, top_k=top_k, filters=filters
        )

        # ── Step 3: Evidence gating ───────────────────────────────────────────
        if len(chunks) < self._min_chunks_required:
            log.warning("answer.insufficient_evidence", chunks_found=len(chunks))

            if trace:
                trace.refusal = True

            return RAGResponse.make_refusal(
                request_id=request_id,
                reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                safety_notes=(
                    "I couldn't find sufficient evidence in the indexed filings "
                    "to answer this question reliably. "
                    "Please ensure the relevant documents have been ingested, "
                    "or try rephrasing your question."
                ),
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # ── Step 4: Generation (STUB — wired on Day 7) ────────────────────────
        log.warning(
            "answer.stub",
            message="AnswerService generation is a stub until Day 7",
            chunks_available=len(chunks),
        )

        # Stub returns a safe placeholder with citations from retrieved chunks
        stub_citations = [chunk.to_citation() for chunk in chunks[:3]]

        latency_ms = (time.monotonic() - start) * 1000

        response = RAGResponse(
            request_id=request_id,
            answer_text=(
                "[STUB] Generation not yet implemented. "
                f"Retrieved {len(chunks)} chunks. "
                "This will be replaced with a real grounded answer on Day 7."
            ),
            citations=stub_citations,
            model_used="stub",
            latency_ms=latency_ms,
        )

        log.info("answer.complete", **response.summary())

        return response
