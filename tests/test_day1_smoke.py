"""
tests/test_day1_smoke.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Day 1 smoke tests — validates that the entire package skeleton is importable,
config loads, and domain contracts work correctly.

These are the "it didn't break overnight" tests that CI runs on every push.
All should pass without any external services (no LLM, no vector DB, no EDGAR).
"""

from __future__ import annotations

import pytest


# ── Import Tests ──────────────────────────────────────────────────────────────

class TestImports:
    """Every module must be importable without circular imports or missing deps."""

    def test_import_core_config(self) -> None:
        from finrag.core.config import Settings, get_settings
        assert Settings is not None
        assert get_settings is not None

    def test_import_core_logging(self) -> None:
        from finrag.core.logging import get_logger, setup_logging
        assert get_logger is not None

    def test_import_core_tracing(self) -> None:
        from finrag.core.tracing import TraceContext, new_trace, get_request_id
        assert TraceContext is not None

    def test_import_domain_models(self) -> None:
        from finrag.domain.models import (
            Citation, ExtractedFact, RAGResponse,
            RetrievedChunk, RefusalReason, FactType
        )
        assert RAGResponse is not None

    def test_import_domain_exceptions(self) -> None:
        from finrag.domain.exceptions import (
            FinRAGError, GuardrailTriggered, InsufficientEvidence,
            RetrievalError, GenerationError, CitationError
        )
        assert FinRAGError is not None

    def test_import_domain_policy(self) -> None:
        from finrag.domain.policy import FinancialQAPolicy, FINANCIAL_QA_POLICY
        assert len(FINANCIAL_QA_POLICY) > 0

    def test_import_adapters_base(self) -> None:
        from finrag.adapters.base import (
            VectorStoreBase, LLMClientBase,
            EmbeddingModelBase, DocumentParserBase
        )
        assert VectorStoreBase is not None

    def test_import_services(self) -> None:
        from finrag.services.retrieval import RetrievalService
        from finrag.services.answer import AnswerService
        assert RetrievalService is not None
        assert AnswerService is not None

    def test_import_agents(self) -> None:
        from finrag.agents.graph import GraphState, build_graph
        assert build_graph is not None

    def test_import_eval(self) -> None:
        from finrag.eval.metrics import (
            GoldEntry, EvalReport, recall_at_k, reciprocal_rank
        )
        assert EvalReport is not None


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    """Config must load with defaults and validate correctly."""

    def test_settings_load_with_defaults(self) -> None:
        from finrag.core.config import get_settings
        get_settings.cache_clear()
        s = get_settings()
        assert s.retrieval_top_k == 6
        assert s.retrieval_hybrid_alpha == 0.5
        assert s.guardrails_enabled is True
        assert s.chroma_collection_name == "finrag_chunks"

    def test_settings_derived_properties(self) -> None:
        from finrag.core.config import get_settings
        s = get_settings()
        # No API key set → openai_available is False
        assert isinstance(s.is_local, bool)
        assert isinstance(s.is_production, bool)
        assert isinstance(s.openai_available, bool)
        assert isinstance(s.llamaparse_available, bool)

    def test_hybrid_alpha_bounds(self) -> None:
        """Alpha must be in [0.0, 1.0] — pydantic should enforce this."""
        from finrag.core.config import Settings
        with pytest.raises(Exception):
            Settings(retrieval_hybrid_alpha=1.5)


# ── Tracing Tests ─────────────────────────────────────────────────────────────

class TestTracing:
    """Trace context must be set, accessible, and cleaned up correctly."""

    def test_new_trace_creates_ulid(self) -> None:
        from finrag.core.tracing import new_trace, get_request_id
        with new_trace(query="test query") as ctx:
            assert len(ctx.request_id) == 26  # ULID length
            assert get_request_id() == ctx.request_id

    def test_request_id_is_no_trace_outside_context(self) -> None:
        from finrag.core.tracing import get_request_id
        assert get_request_id() == "no-trace"

    def test_trace_elapsed_ms_increases(self) -> None:
        import time
        from finrag.core.tracing import new_trace
        with new_trace(query="timing test") as ctx:
            time.sleep(0.05)   # 50ms — well above Windows 15ms timer resolution
            assert ctx.elapsed_ms >= 1   # proves elapsed_ms is non-zero and ticking

    def test_trace_agent_recording(self) -> None:
        from finrag.core.tracing import new_trace
        from finrag.domain.models import AgentRole
        with new_trace(query="agent test") as ctx:
            ctx.record_agent(AgentRole.ROUTER.value)
            ctx.record_agent(AgentRole.RISK_ANALYST.value)
            assert ctx.agent_path == ["router", "risk_analyst"]

    def test_trace_context_restored_after_exit(self) -> None:
        from finrag.core.tracing import new_trace, get_request_id
        with new_trace(query="outer"):
            with new_trace(query="inner") as inner:
                assert get_request_id() == inner.request_id
            # After inner exits, outer is restored — not "no-trace"
        assert get_request_id() == "no-trace"


# ── Domain Model Tests ────────────────────────────────────────────────────────

class TestDomainModels:
    """Pydantic contracts must validate correctly and enforce invariants."""

    def test_citation_requires_non_empty_ids(self) -> None:
        from finrag.domain.models import Citation
        with pytest.raises(Exception):
            Citation(chunk_id="", doc_id="AAPL-2023", section_id="item_7")

    def test_retrieved_chunk_to_citation(self) -> None:
        from finrag.domain.models import RetrievedChunk
        chunk = RetrievedChunk(
            chunk_id="chunk-001",
            doc_id="AAPL-2023",
            section_id="item_7",
            section_title="MD&A",
            text="Revenue was $394.3 billion in fiscal 2023.",
            source_url="https://www.sec.gov/test",
            score=0.92,
        )
        citation = chunk.to_citation()
        assert citation.chunk_id == "chunk-001"
        assert citation.text_preview == "Revenue was $394.3 billion in fiscal 2023."
        assert citation.relevance_score == 0.92

    def test_rag_response_requires_citations_when_not_refusal(self) -> None:
        from finrag.domain.models import RAGResponse
        with pytest.raises(Exception, match="citation"):
            RAGResponse(
                request_id="01ABCDE",
                answer_text="Apple's revenue was $394 billion.",
                citations=[],   # ← violates contract
                refusal=False,
            )

    def test_rag_response_refusal_requires_safety_notes(self) -> None:
        from finrag.domain.models import RAGResponse, RefusalReason
        with pytest.raises(Exception, match="safety_notes"):
            RAGResponse(
                request_id="01ABCDE",
                answer_text="I can't help with that.",
                citations=[],
                refusal=True,
                refusal_reason=RefusalReason.INVESTMENT_ADVICE,
                safety_notes="",   # ← violates contract
            )

    def test_make_refusal_factory(self) -> None:
        from finrag.domain.models import RAGResponse, RefusalReason
        response = RAGResponse.make_refusal(
            request_id="01ABCDE",
            reason=RefusalReason.INVESTMENT_ADVICE,
            safety_notes="I cannot provide investment advice.",
            model_used="gpt-4o-mini",
        )
        assert response.refusal is True
        assert response.citations == []
        assert "investment advice" in response.safety_notes.lower()

    def test_rag_response_summary(self) -> None:
        from finrag.domain.models import RAGResponse, RefusalReason
        r = RAGResponse.make_refusal(
            request_id="01TEST",
            reason=RefusalReason.FORWARD_LOOKING,
            safety_notes="Cannot predict future performance.",
        )
        summary = r.summary()
        assert summary["refusal"] is True
        assert "request_id" in summary
        assert "latency_ms" in summary


# ── Policy Tests ──────────────────────────────────────────────────────────────

class TestFinancialQAPolicy:
    """Policy engine must correctly detect and block unsafe queries."""

    def setup_method(self) -> None:
        from finrag.domain.policy import FinancialQAPolicy
        self.policy = FinancialQAPolicy()

    def test_investment_advice_is_blocked(self) -> None:
        result = self.policy.evaluate("Should I buy Apple stock?")
        assert result.blocked
        assert result.first_block is not None
        assert result.first_block.rule_id == "investment_advice"

    def test_sell_advice_is_blocked(self) -> None:
        result = self.policy.evaluate("Should I sell my Tesla shares?")
        assert result.blocked

    def test_prediction_is_blocked(self) -> None:
        result = self.policy.evaluate("What will Apple's revenue be next year?")
        assert result.blocked
        assert result.first_block.rule_id == "forward_looking_statement"

    def test_legitimate_query_not_blocked(self) -> None:
        legit_queries = [
            "What was Apple's revenue in FY2023?",
            "What are the main risk factors mentioned in the 10-K?",
            "Summarize the business description for Microsoft.",
            "What was the gross margin for fiscal year 2022?",
        ]
        for q in legit_queries:
            result = self.policy.evaluate(q)
            assert not result.blocked, f"Legitimate query incorrectly blocked: {q!r}"

    def test_refusal_safe_response_is_not_empty(self) -> None:
        result = self.policy.evaluate("Is this stock a good investment?")
        assert result.blocked
        assert len(result.first_block.safe_response) > 20


# ── Eval Model Tests ──────────────────────────────────────────────────────────

class TestEvalModels:
    """Evaluation models and metrics must compute correctly."""

    def test_recall_at_k(self) -> None:
        from finrag.eval.metrics import recall_at_k
        retrieved = ["chunk-001", "chunk-002", "chunk-003", "chunk-004"]
        assert recall_at_k(retrieved, "chunk-001", k=1) is True
        assert recall_at_k(retrieved, "chunk-003", k=1) is False
        assert recall_at_k(retrieved, "chunk-003", k=3) is True
        assert recall_at_k(retrieved, "chunk-999", k=6) is False

    def test_reciprocal_rank(self) -> None:
        from finrag.eval.metrics import reciprocal_rank
        retrieved = ["chunk-A", "chunk-B", "chunk-C"]
        assert reciprocal_rank(retrieved, "chunk-A") == pytest.approx(1.0)
        assert reciprocal_rank(retrieved, "chunk-B") == pytest.approx(0.5)
        assert reciprocal_rank(retrieved, "chunk-C") == pytest.approx(1 / 3)
        assert reciprocal_rank(retrieved, "chunk-Z") == 0.0

    def test_eval_report_quality_gates(self) -> None:
        from finrag.eval.metrics import EvalReport
        good = EvalReport(recall_at_3=0.85, citation_coverage=1.0)
        passed, failures = good.passes_quality_gates()
        assert passed
        assert failures == []

        bad = EvalReport(recall_at_3=0.50, citation_coverage=0.80)
        passed, failures = bad.passes_quality_gates()
        assert not passed
        assert len(failures) == 2

    def test_eval_report_markdown(self) -> None:
        from finrag.eval.metrics import EvalReport
        report = EvalReport(
            recall_at_3=0.85,
            citation_coverage=1.0,
            mrr=0.72,
        )
        md = report.to_markdown_table()
        assert "Recall@3" in md
        assert "85.0%" in md
