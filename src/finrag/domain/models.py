"""
finrag.domain.models
~~~~~~~~~~~~~~~~~~~~~
Pydantic contracts that define the "language" of the FinRAG system.

Every layer — retrieval, generation, agents, API — communicates through
these models. No raw dicts cross layer boundaries.

Key models:
  Citation          — a reference to a specific chunk in a source document
  ExtractedFact     — a structured numeric/categorical fact with provenance
  RetrievedChunk    — a chunk returned from the vector store with score
  RAGResponse       — the full response contract returned to the user
  RefusalResponse   — returned when a guardrail blocks the query
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class RefusalReason(str, Enum):
    """Canonical reasons a query can be refused."""
    INVESTMENT_ADVICE = "investment_advice"
    FORWARD_LOOKING = "forward_looking_statement"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OFF_TOPIC = "off_topic"
    MODERATION_FLAG = "moderation_flag"
    CITATION_FAILURE = "citation_failure"


class FactType(str, Enum):
    """Type of extracted financial fact."""
    REVENUE = "revenue"
    NET_INCOME = "net_income"
    EPS = "eps"
    OPERATING_INCOME = "operating_income"
    CASH = "cash_and_equivalents"
    DEBT = "total_debt"
    RISK_FACTOR = "risk_factor"
    ACCOUNTING_POLICY = "accounting_policy"
    OTHER = "other"


class AgentRole(str, Enum):
    """Specialist agents available in the LangGraph workflow."""
    ROUTER = "router"
    RISK_ANALYST = "risk_analyst"
    FINANCIAL_RATIO = "financial_ratio"
    SUMMARIZER = "summarizer"
    AGGREGATOR = "aggregator"


# ── Core Atomic Models ────────────────────────────────────────────────────────

class Citation(BaseModel):
    """
    A reference to a specific chunk within a source document.
    Included in every answer — no citation means no answer.
    """
    chunk_id: str = Field(description="Stable, deterministic chunk identifier")
    doc_id: str = Field(description="Source document ID (e.g., AAPL-2023-0001234567-89)")
    section_id: str = Field(description="Section identifier (e.g., item_7, item_1a)")
    section_title: str = Field(default="", description="Human-readable section name")
    source_url: str = Field(default="", description="Direct URL to the SEC filing")
    text_preview: str = Field(
        default="",
        description="First 200 chars of the cited chunk for UI display",
        max_length=300,
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Retrieval similarity score for this chunk",
    )

    @field_validator("chunk_id", "doc_id", "section_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Citation IDs must not be empty strings")
        return v


class ExtractedFact(BaseModel):
    """
    A structured numeric or categorical fact extracted from the document.
    Grounded facts reduce hallucination risk for financial figures.
    """
    fact_type: FactType = FactType.OTHER
    label: str = Field(description="Human-readable label, e.g. 'Total Revenue FY2023'")
    value: str = Field(description="Raw value as string, e.g. '$394.3 billion'")
    numeric_value: float | None = Field(
        default=None,
        description="Parsed numeric value for downstream computation",
    )
    unit: str = Field(default="", description="e.g. 'USD millions', 'percent'")
    period: str = Field(default="", description="e.g. 'FY2023', 'Q2 2024'")
    cited_chunk_id: str = Field(description="The chunk this fact was extracted from")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (1.0 = exact match from source)",
    )


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with metadata and score."""
    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str = ""
    text: str
    source_url: str = ""
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    retrieval_method: str = Field(
        default="hybrid",
        description="How this chunk was retrieved: 'vector', 'bm25', or 'hybrid'",
    )

    def to_citation(self) -> Citation:
        """Convert a retrieved chunk to a Citation for the response."""
        return Citation(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            section_id=self.section_id,
            section_title=self.section_title,
            source_url=self.source_url,
            text_preview=self.text[:200],
            relevance_score=self.score,
        )


# ── Agent Trace ───────────────────────────────────────────────────────────────

class AgentStep(BaseModel):
    """Records one agent's contribution in a multi-agent trace."""
    agent: AgentRole
    query_rewrite: str = Field(default="", description="How the agent reformulated the query")
    chunks_retrieved: list[str] = Field(
        default_factory=list,
        description="chunk_ids retrieved by this agent",
    )
    reasoning: str = Field(default="", description="Agent's chain-of-thought summary")


# ── Top-Level Response Contracts ──────────────────────────────────────────────

class RAGResponse(BaseModel):
    """
    The complete response contract returned to the user.

    Rules enforced here:
    - If refusal=True, answer_text must explain why (no empty refusals)
    - If refusal=False, citations must be non-empty (grounded answers only)
    """
    # Identity
    request_id: str = Field(description="ULID trace ID for log correlation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of response generation",
    )

    # Core answer
    answer_text: str = Field(description="The final answer shown to the user")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source chunks the answer is grounded in",
    )
    extracted_facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="Structured financial facts extracted from the answer",
    )

    # Safety
    refusal: bool = Field(
        default=False,
        description="True when a guardrail blocked the query",
    )
    refusal_reason: RefusalReason | None = None
    safety_notes: str = Field(
        default="",
        description="Explanation shown to user when refusal=True",
    )

    # Meta / observability
    model_used: str = Field(default="", description="LLM that generated this answer")
    retrieval_method: str = Field(default="hybrid")
    latency_ms: float = Field(default=0.0, description="End-to-end latency in milliseconds")
    agent_trace: list[AgentStep] = Field(
        default_factory=list,
        description="Ordered list of agents that contributed (empty for simple RAG)",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def enforce_citation_contract(self) -> RAGResponse:
        """
        Core invariant: a non-refusal answer MUST have citations.
        A refusal answer MUST have a safety_notes explanation.
        """
        if not self.refusal and not self.citations:
            raise ValueError(
                "RAGResponse contract violation: "
                "non-refusal answers must include at least one citation. "
                "Use refusal=True with InsufficientEvidence if no evidence found."
            )
        if self.refusal and not self.safety_notes:
            raise ValueError(
                "RAGResponse contract violation: "
                "refusals must include safety_notes explaining why."
            )
        return self

    @classmethod
    def make_refusal(
        cls,
        *,
        request_id: str,
        reason: RefusalReason,
        safety_notes: str,
        model_used: str = "",
        latency_ms: float = 0.0,
    ) -> RAGResponse:
        """
        Factory for safe, policy-compliant refusal responses.
        Use this instead of constructing refusals manually.
        """
        return cls(
            request_id=request_id,
            answer_text=safety_notes,
            citations=[],           # validator allows empty on refusal
            refusal=True,
            refusal_reason=reason,
            safety_notes=safety_notes,
            model_used=model_used,
            latency_ms=latency_ms,
        )

    def summary(self) -> dict[str, Any]:
        """Compact dict for logging — excludes full text for brevity."""
        return {
            "request_id": self.request_id,
            "refusal": self.refusal,
            "refusal_reason": self.refusal_reason,
            "citation_count": len(self.citations),
            "fact_count": len(self.extracted_facts),
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
            "answer_preview": self.answer_text[:120],
        }
