"""
finrag.agents.graph
~~~~~~~~~~~~~~~~~~~~
LangGraph multi-agent workflow for FinRAG.

Today (Day 1): Graph structure defined with node stubs.
Day 10: Full implementation with specialist agents, tool calling, shared state.

Agents in the workflow:
  router          → classifies intent → routes to specialist(s)
  risk_analyst    → retrieves + analyzes risk factors (Item 1A)
  financial_ratio → retrieves + computes key financial metrics (Item 8)
  summarizer      → retrieves + summarizes business description (Item 1, 7)
  aggregator      → merges evidence from all agents → final RAGResponse
"""

from __future__ import annotations

from typing import Any, TypedDict

from finrag.core.logging import get_logger
from finrag.domain.models import AgentRole, RAGResponse, RetrievedChunk

log = get_logger(__name__)


# ── Shared Graph State ────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph graph.
    TypedDict ensures type-safety across node boundaries.
    """
    query: str
    request_id: str
    intent: str                        # classified intent from router
    active_agents: list[str]           # which specialist agents to invoke
    retrieved_chunks: list[RetrievedChunk]
    agent_outputs: dict[str, Any]      # keyed by AgentRole value
    final_response: RAGResponse | None
    error: str | None


# ── Node Stubs (will be full implementations on Day 10) ──────────────────────

def router_node(state: GraphState) -> GraphState:
    """
    Classify the query intent and decide which specialist agents to invoke.

    Intent categories:
      - "risk"         → risk_analyst
      - "financials"   → financial_ratio
      - "summary"      → summarizer
      - "multi"        → risk_analyst + financial_ratio (complex queries)
    """
    log.info("agent.router.stub", query_preview=state.get("query", "")[:80])
    # Day 10: LLM-based intent classifier with few-shot examples
    return {
        **state,
        "intent": "stub",
        "active_agents": [AgentRole.SUMMARIZER.value],
    }


def risk_analyst_node(state: GraphState) -> GraphState:
    """
    Specialist: retrieves and analyzes risk factors from Item 1A.
    Rewrites the query for risk-specific retrieval.
    """
    log.info("agent.risk_analyst.stub")
    return {**state, "agent_outputs": {**state.get("agent_outputs", {}), "risk": None}}


def financial_ratio_node(state: GraphState) -> GraphState:
    """
    Specialist: retrieves financial statements (Item 8) and computes ratios.
    Calls a deterministic ratio extraction tool for numbers.
    """
    log.info("agent.financial_ratio.stub")
    return {**state, "agent_outputs": {**state.get("agent_outputs", {}), "financials": None}}


def summarizer_node(state: GraphState) -> GraphState:
    """
    Specialist: retrieves and summarizes business overview (Items 1 and 7).
    """
    log.info("agent.summarizer.stub")
    return {**state, "agent_outputs": {**state.get("agent_outputs", {}), "summary": None}}


def aggregator_node(state: GraphState) -> GraphState:
    """
    Merges all agent outputs into a single unified RAGResponse with
    de-duplicated citations from all agents.
    """
    log.info("agent.aggregator.stub")
    return {**state, "final_response": None}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Day 1: Returns None (LangGraph not yet wired).
    Day 10: Returns a compiled CompiledGraph ready for .invoke().

    Example (Day 10):
        graph = build_graph()
        result = graph.invoke({"query": "What were Apple's risk factors?"})
    """
    log.warning(
        "agent.graph.stub",
        message="LangGraph graph is a stub — will be wired on Day 10",
    )
    # Day 10 implementation:
    # from langgraph.graph import StateGraph, END
    # builder = StateGraph(GraphState)
    # builder.add_node("router", router_node)
    # builder.add_node("risk_analyst", risk_analyst_node)
    # builder.add_node("financial_ratio", financial_ratio_node)
    # builder.add_node("summarizer", summarizer_node)
    # builder.add_node("aggregator", aggregator_node)
    # builder.set_entry_point("router")
    # builder.add_conditional_edges("router", _route_decision)
    # builder.add_edge("aggregator", END)
    # return builder.compile()
    return None
