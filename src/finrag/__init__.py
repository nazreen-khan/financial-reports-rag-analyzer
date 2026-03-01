"""
FinRAG — Financial Reports RAG Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production-grade RAG system for querying SEC 10-K filings with citations.

Architecture:
  finrag.core     — Config, logging, tracing (no business logic)
  finrag.domain   — Models, exceptions, safety policy (pure Python)
  finrag.adapters — External integrations (vector DB, LLM, EDGAR, parsers)
  finrag.services — Business logic (retrieval, answer generation)
  finrag.agents   — LangGraph multi-agent workflows
  finrag.eval     — Evaluation harness and metrics
"""

__version__ = "0.1.0"
__author__ = "FinRAG Team"
