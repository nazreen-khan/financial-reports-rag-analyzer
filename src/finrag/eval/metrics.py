"""
finrag.eval.metrics
~~~~~~~~~~~~~~~~~~~~
Evaluation metrics for retrieval and generation quality.

Today (Day 1): Metric interfaces + data models defined.
Day 8: Full implementation with gold set evaluation harness.

Metrics tracked:
  Retrieval:
    - recall@k:  Did the correct chunk appear in the top-k results?
    - MRR:       Mean Reciprocal Rank of the correct chunk
    - precision@k

  Generation:
    - citation_coverage: % of answers with ≥1 citation (must be 100%)
    - faithfulness:      LLM-judged faithfulness to source chunks
    - numeric_match:     % of numeric facts matching the gold document
    - refusal_accuracy:  % of advice queries correctly refused
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class GoldEntry:
    """
    One entry in the evaluation gold set.
    Created from known facts in the financial documents.
    """
    question_id: str
    question: str
    expected_doc_id: str
    expected_section_id: str
    expected_answer_contains: list[str]   # key phrases/numbers that must appear
    should_refuse: bool = False           # True for advice/prediction queries
    refusal_reason: str | None = None


@dataclass
class RetrievalEvalResult:
    """Results of evaluating retrieval for a single gold entry."""
    question_id: str
    recall_at_1: bool = False
    recall_at_3: bool = False
    recall_at_6: bool = False
    reciprocal_rank: float = 0.0
    top_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GenerationEvalResult:
    """Results of evaluating generation for a single gold entry."""
    question_id: str
    has_citations: bool = False
    numeric_match: bool = False
    faithfulness_score: float | None = None  # GPT-4 judge score 0–1
    correct_refusal: bool | None = None      # None if not a refusal query
    answer_preview: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report across all gold entries."""
    total_questions: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_6: float = 0.0
    mrr: float = 0.0
    citation_coverage: float = 0.0         # Must be 1.0 (quality gate)
    avg_faithfulness: float | None = None
    numeric_match_rate: float = 0.0
    refusal_accuracy: float | None = None
    retrieval_results: list[RetrievalEvalResult] = field(default_factory=list)
    generation_results: list[GenerationEvalResult] = field(default_factory=list)

    def passes_quality_gates(self) -> tuple[bool, list[str]]:
        """
        Check if results meet the minimum quality thresholds.
        Returns (passed, list of failed gate descriptions).
        Used in CI pipeline (Day 13).
        """
        failures: list[str] = []

        if self.recall_at_3 < 0.7:
            failures.append(f"recall@3={self.recall_at_3:.2f} < threshold 0.70")

        if self.citation_coverage < 1.0:
            failures.append(f"citation_coverage={self.citation_coverage:.2f} < 1.00")

        return len(failures) == 0, failures

    def to_markdown_table(self) -> str:
        """Format as markdown table for README embedding."""
        rows = [
            ("Recall@1", f"{self.recall_at_1:.1%}"),
            ("Recall@3", f"{self.recall_at_3:.1%}"),
            ("Recall@6", f"{self.recall_at_6:.1%}"),
            ("MRR", f"{self.mrr:.3f}"),
            ("Citation Coverage", f"{self.citation_coverage:.1%}"),
            ("Numeric Match Rate", f"{self.numeric_match_rate:.1%}"),
        ]
        if self.avg_faithfulness is not None:
            rows.append(("Avg Faithfulness (GPT-4)", f"{self.avg_faithfulness:.2f}"))
        if self.refusal_accuracy is not None:
            rows.append(("Refusal Accuracy", f"{self.refusal_accuracy:.1%}"))

        header = "| Metric | Score |\n|--------|-------|\n"
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return header + body


# ── Metric Functions (stubs — implemented Day 8) ─────────────────────────────

def recall_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_id: str,
    k: int,
) -> bool:
    """Returns True if expected_chunk_id appears in top-k retrieved chunks."""
    return expected_chunk_id in retrieved_chunk_ids[:k]


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_id: str,
) -> float:
    """Returns 1/rank if found, 0 if not found."""
    try:
        rank = retrieved_chunk_ids.index(expected_chunk_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """Compute MRR over a list of reciprocal rank scores."""
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)
