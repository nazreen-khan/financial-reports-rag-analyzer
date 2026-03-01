"""
finrag.domain.policy
~~~~~~~~~~~~~~~~~~~~~
Financial QA Safety Policy — the rules that govern what FinRAG will and
won't answer.

Design philosophy:
  - Rules are data, not code scattered across the codebase
  - Every rule has an ID, description, and example triggers
  - The guardrails service (Day 9) imports this and runs checks
  - Rules are extensible without touching service logic

This module is intentionally pure Python — no LLM calls, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from finrag.domain.models import RefusalReason


# ── Rule Severity ─────────────────────────────────────────────────────────────

class Severity(str, Enum):
    BLOCK = "block"    # Hard block — always refuse, no LLM call made
    WARN  = "warn"     # Soft warn — answer but append disclaimer
    LOG   = "log"      # Log only — no user-facing action


# ── Rule Definition ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyRule:
    """
    A single safety rule.

    Attributes:
        rule_id:        Unique identifier, used in logs and traces
        description:    Human-readable explanation (shown in README / docs)
        severity:       How to react when this rule matches
        refusal_reason: Which RefusalReason to attach on BLOCK
        patterns:       Regex patterns — if ANY match, rule fires
        safe_response:  Template response sent to the user on BLOCK
        examples:       Example queries that should trigger this rule
    """
    rule_id: str
    description: str
    severity: Severity
    refusal_reason: RefusalReason
    patterns: list[str]
    safe_response: str
    examples: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """Returns True if any pattern matches the lowercased input text."""
        lowered = text.lower()
        return any(re.search(p, lowered) for p in self.patterns)


# ── Policy Rules Registry ─────────────────────────────────────────────────────

FINANCIAL_QA_POLICY: list[PolicyRule] = [

    PolicyRule(
        rule_id="investment_advice",
        description=(
            "Block queries asking for buy/sell/hold recommendations or "
            "investment decisions. FinRAG summarizes documents — it does not "
            "provide financial advice."
        ),
        severity=Severity.BLOCK,
        refusal_reason=RefusalReason.INVESTMENT_ADVICE,
        patterns=[
            r"\bshould i (buy|sell|invest|hold|short|purchase)\b",
            r"\bis .+ (a good|a bad|worth) (buy|investment|stock)\b",
            r"\b(buy|sell|invest) (in|into)\b",
            r"\bwould you recommend (buying|selling|investing)\b",
            r"\bwhat stock(s)? should\b",
            r"\bportfolio (advice|recommendation|suggestion)\b",
            r"\b(financial|investment) advice\b",
        ],
        safe_response=(
            "I can summarize and answer questions about information contained "
            "in SEC filings and financial documents, but I'm not able to provide "
            "investment advice or buy/sell recommendations. "
            "For investment decisions, please consult a licensed financial advisor. "
            "\n\nIf you'd like, I can look up specific financial metrics, "
            "risk factors, or business descriptions from the filings instead."
        ),
        examples=[
            "Should I buy Apple stock?",
            "Is Tesla a good investment right now?",
            "What stocks should I invest in based on these filings?",
        ],
    ),

    PolicyRule(
        rule_id="forward_looking_statement",
        description=(
            "Block queries asking FinRAG to predict, forecast, or guarantee "
            "future financial performance. Filings contain forward-looking "
            "statements; we should not amplify them as predictions."
        ),
        severity=Severity.BLOCK,
        refusal_reason=RefusalReason.FORWARD_LOOKING,
        patterns=[
            r"\b(predict|forecast|project|estimate)\b.{0,30}\b(revenue|earnings|profit|stock|price|growth)\b",
            r"\bwill .{0,20} (revenue|earnings|profit|stock|grow|increase|decrease)\b",
            r"\bnext (quarter|year|fiscal).{0,20}(revenue|earnings|profit)\b",
            r"\bguarantee.{0,20}(return|profit|performance)\b",
            r"\bstock (will|going to) (go up|go down|rise|fall|increase|decrease)\b",
            r"\bprice target\b",
            r"\bfuture (earnings|revenue|guidance) (will|should|would)\b",
        ],
        safe_response=(
            "I can only summarize what is reported in the filings — I'm not "
            "able to predict or forecast future financial performance. "
            "Forward-looking statements in filings are management's estimates "
            "and carry significant uncertainty. "
            "\n\nWould you like me to look up the company's most recently "
            "reported guidance or management commentary from their 10-K instead?"
        ),
        examples=[
            "What will Apple's revenue be next year?",
            "Predict Tesla's earnings for Q4.",
            "Will this stock go up?",
        ],
    ),

    PolicyRule(
        rule_id="off_topic",
        description=(
            "Warn (but don't block) when the query appears unrelated to "
            "financial documents. Log for monitoring."
        ),
        severity=Severity.WARN,
        refusal_reason=RefusalReason.OFF_TOPIC,
        patterns=[
            r"\b(weather|recipe|joke|dating|sports score|movie|music|poem)\b",
            r"\bhow (do i|to) (cook|bake|make|draw|write a story)\b",
        ],
        safe_response=(
            "I'm specialized in answering questions about SEC financial filings "
            "and earnings reports. I might not be the best resource for this "
            "particular question, but I'll try to help if there's a relevant "
            "financial angle."
        ),
        examples=[
            "What's the weather like?",
            "Tell me a joke about stocks.",
        ],
    ),
]


# ── Policy Engine ─────────────────────────────────────────────────────────────

class FinancialQAPolicy:
    """
    Evaluates a query against all registered policy rules.

    Usage:
        policy = FinancialQAPolicy()
        result = policy.evaluate("Should I buy Apple stock?")
        if result.blocked:
            return RAGResponse.make_refusal(...)
    """

    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        self._rules = rules if rules is not None else FINANCIAL_QA_POLICY

    def evaluate(self, query: str) -> "PolicyEvaluation":
        """
        Run all rules against the query.
        Returns a PolicyEvaluation with the first BLOCK rule hit (if any),
        and all WARN rules hit.
        """
        blocks: list[PolicyRule] = []
        warnings: list[PolicyRule] = []

        for rule in self._rules:
            if rule.matches(query):
                if rule.severity == Severity.BLOCK:
                    blocks.append(rule)
                elif rule.severity == Severity.WARN:
                    warnings.append(rule)

        return PolicyEvaluation(
            query=query,
            blocks=blocks,
            warnings=warnings,
        )


@dataclass
class PolicyEvaluation:
    """Result of evaluating a query against the policy."""
    query: str
    blocks: list[PolicyRule]
    warnings: list[PolicyRule]

    @property
    def blocked(self) -> bool:
        return len(self.blocks) > 0

    @property
    def first_block(self) -> PolicyRule | None:
        return self.blocks[0] if self.blocks else None

    @property
    def warning_ids(self) -> list[str]:
        return [r.rule_id for r in self.warnings]

    def __repr__(self) -> str:
        return (
            f"PolicyEvaluation("
            f"blocked={self.blocked}, "
            f"block_rules={[r.rule_id for r in self.blocks]}, "
            f"warn_rules={self.warning_ids})"
        )
