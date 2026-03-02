"""
finrag.ingest.corpus_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines the exact corpus to download and index.

Corpus: "Magnificent 7" tech companies, FY2023 + FY2024
  - 7 companies × 2 years = 14 filings
  - ~2,100 pages of high-signal financial text

Why 2023 + 2024 (not 2022 + 2023):
  - All FY2024 10-Ks are filed and available as of March 2026
  - 2024 filings are dominated by AI strategy, AI capex, AI risk factors
    making RAG demos far more topical and interview-relevant
  - 2023→2024 YoY contrast is sharper: NVDA's Blackwell ramp,
    MSFT Copilot monetisation, META efficiency payoff, TSLA margin pressure
  - Recency signals seniority — "most recent available filings" is the
    right answer when an interviewer asks about data freshness

NVDA fiscal year note:
  NVIDIA's fiscal year ends in late January.
  "FY2024" = Feb 2023 – Jan 2024, filed April 2024 ✅
  "FY2025" = Feb 2024 – Jan 2025, filed April 2025 ✅
  We use years 2024 + 2025 for NVDA to align with calendar year labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilingTarget:
    """One filing to download."""
    ticker: str
    year: int
    company_name: str
    industry: str
    fiscal_year_label: str = ""        # e.g. "FY2024" — shown in citations
    demo_questions: list[str] = field(default_factory=list)


# ── Corpus Definition ─────────────────────────────────────────────────────────

CORPUS: list[FilingTarget] = [

    # ── Apple (Fiscal year ends last Saturday of September) ────────────────
    FilingTarget(
        ticker="AAPL",
        year=2023,
        company_name="Apple Inc.",
        industry="Consumer Technology",
        fiscal_year_label="FY2023",
        demo_questions=[
            "What was Apple's total revenue in fiscal year 2023?",
            "What were Apple's main product revenue segments in FY2023?",
            "What supply chain and geopolitical risks did Apple disclose in 2023?",
        ],
    ),
    FilingTarget(
        ticker="AAPL",
        year=2024,
        company_name="Apple Inc.",
        industry="Consumer Technology",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Apple's total revenue in fiscal year 2024?",
            "How did Apple describe its AI strategy and Apple Intelligence in the 2024 10-K?",
            "How did Apple's services revenue grow from 2023 to 2024?",
        ],
    ),

    # ── Microsoft (Fiscal year ends June 30) ───────────────────────────────
    FilingTarget(
        ticker="MSFT",
        year=2023,
        company_name="Microsoft Corporation",
        industry="Cloud / Enterprise Software",
        fiscal_year_label="FY2023",
        demo_questions=[
            "What was Microsoft's total revenue in FY2023?",
            "How did Microsoft describe its AI strategy and OpenAI partnership in FY2023?",
            "What was Microsoft Cloud revenue in FY2023?",
        ],
    ),
    FilingTarget(
        ticker="MSFT",
        year=2024,
        company_name="Microsoft Corporation",
        industry="Cloud / Enterprise Software",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Microsoft's total revenue in FY2024?",
            "How did Copilot and AI products contribute to Microsoft's growth in FY2024?",
            "What were Microsoft's key risk factors around AI in FY2024?",
        ],
    ),

    # ── Alphabet (Fiscal year ends December 31) ────────────────────────────
    FilingTarget(
        ticker="GOOGL",
        year=2023,
        company_name="Alphabet Inc.",
        industry="Digital Advertising / Cloud",
        fiscal_year_label="FY2023",
        demo_questions=[
            "What was Alphabet's total revenue in 2023?",
            "How did Google Cloud revenue grow in 2023?",
            "What antitrust and regulatory risks did Alphabet disclose in 2023?",
        ],
    ),
    FilingTarget(
        ticker="GOOGL",
        year=2024,
        company_name="Alphabet Inc.",
        industry="Digital Advertising / Cloud",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Alphabet's total revenue in 2024?",
            "How did Alphabet describe Gemini and its AI investments in 2024?",
            "What were Alphabet's AI-related risk factors in the 2024 10-K?",
        ],
    ),

    # ── Amazon (Fiscal year ends December 31) ─────────────────────────────
    FilingTarget(
        ticker="AMZN",
        year=2023,
        company_name="Amazon.com Inc.",
        industry="E-Commerce / Cloud",
        fiscal_year_label="FY2023",
        demo_questions=[
            "What was Amazon's net income and operating income in 2023?",
            "How did AWS revenue and operating income perform in 2023?",
            "What were Amazon's key risk factors in its 2023 10-K?",
        ],
    ),
    FilingTarget(
        ticker="AMZN",
        year=2024,
        company_name="Amazon.com Inc.",
        industry="E-Commerce / Cloud",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Amazon's total revenue in 2024?",
            "How did Amazon describe its AI and generative AI investments in 2024?",
            "How did AWS growth compare in 2023 vs 2024?",
        ],
    ),

    # ── Nvidia (Fiscal year ends late January) ─────────────────────────────
    # NVDA FY2024 = Feb 2023–Jan 2024 | FY2025 = Feb 2024–Jan 2025
    # We label by the fiscal year number Nvidia uses officially.
    FilingTarget(
        ticker="NVDA",
        year=2024,
        company_name="NVIDIA Corporation",
        industry="Semiconductors / AI",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Nvidia's total revenue in FY2024?",
            "How did Nvidia's data center segment perform in FY2024?",
            "What export control risks did Nvidia disclose related to China in FY2024?",
        ],
    ),
    FilingTarget(
        ticker="NVDA",
        year=2025,
        company_name="NVIDIA Corporation",
        industry="Semiconductors / AI",
        fiscal_year_label="FY2025",
        demo_questions=[
            "What was Nvidia's revenue in FY2025 and how did it compare to FY2024?",
            "How did Nvidia describe the Blackwell GPU ramp in its FY2025 10-K?",
            "What were Nvidia's key risk factors around supply chain and export controls in FY2025?",
        ],
    ),

    # # ── Meta (Fiscal year ends December 31) ───────────────────────────────
    # FilingTarget(
    #     ticker="META",
    #     year=2023,
    #     company_name="Meta Platforms Inc.",
    #     industry="Social Media / Digital Advertising",
    #     fiscal_year_label="FY2023",
    #     demo_questions=[
    #         "What was Meta's total revenue and operating income in 2023?",
    #         "How did Meta's Year of Efficiency impact its financials in 2023?",
    #         "What were Meta's key risk factors around AI regulation in 2023?",
    #     ],
    # ),
    # FilingTarget(
    #     ticker="META",
    #     year=2024,
    #     company_name="Meta Platforms Inc.",
    #     industry="Social Media / Digital Advertising",
    #     fiscal_year_label="FY2024",
    #     demo_questions=[
    #         "What was Meta's total revenue in 2024?",
    #         "How did Meta describe its AI investments and Llama models in the 2024 10-K?",
    #         "What were Meta's capital expenditure plans for AI infrastructure in 2024?",
    #     ],
    # ),

    # ── Tesla (Fiscal year ends December 31) ──────────────────────────────
    FilingTarget(
        ticker="TSLA",
        year=2023,
        company_name="Tesla Inc.",
        industry="Electric Vehicles / Energy",
        fiscal_year_label="FY2023",
        demo_questions=[
            "What was Tesla's total revenue and gross margin in 2023?",
            "How did Tesla explain the impact of vehicle price reductions on margins in 2023?",
            "What were Tesla's manufacturing and competition risk factors in 2023?",
        ],
    ),
    FilingTarget(
        ticker="TSLA",
        year=2024,
        company_name="Tesla Inc.",
        industry="Electric Vehicles / Energy",
        fiscal_year_label="FY2024",
        demo_questions=[
            "What was Tesla's total revenue and net income in 2024?",
            "How did Tesla describe its Full Self-Driving and autonomy strategy in 2024?",
            "What were Tesla's key competitive and regulatory risks in its 2024 10-K?",
        ],
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_tickers() -> list[str]:
    """Return unique list of tickers in the corpus."""
    return list(dict.fromkeys(f.ticker for f in CORPUS))


def get_years() -> list[int]:
    """Return unique list of years in the corpus."""
    return sorted(set(f.year for f in CORPUS))


def get_filing(ticker: str, year: int) -> FilingTarget | None:
    """Look up a specific filing target."""
    for f in CORPUS:
        if f.ticker == ticker.upper() and f.year == year:
            return f
    return None


def summary() -> str:
    """Human-readable corpus summary."""
    tickers = get_tickers()
    years = get_years()
    lines = [
        f"Corpus: {len(CORPUS)} filings | {len(tickers)} companies | Years: {years}",
        "Data: FY2023 + FY2024 (most recent available as of March 2026)",
        "",
    ]
    for ticker in tickers:
        company_filings = [f for f in CORPUS if f.ticker == ticker]
        name = company_filings[0].company_name
        labels = [f.fiscal_year_label for f in company_filings]
        lines.append(f"  {ticker:6s} {name:38s} ({', '.join(labels)})")

    lines.append("")
    lines.append("Note: NVDA fiscal year ends Jan → FY2024=Feb23-Jan24, FY2025=Feb24-Jan25")
    return "\n".join(lines)
