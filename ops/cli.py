"""
ops/cli.py
~~~~~~~~~~~
Single CLI entrypoint for all FinRAG operations.

Commands:
  finrag ingest   — download SEC filings + build corpus
  finrag index    — embed chunks + build vector index
  finrag query    — query the RAG system interactively
  finrag eval     — run evaluation harness against gold set
  finrag serve    — start the FastAPI server
  finrag info     — show current configuration

Usage (after pip install -e .):
  finrag --help
  finrag query "What was Apple's revenue in FY2023?"
  finrag ingest --ticker AAPL --year 2023
  finrag eval --gold-set data/eval/gold.jsonl
  finrag serve --port 8000

Windows note:
  All paths use pathlib.Path — no hardcoded Unix separators.
  Run from the project root where .env lives.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Bootstrap: ensure src/ is on sys.path when running ops/cli.py directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finrag.core.config import settings
from finrag.core.logging import get_logger, setup_logging
from finrag.core.tracing import new_trace
from finrag.domain.policy import FinancialQAPolicy
from finrag.services.answer import AnswerService

# ── App Setup ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="finrag",
    help="Financial Reports RAG Analyzer — query SEC 10-K filings with citations.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = get_logger(__name__)


# ── Callback: runs before every command ──────────────────────────────────────

@app.callback()
def _startup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Initialize logging and ensure data directories exist."""
    if verbose:
        import os
        os.environ["APP_LOG_LEVEL"] = "DEBUG"
    setup_logging()
    settings.ensure_dirs()


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Stock ticker, e.g. AAPL"),
    year: int = typer.Option(..., "--year", "-y", help="Fiscal year, e.g. 2023"),
    force: bool = typer.Option(False, "--force", help="Re-download even if already exists"),
) -> None:
    """
    Download and parse a SEC 10-K filing into structured sections.

    Steps:
      1. Download HTML filing from EDGAR
      2. Parse into sections (Item 1, 1A, 7, 8…)
      3. Chunk sections with stable chunk_ids
      4. Write to data/processed/

    Example:
      finrag ingest --ticker AAPL --year 2023
    """
    console.print(
        Panel(
            f"[bold cyan]Ingesting[/] {ticker} 10-K ({year})\n"
            f"[dim]Output → {settings.data_processed_dir}[/]",
            title="FinRAG Ingest",
        )
    )
    # Day 2: wire EDGAR downloader
    # Day 3: wire HTML parser
    # Day 4: wire chunker
    log.info("ingest.stub", ticker=ticker, year=year)
    console.print("[yellow]⚠ Ingest pipeline is a stub — will be implemented on Days 2–4.[/]")


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe and rebuild index from scratch"),
) -> None:
    """
    Embed chunks and build/update the vector index.

    Reads from data/processed/chunks.jsonl
    Writes to data/index/chroma/

    Example:
      finrag index
      finrag index --rebuild
    """
    console.print(Panel("[bold cyan]Building vector index[/]", title="FinRAG Index"))
    # Day 5: wire SentenceTransformers + Chroma
    # Day 6: wire BM25 + hierarchical index
    log.info("index.stub")
    console.print("[yellow]⚠ Index pipeline is a stub — will be implemented on Days 5–6.[/]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question about the financial filings"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter by fiscal year"),
    top_k: int = typer.Option(settings.retrieval_top_k, "--top-k", "-k"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON response"),
) -> None:
    """
    Query the RAG system and display a grounded answer with citations.

    Example:
      finrag query "What was Apple's revenue in FY2023?"
      finrag query "What are the main risk factors?" --ticker AAPL --year 2023
    """
    filters = {}
    if ticker:
        filters["ticker"] = ticker
    if year:
        filters["year"] = year

    answer_svc = AnswerService()

    with new_trace(query=question, top_k=top_k) as trace:
        start = time.monotonic()
        response = answer_svc.answer(question, filters=filters or None, top_k=top_k)
        elapsed = (time.monotonic() - start) * 1000

    if json_output:
        console.print_json(response.model_dump_json(indent=2))
        return

    # ── Pretty output ─────────────────────────────────────────────────────────
    status_color = "red" if response.refusal else "green"
    status_label = "REFUSED" if response.refusal else "ANSWERED"

    console.print(
        Panel(
            response.answer_text,
            title=f"[{status_color}]{status_label}[/] — {question[:60]}",
            subtitle=f"[dim]{response.model_used} | {elapsed:.0f}ms | "
                     f"{len(response.citations)} citation(s)[/]",
        )
    )

    if response.refusal and response.safety_notes:
        console.print(f"\n[yellow]Safety note:[/] {response.safety_notes}")

    if response.citations:
        table = Table(title="Citations", show_header=True, header_style="bold magenta")
        table.add_column("Chunk ID", style="dim", width=30)
        table.add_column("Doc ID", width=25)
        table.add_column("Section", width=15)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Preview", width=40)

        for cit in response.citations:
            table.add_row(
                cit.chunk_id[:28],
                cit.doc_id,
                cit.section_id,
                f"{cit.relevance_score:.2f}",
                cit.text_preview[:38] + "…" if cit.text_preview else "",
            )
        console.print(table)

    console.print(f"\n[dim]request_id: {response.request_id}[/]")


@app.command()
def eval(
    gold_set: Path = typer.Option(
        Path("data/eval/gold.jsonl"),
        "--gold-set",
        help="Path to gold evaluation set",
    ),
    output: Path = typer.Option(
        Path("data/eval/results.json"),
        "--output",
        help="Where to write evaluation report",
    ),
) -> None:
    """
    Run the evaluation harness against the gold question set.

    Reports retrieval recall@k, MRR, citation coverage, faithfulness.
    Used as a quality gate in CI (Day 13).

    Example:
      finrag eval
      finrag eval --gold-set data/eval/gold.jsonl --output results.json
    """
    console.print(Panel("[bold cyan]Running evaluation harness[/]", title="FinRAG Eval"))
    log.info("eval.stub", gold_set=str(gold_set))
    console.print("[yellow]⚠ Evaluation harness is a stub — will be implemented on Day 8.[/]")


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host"),
    port: int = typer.Option(settings.api_port, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Hot reload (dev only)"),
) -> None:
    """
    Start the FastAPI server (implemented on Day 11).

    Example:
      finrag serve
      finrag serve --port 8080 --reload
    """
    console.print(
        Panel(
            f"[bold cyan]Starting API server[/] on {host}:{port}",
            title="FinRAG Serve",
        )
    )
    log.info("serve.stub")
    console.print("[yellow]⚠ FastAPI server is a stub — will be implemented on Day 11.[/]")
    # Day 11:
    # import uvicorn
    # from finrag.api.app import create_app
    # uvicorn.run(create_app(), host=host, port=port, reload=reload)


@app.command()
def info() -> None:
    """Show current configuration and system status."""
    setup_logging()

    table = Table(title="FinRAG Configuration", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", width=45)
    table.add_column("Status", width=10)

    checks = [
        ("Environment", settings.app_env.value, "✅"),
        ("LLM Backend", settings.llm_backend.value, "✅"),
        ("OpenAI Model", settings.openai_model, "✅" if settings.openai_available else "⚠ no key"),
        ("Embedding Model", settings.embedding_model, "✅"),
        ("Vector Store", settings.vector_store_backend.value, "✅"),
        ("Chroma Dir", str(settings.chroma_persist_dir),
         "✅" if settings.chroma_persist_dir.exists() else "⚠ not built"),
        ("LlamaParse", "enabled" if settings.llamaparse_enabled else "disabled",
         "✅" if settings.llamaparse_available else "—"),
        ("Guardrails", "enabled" if settings.guardrails_enabled else "disabled", "✅"),
        ("Log Dir", str(settings.app_log_dir), "✅"),
        ("Top-K", str(settings.retrieval_top_k), "✅"),
        ("Hybrid Alpha", str(settings.retrieval_hybrid_alpha), "✅"),
    ]

    for setting, value, status in checks:
        table.add_row(setting, value, status)

    console.print(table)

    # Show policy rules
    policy = FinancialQAPolicy()
    console.print(f"\n[bold]Safety Policy[/]: {len(policy._rules)} rules loaded")
    for rule in policy._rules:
        console.print(f"  [cyan]{rule.rule_id}[/] ({rule.severity.value}) — {rule.description[:60]}…")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
