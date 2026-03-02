"""
ops/cli.py
~~~~~~~~~~~
Single CLI entrypoint for all FinRAG operations.

Commands:
  finrag ingest   — download SEC filings from EDGAR
  finrag corpus   — show corpus definition
  finrag index    — embed chunks + build vector index
  finrag query    — query the RAG system interactively
  finrag eval     — run evaluation harness against gold set
  finrag serve    — start the FastAPI server
  finrag info     — show current configuration + corpus status

Usage (after pip install -e .):
  finrag ingest --ticker AAPL --year 2023
  finrag ingest --all
  finrag query "What was Apple's revenue in FY2023?"

Windows compatible: all paths use pathlib.Path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finrag.core.config import settings
from finrag.core.logging import get_logger, setup_logging
from finrag.core.tracing import new_trace
from finrag.domain.policy import FinancialQAPolicy
from finrag.services.answer import AnswerService

app = typer.Typer(
    name="finrag",
    help="Financial Reports RAG Analyzer — query SEC 10-K filings with citations.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = get_logger(__name__)


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


# ── ingest ────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Single ticker e.g. AAPL"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Fiscal year e.g. 2023"),
    all_corpus: bool = typer.Option(False, "--all", help="Download full Magnificent 7 corpus"),
    force: bool = typer.Option(False, "--force", help="Re-download even if already exists"),
) -> None:
    """
    Download SEC 10-K filings from EDGAR into data/raw/.

    Examples:\n
      finrag ingest --ticker AAPL --year 2023\n
      finrag ingest --all\n
      finrag ingest --all --force
    """
    from finrag.ingest.corpus_config import CORPUS, FilingTarget, summary
    from finrag.ingest.download_edgar_10k import CorpusDownloader

    if not all_corpus and not (ticker and year):
        console.print("[red]Error:[/] Provide --ticker + --year, or use --all")
        raise typer.Exit(1)

    if all_corpus:
        console.print(Panel(summary(), title="[bold cyan]FinRAG Corpus[/]"))
        targets = CORPUS
    else:
        targets = [FilingTarget(
            ticker=ticker.upper(),
            year=year,
            company_name=ticker.upper(),
            industry="",
        )]
        console.print(Panel(
            f"[bold cyan]Downloading[/] {ticker.upper()} 10-K ({year})\n"
            f"[dim]Output → {settings.data_raw_dir}[/]",
            title="FinRAG Ingest",
        ))

    downloader = CorpusDownloader(force=force)
    results = downloader.download_all(targets)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        console.print(f"\n[red]⚠ {len(failed)} filing(s) failed — check logs.[/]")
        raise typer.Exit(1)


# ── corpus ────────────────────────────────────────────────────────────────────

@app.command()
def corpus() -> None:
    """Show the full corpus definition (all 14 planned filings + demo questions)."""
    from finrag.ingest.corpus_config import CORPUS, summary
    console.print(Panel(summary(), title="[bold cyan]Magnificent 7 Corpus[/]"))
    console.print("\n[bold]Sample demo questions per filing:[/]")
    for filing in CORPUS:
        console.print(f"\n  [cyan]{filing.ticker} {filing.year}[/] — {filing.company_name}")
        for q in filing.demo_questions:
            console.print(f"    • {q}")



# ── parse ─────────────────────────────────────────────────────────────────────

@app.command()
def parse(
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Parse single doc by ID"),
    force: bool = typer.Option(False, "--force", help="Re-parse already parsed docs"),
) -> None:
    """
    Parse downloaded 10-K filings into structured sections.

    Reads  : data/raw/{doc_id}/filing.htm
    Writes : data/processed/sections.jsonl

    Uses LlamaParse (if configured) for best quality,
    falls back to BeautifulSoup automatically.

    Examples:\n
      finrag parse\n
      finrag parse --doc-id AAPL-2024-xxx\n
      finrag parse --force
    """
    from finrag.ingest.parse_sections import SectionParser

    parser = SectionParser(force=force)

    console.print(Panel(
        f"[bold cyan]Parsing filings → sections[/]\n"
        f"[dim]Parser: {parser._parser.active_parser} | "
        f"Output: {settings.data_processed_dir / 'sections.jsonl'}[/]",
        title="FinRAG Parse",
    ))

    if doc_id:
        result = parser.parse_one_by_doc_id(doc_id)
        if result.status == "failed":
            console.print(f"[red]❌ Failed:[/] {result.error}")
            raise typer.Exit(1)
        console.print(f"[green]✅[/] {result.doc_id} → {result.sections_found} sections {result.section_ids}")
    else:
        results = parser.parse_all()
        failed = [r for r in results if r.status == "failed"]
        if failed:
            console.print(f"[red]⚠ {len(failed)} filing(s) failed parsing.[/]")
            raise typer.Exit(1)

# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe and rebuild index from scratch"),
) -> None:
    """
    Embed chunks and build/update the vector index.

    Reads from data/processed/chunks.jsonl — Writes to data/index/chroma/
    """
    console.print(Panel("[bold cyan]Building vector index[/]", title="FinRAG Index"))
    log.info("index.stub")
    console.print("[yellow]⚠ Index pipeline is a stub — will be implemented on Days 5–6.[/]")


# ── query ─────────────────────────────────────────────────────────────────────

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

    Example:\n
      finrag query "What was Apple's revenue in FY2023?"\n
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

    status_color = "red" if response.refusal else "green"
    status_label = "REFUSED" if response.refusal else "ANSWERED"

    console.print(Panel(
        response.answer_text,
        title=f"[{status_color}]{status_label}[/] — {question[:60]}",
        subtitle=f"[dim]{response.model_used} | {elapsed:.0f}ms | {len(response.citations)} citation(s)[/]",
    ))

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
                (cit.text_preview[:38] + "…") if cit.text_preview else "",
            )
        console.print(table)

    console.print(f"\n[dim]request_id: {response.request_id}[/]")


# ── eval ──────────────────────────────────────────────────────────────────────

@app.command()
def eval(
    gold_set: Path = typer.Option(Path("data/eval/gold.jsonl"), "--gold-set"),
    output: Path = typer.Option(Path("data/eval/results.json"), "--output"),
) -> None:
    """Run the evaluation harness against the gold question set (Day 8)."""
    console.print(Panel("[bold cyan]Running evaluation harness[/]", title="FinRAG Eval"))
    console.print("[yellow]⚠ Evaluation harness is a stub — will be implemented on Day 8.[/]")


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host"),
    port: int = typer.Option(settings.api_port, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server (Day 11)."""
    console.print(Panel(
        f"[bold cyan]Starting API server[/] on {host}:{port}",
        title="FinRAG Serve",
    ))
    console.print("[yellow]⚠ FastAPI server is a stub — will be implemented on Day 11.[/]")


# ── info ──────────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Show current configuration, corpus download status, and safety policy."""
    setup_logging()

    table = Table(title="FinRAG Configuration", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", width=45)
    table.add_column("Status", width=12)

    checks = [
        ("Environment",    settings.app_env.value,            "✅"),
        ("LLM Backend",    settings.llm_backend.value,        "✅"),
        ("OpenAI Model",   settings.openai_model,             "✅" if settings.openai_available else "⚠ no key"),
        ("Embedding Model",settings.embedding_model,          "✅"),
        ("Vector Store",   settings.vector_store_backend.value,"✅"),
        ("Chroma Dir",     str(settings.chroma_persist_dir),  "✅" if settings.chroma_persist_dir.exists() else "⚠ not built"),
        ("Guardrails",     "enabled" if settings.guardrails_enabled else "disabled", "✅"),
        ("Top-K",          str(settings.retrieval_top_k),     "✅"),
        ("Hybrid Alpha",   str(settings.retrieval_hybrid_alpha),"✅"),
    ]
    for setting, value, status in checks:
        table.add_row(setting, value, status)
    console.print(table)

    # Corpus status
    from finrag.ingest.download_edgar_10k import load_manifest
    manifest = load_manifest()
    if manifest:
        console.print(f"\n[bold]Corpus:[/] {len(manifest)} filing(s) downloaded")
        ct = Table(show_header=True, header_style="bold green")
        ct.add_column("Doc ID", width=35)
        ct.add_column("Status", width=10)
        ct.add_column("Size (KB)", justify="right", width=10)
        ct.add_column("Filed", width=12)
        for entry in manifest:
            ct.add_row(
                entry.get("doc_id", ""),
                entry.get("status", ""),
                str(round(entry.get("file_size_kb", 0))),
                entry.get("filed_date", ""),
            )
        console.print(ct)
    else:
        console.print("\n[yellow]Corpus:[/] No filings yet. Run: [cyan]finrag ingest --all[/]")

    policy = FinancialQAPolicy()
    console.print(f"\n[bold]Safety Policy:[/] {len(policy._rules)} rules loaded")
    for rule in policy._rules:
        console.print(f"  [cyan]{rule.rule_id}[/] ({rule.severity.value})")


if __name__ == "__main__":
    app()
