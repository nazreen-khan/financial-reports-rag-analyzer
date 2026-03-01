"""
finrag.ingest.download_edgar_10k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates downloading the full corpus of 14 SEC 10-K filings.

Outputs:
  data/raw/{doc_id}/filing.htm      — raw HTML filing
  data/raw/{doc_id}/meta.json       — provenance metadata
  data/raw/filings_manifest.jsonl   — master manifest (one line per filing)
  data/raw/download_report.json     — success/failure summary

Usage (via CLI):
  finrag ingest --ticker AAPL --year 2023          # single filing
  finrag ingest --all                               # full corpus
  finrag ingest --all --force                       # re-download everything

Usage (direct):
  python -m finrag.ingest.download_edgar_10k
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finrag.adapters.edgar import EDGARDownloader
from finrag.core.config import settings
from finrag.core.logging import get_logger, setup_logging
from finrag.ingest.corpus_config import CORPUS, FilingTarget, summary

log = get_logger(__name__)

MANIFEST_PATH = settings.data_raw_dir / "filings_manifest.jsonl"
REPORT_PATH = settings.data_raw_dir / "download_report.json"


# ── Result Tracking ───────────────────────────────────────────────────────────

@dataclass
class DownloadResult:
    ticker: str
    year: int
    doc_id: str
    status: str          # "success" | "skipped" | "failed"
    file_path: str
    source_url: str
    sha256: str
    file_size_kb: float
    elapsed_seconds: float
    error: str


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CorpusDownloader:
    """
    Downloads all filings defined in corpus_config.CORPUS.
    Writes a manifest entry for every filing — success or failure.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> None:
        self._output_dir = output_dir or settings.data_raw_dir
        self._force = force
        self._downloader = EDGARDownloader()

    def download_all(
        self,
        targets: list[FilingTarget] | None = None,
    ) -> list[DownloadResult]:
        """
        Download all filings in the corpus (or a subset).

        Args:
            targets: If None, downloads full CORPUS. Pass a subset to filter.

        Returns:
            List of DownloadResult for every attempted filing.
        """
        targets = targets or CORPUS
        results: list[DownloadResult] = []

        log.info(
            "corpus.download.start",
            total=len(targets),
            output_dir=str(self._output_dir),
            force=self._force,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        for i, target in enumerate(targets, 1):
            log.info(
                "corpus.download.progress",
                filing=f"{target.ticker}-{target.year}",
                progress=f"{i}/{len(targets)}",
            )

            result = self._download_one(target)
            results.append(result)

            # Write manifest entry immediately — don't lose progress on crash
            self._append_manifest(result)

            # Brief pause between filings to be a good SEC citizen
            if i < len(targets):
                time.sleep(0.5)

        # Write final report
        self._write_report(results)
        self._print_summary(results)

        return results

    def download_one(self, ticker: str, year: int) -> DownloadResult:
        """Download a single filing by ticker + year."""
        from finrag.ingest.corpus_config import get_filing
        target = get_filing(ticker, year)
        if not target:
            # Create an ad-hoc target if not in corpus
            target = FilingTarget(
                ticker=ticker.upper(),
                year=year,
                company_name=ticker.upper(),
                industry="Unknown",
            )
        result = self._download_one(target)
        self._append_manifest(result)
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _download_one(self, target: FilingTarget) -> DownloadResult:
        start = time.monotonic()

        # If force, remove existing dir
        if self._force:
            import shutil
            ticker_year_dirs = list(self._output_dir.glob(f"{target.ticker}-{target.year}-*"))
            for d in ticker_year_dirs:
                shutil.rmtree(d, ignore_errors=True)
                log.info("corpus.force_delete", path=str(d))

        try:
            manifest = self._downloader.download_10k(
                ticker=target.ticker,
                year=target.year,
                output_dir=self._output_dir,
            )

            elapsed = time.monotonic() - start
            status = "skipped" if elapsed < 0.1 else "success"

            return DownloadResult(
                ticker=target.ticker,
                year=target.year,
                doc_id=manifest["doc_id"],
                status=status,
                file_path=manifest["file_path"],
                source_url=manifest["source_url"],
                sha256=manifest["sha256"],
                file_size_kb=round(manifest["file_size_bytes"] / 1024, 1),
                elapsed_seconds=round(elapsed, 2),
                error="",
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            log.error(
                "corpus.download.failed",
                ticker=target.ticker,
                year=target.year,
                error=str(e),
            )
            return DownloadResult(
                ticker=target.ticker,
                year=target.year,
                doc_id=f"{target.ticker}-{target.year}-FAILED",
                status="failed",
                file_path="",
                source_url="",
                sha256="",
                file_size_kb=0.0,
                elapsed_seconds=round(elapsed, 2),
                error=str(e),
            )

    def _append_manifest(self, result: DownloadResult) -> None:
        """Append one line to the JSONL manifest."""
        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def _write_report(self, results: list[DownloadResult]) -> None:
        """Write a summary JSON report."""
        successes = [r for r in results if r.status in ("success", "skipped")]
        failures = [r for r in results if r.status == "failed"]

        report = {
            "total": len(results),
            "success": len([r for r in results if r.status == "success"]),
            "skipped": len([r for r in results if r.status == "skipped"]),
            "failed": len(failures),
            "total_size_mb": round(sum(r.file_size_kb for r in results) / 1024, 2),
            "failures": [
                {"ticker": r.ticker, "year": r.year, "error": r.error}
                for r in failures
            ],
        }

        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        log.info("corpus.report.written", path=str(REPORT_PATH))

    def _print_summary(self, results: list[DownloadResult]) -> None:
        successes = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failures = sum(1 for r in results if r.status == "failed")
        total_mb = sum(r.file_size_kb for r in results) / 1024

        print("\n" + "=" * 60)
        print(f"  CORPUS DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"  ✅ Downloaded : {successes}")
        print(f"  ⏭  Skipped    : {skipped} (already existed)")
        print(f"  ❌ Failed     : {failures}")
        print(f"  📦 Total size : {total_mb:.1f} MB")
        print(f"  📄 Manifest   : {MANIFEST_PATH}")
        print("=" * 60)

        if failures:
            print("\n  Failed filings:")
            for r in results:
                if r.status == "failed":
                    print(f"    ❌ {r.ticker}-{r.year}: {r.error[:80]}")
            print()


# ── Manifest Reader ───────────────────────────────────────────────────────────

def load_manifest() -> list[dict[str, Any]]:
    """Load all entries from filings_manifest.jsonl."""
    if not MANIFEST_PATH.exists():
        return []
    entries = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_successful_docs() -> list[dict[str, Any]]:
    """Return only successfully downloaded manifest entries."""
    return [e for e in load_manifest() if e.get("status") in ("success", "skipped")]


# ── CLI Entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    settings.ensure_dirs()

    print(summary())
    print()

    downloader = CorpusDownloader()
    results = downloader.download_all()

    failed = [r for r in results if r.status == "failed"]
    sys.exit(1 if failed else 0)
