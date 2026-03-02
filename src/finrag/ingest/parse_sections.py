"""
finrag.ingest.parse_sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates parsing all downloaded filings into structured sections.

Input:  data/raw/{doc_id}/filing.htm  +  data/raw/filings_manifest.jsonl
Output: data/processed/sections.jsonl  (one JSON line per section)

Each output line:
{
  "doc_id":        "AAPL-2024-0000320193",
  "section_id":    "item_1a",
  "section_title": "Risk Factors",
  "text":          "clean text or markdown...",
  "source_url":    "https://www.sec.gov/Archives/...",
  "char_start":    12450,
  "char_end":      34821,
  "word_count":    3241,
  "table_count":   0,
  "parser":        "llamaparse"   or "bs4"
}

Usage:
  python -m finrag.ingest.parse_sections          # parse all downloaded
  python -m finrag.ingest.parse_sections --doc-id AAPL-2024-xxx  # single
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finrag.adapters.parsers import get_parser
from finrag.core.config import settings
from finrag.core.logging import get_logger, setup_logging
from finrag.ingest.download_edgar_10k import get_successful_docs

log = get_logger(__name__)

SECTIONS_PATH = settings.data_processed_dir / "sections.jsonl"
PARSE_REPORT_PATH = settings.data_processed_dir / "parse_report.json"


# ── Result Tracking ───────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    doc_id: str
    status: str          # "success" | "failed" | "skipped"
    sections_found: int
    section_ids: list[str]
    parser_used: str
    elapsed_seconds: float
    error: str


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SectionParser:
    """
    Parses all downloaded filings into sections.jsonl.

    Skips already-parsed docs unless force=True.
    Writes sections immediately after each doc — crash safe.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        force: bool = False,
    ) -> None:
        self._output_path = output_path or SECTIONS_PATH
        self._force = force
        self._parser = get_parser()
        log.info(
            "section_parser.init",
            active_parser=self._parser.active_parser,
            output=str(self._output_path),
        )

    def parse_all(
        self,
        manifest_entries: list[dict[str, Any]] | None = None,
    ) -> list[ParseResult]:
        """
        Parse all successfully downloaded filings.

        Args:
            manifest_entries: If None, loads from filings_manifest.jsonl

        Returns:
            List of ParseResult for every attempted filing.
        """
        entries = manifest_entries or get_successful_docs()

        if not entries:
            log.warning("section_parser.no_docs", hint="Run: finrag ingest --all first")
            return []

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load already-parsed doc_ids to support skip logic
        already_parsed = self._get_already_parsed_doc_ids()

        results: list[ParseResult] = []

        log.info(
            "section_parser.start",
            total=len(entries),
            already_parsed=len(already_parsed),
            force=self._force,
        )

        for i, entry in enumerate(entries, 1):
            print("="*100)
            doc_id = entry.get("doc_id", "")
            print(f"Processing {i}: {doc_id}")
            log.info(
                "section_parser.progress",
                doc_id=doc_id,
                progress=f"{i}/{len(entries)}",
            )

            # Skip if already parsed
            if not self._force and doc_id in already_parsed:
                log.info("section_parser.skip", doc_id=doc_id)
                results.append(ParseResult(
                    doc_id=doc_id,
                    status="skipped",
                    sections_found=0,
                    section_ids=[],
                    parser_used="",
                    elapsed_seconds=0.0,
                    error="",
                ))
                continue

            result = self._parse_one(entry)
            results.append(result)

        self._write_report(results)
        self._print_summary(results)
        return results

    def parse_one_by_doc_id(self, doc_id: str) -> ParseResult:
        """Parse a single filing by its doc_id."""
        docs = get_successful_docs()
        entry = next((d for d in docs if d.get("doc_id") == doc_id), None)
        if not entry:
            raise ValueError(f"doc_id not found in manifest: {doc_id}")
        return self._parse_one(entry)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_one(self, entry: dict[str, Any]) -> ParseResult:
        doc_id = entry.get("doc_id", "")
        file_path = Path(entry.get("file_path", ""))
        source_url = entry.get("source_url", "")
        start = time.monotonic()

        if not file_path.exists():
            log.error("section_parser.file_missing", doc_id=doc_id, path=str(file_path))
            return ParseResult(
                doc_id=doc_id,
                status="failed",
                sections_found=0,
                section_ids=[],
                parser_used="",
                elapsed_seconds=0.0,
                error=f"File not found: {file_path}",
            )

        try:
            sections = self._parser.parse(file_path, doc_id, source_url)
            elapsed = time.monotonic() - start

            # Write sections to JSONL immediately
            print(f"Writing for {doc_id}; sections: {len(sections)}")
            self._append_sections(sections)
            print(f"Writing completed for {doc_id}")
            
            section_ids = [s["section_id"] for s in sections]
            log.info(
                "section_parser.success",
                doc_id=doc_id,
                sections=len(sections),
                section_ids=section_ids,
                elapsed_s=round(elapsed, 2),
            )

            return ParseResult(
                doc_id=doc_id,
                status="success",
                sections_found=len(sections),
                section_ids=section_ids,
                parser_used=sections[0].get("parser", "") if sections else "",
                elapsed_seconds=round(elapsed, 2),
                error="",
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            log.error(
                "section_parser.failed",
                doc_id=doc_id,
                error=str(e),
                elapsed_s=round(elapsed, 2),
            )
            return ParseResult(
                doc_id=doc_id,
                status="failed",
                sections_found=0,
                section_ids=[],
                parser_used="",
                elapsed_seconds=round(elapsed, 2),
                error=str(e),
            )

    def _append_sections(self, sections: list[dict[str, Any]]) -> None:
        """Append sections to JSONL file — one section per line."""
        with open(self._output_path, "a", encoding="utf-8") as f:
            for section in sections:
                f.write(json.dumps(section, ensure_ascii=False) + "\n")

    def _get_already_parsed_doc_ids(self) -> set[str]:
        """Load doc_ids already present in sections.jsonl."""
        if not self._output_path.exists():
            return set()
        doc_ids: set[str] = set()
        try:
            with open(self._output_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        doc_ids.add(entry.get("doc_id", ""))
        except Exception as e:
            log.warning("section_parser.manifest_read_error", error=str(e))
        return doc_ids

    def _write_report(self, results: list[ParseResult]) -> None:
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "failed"]
        skipped = [r for r in results if r.status == "skipped"]

        report = {
            "total": len(results),
            "success": len(successes),
            "skipped": len(skipped),
            "failed": len(failures),
            "total_sections": sum(r.sections_found for r in successes),
            "parsers_used": list({r.parser_used for r in successes if r.parser_used}),
            "sections_per_doc": {
                r.doc_id: r.sections_found for r in successes
            },
            "failures": [
                {"doc_id": r.doc_id, "error": r.error}
                for r in failures
            ],
        }

        PARSE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PARSE_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

    def _print_summary(self, results: list[ParseResult]) -> None:
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "failed"]
        skipped = [r for r in results if r.status == "skipped"]
        total_sections = sum(r.sections_found for r in successes)
        parsers = {r.parser_used for r in successes if r.parser_used}

        print("\n" + "=" * 60)
        print("  SECTION PARSING COMPLETE")
        print("=" * 60)
        print(f"  ✅ Parsed    : {len(successes)} filings")
        print(f"  ⏭  Skipped   : {len(skipped)} (already parsed)")
        print(f"  ❌ Failed    : {len(failures)}")
        print(f"  📄 Sections  : {total_sections} total")
        print(f"  🔧 Parser(s) : {', '.join(parsers) or 'none'}")
        print(f"  📁 Output    : {self._output_path}")
        print("=" * 60)

        if successes:
            print("\n  Sections per filing:")
            for r in successes:
                ids = ", ".join(r.section_ids)
                print(f"    {r.doc_id:40s} {r.sections_found} sections  [{ids}]")

        if failures:
            print("\n  Failures:")
            for r in failures:
                print(f"    ❌ {r.doc_id}: {r.error[:80]}")
        print()


# ── Helpers for downstream use ────────────────────────────────────────────────

def load_sections(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all sections from sections.jsonl."""
    p = path or SECTIONS_PATH
    if not p.exists():
        return []
    sections = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sections.append(json.loads(line))
    return sections


def get_sections_for_doc(doc_id: str) -> list[dict[str, Any]]:
    """Return all sections for a specific doc_id."""
    return [s for s in load_sections() if s.get("doc_id") == doc_id]


def get_sections_by_type(section_id: str) -> list[dict[str, Any]]:
    """Return all sections of a given type across all docs."""
    return [s for s in load_sections() if s.get("section_id") == section_id]


# ── CLI Entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    settings.ensure_dirs()
    parser = SectionParser()
    results = parser.parse_all()
    failed = [r for r in results if r.status == "failed"]
    import sys
    sys.exit(1 if failed else 0)
