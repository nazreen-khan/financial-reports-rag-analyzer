"""
finrag.adapters.parsers_llamaparse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LlamaParse-based parser for SEC 10-K filings.

LlamaParse is the primary parser when available. It significantly outperforms
BeautifulSoup for:
  - Complex financial tables (income statements, balance sheets)
  - Multi-column layouts common in 10-K filings
  - Preserving table structure as clean Markdown rows/columns
  - Handling footnotes and nested table relationships

API: https://cloud.llamaindex.ai (free tier: 1000 pages/day)
Requires: pip install llama-parse

Pipeline:
  1. Upload HTML filing to LlamaParse API
  2. Poll for completion (async job)
  3. Retrieve Markdown result
  4. Run section detector on clean Markdown
  5. Return structured sections
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from finrag.core.config import settings
from finrag.core.logging import get_logger

log = get_logger(__name__)

# Re-use section detection logic from BS4 parser
from finrag.adapters.parsers_bs4 import (
    SECTION_PATTERNS,
    SECTION_TITLES,
    TARGET_SECTIONS,
)


class LlamaParseParser:
    """
    LlamaParse-based parser for SEC 10-K filings.

    Uses LlamaParse cloud API to convert HTML → clean Markdown with
    superior table reconstruction, then runs section detection on the result.

    Falls back to BS4 automatically if:
      - API key not set
      - API call fails
      - Result is unexpectedly empty
    """

    def __init__(self) -> None:
        self._api_key = settings.llamaparse_api_key
        self._enabled = settings.llamaparse_enabled

    def is_available(self) -> bool:
        """Returns True if LlamaParse is configured and importable."""
        if not self._enabled or not self._api_key:
            return False
        try:
            import llama_parse  # noqa: F401
            return True
        except ImportError:
            log.warning(
                "llamaparse.not_installed",
                hint="Run: pip install llama-parse",
            )
            return False

    def parse(
        self,
        file_path: Path,
        doc_id: str,
        source_url: str = "",
    ) -> list[dict[str, Any]]:
        """
        Parse a 10-K filing using LlamaParse.

        Returns structured sections with clean Markdown text.
        Raises RuntimeError if parsing fails (caller should fall back to BS4).
        """
        log.info("parser.llamaparse.start", doc_id=doc_id, file=str(file_path))

        markdown_text = self._call_llamaparse(file_path, doc_id)

        if not markdown_text or len(markdown_text) < 500:
            raise RuntimeError(
                f"LlamaParse returned empty/short result for {doc_id}: "
                f"{len(markdown_text)} chars"
            )

        # Run section detection on the clean Markdown
        boundaries = self._detect_section_boundaries(markdown_text)

        if not boundaries:
            log.warning(
                "parser.llamaparse.no_sections",
                doc_id=doc_id,
                text_length=len(markdown_text),
            )
            # Return whole document — better than nothing
            return self._whole_doc_section(doc_id, markdown_text, source_url)

        sections = self._extract_sections(
            doc_id=doc_id,
            markdown_text=markdown_text,
            boundaries=boundaries,
            source_url=source_url,
        )

        log.info(
            "parser.llamaparse.complete",
            doc_id=doc_id,
            sections_found=len(sections),
            section_ids=[s["section_id"] for s in sections],
            total_chars=len(markdown_text),
        )
        return sections

    # ── LlamaParse API Call ───────────────────────────────────────────────────

    def _call_llamaparse(self, file_path: Path, doc_id: str) -> str:
        """
        Upload file to LlamaParse and retrieve clean Markdown.

        LlamaParse parsing instructions are tuned for SEC 10-K filings:
        - Preserve all financial tables as Markdown
        - Extract section headings (Item 1, 1A, 7, 8…)
        - Keep footnotes inline with relevant tables
        - Do not summarise or paraphrase — return verbatim content
        """
        from llama_parse import LlamaParse
        from llama_index.core import SimpleDirectoryReader

        parsing_instruction = """
        This is a SEC 10-K annual report filing. Please:
        1. Preserve ALL financial tables as proper Markdown tables with aligned columns.
           Tables contain critical financial data (revenue, income, EPS, cash flows).
        2. Keep section headings exactly as they appear (e.g. "Item 1. Business",
           "Item 1A. Risk Factors", "Item 7. MD&A", "Item 8. Financial Statements").
        3. Preserve all numerical values exactly — do not round or abbreviate.
        4. Keep footnotes and table notes inline, marked with (*) or (1) etc.
        5. Do not summarise, paraphrase, or omit any content.
        6. For multi-column financial tables, ensure each row maps correctly
           to its column headers (fiscal year labels like 2024, 2023, 2022).
        """

        parser = LlamaParse(
            api_key=self._api_key,
            result_type="markdown",
            parsing_instruction=parsing_instruction,
            verbose=False,
            language="en",
            # Premium mode for best table extraction
            premium_mode=True,
        )

        log.info("parser.llamaparse.uploading", doc_id=doc_id, size_kb=file_path.stat().st_size // 1024)

        # LlamaParse works best with PDF — convert HTML to temp file approach
        # For HTML files, we use the file reader directly
        file_extractor = {".htm": parser, ".html": parser}
        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor=file_extractor,
        )

        docs = reader.load_data()

        if not docs:
            raise RuntimeError(f"LlamaParse returned no documents for {doc_id}")

        # Concatenate all pages
        full_markdown = "\n\n".join(doc.text for doc in docs if doc.text)

        log.info(
            "parser.llamaparse.received",
            doc_id=doc_id,
            pages=len(docs),
            chars=len(full_markdown),
        )
        return full_markdown

    # ── Section Detection (same logic as BS4, applied to clean Markdown) ─────

    def _detect_section_boundaries(
        self,
        markdown_text: str,
    ) -> list[dict[str, Any]]:
        """
        Find section boundaries in LlamaParse Markdown output.

        LlamaParse preserves headings, so we also check for Markdown
        heading syntax (## Item 7.) in addition to plain text patterns.
        """
        boundaries: list[dict[str, Any]] = []
        text_lower = markdown_text.lower()

        # Add Markdown heading variants to patterns
        md_prefix = r"(?:#{1,4}\s*)?"  # optional ## prefix

        for section_id, patterns in SECTION_PATTERNS.items():
            if section_id not in TARGET_SECTIONS:
                continue

            # Prepend markdown heading prefix to each pattern
            extended_patterns = [
                md_prefix + p for p in patterns
            ] + patterns  # also try without prefix

            for pattern in extended_patterns:
                matches = list(re.finditer(pattern, text_lower))
                if not matches:
                    continue

                # Skip TOC entries (first 3000 chars)
                real_matches = [m for m in matches if m.start() > 3000]
                if not real_matches:
                    real_matches = matches

                match = real_matches[0]
                boundaries.append({
                    "section_id": section_id,
                    "section_title": SECTION_TITLES.get(section_id, section_id),
                    "char_start": match.start(),
                    "pattern_matched": pattern,
                })
                break

        boundaries.sort(key=lambda x: x["char_start"])

        # Deduplicate
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for b in boundaries:
            if b["section_id"] not in seen:
                seen.add(b["section_id"])
                unique.append(b)

        return unique

    def _extract_sections(
        self,
        doc_id: str,
        markdown_text: str,
        boundaries: list[dict[str, Any]],
        source_url: str,
    ) -> list[dict[str, Any]]:
        """Slice Markdown text into sections using detected boundaries."""
        sections: list[dict[str, Any]] = []

        for i, boundary in enumerate(boundaries):
            section_id = boundary["section_id"]
            char_start = boundary["char_start"]
            char_end = (
                boundaries[i + 1]["char_start"]
                if i + 1 < len(boundaries)
                else len(markdown_text)
            )

            text = markdown_text[char_start:char_end].strip()

            if len(text) < 100:
                log.warning(
                    "parser.section.too_short",
                    doc_id=doc_id,
                    section_id=section_id,
                    length=len(text),
                )
                continue

            # Count Markdown tables in this section
            table_count = len(re.findall(r"^\|.+\|$", text, re.MULTILINE))

            sections.append({
                "doc_id": doc_id,
                "section_id": section_id,
                "section_title": boundary["section_title"],
                "text": text,
                "source_url": source_url,
                "char_start": char_start,
                "char_end": char_end,
                "word_count": len(text.split()),
                "table_count": table_count,
                "parser": "llamaparse",
            })

        return sections

    def _whole_doc_section(
        self,
        doc_id: str,
        markdown_text: str,
        source_url: str,
    ) -> list[dict[str, Any]]:
        return [{
            "doc_id": doc_id,
            "section_id": "item_1",
            "section_title": "Full Document",
            "text": markdown_text,
            "source_url": source_url,
            "char_start": 0,
            "char_end": len(markdown_text),
            "word_count": len(markdown_text.split()),
            "table_count": len(re.findall(r"^\|.+\|$", markdown_text, re.MULTILINE)),
            "parser": "llamaparse_no_sections",
        }]

    @property
    def parser_name(self) -> str:
        return "llamaparse"
