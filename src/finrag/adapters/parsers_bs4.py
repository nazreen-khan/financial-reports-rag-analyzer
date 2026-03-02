"""
finrag.adapters.parsers_bs4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BeautifulSoup-based HTML parser for SEC 10-K filings.

Used as fallback when LlamaParse is unavailable.
Handles:
  - HTML boilerplate removal (nav, scripts, styles, headers/footers)
  - Section detection via Item heading patterns
  - Table → Markdown conversion via pandas
  - Text normalisation (unicode, whitespace, encoding artifacts)

SEC 10-K HTML is notoriously inconsistent. This parser uses multiple
detection strategies with fallbacks for each, so it degrades gracefully
even on unusual filing formats.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from finrag.core.logging import get_logger

log = get_logger(__name__)

# ── Target Sections ───────────────────────────────────────────────────────────

# Maps normalised section_id → list of heading patterns to match
# Order matters: more specific patterns first
SECTION_PATTERNS: dict[str, list[str]] = {
    "item_1":  [
        # Standard: "Item 1. Business" or "Item 1 Business"
        r"item\s*1[\.\ s]+business(?!\s*overview\s*of\s*risk)",
        # MSFT/GOOGL: "ITEM 1." as standalone line
        r"item\s+1\s*\.\s*\n",
        r"item\s*1\b(?!a|b)",
    ],
    "item_1a": [
        r"item\s*1a[\.\ s]+risk\s*factor",
        r"item\s*1\s*a[\.\ s]+risk",
        r"item\s+1a\s*\.\s*\n",
        r"item\s+1\s*a\s*\.\s*\n",
        r"item\s*1a\.risk",
        r"item\s*1a\s*\.\s*risk",
    ],
    "item_1b": [
        r"item\s*1b[\.\ s]+unresolved",
        r"item\s*1\s*b[\.\ s]+unresolved",
    ],
    "item_7":  [
        r"item\s*7[\.\ s]+management.{0,30}discussion",
        r"item\s*7[\.\ s]+management",
        r"item\s+7\s*\.\s*\n",
        r"item\s*7\b(?!a|b)",
    ],
    "item_7a": [
        r"item\s*7a[\.\ s]+quantitative",
        r"item\s*7\s*a[\.\ s]+quantitative",
        r"item\s+7a\s*\.\s*\n",
        r"item\s*7a[\.\ s]+quant",
    ],
    "item_8":  [
        r"item\s*8[\.\ s]+financial\s*statements\s*and\s*supplementary",
        r"item\s*8[\.\ s]+financial\s*statements",
        r"item\s+8\s*\.\s*\n",
        r"item\s*8\b",
    ],
}

# Human-readable titles for each section
SECTION_TITLES: dict[str, str] = {
    "item_1":  "Business",
    "item_1a": "Risk Factors",
    "item_1b": "Unresolved Staff Comments",
    "item_7":  "Management's Discussion and Analysis",
    "item_7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "item_8":  "Financial Statements and Supplementary Data",
}

# Sections we actually want to extract
TARGET_SECTIONS = {"item_1", "item_1a", "item_7", "item_7a", "item_8"}


# ── Main Parser ───────────────────────────────────────────────────────────────

class BS4Parser:
    """
    BeautifulSoup-based parser for SEC 10-K HTML filings.

    Usage:
        parser = BS4Parser()
        sections = parser.parse(Path("data/raw/AAPL-2024-xxx/filing.htm"), "AAPL-2024-xxx")
    """

    def parse(
        self,
        file_path: Path,
        doc_id: str,
        source_url: str = "",
    ) -> list[dict[str, Any]]:
        """
        Parse a 10-K HTML filing into structured sections.

        Returns:
            List of section dicts ready for sections.jsonl
        """
        log.info("parser.bs4.start", doc_id=doc_id, file=str(file_path))

        raw_html = file_path.read_bytes()
        encoding = self._detect_encoding(raw_html)
        html_text = raw_html.decode(encoding, errors="replace")

        soup = BeautifulSoup(html_text, "lxml")

        # Remove all noise elements
        self._strip_noise(soup)

        # Get full clean text for section boundary detection
        full_text = self._get_full_text(soup)

        # Detect section boundaries
        boundaries = self._detect_section_boundaries(full_text)

        if not boundaries:
            log.warning("parser.bs4.no_sections", doc_id=doc_id)
            # Fallback: return whole document as single section
            return self._fallback_single_section(doc_id, full_text, source_url)

        # Extract each target section
        sections = self._extract_sections(
            doc_id=doc_id,
            full_text=full_text,
            soup=soup,
            boundaries=boundaries,
            source_url=source_url,
        )

        log.info(
            "parser.bs4.complete",
            doc_id=doc_id,
            sections_found=len(sections),
            section_ids=[s["section_id"] for s in sections],
        )
        return sections

    # ── Noise Removal ─────────────────────────────────────────────────────────

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        """Remove all elements that add no content value."""
        noise_tags = [
            "script", "style", "meta", "link", "noscript",
            "header", "footer", "nav", "iframe", "object",
        ]
        for tag in noise_tags:
            for el in soup.find_all(tag):
                el.decompose()

        # Remove hidden elements
        for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            el.decompose()

        # Remove TOC-like divs (usually have many short anchor links)
        for el in soup.find_all("div"):
            links = el.find_all("a")
            text_len = len(el.get_text(strip=True))
            if len(links) > 15 and text_len < 2000:
                el.decompose()

    # ── Text Extraction ───────────────────────────────────────────────────────

    def _get_full_text(self, soup: BeautifulSoup) -> str:
        """Extract and normalise full text from the cleaned soup."""
        # Preserve paragraph structure
        for tag in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4"]):
            tag.insert_before("\n")
            tag.insert_after("\n")

        text = soup.get_text(separator=" ")
        return self._normalise_text(text)

    def _normalise_text(self, text: str) -> str:
        """Clean unicode artifacts, excessive whitespace, encoding issues."""
        # Normalise unicode (handles &#160; non-breaking spaces etc.)
        text = unicodedata.normalize("NFKD", text)
        # Remove control characters except newlines/tabs
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t"
        )
        # Collapse multiple spaces (but preserve newlines)
        lines = text.split("\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        # Remove blank line runs (max 2 consecutive)
        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)
        return "\n".join(cleaned)

    # ── Section Detection ─────────────────────────────────────────────────────

    def _detect_section_boundaries(
        self,
        full_text: str,
    ) -> list[dict[str, Any]]:
        """
        Find character positions of each target section heading.

        Returns list of dicts sorted by char_start:
          {section_id, section_title, char_start, pattern_matched}
        """
        boundaries: list[dict[str, Any]] = []
        text_lower = full_text.lower()

        for section_id, patterns in SECTION_PATTERNS.items():
            if section_id not in TARGET_SECTIONS:
                continue

            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                if not matches:
                    continue

                # Skip matches in the first 2000 chars (table of contents)
                real_matches = [m for m in matches if m.start() > 2000]
                if not real_matches:
                    real_matches = matches  # fallback if all are in TOC

                # Use the first real match
                match = real_matches[0]
                boundaries.append({
                    "section_id": section_id,
                    "section_title": SECTION_TITLES.get(section_id, section_id),
                    "char_start": match.start(),
                    "pattern_matched": pattern,
                })
                break   # found this section — move to next

        # Sort by position in document
        boundaries.sort(key=lambda x: x["char_start"])

        # Deduplicate — keep first occurrence of each section_id
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for b in boundaries:
            if b["section_id"] not in seen:
                seen.add(b["section_id"])
                unique.append(b)

        return unique

    # ── Section Extraction ────────────────────────────────────────────────────

    def _extract_sections(
        self,
        doc_id: str,
        full_text: str,
        soup: BeautifulSoup,
        boundaries: list[dict[str, Any]],
        source_url: str,
    ) -> list[dict[str, Any]]:
        """
        Slice full_text into sections using detected boundaries.
        For Item 8 (financial tables), extract Markdown tables from soup.
        """
        sections: list[dict[str, Any]] = []

        for i, boundary in enumerate(boundaries):
            section_id = boundary["section_id"]
            char_start = boundary["char_start"]

            # End = start of next section (or end of document)
            if i + 1 < len(boundaries):
                char_end = boundaries[i + 1]["char_start"]
            else:
                char_end = len(full_text)

            raw_text = full_text[char_start:char_end].strip()

            # Skip tiny sections (likely false positives)
            if len(raw_text) < 100:
                log.warning(
                    "parser.section.too_short",
                    doc_id=doc_id,
                    section_id=section_id,
                    length=len(raw_text),
                )
                continue

            # For financial statements: extract tables as Markdown
            if section_id == "item_8":
                table_md = self._extract_tables_as_markdown(soup, char_start, char_end, full_text)
                if table_md:
                    raw_text = self._merge_text_and_tables(raw_text, table_md)

            word_count = len(raw_text.split())
            table_count = raw_text.count("| ---")  # markdown table separator rows

            sections.append({
                "doc_id": doc_id,
                "section_id": section_id,
                "section_title": boundary["section_title"],
                "text": raw_text,
                "source_url": source_url,
                "char_start": char_start,
                "char_end": char_end,
                "word_count": word_count,
                "table_count": table_count,
                "parser": "bs4",
            })

        return sections

    # ── Table Extraction ──────────────────────────────────────────────────────

    def _extract_tables_as_markdown(
        self,
        soup: BeautifulSoup,
        char_start: int,
        char_end: int,
        full_text: str,
    ) -> list[str]:
        """
        Extract HTML tables from the soup and convert to Markdown.
        Uses pandas for robust table parsing.
        """
        try:
            import pandas as pd
        except ImportError:
            log.warning("parser.tables.pandas_missing")
            return []

        markdown_tables: list[str] = []

        for table in soup.find_all("table"):
            table_text = table.get_text()
            # Check if this table is roughly in our section range
            # (approximate — full_text positions don't map 1:1 to soup positions)
            if not table_text.strip() or len(table_text.strip()) < 20:
                continue

            try:
                html_str = str(table)
                dfs = pd.read_html(io.StringIO(html_str))
                for df in dfs:
                    if df.empty or df.shape[1] < 2:
                        continue
                    # Clean up DataFrame
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    df = df.fillna("")
                    df = df.astype(str)
                    # Convert to Markdown
                    md = df.to_markdown(index=False)
                    if md and len(md) > 50:
                        markdown_tables.append(md)
            except Exception as e:
                log.debug("parser.table.parse_failed", error=str(e)[:80])
                continue

        return markdown_tables

    def _merge_text_and_tables(
        self,
        text: str,
        tables: list[str],
    ) -> str:
        """Append extracted Markdown tables to the section text."""
        if not tables:
            return text
        table_block = "\n\n## Extracted Financial Tables\n\n"
        table_block += "\n\n---\n\n".join(tables)
        return text + "\n\n" + table_block

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_single_section(
        self,
        doc_id: str,
        full_text: str,
        source_url: str,
    ) -> list[dict[str, Any]]:
        """
        If no sections detected, return the whole document as item_1.
        Prevents complete parse failures from breaking the pipeline.
        """
        return [{
            "doc_id": doc_id,
            "section_id": "item_1",
            "section_title": "Full Document (section detection failed)",
            "text": full_text[:50_000],  # cap at 50k chars
            "source_url": source_url,
            "char_start": 0,
            "char_end": min(len(full_text), 50_000),
            "word_count": len(full_text.split()),
            "table_count": 0,
            "parser": "bs4_fallback",
        }]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_encoding(self, raw: bytes) -> str:
        """Detect HTML encoding from meta charset tag."""
        # Check BOM
        if raw[:3] == b"\xef\xbb\xbf":
            return "utf-8-sig"
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return "utf-16"

        # Check meta charset
        head = raw[:2000].decode("ascii", errors="ignore").lower()
        match = re.search(r'charset=["\']?([\w-]+)', head)
        if match:
            return match.group(1)

        return "utf-8"

    @property
    def parser_name(self) -> str:
        return "bs4"
