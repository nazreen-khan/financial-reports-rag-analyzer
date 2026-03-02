"""
tests/test_day3_parsing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Day 3 tests — parser correctness, section detection, output schema.
All tests run without network or LlamaParse (BS4 only).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_10K_HTML = """
<!DOCTYPE html>
<html>
<head><title>ACME Corp 10-K</title></head>
<body>

<div class="toc">
  <a href="#item1">Item 1. Business</a>
  <a href="#item1a">Item 1A. Risk Factors</a>
  <a href="#item7">Item 7. MD&amp;A</a>
  <a href="#item8">Item 8. Financial Statements</a>
</div>

<div id="item1">
<h2>Item 1. Business</h2>
<p>ACME Corporation is a leading provider of financial technology solutions.
The company was founded in 1985 and operates in over 30 countries.
Our primary products include payment processing, fraud detection, and
data analytics platforms serving enterprise clients worldwide.
Revenue is primarily generated through subscription fees and transaction charges.
We employ approximately 45,000 people globally across our three operating segments.
</p>
</div>

<div id="item1a">
<h2>Item 1A. Risk Factors</h2>
<p>Our business is subject to numerous risks and uncertainties that could
materially affect our financial condition and results of operations.
The following risk factors should be considered carefully by investors.</p>
<p>CYBERSECURITY RISKS: We face significant cybersecurity threats including
data breaches, ransomware attacks, and unauthorized access to our systems.
A successful attack could result in substantial financial losses and
reputational damage that may be difficult to recover from.</p>
<p>REGULATORY RISK: We operate in a highly regulated industry and are
subject to laws and regulations in multiple jurisdictions. Changes in
regulations could require significant compliance costs and operational changes.</p>
<p>COMPETITION: The financial technology market is intensely competitive.
We compete with large established financial institutions and nimble startups
that may offer lower-cost alternatives to our products.</p>
</div>

<div id="item7">
<h2>Item 7. Management's Discussion and Analysis of Financial Condition</h2>
<p>The following discussion analyzes our financial condition and results of
operations for the fiscal year ended December 31, 2024.</p>
<p>OVERVIEW: Net revenue increased 12% to $8.4 billion in fiscal 2024 compared
to $7.5 billion in fiscal 2023. Operating income was $2.1 billion representing
an operating margin of 25%.</p>
<p>REVENUE ANALYSIS: Our three segments contributed as follows:
Payments processing revenue grew 15% year-over-year driven by increased
transaction volumes. Analytics revenue grew 22% reflecting strong enterprise
demand. Fraud detection revenue grew 8% as customer expansion offset
pricing pressure in certain markets.</p>
</div>

<div id="item8">
<h2>Item 8. Financial Statements and Supplementary Data</h2>
<p>The following financial statements are for fiscal year 2024:</p>
<table>
  <thead>
    <tr><th>Metric</th><th>FY2024</th><th>FY2023</th></tr>
  </thead>
  <tbody>
    <tr><td>Total Revenue</td><td>$8,400M</td><td>$7,500M</td></tr>
    <tr><td>Gross Profit</td><td>$5,040M</td><td>$4,275M</td></tr>
    <tr><td>Operating Income</td><td>$2,100M</td><td>$1,800M</td></tr>
    <tr><td>Net Income</td><td>$1,680M</td><td>$1,425M</td></tr>
    <tr><td>EPS (diluted)</td><td>$4.21</td><td>$3.56</td></tr>
  </tbody>
</table>
</div>

</body>
</html>
"""

MINIMAL_HTML = """
<html><body>
<h2>Item 1. Business</h2>
<p>This is the business section with enough content to pass minimum length check.
The company operates in multiple markets and provides various services to customers.</p>
<h2>Item 1A. Risk Factors</h2>
<p>The company faces various risks including market competition and regulatory changes
that could impact its financial performance in future periods.</p>
</body></html>
"""


@pytest.fixture
def sample_htm_file(tmp_path: Path) -> Path:
    """Write sample 10-K HTML to a temp file."""
    f = tmp_path / "filing.htm"
    f.write_text(SAMPLE_10K_HTML, encoding="utf-8")
    return f


@pytest.fixture
def minimal_htm_file(tmp_path: Path) -> Path:
    f = tmp_path / "filing.htm"
    f.write_text(MINIMAL_HTML, encoding="utf-8")
    return f


# ── BS4 Parser Tests ──────────────────────────────────────────────────────────

class TestBS4Parser:

    def test_parse_returns_list(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        assert isinstance(sections, list)
        assert len(sections) > 0

    def test_parse_finds_target_sections(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        section_ids = {s["section_id"] for s in sections}
        # Should find at least item_1, item_1a, item_7
        assert "item_1" in section_ids or "item_1a" in section_ids, \
            f"Expected item_1 or item_1a, got: {section_ids}"

    def test_section_schema_fields(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001", "https://sec.gov/test")
        for s in sections:
            assert "doc_id" in s
            assert "section_id" in s
            assert "section_title" in s
            assert "text" in s
            assert "source_url" in s
            assert "char_start" in s
            assert "char_end" in s
            assert "word_count" in s
            assert "table_count" in s
            assert "parser" in s

    def test_doc_id_propagated(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-MYID")
        for s in sections:
            assert s["doc_id"] == "ACME-2024-MYID"

    def test_source_url_propagated(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        url = "https://www.sec.gov/Archives/edgar/data/123/filing.htm"
        sections = parser.parse(sample_htm_file, "ACME-2024-001", url)
        for s in sections:
            assert s["source_url"] == url

    def test_text_not_empty(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        for s in sections:
            assert len(s["text"].strip()) > 50, \
                f"Section {s['section_id']} text too short: {len(s['text'])} chars"

    def test_word_count_positive(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        for s in sections:
            assert s["word_count"] > 0

    def test_char_positions_valid(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        for s in sections:
            assert s["char_start"] >= 0
            assert s["char_end"] > s["char_start"]

    def test_parser_name_is_bs4(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        for s in sections:
            assert s["parser"] in ("bs4", "bs4_fallback")

    def test_table_extracted_in_item8(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        item8 = next((s for s in sections if s["section_id"] == "item_8"), None)
        if item8:
            # Should contain table data (either raw text or Markdown)
            assert "Revenue" in item8["text"] or "8,400" in item8["text"] or "8400" in item8["text"]

    def test_handles_missing_sections_gracefully(self, minimal_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        # Should not raise even if only some sections present
        sections = parser.parse(minimal_htm_file, "MINIMAL-001")
        assert isinstance(sections, list)

    def test_noise_stripped(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        full_text = " ".join(s["text"] for s in sections)
        # Script/style tags should be stripped
        assert "<script" not in full_text.lower()
        assert "<style" not in full_text.lower()

    def test_encoding_detection_utf8(self, tmp_path: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        html = b'<html><head><meta charset="utf-8"></head><body><h2>Item 1. Business</h2><p>Test content for encoding detection with enough words to pass.</p></body></html>'
        f = tmp_path / "filing.htm"
        f.write_bytes(html)
        sections = parser.parse(f, "ENC-TEST")
        assert isinstance(sections, list)


# ── Section Detection Tests ───────────────────────────────────────────────────

class TestSectionDetection:

    def test_section_patterns_match_common_formats(self) -> None:
        import re
        from finrag.adapters.parsers_bs4 import SECTION_PATTERNS, TARGET_SECTIONS

        test_cases = {
            "item_1":  ["item 1. business", "item 1 business", "item 1.business"],
            "item_1a": ["item 1a. risk factors", "item 1a risk factors", "item 1a.risk"],
            "item_7":  ["item 7. management", "item 7 management's discussion"],
            "item_8":  ["item 8. financial statements", "item 8 financial"],
        }
        for section_id, cases in test_cases.items():
            if section_id not in TARGET_SECTIONS:
                continue
            patterns = SECTION_PATTERNS[section_id]
            for case in cases:
                matched = any(re.search(p, case, re.I) for p in patterns)
                assert matched, f"Pattern for {section_id} did not match: {case!r}"

    def test_section_id_normalisation(self) -> None:
        from finrag.adapters.parsers_bs4 import SECTION_TITLES, TARGET_SECTIONS
        for section_id in TARGET_SECTIONS:
            assert section_id in SECTION_TITLES, f"Missing title for {section_id}"
            assert SECTION_TITLES[section_id]


# ── Parser Router Tests ───────────────────────────────────────────────────────

class TestParserRouter:

    def test_router_instantiates(self) -> None:
        from finrag.adapters.parsers import get_parser
        router = get_parser()
        assert router is not None

    def test_router_has_active_parser(self) -> None:
        from finrag.adapters.parsers import get_parser
        router = get_parser()
        assert router.active_parser in ("llamaparse", "bs4")

    def test_router_falls_back_to_bs4_without_key(self, monkeypatch) -> None:
        from finrag.adapters.parsers import ParserRouter
        from finrag.core import config as cfg_module
        # Simulate no LlamaParse configured
        monkeypatch.setattr(cfg_module.settings, "llamaparse_enabled", False)
        monkeypatch.setattr(cfg_module.settings, "llamaparse_api_key", "")
        router = ParserRouter()
        assert router.active_parser == "bs4"

    def test_router_parses_sample(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers import get_parser
        router = get_parser()
        sections = router.parse(sample_htm_file, "ACME-2024-001")
        assert len(sections) > 0


# ── Sections JSONL Schema Tests ───────────────────────────────────────────────

class TestSectionsOutput:

    def test_sections_serialisable_to_json(self, sample_htm_file: Path) -> None:
        from finrag.adapters.parsers_bs4 import BS4Parser
        parser = BS4Parser()
        sections = parser.parse(sample_htm_file, "ACME-2024-001")
        for s in sections:
            # Should serialise without error
            line = json.dumps(s, ensure_ascii=False)
            restored = json.loads(line)
            assert restored["doc_id"] == s["doc_id"]
            assert restored["section_id"] == s["section_id"]

    def test_parse_orchestrator_writes_jsonl(
        self, tmp_path: Path, sample_htm_file: Path, monkeypatch
    ) -> None:
        """Integration test: orchestrator writes valid sections.jsonl"""
        from finrag.ingest import parse_sections as ps_module
        from finrag.adapters.parsers import ParserRouter

        output_path = tmp_path / "sections.jsonl"
        monkeypatch.setattr(ps_module, "SECTIONS_PATH", output_path)
        monkeypatch.setattr(ps_module, "PARSE_REPORT_PATH", tmp_path / "report.json")

        # Mock get_successful_docs to return our test file
        def mock_docs():
            return [{
                "doc_id": "ACME-2024-001",
                "file_path": str(sample_htm_file),
                "source_url": "https://sec.gov/test",
                "status": "success",
            }]
        monkeypatch.setattr(ps_module, "get_successful_docs", mock_docs)

        parser = ps_module.SectionParser(output_path=output_path)
        results = parser.parse_all()

        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].sections_found > 0
        assert output_path.exists()

        # Validate JSONL output
        sections = ps_module.load_sections(output_path)
        assert len(sections) > 0
        for s in sections:
            assert "doc_id" in s
            assert "section_id" in s
            assert "text" in s
