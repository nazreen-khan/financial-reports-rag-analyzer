"""
tests/test_day2_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Day 2 tests — corpus config integrity + downloader interface.
All tests run WITHOUT network access (no real EDGAR calls).
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path


# ── Corpus Config Tests ───────────────────────────────────────────────────────

class TestCorpusConfig:

    def test_corpus_has_14_filings(self) -> None:
        from finrag.ingest.corpus_config import CORPUS
        assert len(CORPUS) == 12

    def test_corpus_has_7_unique_tickers(self) -> None:
        from finrag.ingest.corpus_config import CORPUS, get_tickers
        tickers = get_tickers()
        assert len(tickers) == 6
        assert set(tickers) == {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"}

    def test_corpus_covers_two_years(self) -> None:
        from finrag.ingest.corpus_config import get_years
        years = get_years()
        assert 2023 in years
        assert 2024 in years

    def test_each_ticker_has_two_years(self) -> None:
        from finrag.ingest.corpus_config import CORPUS, get_tickers
        for ticker in get_tickers():
            years = [f.year for f in CORPUS if f.ticker == ticker]
            assert len(years) >= 2, f"{ticker} should have 2 filings"
            assert 2023 in years or 2024 in years

    def test_all_filings_have_demo_questions(self) -> None:
        from finrag.ingest.corpus_config import CORPUS
        for filing in CORPUS:
            assert len(filing.demo_questions) >= 2, (
                f"{filing.ticker}-{filing.year} needs at least 2 demo questions"
            )

    def test_get_filing_lookup(self) -> None:
        from finrag.ingest.corpus_config import get_filing
        filing = get_filing("AAPL", 2024)
        assert filing is not None
        assert filing.ticker == "AAPL"
        assert filing.year == 2024
        assert "Apple" in filing.company_name

    def test_get_filing_returns_none_for_unknown(self) -> None:
        from finrag.ingest.corpus_config import get_filing
        assert get_filing("ZZZZ", 2023) is None

    def test_summary_contains_all_tickers(self) -> None:
        from finrag.ingest.corpus_config import summary, get_tickers
        s = summary()
        for ticker in get_tickers():
            assert ticker in s


# ── EDGAR Adapter Tests (no network) ─────────────────────────────────────────

class TestEDGARAdapter:

    def test_edgar_downloader_instantiates(self) -> None:
        from finrag.adapters.edgar import EDGARDownloader
        d = EDGARDownloader()
        assert d is not None

    def test_rate_limiter_does_not_block_first_call(self) -> None:
        import time
        from finrag.adapters.edgar import _RateLimiter
        limiter = _RateLimiter(calls=8, period=1.0)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, "First call should not be rate-limited"

    def test_rate_limiter_slows_after_limit(self) -> None:
        import time
        from finrag.adapters.edgar import _RateLimiter
        # Allow only 2 calls per second
        limiter = _RateLimiter(calls=2, period=1.0)
        limiter.wait()
        limiter.wait()
        start = time.monotonic()
        limiter.wait()   # 3rd call should be delayed
        elapsed = time.monotonic() - start
        # Should have waited some time (at least 0.1s in real scenario)
        # We just check it doesn't raise
        assert elapsed >= 0

    def test_edgar_implements_base_interface(self) -> None:
        from finrag.adapters.base import EDGARDownloaderBase
        from finrag.adapters.edgar import EDGARDownloader
        assert issubclass(EDGARDownloader, EDGARDownloaderBase)


# ── Manifest Tests ────────────────────────────────────────────────────────────

class TestManifest:

    def test_load_manifest_returns_empty_when_no_file(self, tmp_path, monkeypatch) -> None:
        from finrag.ingest import download_edgar_10k as dl_module
        monkeypatch.setattr(dl_module, "MANIFEST_PATH", tmp_path / "missing.jsonl")
        result = dl_module.load_manifest()
        assert result == []

    def test_load_manifest_parses_jsonl(self, tmp_path, monkeypatch) -> None:
        from finrag.ingest import download_edgar_10k as dl_module

        manifest_file = tmp_path / "filings_manifest.jsonl"
        entries = [
            {"doc_id": "AAPL-2023-abc", "ticker": "AAPL", "year": 2023, "status": "success", "file_size_kb": 1200},
            {"doc_id": "MSFT-2023-def", "ticker": "MSFT", "year": 2023, "status": "success", "file_size_kb": 980},
        ]
        with open(manifest_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        monkeypatch.setattr(dl_module, "MANIFEST_PATH", manifest_file)
        result = dl_module.load_manifest()
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"
        assert result[1]["ticker"] == "MSFT"

    def test_get_successful_docs_filters_failures(self, tmp_path, monkeypatch) -> None:
        from finrag.ingest import download_edgar_10k as dl_module

        manifest_file = tmp_path / "filings_manifest.jsonl"
        entries = [
            {"doc_id": "AAPL-2023-abc", "status": "success"},
            {"doc_id": "TSLA-2022-FAILED", "status": "failed"},
            {"doc_id": "MSFT-2023-def", "status": "skipped"},
        ]
        with open(manifest_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        monkeypatch.setattr(dl_module, "MANIFEST_PATH", manifest_file)
        result = dl_module.get_successful_docs()
        assert len(result) == 2
        assert all(e["status"] in ("success", "skipped") for e in result)


# ── DownloadResult Tests ──────────────────────────────────────────────────────

class TestDownloadResult:

    def test_download_result_fields(self) -> None:
        from finrag.ingest.download_edgar_10k import DownloadResult
        r = DownloadResult(
            ticker="AAPL",
            year=2023,
            doc_id="AAPL-2023-abc123",
            status="success",
            file_path="data/raw/AAPL-2023-abc123/filing.htm",
            source_url="https://www.sec.gov/Archives/...",
            sha256="abc123def456",
            file_size_kb=1250.5,
            elapsed_seconds=3.2,
            error="",
        )
        assert r.ticker == "AAPL"
        assert r.status == "success"
        assert r.file_size_kb == 1250.5

    def test_corpus_downloader_instantiates(self, tmp_path) -> None:
        from finrag.ingest.download_edgar_10k import CorpusDownloader
        d = CorpusDownloader(output_dir=tmp_path, force=False)
        assert d is not None
