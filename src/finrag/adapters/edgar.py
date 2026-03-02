"""
finrag.adapters.edgar
~~~~~~~~~~~~~~~~~~~~~~
SEC EDGAR API client for downloading 10-K filings.

SEC Fair-Use Rules (strictly followed):
  - Max 10 requests/second — we use 8 to be safe
  - User-Agent header required: "Name email@domain.com"
  - No bulk downloading — we fetch only what we need

EDGAR API flow for a 10-K:
  1. GET /cgi-bin/browse-edgar → resolve ticker → CIK
  2. GET /cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K → list filings
  3. GET filing index page → find primary document URL
  4. GET primary document (HTML) → save to disk

All downloaded files get:
  - SHA-256 hash for integrity verification
  - Stable doc_id: "{ticker}-{year}-{accession_short}"
  - Full provenance in meta.json

Windows-safe: all paths use pathlib.Path, no Unix-only calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from finrag.adapters.base import EDGARDownloaderBase
from finrag.core.config import settings
from finrag.core.logging import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EDGAR_BASE = "https://data.sec.gov"
EDGAR_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

# Sections we care about — used for targeted extraction in Day 3
TARGET_ITEMS = {"1", "1a", "1A", "7", "7a", "7A", "8"}


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Simple token-bucket rate limiter.
    Ensures we never exceed SEC's 10 req/sec limit.
    """

    def __init__(self, calls: int, period: float) -> None:
        self._calls = calls
        self._period = period
        self._timestamps: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        # Remove timestamps outside the window
        self._timestamps = [t for t in self._timestamps if now - t < self._period]
        if len(self._timestamps) >= self._calls:
            sleep_for = self._period - (now - self._timestamps[0])
            if sleep_for > 0:
                log.debug("edgar.rate_limit.wait", sleep_ms=round(sleep_for * 1000))
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ── EDGAR Client ──────────────────────────────────────────────────────────────

class EDGARDownloader(EDGARDownloaderBase):
    """
    Downloads SEC 10-K filings with rate limiting, retry, and provenance tracking.

    Usage:
        downloader = EDGARDownloader()
        manifest = downloader.download_10k("AAPL", 2023, Path("data/raw"))
    """

    def __init__(self) -> None:
        self._rate_limiter = _RateLimiter(
            calls=settings.edgar_rate_limit_calls,
            period=settings.edgar_rate_limit_period,
        )
        self._client = httpx.Client(
            headers={
                "User-Agent": settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def download_10k(
        self,
        ticker: str,
        year: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Download a 10-K filing for ticker/year. Skips if already downloaded.

        Returns:
            Manifest dict with doc_id, paths, hashes, URLs.
        """
        ticker = ticker.upper()
        log.info("edgar.download.start", ticker=ticker, year=year)

        # Step 1: Resolve CIK
        cik = self._get_cik(ticker)
        log.info("edgar.cik.resolved", ticker=ticker, cik=cik)

        # Step 2: Find the 10-K filing for the target fiscal year
        filing_meta = self._find_10k_filing(cik, ticker, year)
        if not filing_meta:
            raise ValueError(
                f"No 10-K filing found for {ticker} fiscal year {year}. "
                f"Try year-1 — companies file 10-Ks months after fiscal year end."
            )

        accession = filing_meta["accession"]
        accession_short = accession.replace("-", "")[:10]
        doc_id = f"{ticker}-{year}-{accession_short}"

        # Step 3: Check if already downloaded
        doc_dir = output_dir / doc_id
        manifest_path = doc_dir / "meta.json"

        if manifest_path.exists():
            log.info("edgar.download.skip", doc_id=doc_id, reason="already_exists")
            with open(manifest_path) as f:
                return json.load(f)

        # Step 4: Download primary document
        doc_dir.mkdir(parents=True, exist_ok=True)
        primary_url = self._get_primary_doc_url(cik, accession, filing_meta)
        html_path = doc_dir / "filing.htm"

        log.info("edgar.download.fetching", doc_id=doc_id, url=primary_url)
        html_content = self._fetch_with_retry(primary_url)

        with open(html_path, "wb") as f:
            f.write(html_content)

        # Step 5: Compute SHA-256 hash
        sha256 = hashlib.sha256(html_content).hexdigest()

        # Step 6: Build and save manifest
        manifest = {
            "doc_id": doc_id,
            "ticker": ticker,
            "cik": cik,
            "year": year,
            "accession": accession,
            "form_type": filing_meta.get("form", "10-K"),
            "filed_date": filing_meta.get("filed_date", ""),
            "fiscal_year_end": filing_meta.get("fiscal_year_end", ""),
            "file_path": str(html_path),
            "source_url": primary_url,
            "sha256": sha256,
            "file_size_bytes": len(html_content),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        log.info(
            "edgar.download.complete",
            doc_id=doc_id,
            size_kb=round(len(html_content) / 1024),
            sha256_prefix=sha256[:12],
        )
        return manifest

    def list_available(
        self,
        ticker: str,
        form: str = "10-K",
    ) -> list[dict[str, Any]]:
        """List available 10-K filings for a ticker (no download)."""
        ticker = ticker.upper()
        cik = self._get_cik(ticker)
        return self._list_filings(cik, form_type=form)

    # ── Known CIK map (fast path, no network needed for our corpus) ─────────
    _KNOWN_CIKS: dict[str, str] = {
        "AAPL":  "0000320193",
        "MSFT":  "0000789019",
        "GOOGL": "0001652044",
        "AMZN":  "0001018724",
        "NVDA":  "0001045810",
        # "META":  "0001326801",
        "TSLA":  "0001318605",
    }

    # ── Internal: CIK Resolution ──────────────────────────────────────────────

    def _get_cik(self, ticker: str) -> str:
        """
        Resolve ticker → zero-padded 10-digit CIK.

        Strategy (in order):
          1. Hardcoded map for Magnificent 7 (instant, no network)
          2. www.sec.gov/files/company_tickers.json  (correct host)
          3. www.sec.gov/files/company_tickers_exchange.json (backup)
        """
        ticker_upper = ticker.upper()

        # ── Fast path: hardcoded for our corpus ──────────────────────────────
        if ticker_upper in self._KNOWN_CIKS:
            cik = self._KNOWN_CIKS[ticker_upper]
            log.debug("edgar.cik.hardcoded", ticker=ticker_upper, cik=cik)
            return cik

        # ── Network path: correct host is www.sec.gov, NOT data.sec.gov ─────
        ticker_map_urls = [
            "https://www.sec.gov/files/company_tickers.json",
            "https://www.sec.gov/files/company_tickers_exchange.json",
        ]
        for url in ticker_map_urls:
            self._rate_limiter.wait()
            try:
                headers = {"User-Agent": settings.edgar_user_agent}
                resp = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
                resp.raise_for_status()
                data = resp.json()
                for entry in data.values():
                    if isinstance(entry, dict):
                        if entry.get("ticker", "").upper() == ticker_upper:
                            cik_raw = str(entry["cik_str"])
                            return cik_raw.zfill(10)
            except Exception as e:
                log.warning("edgar.cik.map_failed", url=url, ticker=ticker, error=str(e))
                continue

        raise ValueError(
            f"Could not resolve CIK for ticker: {ticker_upper}. "
            f"Add it to _KNOWN_CIKS in edgar.py or verify the ticker spelling."
        )

    # ── Internal: Filing Discovery ────────────────────────────────────────────

    def _list_filings(
        self,
        cik: str,
        form_type: str = "10-K",
    ) -> list[dict[str, Any]]:
        """Fetch submission history and extract 10-K filing metadata."""
        self._rate_limiter.wait()
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"

        resp = self._client.get(url)
        resp.raise_for_status()
        data = resp.json()

        filings: list[dict[str, Any]] = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        fiscal_ends = recent.get("fiscalYearEnd", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == form_type:
                filings.append({
                    "form": form,
                    "accession": accessions[i] if i < len(accessions) else "",
                    "filed_date": filed_dates[i] if i < len(filed_dates) else "",
                    "fiscal_year_end": fiscal_ends[i] if i < len(fiscal_ends) else "",
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                })

        return filings

    def _find_10k_filing(
        self,
        cik: str,
        ticker: str,
        year: int,
    ) -> dict[str, Any] | None:
        """
        Find the 10-K filing that covers fiscal year `year`.

        Key insight: Large companies routinely file through third-party agents
        (e.g. Edgar Filing Services, EDGAR Online). Their accession numbers
        start with the AGENT's CIK, not the company's CIK. This is 100% normal.
        We must NOT filter by accession prefix.

        Matching is purely date-based, using two signals:
          A) fiscal_year_end (MMDD string from EDGAR) — most reliable
          B) filed_date — secondary heuristic

        Filing calendar for each FY end type:
          Dec 31 FY (TSLA, META, GOOGL, AMZN):
            FY2023 filed Jan-Feb 2024  →  filed_year = year+1, month 1-4
          Jun 30 FY (MSFT):
            FY2023 filed Jul-Aug 2023  →  filed_year = year,   month 7-9
          Sep FY  (AAPL):
            FY2024 filed Oct-Nov 2024  →  filed_year = year,   month 10-12
          Jan FY  (NVDA):
            FY2024 (ends Jan 2024) filed Apr 2024 → filed_year = year, month 4
        """
        filings = self._list_filings(cik, "10-K")

        if not filings:
            log.warning("edgar.filing.none_found", ticker=ticker, cik=cik)
            return None

        log.debug(
            "edgar.filing.candidates",
            ticker=ticker,
            year=year,
            count=len(filings),
            dates=[f.get("filed_date") for f in filings[:6]],
        )

        scored: list[tuple[int, dict[str, Any]]] = []

        for filing in filings:
            filed_date = filing.get("filed_date", "")
            fiscal_end  = filing.get("fiscal_year_end", "")   # "MMDD" e.g. "0630"
            accession   = filing.get("accession", "")

            if not filed_date or not accession:
                continue

            filed_year  = int(filed_date[:4])
            filed_month = int(filed_date[5:7])
            score = 0

            # ── Primary signal: fiscal_year_end ───────────────────────────────
            if fiscal_end and len(fiscal_end) >= 4:
                fy_month = int(fiscal_end[:2])   # 1-12
                fy_day   = int(fiscal_end[2:4])

                # Companies whose FY ends in Jan-Jun file in the SAME calendar year
                # e.g. MSFT FY ends Jun 30 → files Jul-Sep of same year
                # e.g. NVDA FY ends Jan 26 → files Apr of same year
                if fy_month <= 6:
                    # filed in same calendar year as FY end
                    if filed_year == year and filed_month > fy_month:
                        score += 10
                    # or very early next year (amended filing)
                    elif filed_year == year + 1 and filed_month <= 3:
                        score += 7

                # Companies whose FY ends Jul-Sep file in same calendar year (Oct-Dec)
                # e.g. AAPL FY ends late Sep → files Oct-Nov same year
                elif fy_month <= 9:
                    if filed_year == year and filed_month >= fy_month:
                        score += 10
                    elif filed_year == year + 1 and filed_month <= 3:
                        score += 7

                # Companies whose FY ends Oct-Dec file NEXT calendar year (Jan-Apr)
                # e.g. TSLA/META/GOOGL/AMZN FY ends Dec 31 → files Jan-Feb year+1
                else:
                    if filed_year == year + 1 and filed_month <= 4:
                        score += 10
                    # Same-year edge case: fiscal year ends in Oct/Nov and
                    # company files very quickly (rare)
                    elif filed_year == year and filed_month >= 11:
                        score += 6

            # ── Secondary signal: filed_date heuristic (no fiscal_end info) ───
            else:
                # Without FY end info, use broad window:
                # FY year Y → filed between Jul Y and Apr Y+1
                if filed_year == year + 1 and 1 <= filed_month <= 4:
                    score += 6
                elif filed_year == year and 7 <= filed_month <= 12:
                    score += 5

            if score > 0:
                scored.append((score, filing))
                log.debug(
                    "edgar.filing.scored",
                    ticker=ticker,
                    accession=accession,
                    filed_date=filed_date,
                    fiscal_end=fiscal_end,
                    score=score,
                )

        if not scored:
            # Nothing matched the date window — log all available filings to help debug
            log.error(
                "edgar.filing.no_match",
                ticker=ticker,
                year=year,
                cik=cik,
                available_filings=[
                    {"filed": f.get("filed_date"), "fiscal_end": f.get("fiscal_year_end")}
                    for f in filings[:8]
                ],
            )
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_filing = scored[0]

        log.info(
            "edgar.filing.selected",
            ticker=ticker,
            year=year,
            accession=best_filing["accession"],
            filed_date=best_filing["filed_date"],
            fiscal_end=best_filing.get("fiscal_year_end"),
            score=best_score,
            candidates_scored=len(scored),
        )
        return best_filing


    # ── Internal: Document URL Resolution ────────────────────────────────────

    def _get_primary_doc_url(
        self,
        cik: str,
        accession: str,
        filing_meta: dict[str, Any],
    ) -> str:
        """
        Build the URL for the primary HTML document of a filing.

        EDGAR archive structure:
        /Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}
        """
        accession_nodash = accession.replace("-", "")
        primary_doc = filing_meta.get("primary_doc", "")

        if primary_doc:
            url = f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/{primary_doc}"
            return url

        # Fallback: fetch filing index to find primary document
        return self._find_primary_doc_from_index(cik, accession_nodash)

    def _find_primary_doc_from_index(
        self,
        cik: str,
        accession_nodash: str,
    ) -> str:
        """Fetch the filing index page to find the primary document."""
        self._rate_limiter.wait()
        index_url = (
            f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/{accession_nodash}-index.htm"
        )

        try:
            resp = self._client.get(index_url)
            resp.raise_for_status()
            content = resp.text

            # Look for the primary 10-K document link
            patterns = [
                r'href="([^"]+\.htm)"[^>]*>[^<]*10-K',
                r'href="([^"]+10k[^"]*\.htm)"',
                r'href="([^"]+annual[^"]*\.htm)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    doc_path = match.group(1)
                    if doc_path.startswith("/"):
                        return f"https://www.sec.gov{doc_path}"
                    return f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/{doc_path}"

        except Exception as e:
            log.warning("edgar.index.parse_failed", error=str(e))

        # Last resort: use JSON index
        self._rate_limiter.wait()
        json_index_url = (
            f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
        )
        return f"{EDGAR_ARCHIVES}/{cik}/{accession_nodash}/{accession_nodash}.htm"

    # ── Internal: HTTP ────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str) -> bytes:
        """Fetch a URL with exponential backoff retry."""
        self._rate_limiter.wait()

        # Use www.sec.gov client headers for archive URLs
        headers = {
            "User-Agent": settings.edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        resp = httpx.get(url, headers=headers, timeout=60.0, follow_redirects=True)

        if resp.status_code == 429:
            log.warning("edgar.rate_limited", url=url)
            time.sleep(10)
            resp = httpx.get(url, headers=headers, timeout=60.0, follow_redirects=True)

        resp.raise_for_status()
        return resp.content

    def __enter__(self) -> EDGARDownloader:
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()