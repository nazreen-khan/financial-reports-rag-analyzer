"""
finrag.adapters.parsers
~~~~~~~~~~~~~~~~~~~~~~~~
Parser router — selects the best available parser for each filing.

Priority:
  1. LlamaParse  (if configured + available) → best quality, especially tables
  2. BS4         (always available)           → good quality, free, local

The router is transparent: callers just call parser.parse() and get the
best result available. The parser_name field in each section records which
parser was actually used, so you can measure quality differences in eval.

Usage:
    from finrag.adapters.parsers import get_parser

    parser = get_parser()
    sections = parser.parse(file_path, doc_id, source_url)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finrag.adapters.parsers_bs4 import BS4Parser
from finrag.core.config import settings
from finrag.core.logging import get_logger

log = get_logger(__name__)


class ParserRouter:
    """
    Routes each filing to the best available parser.

    Tries LlamaParse first. If it fails or is unavailable,
    falls back to BS4 transparently.
    """

    def __init__(self) -> None:
        self._bs4 = BS4Parser()
        self._llamaparse = None

        # Lazily initialise LlamaParse only if configured
        if settings.llamaparse_available:
            try:
                from finrag.adapters.parsers_llamaparse import LlamaParseParser
                lp = LlamaParseParser()
                if lp.is_available():
                    self._llamaparse = lp
                    log.info("parser.router.llamaparse_ready")
                else:
                    log.info("parser.router.llamaparse_unavailable")
            except Exception as e:
                log.warning("parser.router.llamaparse_init_failed", error=str(e))
        else:
            log.info(
                "parser.router.bs4_only",
                reason="LLAMAPARSE_ENABLED=false or no API key",
            )

    def parse(
        self,
        file_path: Path,
        doc_id: str,
        source_url: str = "",
    ) -> list[dict[str, Any]]:
        """
        Parse a filing using the best available parser.

        Tries LlamaParse → falls back to BS4 on any failure.
        """
        # ── Primary: LlamaParse ───────────────────────────────────────────────
        if self._llamaparse is not None:
            try:
                sections = self._llamaparse.parse(file_path, doc_id, source_url)
                if sections:
                    log.info(
                        "parser.router.used_llamaparse",
                        doc_id=doc_id,
                        sections=len(sections),
                    )
                    return sections
                log.warning(
                    "parser.router.llamaparse_empty",
                    doc_id=doc_id,
                    fallback="bs4",
                )
            except Exception as e:
                log.warning(
                    "parser.router.llamaparse_failed",
                    doc_id=doc_id,
                    error=str(e)[:200],
                    fallback="bs4",
                )

        # ── Fallback: BS4 ─────────────────────────────────────────────────────
        log.info("parser.router.using_bs4", doc_id=doc_id)
        return self._bs4.parse(file_path, doc_id, source_url)

    @property
    def active_parser(self) -> str:
        """Returns name of the primary active parser."""
        return "llamaparse" if self._llamaparse else "bs4"


def get_parser() -> ParserRouter:
    """Factory function — returns a configured ParserRouter."""
    return ParserRouter()
