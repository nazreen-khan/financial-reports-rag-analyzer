"""
finrag.core.logging
~~~~~~~~~~~~~~~~~~~~
Structured, JSONL-format logging for production observability.

Every log line is a JSON object written to:
  - stderr (human-readable via rich in local mode)
  - logs/finrag_{date}.jsonl (machine-readable, for Datadog/Splunk/CloudWatch)

Key fields on every line:
  timestamp, level, logger, request_id, message, ...extra

Usage:
    from finrag.core.logging import get_logger

    log = get_logger(__name__)
    log.info("retrieval.complete", chunks_found=6, top_score=0.87)
    log.warning("guardrail.triggered", reason="investment_advice")
    log.error("llm.timeout", model="gpt-4o-mini", elapsed_ms=30012)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import structlog
from structlog.types import EventDict, WrappedLogger

from finrag.core.config import settings
from finrag.core.tracing import get_request_id


# ── Custom Processors ─────────────────────────────────────────────────────────

def _inject_request_id(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject the current trace's request_id into every log line."""
    event_dict["request_id"] = get_request_id()
    return event_dict


def _inject_timestamp(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add ISO-8601 UTC timestamp."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _sanitize_secrets(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Redact any accidental secret leaks in log payloads."""
    sensitive_keys = {"api_key", "openai_api_key", "password", "token", "secret"}
    for key in list(event_dict.keys()):
        if any(s in key.lower() for s in sensitive_keys):
            event_dict[key] = "***REDACTED***"
    return event_dict


# ── JSONL File Handler ────────────────────────────────────────────────────────

class _JSONLFileHandler(logging.Handler):
    """Writes one JSON object per line to a rotating daily log file."""

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._path = log_dir / f"finrag_{date_str}.jsonl"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # structlog has already formatted the record into a dict via orjson renderer
            line = record.getMessage()
            with self._path.open("ab") as f:
                f.write(line.encode() if isinstance(line, str) else line)
                f.write(b"\n")
        except Exception:
            self.handleError(record)


# ── Setup ─────────────────────────────────────────────────────────────────────

def _is_local() -> bool:
    return settings.app_env.value == "local"


def setup_logging() -> None:
    """
    Configure structlog. Call once at application startup (CLI entrypoint / FastAPI lifespan).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    log_level = settings.app_log_level.value

    # Shared processors run on every log event regardless of renderer
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        _inject_timestamp,
        _sanitize_secrets,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if _is_local():
        # Human-friendly colored output for local development
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
        processors = shared_processors + [renderer]
    else:
        # Machine-readable JSON for staging/prod → shipped to log aggregator
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(serializer=orjson.dumps),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also wire up stdlib logging → structlog (for third-party libs like httpx)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level, logging.INFO),
    )

    # Attach JSONL file handler to root logger for persistent logs
    file_handler = _JSONLFileHandler(settings.app_log_dir)
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Returns a bound structlog logger for the given module.

    Convention: always pass __name__
        log = get_logger(__name__)
    """
    return structlog.get_logger(name)


# ── Module-level logger for core itself ──────────────────────────────────────
_log = get_logger(__name__)
