"""
finrag.core.tracing
~~~~~~~~~~~~~~~~~~~
Request-scoped trace context using Python contextvars.

Every log line emitted during a single query shares the same request_id.
This is the foundation for distributed tracing and production debugging.

Usage:
    from finrag.core.tracing import trace_context, new_trace

    with new_trace(query="What was Apple's revenue?") as ctx:
        # ctx.request_id is a ULID string — sortable + unique
        retrieve(...)   # all logs inside here carry ctx.request_id
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generator

from ulid import ULID


# ── Trace Dataclass ───────────────────────────────────────────────────────────

@dataclass
class TraceContext:
    """Immutable snapshot of metadata for one request."""
    request_id: str
    query: str
    model: str = ""
    top_k: int = 0
    start_time: float = field(default_factory=time.monotonic)
    refusal: bool = False
    safety_triggered: bool = False
    agent_path: list[str] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000

    def record_agent(self, agent_name: str) -> None:
        """Track which agents fired during this request."""
        self.agent_path.append(agent_name)

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "query_preview": self.query[:120],
            "model": self.model,
            "top_k": self.top_k,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "refusal": self.refusal,
            "safety_triggered": self.safety_triggered,
            "agent_path": self.agent_path,
        }


# ── ContextVar (thread + async safe) ─────────────────────────────────────────

_current_trace: ContextVar[TraceContext | None] = ContextVar(
    "_current_trace", default=None
)


def get_current_trace() -> TraceContext | None:
    """Returns the active TraceContext for this coroutine/thread, or None."""
    return _current_trace.get()


def get_request_id() -> str:
    """Returns current request_id or 'no-trace' if called outside a trace."""
    ctx = _current_trace.get()
    return ctx.request_id if ctx else "no-trace"


# ── Context manager ────────────────────────────────────────────────────────────

@contextmanager
def new_trace(
    query: str,
    model: str = "",
    top_k: int = 0,
) -> Generator[TraceContext, None, None]:
    """
    Creates a new TraceContext and binds it to the current execution context.

    Example:
        with new_trace(query="What was revenue?", model="gpt-4o-mini") as ctx:
            answer = rag_pipeline(query)
            # ctx.elapsed_ms is available here
    """
    ctx = TraceContext(
        request_id=str(ULID()),
        query=query,
        model=model,
        top_k=top_k,
    )
    token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        _current_trace.reset(token)
