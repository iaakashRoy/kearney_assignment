"""Lightweight trace bus for the agentic graph.

The graph nodes call :func:`emit` to publish structured events
(planner.start, executor.tool, reflector.verdict, etc.). When an HTTP
request opts into streaming, the route installs a callback via
:func:`use_emitter` (a :class:`contextvars.ContextVar`) that forwards
events to an asyncio queue for SSE delivery. Outside that context the
emitter is a no-op, so logging-only callers are unaffected.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

Emitter = Callable[[str, dict[str, Any]], None]

_current: ContextVar[Emitter | None] = ContextVar("agent_trace_emitter", default=None)


def emit(event: str, **payload: Any) -> None:
    """Publish an event to the active emitter, if any."""
    cb = _current.get()
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:  # noqa: BLE001 — never let tracing break the pipeline
        pass


@contextmanager
def use_emitter(cb: Emitter) -> Iterator[None]:
    """Install ``cb`` as the active emitter for the duration of the block."""
    token = _current.set(cb)
    try:
        yield
    finally:
        _current.reset(token)
