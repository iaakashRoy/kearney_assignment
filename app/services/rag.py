"""
Public entry point for analytical (RAG) queries.

The actual orchestration lives in :mod:`app.services.agents.graph` — a
LangGraph state machine with four cooperating nodes (planner, tool_executor,
reflector, synthesizer) and two tools (``retriever_tool``, ``sql_tool``).

This module is kept as a thin wrapper so existing callers
(``app/api/routes/query.py``) and tests continue to work unchanged.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.models.schemas import AnalyticalQueryResponse
from app.services.agents import run_analytical_query
from app.services.agents.tools import (
    components_tool,
    retriever_tool,
    sql_tool,
)

logger = get_logger(__name__)


def analytical_query(question: str) -> AnalyticalQueryResponse:
    """Run the agentic graph and return a grounded, cited answer."""
    return run_analytical_query(question)


# ─── Backwards-compatible helpers (used by older code paths / tests) ──────────

def retrieve(question: str) -> dict:
    """
    Legacy single-shot hybrid retrieval — kept so that ad-hoc callers and
    older tests keep working. New code should use the agentic graph instead.
    """
    chunks = retriever_tool(question)
    return {
        "chunks": chunks,
        "components": components_tool(),
        "sub_queries": [question],
    }


__all__ = [
    "analytical_query",
    "retrieve",
    "retriever_tool",
    "sql_tool",
    "components_tool",
]
