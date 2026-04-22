"""
LangGraph state machine that orchestrates the analytical query pipeline.

  START ──▶ planner ──▶ tool_executor ──▶ reflector
                              ▲                │
                              │                ▼
                              └──── (loop) ──┐ │
                                             │ │
                                          synthesizer ──▶ END

Each node mutates a small ``AgentState`` (TypedDict). Mutations to
list-typed fields use ``operator.add`` reducers so multi-hop iterations
accumulate evidence rather than overwrite it.

Node implementations live in :mod:`app.services.agents.nodes` — this
module only assembles the graph and exposes the public entry point
:func:`run_analytical_query`.
"""
from __future__ import annotations

import time
import uuid

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.models.schemas import AnalyticalQueryResponse
from app.services.agents.helpers import short
from app.services.agents.nodes import (
    planner_node,
    reflector_node,
    synthesizer_node,
    tool_executor_node,
)
from app.services.agents.state import AgentState, MAX_HOPS
from app.services.agents.trace import emit

logger = get_logger(__name__)


# ─── Routing ──────────────────────────────────────────────────────────────────

def _route_after_reflection(state: AgentState) -> str:
    refl = state.get("reflection") or {}
    if not refl.get("sufficient", True) and state.get("pending_sub_queries"):
        return "tool_executor"
    return "synthesizer"


# ─── Graph factory ────────────────────────────────────────────────────────────

_compiled_graph = None


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("tool_executor", tool_executor_node)
    g.add_node("reflector", reflector_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "tool_executor")
    g.add_edge("tool_executor", "reflector")
    g.add_conditional_edges(
        "reflector",
        _route_after_reflection,
        {"tool_executor": "tool_executor", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)
    return g.compile()


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


# ─── Public entry point ───────────────────────────────────────────────────────

def run_analytical_query(question: str, max_hops: int = MAX_HOPS) -> AnalyticalQueryResponse:
    """
    Execute the agentic graph end-to-end and return the final response.

    On catastrophic failure (e.g. LangGraph itself crashes) we still return a
    valid :class:`AnalyticalQueryResponse` so the API contract holds.
    """
    qid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    logger.info(
        "[%s] ═══ analytical_query start ═══  max_hops=%d  question=%r",
        qid, max(1, max_hops), short(question, 300),
    )
    emit("run.start", qid=qid, question=question, max_hops=max(1, max_hops))
    initial: AgentState = {
        "question": question,
        "max_iterations": max(1, max_hops),
        "qid": qid,
    }
    try:
        final_state = _get_graph().invoke(initial)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] agent graph crashed for question=%r", qid, question)
        return AnalyticalQueryResponse(
            answer="The analytical pipeline encountered an unexpected error.",
            reasoning=None,
            confidence="none",
            grounding_warning=str(exc),
            plan=[question],
            iterations=0,
            sql_used=False,
        )

    final = final_state.get("final")
    if final:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[%s] ═══ analytical_query done in %.0fms  iterations=%s confidence=%s sources=%d ═══",
            qid, elapsed_ms,
            final.get("iterations"), final.get("confidence"), len(final.get("sources") or []),
        )
        emit("run.done", qid=qid, elapsed_ms=int(elapsed_ms),
             iterations=final.get("iterations"), confidence=final.get("confidence"),
             sources=len(final.get("sources") or []), final=final)
        return AnalyticalQueryResponse.model_validate(final)

    # Should not happen, but fail safe.
    logger.warning("[%s] graph terminated with empty final state", qid)
    return AnalyticalQueryResponse(
        answer="No answer was produced by the pipeline.",
        confidence="none",
        grounding_warning="empty terminal state",
        plan=[question],
        iterations=final_state.get("iteration", 0),
        sql_used=bool(final_state.get("sql_attempts")),
    )
