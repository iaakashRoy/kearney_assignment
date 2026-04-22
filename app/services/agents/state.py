"""
Shared state definitions and constants for the agentic graph.

The :class:`AgentState` ``TypedDict`` is the single payload that flows between
the planner, executor, reflector and synthesizer nodes. List-typed fields use
``operator.add`` reducers so multi-hop iterations *append* evidence rather
than overwrite it.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


# Planner hop + at most one reflector-driven extra hop.
MAX_HOPS = 2


class Plan(TypedDict, total=False):
    rationale: str
    sub_queries: list[str]
    use_retriever: bool
    use_sql: bool
    sql_question: str | None


class Retrieval(TypedDict):
    sub_query: str
    chunks: list[dict]


class SqlAttempt(TypedDict):
    question: str
    response: dict


class Reflection(TypedDict):
    sufficient: bool
    missing: str
    next_sub_queries: list[str]


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    max_iterations: int
    qid: str  # short correlation id for log lines
    # planner
    plan: Plan | None
    # working set
    pending_sub_queries: list[str]
    iteration: int
    # accumulated evidence (use ``add`` reducer so multi-hop iterations append)
    retrievals: Annotated[list[Retrieval], add]
    sql_attempts: Annotated[list[SqlAttempt], add]
    components: list[dict]
    errors: Annotated[list[str], add]
    # reflector
    reflection: Reflection | None
    # output
    final: dict | None
