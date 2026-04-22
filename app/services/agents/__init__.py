"""LangGraph-based agentic RAG pipeline.

The graph implements four cooperating nodes with distinct responsibilities:

* ``planner``      – analyses the question, decides which tools to invoke and
                     decomposes it into focused sub-queries (Chain-of-Thought).
* ``tool_executor`` – calls the ``retriever_tool`` and (optionally) ``sql_tool``
                     for the currently pending sub-queries.
* ``reflector``    – judges whether the evidence is sufficient and, if not,
                     proposes the next hop of sub-queries (multi-hop).
* ``synthesizer``  – grounded CoT reasoning that returns the final
                     :class:`~app.models.schemas.AnalyticalQueryResponse`.

Public entry point: :func:`run_analytical_query`.
"""
from app.services.agents.graph import run_analytical_query

__all__ = ["run_analytical_query"]
