"""
Tool-executor node — runs the retriever (per pending sub-query) and SQL tools.
"""
from __future__ import annotations

import time

from app.core.logging import get_logger
from app.services.agents import tools
from app.services.agents.helpers import short
from app.services.agents.state import AgentState, Retrieval, SqlAttempt
from app.services.agents.trace import emit

logger = get_logger(__name__)


def tool_executor_node(state: AgentState) -> dict:
    """Run the retriever (per pending sub-query) and SQL tools."""
    plan = state.get("plan") or {}
    pending: list[str] = state.get("pending_sub_queries") or []
    iteration: int = state.get("iteration", 0)
    qid = state.get("qid", "-")
    new_errors: list[str] = []
    t0 = time.perf_counter()

    hop_label = "hop-1 (initial)" if iteration == 0 else f"hop-{iteration + 1} (multi-hop reflection-driven)"
    logger.info("[%s] ▶ EXECUTOR %s  pending=%d", qid, hop_label, len(pending))
    emit("node.start", node="executor", qid=qid, hop=iteration + 1, hop_label=hop_label, pending=pending)

    new_retrievals: list[Retrieval] = []
    if plan.get("use_retriever") and pending:
        for sq in pending:
            t_sq = time.perf_counter()
            try:
                chunks = tools.retriever_tool(sq)
                new_retrievals.append({"sub_query": sq, "chunks": chunks})
                if chunks:
                    top = chunks[0]
                    logger.info(
                        "[%s]   retriever_tool(%r) → %d chunks  top=[%s p%s dist=%.3f via=%s] %s",
                        qid, short(sq, 80), len(chunks),
                        top.get("source_file", "?"),
                        top.get("page_number") or "?",
                        top.get("_distance", 1.0),
                        "+".join(top.get("_retrieval_sources", []) or ["?"]),
                        short(top.get("chunk_text", ""), 120),
                    )
                    emit(
                        "executor.tool", qid=qid, tool="retriever", sub_query=sq,
                        chunks=len(chunks),
                        top_file=top.get("source_file"),
                        top_page=top.get("page_number"),
                        top_distance=top.get("_distance"),
                        top_via=top.get("_retrieval_sources", []),
                        top_excerpt=short(top.get("chunk_text", ""), 220),
                    )
                else:
                    logger.info("[%s]   retriever_tool(%r) → 0 chunks (no hits)", qid, short(sq, 80))
                    emit("executor.tool", qid=qid, tool="retriever", sub_query=sq, chunks=0)
                logger.debug("[%s]   retriever_tool(%r) took %.0fms", qid, sq, (time.perf_counter() - t_sq) * 1000)
            except Exception as exc:  # noqa: BLE001
                new_errors.append(f"retriever_tool({sq!r}) failed: {exc}")
                new_retrievals.append({"sub_query": sq, "chunks": []})
                logger.warning("[%s]   retriever_tool(%r) FAILED: %s", qid, short(sq, 80), exc)
                emit("executor.tool_error", qid=qid, tool="retriever", sub_query=sq, error=str(exc))

    # SQL only fires on the first hop (iteration == 0) to avoid duplicate work.
    new_sql: list[SqlAttempt] = []
    if iteration == 0 and plan.get("use_sql") and plan.get("sql_question"):
        try:
            resp = tools.sql_tool(plan["sql_question"])
            new_sql.append({"question": plan["sql_question"], "response": resp.model_dump()})
            row_count = len(resp.result or [])
            if resp.error:
                logger.warning(
                    "[%s]   sql_tool(%r) returned error: %s",
                    qid, short(plan["sql_question"], 80), resp.error,
                )
                emit("executor.tool_error", qid=qid, tool="sql",
                     question=plan["sql_question"], error=resp.error)
            else:
                logger.info(
                    "[%s]   sql_tool(%r) → %d rows  sql=%s",
                    qid, short(plan["sql_question"], 80), row_count, short(resp.sql, 200),
                )
                emit("executor.tool", qid=qid, tool="sql",
                     question=plan["sql_question"], rows=row_count, sql=resp.sql)
        except Exception as exc:  # noqa: BLE001
            new_errors.append(f"sql_tool failed: {exc}")
            logger.warning("[%s]   sql_tool FAILED: %s", qid, exc)
            emit("executor.tool_error", qid=qid, tool="sql", error=str(exc))

    # Component catalog is small — load once on the first hop.
    components = state.get("components") or []
    if iteration == 0 and not components:
        components = tools.components_tool()
        logger.info("[%s]   components_tool() → %d catalog rows", qid, len(components))
        emit("executor.tool", qid=qid, tool="components", rows=len(components))

    total_chunks = sum(len(r["chunks"]) for r in new_retrievals)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] ✓ EXECUTOR %s done in %.0fms  hop_chunks=%d sql_attempts=%d errors=%d",
        qid, hop_label, elapsed, total_chunks, len(new_sql), len(new_errors),
    )
    emit("node.done", node="executor", qid=qid, hop=iteration + 1,
         hop_chunks=total_chunks, sql_attempts=len(new_sql),
         errors=len(new_errors), elapsed_ms=int(elapsed))
    return {
        "retrievals": new_retrievals,
        "sql_attempts": new_sql,
        "components": components,
        "errors": new_errors,
        "iteration": iteration + 1,
        "pending_sub_queries": [],
    }
