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

The graph is deliberately compact — every node has a single responsibility
and is independently testable.
"""
from __future__ import annotations

import json
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.schemas import AnalyticalQueryResponse, SourceReference
from app.services import llm
from app.services.agents import tools
from app.services.agents.prompts import (
    PLANNER_PROMPT,
    REFLECTOR_PROMPT,
    SYNTH_SYSTEM_PROMPT,
    SYNTH_USER_PROMPT,
)
from app.utils.text import extract_json

logger = get_logger(__name__)

_MAX_HOPS = 2  # planner hop + at most one reflector-driven extra hop


# ─── State ────────────────────────────────────────────────────────────────────

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _flatten_chunks(retrievals: list[Retrieval]) -> list[dict]:
    """Merge multi-hop retrievals, deduping on (source_file, chunk_index)."""
    best: dict[tuple, dict] = {}
    for r in retrievals:
        for c in r.get("chunks", []):
            key = (c.get("source_file"), c.get("chunk_index"))
            existing = best.get(key)
            if existing is None or c.get("_rrf_score", 0) > existing.get("_rrf_score", 0):
                # remember which sub-query first surfaced this chunk
                merged = dict(c)
                merged.setdefault("_sub_query", r.get("sub_query", ""))
                best[key] = merged
    out = list(best.values())
    out.sort(key=lambda c: (-c.get("_rrf_score", 0.0), c.get("_distance", 1.0)))
    return out[: settings.rag_top_k * 2]


def _determine_confidence(chunks: list[dict]) -> tuple[str, str | None]:
    if not chunks:
        return "none", None
    best = min(c.get("_distance", 1.0) for c in chunks)
    if best > settings.rag_distance_low:
        return "none", None
    if best > settings.rag_distance_medium:
        return "low", "Partial evidence only — answer may be incomplete"
    if best > settings.rag_distance_high:
        return "medium", None
    return "high", None


def _format_evidence_brief(retrievals: list[Retrieval], char_limit: int = 240) -> str:
    """Compact evidence summary for the reflector node — keeps prompt small."""
    if not retrievals:
        return "(no chunks retrieved yet)"
    lines: list[str] = []
    for r in retrievals:
        lines.append(f"## sub_query: {r['sub_query']}")
        if not r.get("chunks"):
            lines.append("  (no hits)")
            continue
        for c in r["chunks"][:3]:  # top-3 per sub-query is enough for reflection
            snippet = (c.get("chunk_text") or "").strip().replace("\n", " ")[:char_limit]
            lines.append(
                f"  - [{c.get('source_file', '?')} | dist={c.get('_distance', 1):.3f}] {snippet}"
            )
    return "\n".join(lines)


# ─── Nodes ────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Planner agent — decomposes the question and selects tools."""
    question = state["question"]

    fallback_plan: Plan = {
        "rationale": "Planner unavailable — falling back to single-hop hybrid retrieval.",
        "sub_queries": [question],
        "use_retriever": True,
        "use_sql": False,
        "sql_question": None,
    }

    try:
        raw = llm.generate_text(PLANNER_PROMPT.format(question=question), max_tokens=600)
        data = extract_json(raw)
        if not data:
            raise ValueError("planner returned no JSON")
        sub_qs_raw = data.get("sub_queries") or []
        sub_queries = [s.strip() for s in sub_qs_raw if isinstance(s, str) and s.strip()]
        if not sub_queries:
            sub_queries = [question]
        # cap to 3 and ensure original question is represented
        sub_queries = list(dict.fromkeys(sub_queries))[:3]

        sql_q = data.get("sql_question")
        plan: Plan = {
            "rationale": str(data.get("rationale", "")),
            "sub_queries": sub_queries,
            "use_retriever": bool(data.get("use_retriever", True)),
            "use_sql": bool(data.get("use_sql", False)),
            "sql_question": sql_q if isinstance(sql_q, str) and sql_q.strip() else None,
        }
        if not plan["use_retriever"] and not plan["use_sql"]:
            # Defensive: if the LLM disabled everything, force RAG.
            plan["use_retriever"] = True
        if plan["use_sql"] and not plan["sql_question"]:
            plan["sql_question"] = question

        logger.info(
            "Planner: hops=%d use_retriever=%s use_sql=%s sub_queries=%s",
            len(sub_queries), plan["use_retriever"], plan["use_sql"], sub_queries,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Planner LLM failed (%s) — using fallback plan", exc)
        plan = fallback_plan

    return {
        "plan": plan,
        "pending_sub_queries": list(plan["sub_queries"]),
        "iteration": 0,
        "components": [],
        "retrievals": [],
        "sql_attempts": [],
        "errors": [],
    }


def tool_executor_node(state: AgentState) -> dict:
    """Run the retriever (per pending sub-query) and SQL tools."""
    plan = state.get("plan") or {}
    pending: list[str] = state.get("pending_sub_queries") or []
    iteration: int = state.get("iteration", 0)
    new_errors: list[str] = []

    new_retrievals: list[Retrieval] = []
    if plan.get("use_retriever") and pending:
        for sq in pending:
            try:
                chunks = tools.retriever_tool(sq)
                new_retrievals.append({"sub_query": sq, "chunks": chunks})
                logger.debug("retriever_tool(%r) -> %d chunks", sq, len(chunks))
            except Exception as exc:  # noqa: BLE001
                new_errors.append(f"retriever_tool({sq!r}) failed: {exc}")
                new_retrievals.append({"sub_query": sq, "chunks": []})

    # SQL only fires on the first hop (iteration == 0) to avoid duplicate work.
    new_sql: list[SqlAttempt] = []
    if iteration == 0 and plan.get("use_sql") and plan.get("sql_question"):
        try:
            resp = tools.sql_tool(plan["sql_question"])
            new_sql.append({"question": plan["sql_question"], "response": resp.model_dump()})
        except Exception as exc:  # noqa: BLE001
            new_errors.append(f"sql_tool failed: {exc}")

    # Component catalog is small — load once on the first hop.
    components = state.get("components") or []
    if iteration == 0 and not components:
        components = tools.components_tool()

    return {
        "retrievals": new_retrievals,
        "sql_attempts": new_sql,
        "components": components,
        "errors": new_errors,
        "iteration": iteration + 1,
        "pending_sub_queries": [],
    }


def reflector_node(state: AgentState) -> dict:
    """Critic agent — decides if another retrieval hop is warranted."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", _MAX_HOPS)
    retrievals: list[Retrieval] = state.get("retrievals", []) or []

    # Stop conditions that don't need an LLM call.
    if iteration >= max_iter:
        return {"reflection": {"sufficient": True, "missing": "", "next_sub_queries": []}}

    total_chunks = sum(len(r.get("chunks", [])) for r in retrievals)
    if total_chunks == 0:
        # Nothing retrieved — no point in another hop, let synthesizer report empty.
        return {
            "reflection": {
                "sufficient": True,
                "missing": "no relevant chunks retrieved on first hop",
                "next_sub_queries": [],
            }
        }

    executed = "\n".join(f"  - {r['sub_query']}" for r in retrievals)
    evidence = _format_evidence_brief(retrievals)

    try:
        raw = llm.generate_text(
            REFLECTOR_PROMPT.format(
                question=state["question"], executed=executed, evidence=evidence
            ),
            max_tokens=500,
        )
        data = extract_json(raw)
        if not data:
            raise ValueError("reflector returned no JSON")
        sufficient = bool(data.get("sufficient", True))
        missing = str(data.get("missing", "") or "")
        nxt_raw = data.get("next_sub_queries") or []
        nxt = [s.strip() for s in nxt_raw if isinstance(s, str) and s.strip()]
        # Drop any sub-queries we've already tried.
        already = {r["sub_query"] for r in retrievals}
        nxt = [q for q in nxt if q not in already][:2]

        if sufficient or not nxt:
            return {"reflection": {"sufficient": True, "missing": missing, "next_sub_queries": []}}

        logger.info("Reflector requested another hop with %d new sub-queries: %s", len(nxt), nxt)
        return {
            "reflection": {"sufficient": False, "missing": missing, "next_sub_queries": nxt},
            "pending_sub_queries": nxt,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reflector LLM failed (%s); proceeding to synthesis", exc)
        return {"reflection": {"sufficient": True, "missing": "", "next_sub_queries": []}}


def synthesizer_node(state: AgentState) -> dict:
    """Reasoning agent — produces the final grounded answer."""
    question = state["question"]
    plan: Plan = state.get("plan") or {}
    components = state.get("components") or []
    sql_attempts = state.get("sql_attempts") or []
    chunks = _flatten_chunks(state.get("retrievals") or [])

    # Build per-file maps for source-score and page resolution.
    file_distance_map: dict[str, float] = {}
    file_page_map: dict[str, int | None] = {}
    for c in chunks:
        sf = c.get("source_file", "")
        dist = c.get("_distance", 1.0)
        if sf not in file_distance_map or dist < file_distance_map[sf]:
            file_distance_map[sf] = dist
            page = c.get("page_number")
            file_page_map[sf] = int(page) if page else None

    chunks_text = "\n\n".join(
        f"[{c.get('source_file', 'unknown')} | "
        f"page={c.get('page_number') or '?'} | "
        f"chunk_type={c.get('chunk_type', '?')} | "
        f"dist={c.get('_distance', 1):.3f} | "
        f"matched_via={'+'.join(c.get('_retrieval_sources', []) or ['vector'])} | "
        f"sub_query={c.get('_sub_query', '')!r}]\n"
        f"{c.get('chunk_text', '')}"
        for c in chunks
    ) or "(no document chunks retrieved)"

    components_text = "\n".join(
        f"- {r.get('source_file')}: {r.get('component_type')} / {r.get('material')} / "
        f"${r.get('avg_cost_usd')} avg / confidence={r.get('confidence')}"
        for r in components
    ) or "(no component catalog rows)"

    if sql_attempts:
        sql_lines = []
        for a in sql_attempts:
            resp = a.get("response", {})
            err = resp.get("error")
            sql_lines.append(f"## sql_question: {a.get('question')}")
            sql_lines.append(f"   sql: {resp.get('sql')}")
            if err:
                sql_lines.append(f"   error: {err}")
            else:
                rows = resp.get("result") or []
                sql_lines.append(f"   rows ({len(rows)}): {json.dumps(rows[:10], default=str)}")
        sql_text = "\n".join(sql_lines)
    else:
        sql_text = "(SQL tool not invoked)"

    plan_summary = (
        f"rationale: {plan.get('rationale', '')}\n"
        f"sub_queries: {plan.get('sub_queries', [])}\n"
        f"use_retriever={plan.get('use_retriever')} use_sql={plan.get('use_sql')}"
    )

    prompt = SYNTH_SYSTEM_PROMPT + "\n\n" + SYNTH_USER_PROMPT.format(
        question=question,
        plan_summary=plan_summary,
        chunks_text=chunks_text,
        components_text=components_text,
        sql_text=sql_text,
    )

    # ── Call LLM with graceful fallback ──
    try:
        raw = llm.generate_text(prompt, max_tokens=2048)
        result = extract_json(raw) or {
            "answer": raw,
            "reasoning": None,
            "sources": [],
            "confidence": "low",
            "grounding_warning": "Could not parse structured response from LLM",
        }
    except LLMError as exc:
        logger.error("Synthesizer LLM failed: %s", exc)
        return {
            "final": AnalyticalQueryResponse(
                answer="Reasoning service is temporarily unavailable. Please try again later.",
                reasoning=None,
                confidence="none",
                grounding_warning=str(exc),
                plan=plan.get("sub_queries"),
                iterations=state.get("iteration", 0),
                sql_used=bool(sql_attempts),
            ).model_dump()
        }

    # ── Confidence is GROUND TRUTH from retrieval distances, not LLM opinion ──
    confidence, warning = _determine_confidence(chunks)
    if confidence == "none":
        result["answer"] = (
            "The requested information is not present in the ingested corpus. "
            + (result.get("answer") or "")
        ).strip()
    if warning and not result.get("grounding_warning"):
        result["grounding_warning"] = warning

    # Surface fault-tolerance signals if any tool failed mid-flight.
    errors = state.get("errors") or []
    if errors and not result.get("grounding_warning"):
        result["grounding_warning"] = "; ".join(errors[:3])

    # ── Resolve source scores from real distances ──
    sources: list[SourceReference] = []
    for s in result.get("sources", []):
        if not isinstance(s, dict):
            continue
        file_name = s.get("file", "unknown")
        actual_dist = file_distance_map.get(file_name)
        score = round(1.0 - actual_dist, 4) if actual_dist is not None else s.get("score")
        page = file_page_map.get(file_name)
        if page is None and isinstance(s.get("page"), (int, float)):
            page = int(s["page"])
        sources.append(
            SourceReference(
                file=file_name,
                excerpt=s.get("excerpt", ""),
                score=score,
                page=page,
            )
        )

    response = AnalyticalQueryResponse(
        answer=_coerce_to_str(result.get("answer", "")),
        reasoning=_coerce_to_str(result.get("reasoning"))
        if result.get("reasoning") is not None
        else None,
        sources=sources,
        confidence=confidence,
        grounding_warning=result.get("grounding_warning"),
        plan=plan.get("sub_queries"),
        iterations=state.get("iteration", 0),
        sql_used=bool(sql_attempts),
    )
    return {"final": response.model_dump()}


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

def run_analytical_query(question: str, max_hops: int = _MAX_HOPS) -> AnalyticalQueryResponse:
    """
    Execute the agentic graph end-to-end and return the final response.

    On catastrophic failure (e.g. LangGraph itself crashes) we still return a
    valid :class:`AnalyticalQueryResponse` so the API contract holds.
    """
    initial: AgentState = {
        "question": question,
        "max_iterations": max(1, max_hops),
    }
    try:
        final_state = _get_graph().invoke(initial)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent graph crashed for question=%r", question)
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
        return AnalyticalQueryResponse.model_validate(final)

    # Should not happen, but fail safe.
    return AnalyticalQueryResponse(
        answer="No answer was produced by the pipeline.",
        confidence="none",
        grounding_warning="empty terminal state",
        plan=[question],
        iterations=final_state.get("iteration", 0),
        sql_used=bool(final_state.get("sql_attempts")),
    )
