"""
Synthesizer node — produces the final grounded answer.
"""
from __future__ import annotations

import json
import time

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.schemas import AnalyticalQueryResponse, SourceReference
from app.services import llm
from app.services.agents.helpers import (
    coerce_to_str,
    determine_confidence,
    flatten_chunks,
    short,
)
from app.services.agents.prompts import SYNTH_SYSTEM_PROMPT, SYNTH_USER_PROMPT
from app.services.agents.state import AgentState, Plan
from app.services.agents.trace import emit
from app.utils.text import extract_json

logger = get_logger(__name__)


def synthesizer_node(state: AgentState) -> dict:
    """Reasoning agent — produces the final grounded answer."""
    question = state["question"]
    plan: Plan = state.get("plan") or {}
    components = state.get("components") or []
    sql_attempts = state.get("sql_attempts") or []
    chunks = flatten_chunks(state.get("retrievals") or [])
    qid = state.get("qid", "-")
    t0 = time.perf_counter()
    logger.info(
        "[%s] ▶ SYNTHESIZER  unique_chunks=%d components=%d sql_attempts=%d",
        qid, len(chunks), len(components), len(sql_attempts),
    )
    emit("node.start", node="synthesizer", qid=qid,
         unique_chunks=len(chunks), components=len(components),
         sql_attempts=len(sql_attempts))

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
        logger.error("[%s]   synthesizer LLM failed: %s", qid, exc)
        logger.info("[%s] ✓ SYNTHESIZER done in %.0fms (LLM error — returned fallback)", qid, (time.perf_counter() - t0) * 1000)
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
    confidence, warning = determine_confidence(chunks)
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
        answer=coerce_to_str(result.get("answer", "")),
        reasoning=coerce_to_str(result.get("reasoning"))
        if result.get("reasoning") is not None
        else None,
        sources=sources,
        confidence=confidence,
        grounding_warning=result.get("grounding_warning"),
        plan=plan.get("sub_queries"),
        iterations=state.get("iteration", 0),
        sql_used=bool(sql_attempts),
    )

    # ── Log the LLM's chain-of-thought + final shape ──
    if response.reasoning:
        logger.info("[%s]   CoT reasoning : %s", qid, short(response.reasoning, 400))
        emit("synthesizer.reasoning", qid=qid, reasoning=response.reasoning)
    logger.info("[%s]   answer        : %s", qid, short(response.answer, 320))
    logger.info(
        "[%s]   confidence=%s  sources=%d  warning=%s",
        qid, response.confidence, len(response.sources),
        short(response.grounding_warning, 120) if response.grounding_warning else "none",
    )
    emit("synthesizer.answer", qid=qid,
         answer=response.answer, confidence=response.confidence,
         sources=len(response.sources),
         grounding_warning=response.grounding_warning)
    for i, src in enumerate(response.sources[:5], 1):
        logger.info(
            "[%s]   source[%d]    : %s p%s score=%s | %s",
            qid, i, src.file, src.page if src.page else "?",
            f"{src.score:.3f}" if isinstance(src.score, (int, float)) else src.score,
            short(src.excerpt, 140),
        )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[%s] ✓ SYNTHESIZER done in %.0fms", qid, elapsed)
    emit("node.done", node="synthesizer", qid=qid, elapsed_ms=int(elapsed))
    return {"final": response.model_dump()}
