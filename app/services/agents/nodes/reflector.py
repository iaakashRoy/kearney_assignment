"""
Reflector node — judges if more retrieval is warranted (multi-hop).
"""
from __future__ import annotations

import time

from app.core.logging import get_logger
from app.services import llm
from app.services.agents.helpers import format_evidence_brief, short
from app.services.agents.prompts import REFLECTOR_PROMPT
from app.services.agents.state import AgentState, MAX_HOPS, Retrieval
from app.services.agents.trace import emit
from app.utils.text import extract_json

logger = get_logger(__name__)


def reflector_node(state: AgentState) -> dict:
    """Critic agent — decides if another retrieval hop is warranted."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_HOPS)
    retrievals: list[Retrieval] = state.get("retrievals", []) or []
    qid = state.get("qid", "-")
    t0 = time.perf_counter()
    total_so_far = sum(len(r.get("chunks", [])) for r in retrievals)
    logger.info(
        "[%s] ▶ REFLECTOR iteration=%d/%d total_chunks=%d",
        qid, iteration, max_iter, total_so_far,
    )
    emit("node.start", node="reflector", qid=qid,
         iteration=iteration, max_iterations=max_iter, total_chunks=total_so_far)

    # Stop conditions that don't need an LLM call.
    if iteration >= max_iter:
        logger.info("[%s]   verdict: SUFFICIENT (max iterations reached)", qid)
        emit("reflector.verdict", qid=qid, sufficient=True,
             reason="max iterations reached", next_sub_queries=[])
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] ✓ REFLECTOR done in %.0fms", qid, elapsed)
        emit("node.done", node="reflector", qid=qid, elapsed_ms=int(elapsed))
        return {"reflection": {"sufficient": True, "missing": "", "next_sub_queries": []}}

    total_chunks = sum(len(r.get("chunks", [])) for r in retrievals)
    if total_chunks == 0:
        # Nothing retrieved — no point in another hop, let synthesizer report empty.
        logger.info("[%s]   verdict: SUFFICIENT (no chunks retrieved — nothing to refine)", qid)
        emit("reflector.verdict", qid=qid, sufficient=True,
             reason="no chunks retrieved", next_sub_queries=[])
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] ✓ REFLECTOR done in %.0fms", qid, elapsed)
        emit("node.done", node="reflector", qid=qid, elapsed_ms=int(elapsed))
        return {
            "reflection": {
                "sufficient": True,
                "missing": "no relevant chunks retrieved on first hop",
                "next_sub_queries": [],
            }
        }

    executed = "\n".join(f"  - {r['sub_query']}" for r in retrievals)
    evidence = format_evidence_brief(retrievals)

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

        if missing:
            logger.info("[%s]   missing       : %s", qid, short(missing, 240))

        if sufficient or not nxt:
            logger.info("[%s]   verdict: SUFFICIENT → proceed to synthesizer", qid)
            emit("reflector.verdict", qid=qid, sufficient=True,
                 missing=missing, next_sub_queries=[])
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("[%s] ✓ REFLECTOR done in %.0fms", qid, elapsed)
            emit("node.done", node="reflector", qid=qid, elapsed_ms=int(elapsed))
            return {"reflection": {"sufficient": True, "missing": missing, "next_sub_queries": []}}

        logger.info(
            "[%s]   verdict: INSUFFICIENT → multi-hop with %d new sub-quer%s",
            qid, len(nxt), "y" if len(nxt) == 1 else "ies",
        )
        for i, q in enumerate(nxt, 1):
            logger.info("[%s]   next_sub_query[%d]: %s", qid, i, short(q, 200))
        emit("reflector.verdict", qid=qid, sufficient=False,
             missing=missing, next_sub_queries=nxt)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] ✓ REFLECTOR done in %.0fms", qid, elapsed)
        emit("node.done", node="reflector", qid=qid, elapsed_ms=int(elapsed))
        return {
            "reflection": {"sufficient": False, "missing": missing, "next_sub_queries": nxt},
            "pending_sub_queries": nxt,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s]   reflector LLM failed (%s); proceeding to synthesis", qid, exc)
        emit("reflector.error", qid=qid, error=str(exc))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("[%s] ✓ REFLECTOR done in %.0fms", qid, elapsed)
        emit("node.done", node="reflector", qid=qid, elapsed_ms=int(elapsed))
        return {"reflection": {"sufficient": True, "missing": "", "next_sub_queries": []}}
