"""
Retrieval-Augmented Generation (RAG) pipeline.

Agent 1 — Retrieval: decompose question into sub-queries, embed each, merge
           and de-duplicate chunks from LanceDB, fetch component rows.
Agent 2 — Reasoning: build a grounded Chain-of-Thought prompt, call the LLM,
           resolve source scores from real retrieval distances, post-process confidence.
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.db import sqlite as db
from app.db import vector_store
from app.models.schemas import AnalyticalQueryResponse, SourceReference
from app.services import embedding, llm
from app.utils.text import extract_json

logger = get_logger(__name__)

_DECOMPOSE_PROMPT = """\
You are a query-planning specialist.
Decompose the analytical question below into 2–3 focused sub-questions whose
answers, taken together, fully address the original question.
If the question is already simple and self-contained, return only the original.

Question: {question}

Return ONLY a JSON array of strings — no explanation:
["sub-question 1", "sub-question 2"]
"""

_SYSTEM_PROMPT = """\
You are an expert analyst for an enterprise intelligence platform.

Guidelines:
- Answer using ONLY the provided context — never fabricate information.
- Think step-by-step: record your reasoning in the \"reasoning\" field FIRST, then write the \"answer\".
- Cite specific source files for every fact you use.
- If the answer requires comparing document chunks and component data, synthesise across both.
- If context is insufficient, say so clearly and state what information is missing.\
"""

_USER_PROMPT = """\
Question: {question}

--- Document Chunks (source_file | chunk_type | retrieval_distance) ---
{chunks_text}

--- Component Data ---
{components_text}

Respond with ONLY this JSON object:
{{
  "reasoning": "<chain-of-thought: step 1 identify relevant evidence → step 2 assess gaps → step 3 synthesise>",
  "answer": "<final detailed answer citing sources>",
  "sources": [
    {{"file": "<source_file>", "page": <int page number or null>, "excerpt": "<exact short quote from that source>", "score": <float 0-1>}}
  ],
  "confidence": "<high|medium|low|none>",
  "grounding_warning": null
}}

Confidence levels (based on retrieval distances):
- \"none\"   → all distances > {dist_low}  (or no chunks found)
- \"low\"    → best distance {dist_med}–{dist_low}
- \"medium\" → best distance {dist_high}–{dist_med}
- \"high\"   → best distance < {dist_high}\
"""


# ─── Confidence scoring ───────────────────────────────────────────────────────

def _coerce_to_str(value) -> str:
    """LLMs occasionally return objects/lists where a string is expected.
    Serialise non-strings to JSON so the API contract still holds."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _determine_confidence(chunks: list[dict]) -> tuple[str, str | None]:
    """Return (confidence_label, optional_warning) based on best retrieval distance."""
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


# ─── Agent 1: Retrieval ───────────────────────────────────────────────────────

def _decompose_question(question: str) -> list[str]:
    """
    Sub-role of the Retrieval Agent: use the LLM to decompose a complex question
    into 2–3 focused sub-queries for independent retrieval.
    Falls back to [question] on any failure.
    """
    try:
        raw = llm.generate_text(_DECOMPOSE_PROMPT.format(question=question), max_tokens=512)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            sub_qs = json.loads(match.group())
            if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
                # Deduplicate, always include the original, cap at 3 total
                unique = list(dict.fromkeys([question] + [q for q in sub_qs if q != question]))
                return unique[:3]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Question decomposition failed (%s) — using original only", exc)
    return [question]


def retrieve(question: str) -> dict:
    """
    Hybrid multi-hop retrieval.

    1. Decompose the question into sub-queries (Retrieval Agent sub-role).
    2. For each sub-query run BOTH:
         - vector search (cosine over MiniLM embeddings in LanceDB)
         - keyword search (BM25 over chunks_fts in SQLite)
    3. Fuse the two ranked lists per sub-query with Reciprocal Rank Fusion
       (RRF, k=60) so an answer can be supported by either lexical or semantic
       evidence.
    4. Merge across sub-queries, de-duplicate by (source_file, chunk_index),
       sort by best (lowest) vector distance for the confidence calculator,
       and cap at ``2 × rag_top_k`` chunks.
    """
    sub_queries = _decompose_question(question)
    logger.debug("Multi-hop sub-queries: %s", sub_queries)

    # _key → best representative chunk dict across all sub-queries
    best_chunk: dict[tuple, dict] = {}
    # _key → accumulated RRF score
    rrf_scores: dict[tuple, float] = {}
    rrf_k = 60  # standard RRF constant

    for sq in sub_queries:
        # --- vector search ---
        try:
            q_vector = embedding.embed([sq])[0]
            vec_hits = vector_store.search(q_vector, limit=settings.rag_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed for '%s': %s", sq, exc)
            vec_hits = []

        # --- keyword search ---
        try:
            kw_hits = db.keyword_search_chunks(sq, limit=settings.rag_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keyword search failed for '%s': %s", sq, exc)
            kw_hits = []

        for ranked, source_label in ((vec_hits, "vector"), (kw_hits, "keyword")):
            for rank, chunk in enumerate(ranked):
                key = (chunk.get("source_file"), chunk.get("chunk_index"))
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                # Keep the chunk that carries the most useful metadata: prefer
                # the vector hit (it has _distance), but back-fill missing
                # fields from whichever source produced it first.
                existing = best_chunk.get(key)
                if existing is None:
                    chunk = dict(chunk)
                    chunk.setdefault("_retrieval_sources", set())
                    chunk["_retrieval_sources"].add(source_label)
                    best_chunk[key] = chunk
                else:
                    existing["_retrieval_sources"].add(source_label)
                    # If the new hit has a vector distance and the existing one
                    # doesn't, copy it over so confidence scoring works.
                    if "_distance" not in existing and "_distance" in chunk:
                        existing["_distance"] = chunk["_distance"]

    merged_chunks: list[dict] = []
    for key, chunk in best_chunk.items():
        chunk["_rrf_score"] = rrf_scores[key]
        # Ensure every chunk has a _distance for confidence scoring; keyword-only
        # hits get a neutral distance just below the "low" threshold so they
        # surface as evidence without inflating confidence.
        chunk.setdefault("_distance", min(settings.rag_distance_low, 0.6))
        chunk["_retrieval_sources"] = sorted(chunk.get("_retrieval_sources", []))
        merged_chunks.append(chunk)

    # Order by RRF (best first), then by vector distance as a tiebreaker
    merged_chunks.sort(key=lambda c: (-c.get("_rrf_score", 0.0), c.get("_distance", 1.0)))
    merged_chunks = merged_chunks[: settings.rag_top_k * 2]

    components = db.fetch_all_components()
    logger.debug(
        "Hybrid retrieval: %d unique chunks across %d sub-queries (vector+keyword RRF), %d components",
        len(merged_chunks), len(sub_queries), len(components),
    )
    return {"chunks": merged_chunks, "components": components, "sub_queries": sub_queries}


# ─── Agent 2: Reasoning ───────────────────────────────────────────────────────

def reason(question: str, context: dict) -> AnalyticalQueryResponse:
    """Build a grounded CoT prompt, call the LLM, and return a structured response."""
    chunks = context.get("chunks", [])
    components = context.get("components", [])

    # Build a lookup: source_file → best retrieval distance (used to produce real scores)
    file_distance_map: dict[str, float] = {}
    # source_file → page number of the best (lowest-distance) chunk
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
        f"matched_via={'+'.join(c.get('_retrieval_sources', []) or ['vector'])}]\n"
        f"{c.get('chunk_text', '')}"
        for c in chunks
    ) or "(no document chunks retrieved)"

    components_text = "\n".join(
        f"- {r.get('source_file')}: {r.get('component_type')} / {r.get('material')} / "
        f"${r.get('avg_cost_usd')} avg / confidence={r.get('confidence')}"
        for r in components
    ) or "(no component data)"

    prompt = _SYSTEM_PROMPT + "\n\n" + _USER_PROMPT.format(
        question=question,
        chunks_text=chunks_text,
        components_text=components_text,
        dist_high=settings.rag_distance_high,
        dist_med=settings.rag_distance_medium,
        dist_low=settings.rag_distance_low,
    )

    # LLM call with fallback
    try:
        raw = llm.generate_text(prompt, max_tokens=2048)
        result = extract_json(raw)
        if not result:
            result = {
                "answer": raw,
                "reasoning": None,
                "sources": [],
                "confidence": "low",
                "grounding_warning": "Could not parse structured response from LLM",
            }
    except LLMError as exc:
        logger.error("LLM reasoning failed: %s", exc)
        return AnalyticalQueryResponse(
            answer="Reasoning service is temporarily unavailable. Please try again later.",
            reasoning=None,
            confidence="none",
            grounding_warning=str(exc),
        )

    # Override confidence with actual retrieval distances (ground truth, not LLM opinion)
    confidence, warning = _determine_confidence(chunks)
    if confidence == "none":
        result["answer"] = (
            "The requested information is not present in the ingested corpus. "
            + (result.get("answer") or "")
        ).strip()
    if warning and not result.get("grounding_warning"):
        result["grounding_warning"] = warning

    # Resolve source scores from actual retrieval distances: similarity = 1 − cosine_distance
    sources: list[SourceReference] = []
    for s in result.get("sources", []):
        if not isinstance(s, dict):
            continue
        file_name = s.get("file", "unknown")
        actual_dist = file_distance_map.get(file_name)
        score = round(1.0 - actual_dist, 4) if actual_dist is not None else s.get("score")
        # Prefer the page derived from real retrieval; fall back to whatever the
        # LLM produced (often missing) if we have no chunk match.
        page = file_page_map.get(file_name)
        if page is None and isinstance(s.get("page"), (int, float)):
            page = int(s["page"])
        sources.append(SourceReference(
            file=file_name,
            excerpt=s.get("excerpt", ""),
            score=score,
            page=page,
        ))

    return AnalyticalQueryResponse(
        answer=_coerce_to_str(result.get("answer", "")),
        reasoning=_coerce_to_str(result.get("reasoning")) if result.get("reasoning") is not None else None,
        sources=sources,
        confidence=confidence,
        grounding_warning=result.get("grounding_warning"),
    )


# ─── Public entry point ───────────────────────────────────────────────────────

def analytical_query(question: str) -> AnalyticalQueryResponse:
    context = retrieve(question)
    return reason(question, context)
