"""
Tools exposed to the agentic graph.

Each tool is a plain Python function with strict typing so it can be
audited / mocked / re-used outside LangGraph. The graph nodes call these
functions directly; LangGraph's tool-calling abstractions are intentionally
NOT used here so we keep the behaviour deterministic and inspectable.
"""
from __future__ import annotations

from app.config import settings
from app.core.logging import get_logger
from app.db import sqlite as db
from app.db import vector_store
from app.models.schemas import StructuredQueryResponse
from app.services import embedding, sql_agent

logger = get_logger(__name__)

_RRF_K = 60  # standard Reciprocal Rank Fusion constant


# ─── Retriever tool ───────────────────────────────────────────────────────────

def retriever_tool(sub_query: str, top_k: int | None = None) -> list[dict]:
    """
    Hybrid retrieval for a single sub-query.

    Combines:
    * vector search over LanceDB (cosine distance on MiniLM embeddings)
    * BM25 keyword search over the SQLite ``chunks_fts`` table

    The two ranked lists are fused with Reciprocal Rank Fusion (RRF, k=60)
    so the final ordering rewards chunks that surface in BOTH modalities.

    Each returned chunk dict carries:
      * its original metadata (``source_file``, ``chunk_index``,
        ``chunk_text``, ``chunk_type``, ``page_number``, ...)
      * ``_distance``       – best cosine distance (1.0 fallback if BM25-only)
      * ``_rrf_score``      – fused relevance score (higher = better)
      * ``_retrieval_sources`` – sorted list of {"vector", "keyword"}
      * ``_sub_query``      – the sub-query that surfaced this chunk
    """
    k = top_k or settings.rag_top_k

    # ── vector search (best-effort) ──
    try:
        q_vec = embedding.embed([sub_query])[0]
        vec_hits = vector_store.search(q_vec, limit=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retriever_tool: vector search failed for %r: %s", sub_query, exc)
        vec_hits = []

    # ── keyword search (best-effort) ──
    try:
        kw_hits = db.keyword_search_chunks(sub_query, limit=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retriever_tool: keyword search failed for %r: %s", sub_query, exc)
        kw_hits = []

    if not vec_hits and not kw_hits:
        return []

    best: dict[tuple, dict] = {}
    rrf: dict[tuple, float] = {}

    for ranked, label in ((vec_hits, "vector"), (kw_hits, "keyword")):
        for rank, chunk in enumerate(ranked):
            key = (chunk.get("source_file"), chunk.get("chunk_index"))
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            existing = best.get(key)
            if existing is None:
                merged = dict(chunk)
                merged["_retrieval_sources"] = {label}
                merged["_sub_query"] = sub_query
                best[key] = merged
            else:
                existing["_retrieval_sources"].add(label)
                if "_distance" not in existing and "_distance" in chunk:
                    existing["_distance"] = chunk["_distance"]

    out: list[dict] = []
    for key, chunk in best.items():
        chunk["_rrf_score"] = rrf[key]
        # BM25-only hits get a neutral distance just below the "low" threshold
        # so they remain visible without inflating confidence.
        chunk.setdefault("_distance", min(settings.rag_distance_low, 0.6))
        chunk["_retrieval_sources"] = sorted(chunk["_retrieval_sources"])
        out.append(chunk)

    out.sort(key=lambda c: (-c.get("_rrf_score", 0.0), c.get("_distance", 1.0)))
    return out[:k]


# ─── SQL tool ─────────────────────────────────────────────────────────────────

def sql_tool(question: str) -> StructuredQueryResponse:
    """
    Natural-language → SQL over the structured catalog.

    Thin wrapper around :func:`app.services.sql_agent.structured_query` so the
    agent graph can treat structured retrieval as just another tool.
    """
    return sql_agent.structured_query(question)


# ─── Components tool ──────────────────────────────────────────────────────────

def components_tool() -> list[dict]:
    """Return every component row from the catalog (cheap; cached per call)."""
    try:
        return db.fetch_all_components()
    except Exception as exc:  # noqa: BLE001
        logger.warning("components_tool failed: %s", exc)
        return []
