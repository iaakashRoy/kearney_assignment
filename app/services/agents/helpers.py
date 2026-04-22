"""
Pure helper functions used by the agentic graph nodes.

Kept side-effect free so they can be unit-tested in isolation.
"""
from __future__ import annotations

import json
from typing import Any

from app.config import settings

from app.services.agents.state import Retrieval


def short(text: Any, n: int = 200) -> str:
    """Collapse whitespace and truncate text for log lines."""
    s = str(text or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def flatten_chunks(retrievals: list[Retrieval]) -> list[dict]:
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


def determine_confidence(chunks: list[dict]) -> tuple[str, str | None]:
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


def format_evidence_brief(retrievals: list[Retrieval], char_limit: int = 240) -> str:
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
