"""
Stream B — component photo ingestion.

Classifies the component via vision LLM, joins with the benchmark cost CSV,
and persists a row to the SQLite ``components`` table.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.db import sqlite as db
from app.models.schemas import ComponentClassification, IngestResult
from app.services import llm
from app.services.ingestion.cost import lookup_cost

logger = get_logger(__name__)


def ingest_component(source_file: str, file_bytes: bytes, suffix: str) -> IngestResult:
    raw = llm.classify_component_image(file_bytes, suffix)
    classification = ComponentClassification(
        component_type=raw.get("component_type"),
        material=raw.get("material"),
        size_category=raw.get("size_category"),
        confidence=float(raw.get("confidence") or 0.0),
        reasoning=raw.get("reasoning") or "",
    )

    cost_row = lookup_cost(
        classification.component_type or "",
        classification.material or "",
        classification.size_category or "",
    )

    db.insert_component(
        source_file=source_file,
        component_type=classification.component_type,
        material=classification.material,
        size_category=classification.size_category,
        min_cost_usd=float(cost_row["min_cost_usd"]) if cost_row else None,
        max_cost_usd=float(cost_row["max_cost_usd"]) if cost_row else None,
        avg_cost_usd=float(cost_row["avg_cost_usd"]) if cost_row else None,
        manufacturing_process=cost_row.get("manufacturing_process") if cost_row else None,
        confidence=classification.confidence,
        low_confidence=1 if classification.confidence < 0.65 else 0,
        reasoning=classification.reasoning,
    )

    avg = float(cost_row["avg_cost_usd"]) if cost_row else None
    logger.info(
        "Component ingested: %s → %s / %s / $%s avg",
        source_file,
        classification.component_type,
        classification.material,
        avg,
    )
    return IngestResult(
        source_file=source_file,
        input_type="component_photo",
        status="ok",
        detail=f"{classification.component_type} / {classification.material} / ${avg} avg",
    )
