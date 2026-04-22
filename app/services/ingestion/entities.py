"""
LLM-driven entity / relationship / summary extraction for documents.
"""
from __future__ import annotations

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.schemas import DocumentStructure, ExtractionResult, RelationshipRecord
from app.services import llm
from app.utils.text import extract_json

logger = get_logger(__name__)

ENTITY_PROMPT = """\
Extract information from the document below and return ONLY a JSON object.

Required JSON format:
{{
  "doc_type": "<primary: one of: Financial Report, Meeting Notes, Project Update, Executive Summary>",
  "doc_type_confidence": <float 0-1>,
  "secondary_doc_type": "<secondary best-fit from same list, or null>",
  "secondary_doc_type_confidence": <float 0-1 or null>,
  "entities": [
    {{
      "type": "<PERSON|ORG|DATE|MONEY|PROJECT>",
      "value": "<extracted text>",
      "normalized": "<canonical form: ISO 8601 for DATE e.g. 2024-03-15, currency symbol + amount for MONEY e.g. $1,200,000>"
    }}
  ],
  "relationships": [
    {{
      "subject": "<entity value>",
      "predicate": "<assigned_to|member_of|has_budget|owns|reports_to|associated_with>",
      "object": "<entity value>"
    }}
  ],
  "summary": "<2-sentence summary>"
}}

Rules:
- DATE normalized field must be ISO 8601 (YYYY-MM-DD). Use best estimate if only month/year given.
- MONEY normalized field must start with currency symbol and include full numeric value.
- Relationships must only reference values that appear in the entities list.
- Extract ALL named persons, organizations, dates, monetary amounts, and project names.

Document:
{text}

Return ONLY the JSON object, no other text.\
"""

ENTITY_DEFAULTS: dict = {
    "doc_type": "Project Update",
    "doc_type_confidence": 0.5,
    "secondary_doc_type": None,
    "secondary_doc_type_confidence": None,
    "entities": [],
    "relationships": [],
    "summary": "Could not extract summary.",
}


def extract_entities(text: str, structure: DocumentStructure | None = None) -> ExtractionResult:
    try:
        # Entity extraction returns a large JSON object; allow generous budget so
        # the model is not truncated mid-output (especially for entity-rich docs).
        raw = llm.generate_text(ENTITY_PROMPT.format(text=text[:4000]), max_tokens=2048)
        data = extract_json(raw)
    except LLMError:
        logger.warning("Entity extraction LLM call failed — using defaults")
        data = {}

    entities_raw = data.get("entities") or []
    relationships_raw = data.get("relationships") or []

    return ExtractionResult(
        doc_type=data.get("doc_type") or ENTITY_DEFAULTS["doc_type"],
        doc_type_confidence=data.get("doc_type_confidence") or ENTITY_DEFAULTS["doc_type_confidence"],
        secondary_doc_type=data.get("secondary_doc_type"),
        secondary_doc_type_confidence=data.get("secondary_doc_type_confidence"),
        entities=[
            {"type": e.get("type", ""), "value": e.get("value", ""), "normalized": e.get("normalized")}
            for e in entities_raw
            if isinstance(e, dict) and e.get("value")
        ],
        relationships=[
            RelationshipRecord(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
            )
            for r in relationships_raw
            if isinstance(r, dict) and r.get("subject") and r.get("predicate") and r.get("object")
        ],
        summary=data.get("summary") or ENTITY_DEFAULTS["summary"],
        structure=structure,
    )
