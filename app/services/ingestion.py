"""
Ingestion service — routes uploaded files through the correct processing stream.

Stream A (digital_doc / scanned_doc): entity extraction → SQLite + LanceDB vector store.
Stream B (component_photo): vision classification + cost lookup → SQLite.
"""
from __future__ import annotations

import csv
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from docling.document_converter import DocumentConverter

from app.config import settings
from app.core.exceptions import EmbeddingError, IngestionError, LLMError
from app.core.logging import get_logger
from app.db import sqlite as db
from app.db import vector_store
from app.models.schemas import (
    ComponentClassification,
    DocumentStructure,
    ExtractionResult,
    IngestResult,
    RelationshipRecord,
)
from app.services import embedding, llm
from app.utils.image_preprocess import preprocess_for_ocr
from app.utils.text import chunk_text, extract_json, parse_markdown_structure, structure_aware_chunks

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg"})

_ENTITY_PROMPT = """\
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


# ─── Benchmark data ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_benchmark() -> list[dict]:
    """Load benchmark cost data once; returns [] if not found."""
    csv_path = settings.benchmark_csv_path
    if csv_path:
        candidates = [Path(csv_path)]
    else:
        # Bundled inside the package (app/data/) and the legacy /-mounted Docker location.
        candidates = [
            Path(__file__).resolve().parent.parent / "data" / "benchmark_component_costs.csv",
            Path("/benchmark_component_costs.csv"),
        ]

    for path in candidates:
        if path.exists():
            logger.info("Loaded benchmark CSV from %s", path)
            with open(path, newline="") as fh:
                return list(csv.DictReader(fh))

    logger.warning("benchmark_component_costs.csv not found — cost lookup disabled")
    return []


# ─── PDF extraction ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_docling() -> DocumentConverter:
    logger.info("Initialising Docling document converter")
    return DocumentConverter()


def _extract_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = _get_docling().convert(tmp_path)
        return result.document.export_to_markdown()
    except Exception as exc:
        raise IngestionError(f"PDF extraction failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)


def _extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """
    Convert a PDF and return ``[(page_number, markdown), ...]`` so chunks can
    later be attributed to the page they came from. Pages with no extractable
    text are skipped. Falls back to a single ``(1, full_markdown)`` entry if
    Docling cannot enumerate pages.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = _get_docling().convert(tmp_path).document
        pages: list[tuple[int, str]] = []
        page_nos: list[int] = []
        try:
            page_nos = sorted(int(p) for p in (doc.pages or {}).keys())
        except Exception:  # noqa: BLE001
            page_nos = []
        if not page_nos:
            return [(1, doc.export_to_markdown())]
        for pno in page_nos:
            try:
                md = doc.export_to_markdown(page_no=pno).strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("export_to_markdown(page_no=%s) failed: %s", pno, exc)
                md = ""
            if md:
                pages.append((pno, md))
        return pages or [(1, doc.export_to_markdown())]
    except Exception as exc:
        raise IngestionError(f"PDF extraction failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)


def _save_source(filename: str, file_bytes: bytes) -> None:
    """Persist the original upload to ``settings.sources_dir`` for preview rendering."""
    try:
        out_dir = Path(settings.sources_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Strip any path components to avoid traversal.
        safe = Path(filename).name
        (out_dir / safe).write_bytes(file_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist source file %s: %s", filename, exc)


# ─── Entity extraction ────────────────────────────────────────────────────────

_ENTITY_DEFAULTS: dict = {
    "doc_type": "Project Update",
    "doc_type_confidence": 0.5,
    "secondary_doc_type": None,
    "secondary_doc_type_confidence": None,
    "entities": [],
    "relationships": [],
    "summary": "Could not extract summary.",
}


def _extract_entities(text: str, structure: DocumentStructure | None = None) -> ExtractionResult:
    try:
        # Entity extraction returns a large JSON object; allow generous budget so
        # the model is not truncated mid-output (especially for entity-rich docs).
        raw = llm.generate_text(_ENTITY_PROMPT.format(text=text[:4000]), max_tokens=2048)
        data = extract_json(raw)
    except LLMError:
        logger.warning("Entity extraction LLM call failed — using defaults")
        data = {}

    entities_raw = data.get("entities") or []
    relationships_raw = data.get("relationships") or []

    return ExtractionResult(
        doc_type=data.get("doc_type") or _ENTITY_DEFAULTS["doc_type"],
        doc_type_confidence=data.get("doc_type_confidence") or _ENTITY_DEFAULTS["doc_type_confidence"],
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
        summary=data.get("summary") or _ENTITY_DEFAULTS["summary"],
        structure=structure,
    )


# ─── Cost lookup ──────────────────────────────────────────────────────────────

def _lookup_cost(component_type: str, material: str, size_category: str) -> dict | None:
    benchmark = _load_benchmark()
    # Exact match
    for row in benchmark:
        if (
            row.get("component_type") == component_type
            and row.get("material") == material
            and row.get("size_category") == size_category
        ):
            return row
    # Fallback: component_type + material
    for row in benchmark:
        if row.get("component_type") == component_type and row.get("material") == material:
            return row
    # Fallback: component_type only
    for row in benchmark:
        if row.get("component_type") == component_type:
            return row
    return None


# ─── Stream A: document ingestion ─────────────────────────────────────────────

def _ingest_document(
    source_file: str,
    input_type: str,
    pages: list[tuple[int, str]],
) -> IngestResult:
    """
    Persist a document made up of one or more (page_number, markdown) pages.

    Each page is chunked independently so every chunk carries the page it came
    from — required to render a focused preview in the UI.
    """
    raw_text = "\n\n".join(md for _, md in pages)
    structure = parse_markdown_structure(raw_text)
    extraction = _extract_entities(raw_text, structure=structure)

    doc_id = db.insert_document(
        source_file=source_file,
        input_type=input_type,
        doc_type=extraction.doc_type,
        doc_type_confidence=extraction.doc_type_confidence,
        secondary_doc_type=extraction.secondary_doc_type,
        secondary_doc_type_confidence=extraction.secondary_doc_type_confidence,
        summary=extraction.summary,
        raw_text=raw_text,
    )

    entity_dicts = [e.model_dump() for e in extraction.entities]
    if entity_dicts:
        db.insert_entities(doc_id, entity_dicts)

    relationship_dicts = [r.model_dump() for r in extraction.relationships]
    if relationship_dicts:
        db.insert_relationships(doc_id, relationship_dicts)

    # Per-page structure-aware chunking — preserves the originating page number
    # on every chunk so the UI can render an accurate preview.
    page_chunks: list[tuple[str, str, int]] = []  # (text, chunk_type, page)
    for page_no, page_md in pages:
        for ctext, ctype in structure_aware_chunks(
            page_md, settings.chunk_size, settings.chunk_overlap
        ):
            page_chunks.append((ctext, ctype, page_no))

    if page_chunks:
        texts = [t for t, _, _ in page_chunks]
        types = [ct for _, ct, _ in page_chunks]
        page_nums = [pn for _, _, pn in page_chunks]
        vectors = embedding.embed(texts)
        chunk_records = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_type": chunk_type,
                "source_file": source_file,
                "chunk_text": chunk,
                "page_number": page_num,
                "vector": vec,
            }
            for i, (chunk, chunk_type, page_num, vec) in enumerate(
                zip(texts, types, page_nums, vectors)
            )
        ]
        vector_store.add_chunks(chunk_records)
        # Mirror chunks into SQLite FTS5 so the analytical query can do
        # keyword (BM25) search alongside vector search.
        db.insert_chunks_fts(chunk_records)

    n_chunks = len(page_chunks)
    n_pages = len(pages)
    logger.info(
        "Document ingested: %s (%d pages, %d entities, %d relationships, %d chunks, %d headers, %d tables)",
        source_file,
        n_pages,
        len(entity_dicts),
        len(relationship_dicts),
        n_chunks,
        len(structure.headers),
        len(structure.tables),
    )
    return IngestResult(
        source_file=source_file,
        input_type=input_type,
        status="ok",
        detail=(
            f"{n_pages} pages, {len(entity_dicts)} entities, "
            f"{len(relationship_dicts)} relationships, {n_chunks} chunks, "
            f"{len(structure.headers)} headers, {len(structure.tables)} tables"
        ),
        structure=structure,
    )


# ─── Stream B: component photo ingestion ──────────────────────────────────────

def _ingest_component(source_file: str, file_bytes: bytes, suffix: str) -> IngestResult:
    raw = llm.classify_component_image(file_bytes, suffix)
    classification = ComponentClassification(
        component_type=raw.get("component_type"),
        material=raw.get("material"),
        size_category=raw.get("size_category"),
        confidence=float(raw.get("confidence") or 0.0),
        reasoning=raw.get("reasoning") or "",
    )

    cost_row = _lookup_cost(
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


# ─── Public entry point ───────────────────────────────────────────────────────

def ingest_file(filename: str, file_bytes: bytes) -> IngestResult:
    """
    Process a single uploaded file end-to-end.

    Returns an IngestResult with status ``"ok"``, ``"error"``, or ``"skipped"``.
    Never raises — errors are captured into the result.
    """
    ext = Path(filename).suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported file extension: %s", ext)
        return IngestResult(
            source_file=filename,
            input_type="unknown",
            status="skipped",
            detail=f"Unsupported extension: {ext!r}",
        )

    logger.info("Ingesting file: %s", filename)
    try:
        # Persist the original upload so the UI can render previews. Best-effort.
        _save_source(filename, file_bytes)

        if ext == ".pdf":
            pages = _extract_pdf_pages(file_bytes)
            return _ingest_document(filename, "digital_doc", pages)

        # Pre-process the image to improve OCR quality, then detect type.
        # Original bytes are preserved for component classification (keeps colour).
        preprocessed_bytes = preprocess_for_ocr(file_bytes, suffix=ext)
        input_type, raw_text = llm.ocr_image(preprocessed_bytes, suffix=ext)
        if input_type == "scanned_doc":
            return _ingest_document(filename, input_type, [(1, raw_text)])
        # Component photos: classify using original (colour) bytes.
        return _ingest_component(filename, file_bytes, ext)

    except (LLMError, EmbeddingError, IngestionError) as exc:
        logger.error("Ingestion error for %s: %s", filename, exc)
        return IngestResult(source_file=filename, input_type="unknown", status="error", detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error ingesting %s", filename)
        return IngestResult(source_file=filename, input_type="unknown", status="error", detail=str(exc))
