"""
Stream A — document ingestion (digital_doc / scanned_doc).

Persists the document, extracts entities/relationships, and writes
structure-aware chunks to both LanceDB (vector) and SQLite FTS5 (keyword).
"""
from __future__ import annotations

from app.config import settings
from app.core.logging import get_logger
from app.db import sqlite as db
from app.db import vector_store
from app.models.schemas import IngestResult
from app.services import embedding
from app.services.ingestion.entities import extract_entities
from app.utils.text import parse_markdown_structure, structure_aware_chunks

logger = get_logger(__name__)


def ingest_document(
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
    extraction = extract_entities(raw_text, structure=structure)

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
