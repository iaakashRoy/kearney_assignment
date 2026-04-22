"""LanceDB vector store for document chunks."""
from __future__ import annotations

from functools import lru_cache

import lancedb
import pyarrow as pa

from app.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)

_TABLE_NAME = "chunks"
_VECTOR_DIM = 384


@lru_cache(maxsize=1)
def _get_db() -> lancedb.LanceDBConnection:
    logger.info("Connecting to LanceDB at %s", settings.lancedb_path)
    return lancedb.connect(settings.lancedb_path)


def _expected_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("document_id", pa.int64()),
            pa.field("chunk_index", pa.int64()),
            pa.field("chunk_type", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("chunk_text", pa.string()),
            pa.field("page_number", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), _VECTOR_DIM)),
        ]
    )


def _get_or_create_table(db: lancedb.LanceDBConnection) -> lancedb.table.LanceTable:
    if _TABLE_NAME not in db.table_names():
        logger.info("Creating LanceDB table '%s'", _TABLE_NAME)
        return db.create_table(_TABLE_NAME, schema=_expected_schema())
    table = db.open_table(_TABLE_NAME)
    # Auto-migrate if older table is missing page_number — old vectors are
    # dropped (RAG would still work, just without page previews until re-ingest).
    existing_fields = {f.name for f in table.schema}
    if "page_number" not in existing_fields:
        logger.warning(
            "LanceDB table '%s' is missing 'page_number'; recreating (old chunks dropped). "
            "Re-ingest documents to enable PDF page previews.",
            _TABLE_NAME,
        )
        db.drop_table(_TABLE_NAME)
        return db.create_table(_TABLE_NAME, schema=_expected_schema())
    return table


def add_chunks(chunks: list[dict]) -> None:
    """Persist a list of chunk dicts to the vector store."""
    if not chunks:
        return
    try:
        db = _get_db()
        table = _get_or_create_table(db)
        # Normalise: every row must carry page_number (default 0 = unknown).
        for c in chunks:
            c.setdefault("page_number", 0)
        table.add(chunks)
        logger.debug("Added %d chunks to vector store", len(chunks))
    except Exception as exc:
        raise VectorStoreError(f"Failed to add chunks: {exc}") from exc


def search(vector: list[float], limit: int = 5) -> list[dict]:
    """Return top-k chunks by cosine similarity. Returns [] when the table is empty."""
    try:
        db = _get_db()
        if _TABLE_NAME not in db.table_names():
            logger.debug("Vector store table does not exist yet — returning empty results")
            return []
        table = db.open_table(_TABLE_NAME)
        return table.search(vector).metric("cosine").limit(limit).to_list()
    except Exception as exc:
        raise VectorStoreError(f"Vector search failed: {exc}") from exc


def reset() -> None:
    """Drop the chunks table (used in tests / re-indexing)."""
    try:
        db = _get_db()
        if _TABLE_NAME in db.table_names():
            db.drop_table(_TABLE_NAME)
        logger.info("Vector store table '%s' dropped", _TABLE_NAME)
    except Exception as exc:
        raise VectorStoreError(f"Failed to reset vector store: {exc}") from exc


def delete_by_source_file(source_file: str) -> None:
    """Remove every chunk vector belonging to *source_file*. No-op if missing."""
    if not source_file:
        return
    try:
        db = _get_db()
        if _TABLE_NAME not in db.table_names():
            return
        table = db.open_table(_TABLE_NAME)
        # LanceDB accepts SQL-style predicates; escape single quotes defensively.
        safe = source_file.replace("'", "''")
        table.delete(f"source_file = '{safe}'")
        logger.debug("Deleted vectors for source_file=%s", source_file)
    except Exception as exc:  # noqa: BLE001
        # Vector deletion is best-effort: we still want SQL deletes to commit.
        logger.warning("delete_by_source_file failed for %s: %s", source_file, exc)
