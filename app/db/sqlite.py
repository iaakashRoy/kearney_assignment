"""SQLite data-access layer."""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.config import settings
from app.core.exceptions import DatabaseError
from app.core.logging import get_logger

logger = get_logger(__name__)

# schema.sql lives next to this module, inside the db package.
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a SQLite connection with foreign-key enforcement.
    Auto-commits on clean exit, rolls back on exception.

    WAL journal mode + a generous busy timeout let multiple worker processes
    (used by the async ingestion job manager) write concurrently without
    "database is locked" errors.
    """
    conn = sqlite3.connect(settings.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables from schema.sql (idempotent) and run additive migrations."""
    logger.info("Initialising SQLite schema at %s", settings.db_path)
    schema = _SCHEMA_PATH.read_text()
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.executescript(schema)
        conn.commit()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Schema initialisation failed: {exc}") from exc
    finally:
        conn.close()

    # Additive column migrations — safe to run on existing databases.
    # SQLite raises OperationalError ("duplicate column name") if the column
    # already exists; we catch and ignore that specific case.
    _run_migrations()


_MIGRATIONS: list[str] = [
    "ALTER TABLE documents ADD COLUMN secondary_doc_type TEXT",
    "ALTER TABLE documents ADD COLUMN secondary_doc_type_confidence REAL",
]


def _ensure_chunks_fts_has_page() -> None:
    """
    FTS5 virtual tables can't be ALTERed. If the existing chunks_fts was created
    by an earlier version that lacked ``page_number``, drop and recreate it so
    new ingests (which expect the column) don't fail. Existing chunk vectors
    in LanceDB are preserved; the FTS index will be rebuilt on next ingest.
    """
    conn = sqlite3.connect(settings.db_path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if row and row[0] and "page_number" not in row[0]:
            logger.info("Recreating chunks_fts with page_number column")
            conn.execute("DROP TABLE chunks_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_text, source_file UNINDEXED, chunk_type UNINDEXED, "
                "document_id UNINDEXED, chunk_index UNINDEXED, page_number UNINDEXED, "
                "tokenize='porter unicode61')"
            )
            conn.commit()
    finally:
        conn.close()


def _run_migrations() -> None:
    """Apply additive ALTER TABLE migrations; silently skip already-applied ones."""
    conn = sqlite3.connect(settings.db_path)
    try:
        for stmt in _MIGRATIONS:
            try:
                conn.execute(stmt)
                conn.commit()
                logger.info("Migration applied: %s", stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" in str(exc).lower():
                    pass  # already applied
                else:
                    raise
    finally:
        conn.close()
    _ensure_chunks_fts_has_page()


def get_schema_ddl() -> str:
    """Return the raw schema DDL string."""
    return _SCHEMA_PATH.read_text()


def get_sample_rows(limit: int = 3) -> dict[str, list[dict]]:
    """Return up to *limit* rows from each table (used for LLM context)."""
    tables = ["documents", "entities", "components"]
    samples: dict[str, list[dict]] = {}
    try:
        with get_connection() as conn:
            for table in tables:
                # Table names come from a controlled constant — no injection risk
                rows = conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,)).fetchall()  # noqa: S608
                samples[table] = [dict(r) for r in rows]
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to fetch sample rows: {exc}") from exc
    return samples


# ─── Documents ────────────────────────────────────────────────────────────────

def insert_document(
    source_file: str,
    input_type: str,
    doc_type: str | None,
    doc_type_confidence: float | None,
    secondary_doc_type: str | None,
    secondary_doc_type_confidence: float | None,
    summary: str | None,
    raw_text: str,
) -> int:
    sql = """
        INSERT INTO documents
            (source_file, input_type, doc_type, doc_type_confidence,
             secondary_doc_type, secondary_doc_type_confidence, summary, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            input_type                    = excluded.input_type,
            doc_type                      = excluded.doc_type,
            doc_type_confidence           = excluded.doc_type_confidence,
            secondary_doc_type            = excluded.secondary_doc_type,
            secondary_doc_type_confidence = excluded.secondary_doc_type_confidence,
            summary                       = excluded.summary,
            raw_text                      = excluded.raw_text
        RETURNING id
    """
    try:
        with get_connection() as conn:
            row = conn.execute(
                sql,
                (
                    source_file, input_type, doc_type, doc_type_confidence,
                    secondary_doc_type, secondary_doc_type_confidence, summary, raw_text,
                ),
            ).fetchone()
            return row["id"]
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"insert_document failed: {exc}") from exc


def insert_entities(document_id: int, entities: list[dict]) -> None:
    sql = "INSERT INTO entities (document_id, entity_type, value, normalized) VALUES (?, ?, ?, ?)"
    rows = [(document_id, e["type"], e["value"], e.get("normalized")) for e in entities]
    try:
        with get_connection() as conn:
            conn.executemany(sql, rows)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"insert_entities failed: {exc}") from exc


def insert_relationships(document_id: int, relationships: list[dict]) -> None:
    """Persist extracted entity relationships for a document."""
    if not relationships:
        return
    sql = "INSERT INTO relationships (document_id, subject, predicate, object) VALUES (?, ?, ?, ?)"
    rows = [
        (document_id, r["subject"], r["predicate"], r["object"])
        for r in relationships
        if r.get("subject") and r.get("predicate") and r.get("object")
    ]
    try:
        with get_connection() as conn:
            conn.executemany(sql, rows)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"insert_relationships failed: {exc}") from exc


# ─── Components ───────────────────────────────────────────────────────────────

def insert_component(
    source_file: str,
    component_type: str | None,
    material: str | None,
    size_category: str | None,
    min_cost_usd: float | None,
    max_cost_usd: float | None,
    avg_cost_usd: float | None,
    manufacturing_process: str | None,
    confidence: float | None,
    low_confidence: int,
    reasoning: str | None,
) -> None:
    sql = """
        INSERT INTO components
            (source_file, component_type, material, size_category,
             min_cost_usd, max_cost_usd, avg_cost_usd, manufacturing_process,
             confidence, low_confidence, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            component_type        = excluded.component_type,
            material              = excluded.material,
            size_category         = excluded.size_category,
            min_cost_usd          = excluded.min_cost_usd,
            max_cost_usd          = excluded.max_cost_usd,
            avg_cost_usd          = excluded.avg_cost_usd,
            manufacturing_process = excluded.manufacturing_process,
            confidence            = excluded.confidence,
            low_confidence        = excluded.low_confidence,
            reasoning             = excluded.reasoning
    """
    try:
        with get_connection() as conn:
            conn.execute(
                sql,
                (
                    source_file, component_type, material, size_category,
                    min_cost_usd, max_cost_usd, avg_cost_usd, manufacturing_process,
                    confidence, low_confidence, reasoning,
                ),
            )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"insert_component failed: {exc}") from exc


def fetch_all_components() -> list[dict]:
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM components").fetchall()
            return [dict(r) for r in rows]
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"fetch_all_components failed: {exc}") from exc


# ─── Catalog / data-marketplace queries ───────────────────────────────────────

def list_documents() -> list[dict]:
    """
    Catalog rows for ingested documents, with per-document counts of entities,
    relationships, and chunks (joined via the FTS mirror table, which is
    populated for every chunk we store in LanceDB).
    """
    sql = """
        SELECT
            d.id,
            d.source_file,
            d.input_type,
            d.doc_type,
            d.doc_type_confidence,
            d.secondary_doc_type,
            d.summary,
            d.created_at,
            (SELECT COUNT(*) FROM entities      e WHERE e.document_id = d.id) AS n_entities,
            (SELECT COUNT(*) FROM relationships r WHERE r.document_id = d.id) AS n_relationships,
            (SELECT COUNT(*) FROM chunks_fts    c WHERE c.document_id = d.id) AS n_chunks
        FROM documents d
        ORDER BY d.created_at DESC, d.id DESC
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"list_documents failed: {exc}") from exc


def list_components() -> list[dict]:
    sql = """
        SELECT id, source_file, component_type, material, size_category,
               min_cost_usd, max_cost_usd, avg_cost_usd, manufacturing_process,
               confidence, low_confidence, created_at
        FROM components
        ORDER BY created_at DESC, id DESC
    """
    try:
        with get_connection() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"list_components failed: {exc}") from exc


def get_document_source_file(document_id: int) -> str | None:
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT source_file FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return row["source_file"] if row else None
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"get_document_source_file failed: {exc}") from exc


def delete_document(document_id: int) -> str | None:
    """
    Remove a document and its dependent rows (entities/relationships cascade,
    chunks_fts is cleared explicitly). Returns the deleted ``source_file`` so
    the caller can also delete the matching vectors from LanceDB.
    """
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT source_file FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not row:
                return None
            source_file = row["source_file"]
            # entities / relationships are removed by ON DELETE CASCADE
            conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents  WHERE id = ?", (document_id,))
            return source_file
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"delete_document failed: {exc}") from exc


def delete_component(component_id: int) -> str | None:
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT source_file FROM components WHERE id = ?", (component_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM components WHERE id = ?", (component_id,))
            return row["source_file"]
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"delete_component failed: {exc}") from exc


def delete_all() -> dict:
    """Wipe every ingested row from SQLite. Returns per-table delete counts."""
    counts: dict[str, int] = {}
    try:
        with get_connection() as conn:
            for table in ("chunks_fts", "relationships", "entities", "documents", "components"):
                cur = conn.execute(f"DELETE FROM {table}")
                counts[table] = cur.rowcount if cur.rowcount is not None else 0
        return counts
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"delete_all failed: {exc}") from exc


# ─── Chunk keyword index (FTS5) ───────────────────────────────────────────────

def insert_chunks_fts(chunks: list[dict]) -> None:
    """
    Mirror chunk text into the FTS5 virtual table for BM25 keyword search.

    Each *chunk* dict must expose: ``chunk_text``, ``source_file``, ``chunk_type``,
    ``document_id``, ``chunk_index``, ``page_number``. Failures are logged but
    never raised — the vector store remains the authoritative chunk store.
    """
    if not chunks:
        return
    sql = (
        "INSERT INTO chunks_fts (chunk_text, source_file, chunk_type, document_id, chunk_index, page_number) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    rows = [
        (
            c.get("chunk_text", ""),
            c.get("source_file", ""),
            c.get("chunk_type", ""),
            int(c.get("document_id", 0) or 0),
            int(c.get("chunk_index", 0) or 0),
            int(c.get("page_number", 0) or 0),
        )
        for c in chunks
    ]
    try:
        with get_connection() as conn:
            conn.executemany(sql, rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("insert_chunks_fts failed (keyword search will be incomplete): %s", exc)


_FTS_SAFE = re.compile(r"[A-Za-z0-9_]+")


def _build_fts_match_query(question: str) -> str:
    """
    Convert a free-form question into a safe FTS5 MATCH expression.

    - Extract alphanumeric tokens (drops punctuation that FTS5 treats specially).
    - Drop tokens shorter than 3 chars and a small stop-word set.
    - Join with OR so any keyword can match.
    - Append a prefix match (``token*``) for the longest token to catch stems.

    Returns an empty string when no usable tokens remain.
    """
    stop = {"the", "and", "for", "are", "with", "from", "this", "that",
            "what", "which", "who", "how", "why", "does", "did", "was",
            "were", "has", "have", "had", "any", "all", "into", "out"}
    tokens = [t.lower() for t in _FTS_SAFE.findall(question or "")]
    tokens = [t for t in tokens if len(t) >= 3 and t not in stop]
    if not tokens:
        return ""
    # Deduplicate, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    longest = max(uniq, key=len)
    parts = [f'"{t}"' for t in uniq]
    parts.append(f'"{longest}"*')
    return " OR ".join(parts)


def keyword_search_chunks(question: str, limit: int = 5) -> list[dict]:
    """
    BM25 keyword search over chunks_fts. Returns up to *limit* chunks ordered
    by ascending BM25 score (lower is more relevant in SQLite FTS5).

    Each result dict contains: ``source_file``, ``chunk_text``, ``chunk_type``,
    ``document_id``, ``chunk_index``, ``_bm25``.
    """
    match = _build_fts_match_query(question)
    if not match:
        return []
    sql = (
        "SELECT source_file, chunk_text, chunk_type, document_id, chunk_index, page_number, "
        "       bm25(chunks_fts) AS bm25 "
        "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25 LIMIT ?"
    )
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, (match, limit)).fetchall()
            return [
                {
                    "source_file": r["source_file"],
                    "chunk_text": r["chunk_text"],
                    "chunk_type": r["chunk_type"],
                    "document_id": r["document_id"],
                    "chunk_index": r["chunk_index"],
                    "page_number": r["page_number"],
                    "_bm25": r["bm25"],
                }
                for r in rows
            ]
    except sqlite3.OperationalError as exc:
        # Malformed MATCH or table missing — fall back gracefully
        logger.debug("keyword_search_chunks skipped (%s)", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("keyword_search_chunks failed: %s", exc)
        return []
