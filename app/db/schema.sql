CREATE TABLE IF NOT EXISTS documents (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file                   TEXT    NOT NULL UNIQUE,
    input_type                    TEXT    NOT NULL,
    doc_type                      TEXT,
    doc_type_confidence           REAL,
    secondary_doc_type            TEXT,
    secondary_doc_type_confidence REAL,
    summary                       TEXT,
    raw_text                      TEXT,
    created_at                    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_type TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    normalized  TEXT
);

-- Indexes for fast entity lookups and cross-domain joins (Layer 4)
CREATE INDEX IF NOT EXISTS idx_entities_document_id  ON entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type         ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_type_value   ON entities(entity_type, value);

CREATE TABLE IF NOT EXISTS relationships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subject     TEXT    NOT NULL,
    predicate   TEXT    NOT NULL,
    object      TEXT    NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_relationships_document_id ON relationships(document_id);
CREATE INDEX IF NOT EXISTS idx_relationships_subject      ON relationships(subject);
CREATE INDEX IF NOT EXISTS idx_relationships_predicate    ON relationships(predicate);

CREATE TABLE IF NOT EXISTS components (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file           TEXT    NOT NULL UNIQUE,
    component_type        TEXT,
    material              TEXT,
    size_category         TEXT,
    min_cost_usd          REAL,
    max_cost_usd          REAL,
    avg_cost_usd          REAL,
    manufacturing_process TEXT,
    confidence            REAL,
    low_confidence        INTEGER DEFAULT 0,
    reasoning             TEXT,
    created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes to support cross-domain queries (e.g. component cost vs project budget)
CREATE INDEX IF NOT EXISTS idx_components_type          ON components(component_type);
CREATE INDEX IF NOT EXISTS idx_components_type_material ON components(component_type, material);
CREATE INDEX IF NOT EXISTS idx_components_low_conf      ON components(low_confidence);

-- ── Keyword search over document chunks (FTS5) ────────────────────────────────
-- Used by the analytical-query hybrid retriever (BM25 keyword + vector cosine).
-- chunk_text is tokenised; the rest are unindexed so they round-trip back to
-- the application without re-querying the chunks table.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_text,
    source_file   UNINDEXED,
    chunk_type    UNINDEXED,
    document_id   UNINDEXED,
    chunk_index   UNINDEXED,
    page_number   UNINDEXED,
    tokenize = 'porter unicode61'
);
