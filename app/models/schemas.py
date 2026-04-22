"""Pydantic models for API requests, responses, and internal data structures."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Requests ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


# ─── Internal data models ──────────────────────────────────────────────────────

class EntityRecord(BaseModel):
    type: str
    value: str
    normalized: str | None = None


class RelationshipRecord(BaseModel):
    """A directed relationship between two entities: subject → predicate → object."""
    subject: str
    predicate: str
    object: str


class TableData(BaseModel):
    """A single extracted table: column headers + data rows."""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class DocumentStructure(BaseModel):
    """Detected structural elements within a document."""
    headers: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    doc_type: str | None = None
    doc_type_confidence: float | None = None
    secondary_doc_type: str | None = None
    secondary_doc_type_confidence: float | None = None
    entities: list[EntityRecord] = Field(default_factory=list)
    relationships: list[RelationshipRecord] = Field(default_factory=list)
    summary: str | None = None
    structure: DocumentStructure | None = None


class ComponentClassification(BaseModel):
    component_type: str | None = None
    material: str | None = None
    size_category: str | None = None
    confidence: float = 0.0
    reasoning: str | None = None


class ChunkRecord(BaseModel):
    document_id: int
    chunk_index: int
    source_file: str
    chunk_text: str
    vector: list[float]


# ─── API Responses ─────────────────────────────────────────────────────────────

class IngestResult(BaseModel):
    source_file: str
    input_type: str
    status: str  # "ok" | "error" | "skipped"
    detail: str
    structure: DocumentStructure | None = None


class UploadResponse(BaseModel):
    ingested: int
    results: list[IngestResult]


class SourceReference(BaseModel):
    file: str
    excerpt: str
    score: float | None = None
    page: int | None = None


class AnalyticalQueryResponse(BaseModel):
    answer: str
    reasoning: str | None = None  # chain-of-thought steps from the reasoning agent
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: str  # high | medium | low | none
    grounding_warning: str | None = None
    # ── Agentic-pipeline transparency (LangGraph) ─────────────────────────────
    plan: list[str] | None = None        # sub-queries the planner produced
    iterations: int | None = None        # number of retrieval hops actually run
    sql_used: bool | None = None         # whether the sql_tool was invoked


class StructuredQueryResponse(BaseModel):
    question: str
    sql: str | None = None
    result: list[dict[str, Any]] = Field(default_factory=list)
    assumption: str | None = None
    error: str | None = None


# ─── Async ingestion job tracking ─────────────────────────────────────────────

class JobFileStatus(BaseModel):
    index: int
    filename: str
    status: str  # pending | processing | ok | error | skipped
    input_type: str | None = None
    detail: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class JobStatus(BaseModel):
    job_id: str
    created_at: float
    updated_at: float
    total: int
    completed: int
    done: bool
    files: list[JobFileStatus] = Field(default_factory=list)


class JobSubmitResponse(BaseModel):
    job_id: str
    total: int


# ─── Data marketplace / catalog ───────────────────────────────────────────────

class CatalogDocument(BaseModel):
    id: int
    source_file: str
    input_type: str
    doc_type: str | None = None
    doc_type_confidence: float | None = None
    secondary_doc_type: str | None = None
    summary: str | None = None
    created_at: str | None = None
    n_entities: int = 0
    n_relationships: int = 0
    n_chunks: int = 0


class CatalogComponent(BaseModel):
    id: int
    source_file: str
    component_type: str | None = None
    material: str | None = None
    size_category: str | None = None
    min_cost_usd: float | None = None
    max_cost_usd: float | None = None
    avg_cost_usd: float | None = None
    manufacturing_process: str | None = None
    confidence: float | None = None
    low_confidence: int | None = None
    created_at: str | None = None


class CatalogResponse(BaseModel):
    documents: list[CatalogDocument] = Field(default_factory=list)
    components: list[CatalogComponent] = Field(default_factory=list)
    total_documents: int = 0
    total_components: int = 0


class DeleteResponse(BaseModel):
    deleted: bool
    kind: str  # "document" | "component" | "all"
    id: int | None = None
    source_file: str | None = None
    counts: dict[str, int] | None = None
