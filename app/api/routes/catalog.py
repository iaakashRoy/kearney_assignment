"""Data marketplace endpoints — list, delete, and preview ingested artefacts."""
from __future__ import annotations

import io
import mimetypes
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.core.logging import get_logger
from app.db import sqlite as db
from app.db import vector_store
from app.models.schemas import (
    CatalogComponent,
    CatalogDocument,
    CatalogResponse,
    DeleteResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["catalog"])


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# pypdfium2 / libpdfium is NOT thread-safe. FastAPI runs sync handlers in a
# threadpool, and the marketplace UI fires many preview requests in parallel,
# which can abort the interpreter (SIGTRAP). Serialize all pdfium calls.
_PDFIUM_LOCK = threading.Lock()


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    """Return every ingested document and component for the marketplace UI."""
    docs = db.list_documents()
    comps = db.list_components()
    return CatalogResponse(
        documents=[CatalogDocument(**d) for d in docs],
        components=[CatalogComponent(**c) for c in comps],
        total_documents=len(docs),
        total_components=len(comps),
    )


@router.delete("/catalog/document/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: int) -> DeleteResponse:
    source_file = db.delete_document(document_id)
    if source_file is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    # Best-effort: remove the matching vectors and on-disk source.
    vector_store.delete_by_source_file(source_file)
    _unlink_source(source_file)
    return DeleteResponse(deleted=True, kind="document", id=document_id, source_file=source_file)


@router.delete("/catalog/component/{component_id}", response_model=DeleteResponse)
def delete_component(component_id: int) -> DeleteResponse:
    source_file = db.delete_component(component_id)
    if source_file is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
    _unlink_source(source_file)
    return DeleteResponse(deleted=True, kind="component", id=component_id, source_file=source_file)


@router.delete("/catalog", response_model=DeleteResponse)
def delete_all() -> DeleteResponse:
    """Wipe the entire catalog — SQLite tables and the vector store."""
    counts = db.delete_all()
    try:
        vector_store.reset()
    except Exception:  # noqa: BLE001
        # SQLite truncation already succeeded; surface the failure in counts only.
        counts["vector_store_reset"] = 0
    else:
        counts["vector_store_reset"] = 1
    counts["sources_removed"] = _wipe_sources_dir()
    return DeleteResponse(deleted=True, kind="all", counts=counts)


def _unlink_source(file_name: str) -> None:
    try:
        path = Path(settings.sources_dir) / Path(file_name).name
        if path.is_file():
            path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete source file %s: %s", file_name, exc)


def _wipe_sources_dir() -> int:
    removed = 0
    try:
        sources_dir = Path(settings.sources_dir)
        if sources_dir.is_dir():
            for child in sources_dir.iterdir():
                if child.is_file():
                    child.unlink()
                    removed += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sources-dir wipe failed: %s", exc)
    return removed


# ─── Source previews ──────────────────────────────────────────────────────────

def _resolve_source_path(file_name: str) -> Path:
    """Resolve *file_name* to a path inside ``settings.sources_dir``, blocking traversal."""
    sources_dir = Path(settings.sources_dir).resolve()
    # Strip directory components — the catalog only stores plain filenames.
    safe_name = Path(file_name).name
    candidate = (sources_dir / safe_name).resolve()
    if sources_dir not in candidate.parents and candidate != sources_dir:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Source not found: {safe_name}")
    return candidate


@router.get("/catalog/source")
def get_source(
    file: str = Query(..., description="Source filename as stored in the catalog"),
    page: int | None = Query(None, ge=1, description="1-based page number; PDFs only"),
    scale: float | None = Query(None, gt=0.5, le=4.0, description="PDF render scale (defaults to settings.preview_dpi_scale)"),
):
    """
    Stream a preview of an ingested source.

    - **PDFs** with a ``page`` query render that page to PNG using pypdfium2.
    - **PDFs** without ``page`` return the original PDF (browser can render it).
    - **Images** are returned as-is.
    """
    path = _resolve_source_path(file)
    suffix = path.suffix.lower()

    if suffix in _IMAGE_EXTS:
        mime, _ = mimetypes.guess_type(str(path))
        return FileResponse(path, media_type=mime or "application/octet-stream")

    if suffix == ".pdf":
        if page is None:
            return FileResponse(path, media_type="application/pdf")
        return _render_pdf_page(path, page, scale or settings.preview_dpi_scale)

    raise HTTPException(status_code=415, detail=f"Unsupported source type: {suffix}")


def _render_pdf_page(path: Path, page_no: int, scale: float) -> Response:
    """Render a single PDF page to PNG bytes."""
    try:
        import pypdfium2 as pdfium  # imported lazily — only needed for PDF previews
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="pypdfium2 is not installed; cannot render PDF previews",
        ) from exc

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            if page_no < 1 or page_no > len(pdf):
                raise HTTPException(
                    status_code=404,
                    detail=f"Page {page_no} out of range (1..{len(pdf)})",
                )
            page = pdf[page_no - 1]
            try:
                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil()
                buf = io.BytesIO()
                pil.save(buf, format="PNG", optimize=True)
                return Response(content=buf.getvalue(), media_type="image/png")
            finally:
                page.close()
        finally:
            pdf.close()
