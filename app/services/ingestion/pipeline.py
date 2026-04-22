"""
Public dispatcher: routes a single uploaded file to the correct stream.
"""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import EmbeddingError, IngestionError, LLMError
from app.core.logging import get_logger
from app.models.schemas import IngestResult
from app.services import llm
from app.services.ingestion.component import ingest_component
from app.services.ingestion.document import ingest_document
from app.services.ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    extract_pdf_pages,
    save_source,
)
from app.utils.image_preprocess import preprocess_for_ocr

logger = get_logger(__name__)


def ingest_file(filename: str, file_bytes: bytes) -> IngestResult:
    """
    Process a single uploaded file end-to-end.

    Returns an IngestResult with status ``"ok"``, ``"error"``, or ``"skipped"``.
    Never raises — errors are captured into the result.
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
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
        save_source(filename, file_bytes)

        if ext == ".pdf":
            pages = extract_pdf_pages(file_bytes)
            return ingest_document(filename, "digital_doc", pages)

        # Pre-process the image to improve OCR quality, then detect type.
        # Original bytes are preserved for component classification (keeps colour).
        preprocessed_bytes = preprocess_for_ocr(file_bytes, suffix=ext)
        input_type, raw_text = llm.ocr_image(preprocessed_bytes, suffix=ext)
        if input_type == "scanned_doc":
            return ingest_document(filename, input_type, [(1, raw_text)])
        # Component photos: classify using original (colour) bytes.
        return ingest_component(filename, file_bytes, ext)

    except (LLMError, EmbeddingError, IngestionError) as exc:
        logger.error("Ingestion error for %s: %s", filename, exc)
        return IngestResult(source_file=filename, input_type="unknown", status="error", detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error ingesting %s", filename)
        return IngestResult(source_file=filename, input_type="unknown", status="error", detail=str(exc))
