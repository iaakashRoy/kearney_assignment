"""
PDF / image extraction utilities (powered by Docling) and source-file
persistence helpers.
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from docling.document_converter import DocumentConverter

from app.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg"})


@lru_cache(maxsize=1)
def get_docling() -> DocumentConverter:
    logger.info("Initialising Docling document converter")
    return DocumentConverter()


def extract_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = get_docling().convert(tmp_path)
        return result.document.export_to_markdown()
    except Exception as exc:
        raise IngestionError(f"PDF extraction failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)


def extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
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
        doc = get_docling().convert(tmp_path).document
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


def save_source(filename: str, file_bytes: bytes) -> None:
    """Persist the original upload to ``settings.sources_dir`` for preview rendering."""
    try:
        out_dir = Path(settings.sources_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Strip any path components to avoid traversal.
        safe = Path(filename).name
        (out_dir / safe).write_bytes(file_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist source file %s: %s", filename, exc)
