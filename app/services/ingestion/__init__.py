"""
Ingestion service — routes uploaded files through the correct processing stream.

* Stream A (digital_doc / scanned_doc): entity extraction → SQLite + LanceDB
  vector store. Implemented in :mod:`app.services.ingestion.document`.
* Stream B (component_photo): vision classification + cost lookup → SQLite.
  Implemented in :mod:`app.services.ingestion.component`.

The :func:`ingest_file` dispatcher in :mod:`app.services.ingestion.pipeline`
selects the right stream based on file type / OCR result.
"""
from app.services.ingestion.pipeline import ingest_file

__all__ = ["ingest_file"]
