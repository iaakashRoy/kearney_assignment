FROM python:3.11-slim

WORKDIR /app

# System dependencies for Docling (poppler for PDF rendering, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY platform/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Purge any stale HuggingFace / Docling model cache that may have been
# inherited from a parent image layer before pre-warming fresh copies.
# This avoids the "storage has wrong byte size" safetensors mismatch that
# occurs when a cached config and weights drift out of sync across versions.
RUN rm -rf /root/.cache/huggingface /root/.cache/docling /tmp/* /var/tmp/*

# Pre-warm Docling layout models (~300 MB, cached in this layer)
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

COPY platform/ .

# Remove any stray DB / vector-store / source files that may have been
# committed to the build context. Runtime data lives on the mounted
# /data volume, never inside the image.
RUN rm -rf /app/data/*.db /app/data/*.db-journal /app/data/*.sqlite \
           /app/data/platform_lancedb /app/data/platform_sources \
           /app/*.db /app/*.db-journal /app/*.sqlite

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]