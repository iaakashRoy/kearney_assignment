FROM python:3.11-slim

WORKDIR /app

# System dependencies for Docling (poppler for PDF rendering, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# BuildKit cache mount keeps the pip download cache across rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Purge any stale model cache inherited from a parent image layer, then
# pre-warm Docling layout models (~300 MB) — both in one layer so the
# downloaded weights are committed and the purge is not a wasted layer.
RUN rm -rf /root/.cache/huggingface /root/.cache/docling /tmp/* /var/tmp/* && \
    python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

COPY . .

# Remove any stray DB / vector-store / source files that may have been
# committed to the build context. Runtime data lives on the mounted
# /data volume, never inside the image.
RUN rm -rf /app/data/*.db /app/data/*.db-journal /app/data/*.sqlite \
           /app/data/platform_lancedb /app/data/platform_sources \
           /app/*.db /app/*.db-journal /app/*.sqlite

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]