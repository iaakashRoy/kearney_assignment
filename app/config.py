from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve .env relative to this file (platform/app/config.py → platform/.env)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = "platform.db"
    lancedb_path: str = "platform_lancedb"
    benchmark_csv_path: str = ""  # auto-detected if empty
    # Directory where original uploaded files are kept on disk so the UI can
    # render previews (PDF page renders, image source thumbnails).
    sources_dir: str = "platform_sources"
    preview_dpi_scale: float = 1.5  # pypdfium2 render scale for PDF previews

    # ── Text processing ───────────────────────────────────────────────────────
    ocr_text_threshold: int = 100
    chunk_size: int = 1600
    chunk_overlap: int = 200

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Groq API ──────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_text_model: str = "openai/gpt-oss-20b"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── LLM retry ─────────────────────────────────────────────────────────────
    llm_max_retries: int = 3
    llm_retry_min_wait: float = 1.0
    llm_retry_max_wait: float = 10.0

    # ── RAG thresholds ────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_distance_high: float = 0.30
    rag_distance_medium: float = 0.50
    rag_distance_low: float = 0.65

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


settings = Settings()
