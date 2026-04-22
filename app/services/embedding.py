"""Sentence-Transformers embedding model — CPU-friendly, loaded once."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.

    Returns:
        A list of float vectors (one per input text).
    Raises:
        EmbeddingError: if the model fails.
    """
    if not texts:
        return []
    try:
        model = _get_model()
        arrays = model.encode(texts, convert_to_numpy=True)
        return arrays.tolist()
    except Exception as exc:
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
