"""
Groq LLM client — text generation and vision inference.

Public surface (re-exported for backwards compatibility):

* :func:`get_groq_client`       – cached client factory
* :func:`generate_text`         – text generation
* :func:`ocr_image`             – vision OCR
* :func:`classify_component_image` – vision classification

All functions raise :class:`~app.core.exceptions.LLMError` on failure
(after exhausting retries).
"""
from app.services.llm.client import get_groq_client
from app.services.llm.text import generate_text
from app.services.llm.vision import classify_component_image, ocr_image

__all__ = [
    "get_groq_client",
    "generate_text",
    "ocr_image",
    "classify_component_image",
]
