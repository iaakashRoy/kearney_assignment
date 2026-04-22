"""
Groq LLM client — text generation and vision inference.

All public functions raise LLMError on failure (after exhausting retries).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import groq
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.utils.text import extract_json, image_to_data_url

logger = get_logger(__name__)

_RETRIABLE_ERRORS = (
    groq.APIConnectionError,
    groq.RateLimitError,
    groq.InternalServerError,
)


@lru_cache(maxsize=1)
def get_groq_client() -> groq.Groq:
    api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not configured")
    return groq.Groq(api_key=api_key)


# ─── Text generation ──────────────────────────────────────────────────────────

def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """Call Groq text model and return the full response string."""

    @retry(
        retry=retry_if_exception_type(_RETRIABLE_ERRORS),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call() -> str:
        client = get_groq_client()
        kwargs: dict = dict(
            model=settings.groq_text_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            stream=True,
        )
        # gpt-oss is a reasoning model — minimise reasoning tokens so the
        # full token budget is available for the actual JSON answer.
        if "gpt-oss" in settings.groq_text_model:
            kwargs["reasoning_effort"] = "low"
        completion = client.chat.completions.create(**kwargs)
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)

    try:
        return _call()
    except _RETRIABLE_ERRORS as exc:
        raise LLMError(f"Text generation failed after retries: {exc}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected LLM error: {exc}") from exc


# ─── Vision: OCR ──────────────────────────────────────────────────────────────

def ocr_image(image_bytes: bytes, suffix: str = ".jpg") -> tuple[str, str]:
    """
    OCR an image using Groq vision.

    Returns:
        (input_type, extracted_text)
        input_type is ``"scanned_doc"`` when substantial text is found,
        otherwise ``"component_photo"``.
    """
    data_url = image_to_data_url(image_bytes, suffix)

    @retry(
        retry=retry_if_exception_type(_RETRIABLE_ERRORS),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        client = get_groq_client()
        return client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Examine this image carefully.\n\n"
                                "IF this image contains a document, form, report, invoice, table, "
                                "or any text-heavy content:\n"
                                "  1. Extract ALL text, preserving the document structure.\n"
                                "  2. Mark headings with Markdown heading syntax: # for H1, ## for H2, ### for H3.\n"
                                "  3. Format every table using Markdown pipe syntax:\n"
                                "     | Column A | Column B |\n"
                                "     |----------|----------|\n"
                                "     | value    | value    |\n"
                                "  4. Separate paragraphs with a blank line.\n"
                                "  5. Return ONLY the structured Markdown text — no commentary.\n\n"
                                "IF this image is a physical component, hardware part, or product photo "
                                "with little or no document text, respond with exactly: NO_TEXT"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=2048,
        )

    try:
        completion = _call()
    except _RETRIABLE_ERRORS as exc:
        raise LLMError(f"Vision OCR failed after retries: {exc}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected vision error: {exc}") from exc

    extracted = (completion.choices[0].message.content or "").strip()
    if extracted == "NO_TEXT" or len(extracted) < settings.ocr_text_threshold:
        return "component_photo", ""
    return "scanned_doc", extracted


# ─── Vision: component classification ────────────────────────────────────────

_CLASSIFY_PROMPT = (
    "You are an automotive seat component analyst. Examine this image and classify the component.\n"
    "Return ONLY a JSON object (no other text):\n"
    '{"component_type": "<one of: Seat Track Assembly, Slide Rail Mechanism, Seat Frame Structure, '
    "Seat Recliner Mechanism, Seat Back Frame, Seat Cushion Pan, Wire Spring Support, Mounting Bracket, "
    'Height Adjuster Mechanism, Lumbar Support Frame>", '
    '"material": "<one of: Cold-Rolled Steel (1008), High-Strength Low-Alloy Steel (HSLA), '
    "Tubular Steel (ASTM A513), Stamped Steel Sheet, Spring Steel Wire (ASTM A228), "
    'Aluminum Alloy (6061)>", '
    '"size_category": "<brief size description>", '
    '"confidence": <float 0.0-1.0>, '
    '"reasoning": "<one sentence>"}'
)


def classify_component_image(image_bytes: bytes, suffix: str = ".jpg") -> dict:
    """Classify a component photo via Groq vision. Returns a raw dict."""
    data_url = image_to_data_url(image_bytes, suffix)

    @retry(
        retry=retry_if_exception_type(_RETRIABLE_ERRORS),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        client = get_groq_client()
        return client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _CLASSIFY_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=256,
        )

    try:
        completion = _call()
    except _RETRIABLE_ERRORS as exc:
        raise LLMError(f"Component classification failed after retries: {exc}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected classification error: {exc}") from exc

    raw = (completion.choices[0].message.content or "").strip()
    result = extract_json(raw)
    if not result:
        logger.warning("Empty classification result — returning zero-confidence fallback")
        return {
            "component_type": None,
            "material": None,
            "size_category": None,
            "confidence": 0.0,
            "reasoning": raw or "No response from model",
        }
    return result
