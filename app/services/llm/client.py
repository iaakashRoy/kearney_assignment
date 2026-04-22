"""
Cached Groq client and shared retry configuration.
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

logger = get_logger(__name__)

RETRIABLE_ERRORS = (
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


def with_retry(func):
    """Decorator that applies the standard Groq retry policy to *func*."""
    return retry(
        retry=retry_if_exception_type(RETRIABLE_ERRORS),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
