"""
Text generation via the Groq chat completions API (streaming).
"""
from __future__ import annotations

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.services.llm.client import RETRIABLE_ERRORS, get_groq_client, with_retry

logger = get_logger(__name__)


def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """Call Groq text model and return the full response string."""

    @with_retry
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
    except RETRIABLE_ERRORS as exc:
        raise LLMError(f"Text generation failed after retries: {exc}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected LLM error: {exc}") from exc
