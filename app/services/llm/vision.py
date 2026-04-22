"""
Vision endpoints — OCR and component classification.
"""
from __future__ import annotations

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.services.llm.client import RETRIABLE_ERRORS, get_groq_client, with_retry
from app.utils.text import extract_json, image_to_data_url

logger = get_logger(__name__)


# ─── OCR ──────────────────────────────────────────────────────────────────────

def ocr_image(image_bytes: bytes, suffix: str = ".jpg") -> tuple[str, str]:
    """
    OCR an image using Groq vision.

    Returns:
        (input_type, extracted_text)
        input_type is ``"scanned_doc"`` when substantial text is found,
        otherwise ``"component_photo"``.
    """
    data_url = image_to_data_url(image_bytes, suffix)

    @with_retry
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
    except RETRIABLE_ERRORS as exc:
        raise LLMError(f"Vision OCR failed after retries: {exc}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected vision error: {exc}") from exc

    extracted = (completion.choices[0].message.content or "").strip()
    if extracted == "NO_TEXT" or len(extracted) < settings.ocr_text_threshold:
        return "component_photo", ""
    return "scanned_doc", extracted


# ─── Component classification ────────────────────────────────────────────────

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

    @with_retry
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
    except RETRIABLE_ERRORS as exc:
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
