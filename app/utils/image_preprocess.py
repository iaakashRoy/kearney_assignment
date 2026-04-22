"""
Image pre-processing for scanned documents.

Pipeline: skew correction → noise reduction → contrast enhancement (CLAHE).
Applied before OCR to improve text extraction quality.
Falls back to original bytes on any error so ingestion never hard-fails.
"""
from __future__ import annotations

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def _detect_skew_angle(gray: "np.ndarray") -> float:
    """
    Estimate document skew angle (degrees) via minAreaRect on thresholded foreground pixels.
    Returns 0.0 when not enough foreground pixels are found.
    """
    import cv2  # local import — opencv is optional at module load time

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); normalise to (-45, 45]
    if angle < -45.0:
        angle = 90.0 + angle
    return float(angle)


def preprocess_for_ocr(image_bytes: bytes, suffix: str = ".jpg") -> bytes:
    """
    Pre-process an image to improve OCR quality.

    Steps
    -----
    1. Skew correction — only applied when tilt > 0.5°
    2. Noise reduction — ``cv2.fastNlMeansDenoising``
    3. Contrast enhancement — CLAHE on grayscale

    Returns preprocessed bytes (grayscale) in the same format as the input.
    Falls back to *image_bytes* unchanged on any error.
    """
    try:
        import cv2  # local import

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not decode image for pre-processing — using original")
            return image_bytes

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 1. Skew correction ────────────────────────────────────────────────
        angle = _detect_skew_angle(gray)
        if abs(angle) > 0.5:
            logger.debug("Correcting skew: %.2f°", angle)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 2. Noise reduction ────────────────────────────────────────────────
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # ── 3. Contrast enhancement (CLAHE) ───────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Encode back in the original format
        ext = suffix.lstrip(".").lower()
        encode_ext = ".png" if ext == "png" else ".jpg"
        success, buf = cv2.imencode(encode_ext, enhanced)
        if not success:
            logger.warning("Failed to encode pre-processed image — using original")
            return image_bytes

        logger.debug("Image pre-processed successfully (skew=%.2f°)", angle)
        return buf.tobytes()

    except Exception as exc:
        logger.warning("Image pre-processing failed (%s) — using original bytes", exc)
        return image_bytes
