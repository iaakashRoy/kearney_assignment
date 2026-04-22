"""Text utility helpers: chunking, JSON extraction, image encoding, structure parsing."""
from __future__ import annotations

import base64
import json
import re

from app.core.logging import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 200) -> list[str]:
    """
    Character-based text splitter (~400 token chunks at ~4 chars/token).
    Returns an empty list for empty input.
    """
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def extract_json(text: str) -> dict:
    """
    Robustly parse the first JSON object from LLM output that may contain
    surrounding prose or markdown fences.
    Returns an empty dict if no valid JSON is found.
    """
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning("Could not extract JSON from LLM output (preview: %s)", text[:200])
    return {}


def image_to_data_url(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Encode raw image bytes as a base64 data URL for vision APIs."""
    ext = suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def structure_aware_chunks(
    text: str,
    chunk_size: int = 1600,
    overlap: int = 200,
) -> list[tuple[str, str]]:
    """
    Split *text* into chunks whose boundaries respect document structure.

    Strategy
    --------
    1. Split at Markdown heading lines (``#`` prefix) → one section per heading.
       The heading text itself is prepended to its section so each chunk is
       self-contained.
    2. Detect pipe-table blocks within each section and keep them intact as a
       single ``"table"`` chunk (tables must not be split mid-row).
    3. For remaining prose, apply character-based splitting with *overlap*.
    4. Any section that fits within *chunk_size* is kept as a single chunk.

    Returns
    -------
    list of (chunk_text, chunk_type) tuples.
    chunk_type is one of: ``"header_section"``, ``"table"``, ``"paragraph"``.
    """
    if not text:
        return []

    results: list[tuple[str, str]] = []

    # ── Split into heading-delimited sections ──────────────────────────────────
    heading_re = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    boundaries = [m.start() for m in heading_re.finditer(text)]

    # Add a virtual boundary at position 0 if the text doesn't start with a heading
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    sections: list[tuple[str, str]] = []  # (section_text, chunk_type_hint)
    for i in range(len(boundaries) - 1):
        section = text[boundaries[i]: boundaries[i + 1]].strip()
        if not section:
            continue
        hint = "header_section" if heading_re.match(section) else "paragraph"
        sections.append((section, hint))

    # ── Process each section ───────────────────────────────────────────────────
    for section_text, hint in sections:
        lines = section_text.splitlines()

        # Separate table blocks from prose within the section
        i = 0
        prose_lines: list[str] = []

        def _flush_prose(prose: list[str]) -> None:
            prose_text = "\n".join(prose).strip()
            if not prose_text:
                return
            if len(prose_text) <= chunk_size:
                results.append((prose_text, hint))
            else:
                for c in chunk_text(prose_text, chunk_size, overlap):
                    results.append((c, hint))

        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("|"):
                # Flush accumulated prose first
                _flush_prose(prose_lines)
                prose_lines = []

                # Collect the full table block
                table_lines: list[str] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                table_block = "\n".join(table_lines)
                # Tables are kept whole; if truly enormous, split by rows
                if len(table_block) <= chunk_size:
                    results.append((table_block, "table"))
                else:
                    # Keep header + separator, split rows into sub-tables
                    header_rows = table_lines[:2]  # header + separator row
                    header_str = "\n".join(header_rows)
                    batch: list[str] = list(header_rows)
                    for row in table_lines[2:]:
                        batch.append(row)
                        candidate = "\n".join(batch)
                        if len(candidate) > chunk_size and len(batch) > len(header_rows) + 1:
                            results.append(("\n".join(batch[:-1]), "table"))
                            batch = list(header_rows) + [row]
                    if len(batch) > len(header_rows):
                        results.append(("\n".join(batch), "table"))
            else:
                prose_lines.append(line)
                i += 1

        _flush_prose(prose_lines)

    return results if results else [(text, "paragraph")]


def parse_markdown_structure(text: str) -> "DocumentStructure":
    """
    Parse markdown text into a DocumentStructure with headers, paragraphs, and tables.

    - Lines starting with ``#`` are treated as headers (level stripped).
    - Pipe-delimited lines are parsed as structured tables (first row = column headers,
      separator rows ``|---|`` are discarded, remaining rows become data rows).
    - Remaining non-blank lines are grouped into paragraphs by blank-line boundaries.

    Returns a :class:`~app.models.schemas.DocumentStructure`.
    """
    # Import here to avoid a top-level circular dependency risk
    from app.models.schemas import DocumentStructure, TableData  # noqa: PLC0415

    headers: list[str] = []
    paragraphs: list[str] = []
    tables: list[TableData] = []

    lines = text.splitlines()
    current_para: list[str] = []

    def _flush_paragraph() -> None:
        para = " ".join(current_para).strip()
        if para:
            paragraphs.append(para)
        current_para.clear()

    def _parse_table_row(row_str: str) -> list[str]:
        return [cell.strip() for cell in row_str.strip().strip("|").split("|")]

    def _is_separator_row(row_str: str) -> bool:
        """True for rows like |---|:---|:---:| that mark the header/body boundary."""
        return bool(re.match(r"^[\s|:\-]+$", row_str))

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Headers ──────────────────────────────────────────────────────────
        if stripped.startswith("#"):
            _flush_paragraph()
            header_text = stripped.lstrip("#").strip()
            if header_text:
                headers.append(header_text)

        # ── Tables ───────────────────────────────────────────────────────────
        elif stripped.startswith("|"):
            _flush_paragraph()
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1  # compensate for the outer i += 1

            if table_lines:
                col_headers = _parse_table_row(table_lines[0])
                data_rows: list[list[str]] = []
                for tl in table_lines[1:]:
                    if _is_separator_row(tl):
                        continue
                    data_rows.append(_parse_table_row(tl))
                tables.append(TableData(headers=col_headers, rows=data_rows))

        # ── Blank line → paragraph boundary ──────────────────────────────────
        elif not stripped:
            _flush_paragraph()

        # ── Regular text ──────────────────────────────────────────────────────
        else:
            current_para.append(stripped)

        i += 1

    _flush_paragraph()

    return DocumentStructure(headers=headers, paragraphs=paragraphs, tables=tables)
