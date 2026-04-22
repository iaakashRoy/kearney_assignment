"""
Prompt template and LLM call used by the SQL agent.
"""
from __future__ import annotations

import json
import re

from app.db import sqlite as db
from app.services.llm import generate_text
from app.utils.text import extract_json

SYSTEM_PROMPT = """\
You are an expert SQLite query generator for a document intelligence platform.

Schema:
{schema}

Sample rows:
{samples}

Rules:
- Return ONLY a valid SQLite SELECT statement — no markdown fences, no explanation.
- Use only tables and columns present in the schema.
- If the question is ambiguous (e.g. "latest"), make a reasonable assumption and explain it.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE.

Respond with a JSON object:
{{
  "sql": "<SELECT statement>",
  "assumption": null | "<explanation if you made an assumption>"
}}\
"""


def call_llm(
    question: str,
    prior_sql: str | None = None,
    prior_error: str | None = None,
) -> dict:
    """Ask the LLM for a SQL statement; falls back to extracting any SELECT."""
    schema = db.get_schema_ddl()
    samples = json.dumps(db.get_sample_rows(3), indent=2, default=str)
    system_msg = SYSTEM_PROMPT.format(schema=schema, samples=samples)

    parts = [f"Question: {question}"]
    if prior_sql and prior_error:
        parts.append(
            f"\nPrevious attempt failed.\n"
            f"SQL: {prior_sql}\n"
            f"Error: {prior_error}\n"
            f"Please correct the SQL and retry."
        )

    raw = generate_text(system_msg + "\n\n" + "\n".join(parts), max_tokens=1024)
    result = extract_json(raw)

    if not result or "sql" not in result:
        # Fallback: grab any SELECT from the raw text
        sql_match = re.search(r"SELECT\b.*", raw, re.IGNORECASE | re.DOTALL)
        result = {
            "sql": sql_match.group().strip() if sql_match else "",
            "assumption": None,
        }
    return result
