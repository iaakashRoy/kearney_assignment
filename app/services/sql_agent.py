"""
Natural-language → SQL agent with validation and one-shot self-correction retry.
"""
from __future__ import annotations

import json
import re
import sqlite3

import sqlparse

from app.core.exceptions import LLMError, SQLValidationError
from app.core.logging import get_logger
from app.db import sqlite as db
from app.models.schemas import StructuredQueryResponse
from app.services.llm import generate_text
from app.utils.text import extract_json

logger = get_logger(__name__)

_BLOCKED_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """\
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


# ─── SQL validation ───────────────────────────────────────────────────────────

def _validate_sql(sql: str) -> None:
    """
    Raise SQLValidationError if the SQL contains disallowed statements or is unparseable.
    """
    if not sql:
        raise SQLValidationError("LLM returned empty SQL")
    if _BLOCKED_PATTERN.search(sql):
        raise SQLValidationError(f"Disallowed statement in generated SQL: {sql[:120]!r}")
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            raise SQLValidationError("SQL appears empty after parsing")
    except SQLValidationError:
        raise
    except Exception as exc:
        raise SQLValidationError(f"SQL parse error: {exc}") from exc


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(
    question: str,
    prior_sql: str | None = None,
    prior_error: str | None = None,
) -> dict:
    schema = db.get_schema_ddl()
    samples = json.dumps(db.get_sample_rows(3), indent=2, default=str)
    system_msg = _SYSTEM_PROMPT.format(schema=schema, samples=samples)

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


# ─── Public entry point ───────────────────────────────────────────────────────

def structured_query(question: str) -> StructuredQueryResponse:
    """
    Translate *question* to SQL, validate, execute, and return results.
    Automatically retries once with error context on validation or execution failure.
    """
    sql: str | None = None
    assumption: str | None = None
    error: str | None = None

    for attempt in range(2):
        try:
            llm_response = _call_llm(question, sql, error)
            sql = llm_response.get("sql", "").strip()
            assumption = llm_response.get("assumption")

            _validate_sql(sql)

            with db.get_connection() as conn:
                # Validate tables and columns exist without executing the query.
                # EXPLAIN QUERY PLAN raises sqlite3.OperationalError for unknown
                # tables or columns, which triggers the retry loop below.
                conn.execute(f"EXPLAIN QUERY PLAN {sql}")  # noqa: S608 — read-only plan only
                rows = conn.execute(sql).fetchall()
                result_rows = [dict(r) for r in rows]

            logger.info("Structured query OK (attempt %d): %s", attempt + 1, sql)
            return StructuredQueryResponse(
                question=question,
                sql=sql,
                result=result_rows,
                assumption=assumption,
                error=None,
            )

        except SQLValidationError as exc:
            error = str(exc)
            logger.warning("SQL validation failed (attempt %d): %s", attempt + 1, error)

        except sqlite3.Error as exc:
            error = str(exc)
            logger.warning("SQL execution failed (attempt %d): %s | SQL: %s", attempt + 1, error, sql)

        except LLMError as exc:
            logger.error("LLM call failed during SQL generation: %s", exc)
            return StructuredQueryResponse(
                question=question,
                sql=sql,
                result=[],
                assumption=assumption,
                error=str(exc),
            )

        except Exception as exc:
            logger.exception("Unexpected error during structured query")
            return StructuredQueryResponse(
                question=question,
                sql=sql,
                result=[],
                assumption=assumption,
                error=str(exc),
            )

    return StructuredQueryResponse(
        question=question,
        sql=sql,
        result=[],
        assumption=assumption,
        error=error or "Max retries exceeded",
    )
