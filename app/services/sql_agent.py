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
from app.services.agents.trace import emit
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
    import time
    import uuid

    qid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    logger.info("[%s] ═══ structured_query start ═══ question=%r", qid, question[:200])
    emit("run.start", qid=qid, question=question, mode="structured")

    sql: str | None = None
    assumption: str | None = None
    error: str | None = None

    for attempt in range(2):
        attempt_label = "initial" if attempt == 0 else "retry (self-correction)"
        logger.info("[%s] ▶ SQL_AGENT attempt=%d (%s)", qid, attempt + 1, attempt_label)
        emit("node.start", node="sql_agent", qid=qid,
             attempt=attempt + 1, attempt_label=attempt_label,
             prior_error=error)
        t_attempt = time.perf_counter()
        try:
            llm_response = _call_llm(question, sql, error)
            sql = llm_response.get("sql", "").strip()
            assumption = llm_response.get("assumption")
            logger.info("[%s]   LLM proposed SQL: %s", qid, sql[:200])
            if assumption:
                logger.info("[%s]   assumption     : %s", qid, assumption[:200])
            emit("sql.proposed", qid=qid, attempt=attempt + 1,
                 sql=sql, assumption=assumption)

            _validate_sql(sql)
            emit("sql.validated", qid=qid, attempt=attempt + 1)
            logger.info("[%s]   validation     : OK (no disallowed statements)", qid)

            with db.get_connection() as conn:
                conn.execute(f"EXPLAIN QUERY PLAN {sql}")  # noqa: S608
                rows = conn.execute(sql).fetchall()
                result_rows = [dict(r) for r in rows]

            elapsed = (time.perf_counter() - t_attempt) * 1000
            logger.info(
                "[%s]   execution      : OK → %d rows in %.0fms",
                qid, len(result_rows), elapsed,
            )
            emit("sql.executed", qid=qid, attempt=attempt + 1,
                 rows=len(result_rows), elapsed_ms=int(elapsed),
                 columns=list(result_rows[0].keys()) if result_rows else [])
            emit("node.done", node="sql_agent", qid=qid,
                 attempt=attempt + 1, elapsed_ms=int(elapsed))

            response = StructuredQueryResponse(
                question=question, sql=sql, result=result_rows,
                assumption=assumption, error=None,
            )
            run_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "[%s] ═══ structured_query done in %.0fms attempts=%d rows=%d ═══",
                qid, run_ms, attempt + 1, len(result_rows),
            )
            emit("run.done", qid=qid, elapsed_ms=int(run_ms),
                 attempts=attempt + 1, rows=len(result_rows),
                 final=response.model_dump())
            return response

        except SQLValidationError as exc:
            error = str(exc)
            logger.warning("[%s]   validation FAILED (attempt %d): %s", qid, attempt + 1, error)
            emit("sql.error", qid=qid, attempt=attempt + 1,
                 stage="validation", error=error, sql=sql)
            emit("node.done", node="sql_agent", qid=qid, attempt=attempt + 1,
                 elapsed_ms=int((time.perf_counter() - t_attempt) * 1000),
                 error=error)

        except sqlite3.Error as exc:
            error = str(exc)
            logger.warning("[%s]   execution FAILED (attempt %d): %s | SQL: %s",
                           qid, attempt + 1, error, sql)
            emit("sql.error", qid=qid, attempt=attempt + 1,
                 stage="execution", error=error, sql=sql)
            emit("node.done", node="sql_agent", qid=qid, attempt=attempt + 1,
                 elapsed_ms=int((time.perf_counter() - t_attempt) * 1000),
                 error=error)

        except LLMError as exc:
            logger.error("[%s]   LLM failed: %s", qid, exc)
            emit("sql.error", qid=qid, attempt=attempt + 1,
                 stage="llm", error=str(exc))
            response = StructuredQueryResponse(
                question=question, sql=sql, result=[],
                assumption=assumption, error=str(exc),
            )
            emit("run.done", qid=qid,
                 elapsed_ms=int((time.perf_counter() - t0) * 1000),
                 attempts=attempt + 1, rows=0, final=response.model_dump())
            return response

        except Exception as exc:
            logger.exception("[%s] Unexpected error during structured query", qid)
            emit("sql.error", qid=qid, attempt=attempt + 1,
                 stage="unexpected", error=str(exc))
            response = StructuredQueryResponse(
                question=question, sql=sql, result=[],
                assumption=assumption, error=str(exc),
            )
            emit("run.done", qid=qid,
                 elapsed_ms=int((time.perf_counter() - t0) * 1000),
                 attempts=attempt + 1, rows=0, final=response.model_dump())
            return response

    response = StructuredQueryResponse(
        question=question, sql=sql, result=[],
        assumption=assumption,
        error=error or "Max retries exceeded",
    )
    run_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] ═══ structured_query exhausted retries in %.0fms ═══ error=%s",
        qid, run_ms, response.error,
    )
    emit("run.done", qid=qid, elapsed_ms=int(run_ms),
         attempts=2, rows=0, final=response.model_dump())
    return response
