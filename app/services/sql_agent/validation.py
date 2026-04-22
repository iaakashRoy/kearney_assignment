"""
Defensive SQL validation — blocks DML/DDL and ensures the statement parses.
"""
from __future__ import annotations

import re

import sqlparse

from app.core.exceptions import SQLValidationError

BLOCKED_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> None:
    """
    Raise SQLValidationError if the SQL contains disallowed statements or is
    unparseable.
    """
    if not sql:
        raise SQLValidationError("LLM returned empty SQL")
    if BLOCKED_PATTERN.search(sql):
        raise SQLValidationError(f"Disallowed statement in generated SQL: {sql[:120]!r}")
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            raise SQLValidationError("SQL appears empty after parsing")
    except SQLValidationError:
        raise
    except Exception as exc:
        raise SQLValidationError(f"SQL parse error: {exc}") from exc
