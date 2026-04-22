"""
Natural-language → SQL agent with validation and one-shot self-correction retry.

Public surface (re-exported):

* :func:`structured_query` – translate a question to SQL and execute it.
"""
from app.services.sql_agent.runner import structured_query

__all__ = ["structured_query"]
