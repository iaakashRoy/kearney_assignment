"""Custom exception hierarchy for the platform."""


class AppError(Exception):
    """Base exception for all application errors."""


class IngestionError(AppError):
    """Raised when file ingestion fails."""


class UnsupportedFileTypeError(IngestionError):
    """Raised for unsupported file extensions."""


class LLMError(AppError):
    """Raised when an LLM call fails after retries."""


class EmbeddingError(AppError):
    """Raised when embedding generation fails."""


class DatabaseError(AppError):
    """Raised on SQLite operation failures."""


class VectorStoreError(AppError):
    """Raised on LanceDB operation failures."""


class SQLValidationError(AppError):
    """Raised when generated SQL fails safety validation."""


class SQLGenerationError(AppError):
    """Raised when the LLM fails to produce usable SQL."""
