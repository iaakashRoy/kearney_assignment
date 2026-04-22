import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured, human-readable format."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "app.log"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Propagate uvicorn's own loggers through the same handlers
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_log = logging.getLogger(uvicorn_logger)
        uv_log.handlers = handlers
        uv_log.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
