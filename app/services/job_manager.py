"""
Background ingestion job manager.

Runs each uploaded file through :func:`app.services.ingestion.ingest_file` in a
separate worker **process** (``ProcessPoolExecutor``) so that:

- Multiple files are processed in parallel (true parallelism, not just async I/O).
- The HTTP request returns immediately with a ``job_id``; the UI polls
  :func:`get_job` for live per-file progress.
- A single slow / hanging file does not block the rest of the batch.

State is held in-process (a module-level dict guarded by a lock). Sufficient
for a single-worker FastAPI deployment; swap for Redis if you scale out.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from app.core.logging import get_logger
from app.services.ingestion import ingest_file

logger = get_logger(__name__)


# ─── Worker pool (lazy singleton) ─────────────────────────────────────────────

_pool_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None


def _default_workers() -> int:
    # Cap at 4 to avoid hammering the LLM API rate limits / model cold-start RAM.
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu))


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            workers = int(os.environ.get("INGEST_WORKERS", _default_workers()))
            logger.info("Starting ingestion ProcessPoolExecutor with %d workers", workers)
            _pool = ProcessPoolExecutor(max_workers=workers)
        return _pool


def shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


# ─── Job state ────────────────────────────────────────────────────────────────

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _new_job_record(filenames: list[str]) -> dict[str, Any]:
    return {
        "job_id": uuid.uuid4().hex,
        "created_at": _now(),
        "updated_at": _now(),
        "total": len(filenames),
        "completed": 0,
        "done": False,
        "files": [
            {
                "index": i,
                "filename": fn,
                "status": "pending",   # pending → processing → ok | error | skipped
                "input_type": None,
                "detail": None,
                "started_at": None,
                "finished_at": None,
            }
            for i, fn in enumerate(filenames)
        ],
    }


def _set_file_status(job_id: str, index: int, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["files"][index].update(fields)
        job["updated_at"] = _now()
        if all(f["status"] in {"ok", "error", "skipped"} for f in job["files"]):
            job["done"] = True
        job["completed"] = sum(
            1 for f in job["files"] if f["status"] in {"ok", "error", "skipped"}
        )


def _make_done_callback(job_id: str, index: int):
    def _cb(fut: Future) -> None:
        try:
            result = fut.result()
            _set_file_status(
                job_id,
                index,
                status=result.status,
                input_type=result.input_type,
                detail=result.detail,
                finished_at=_now(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingest worker crashed for job %s file %d", job_id, index)
            _set_file_status(
                job_id,
                index,
                status="error",
                detail=f"Worker crashed: {exc}",
                finished_at=_now(),
            )
    return _cb


# ─── Public API ───────────────────────────────────────────────────────────────

def submit_job(files: list[tuple[str, bytes]]) -> str:
    """
    Schedule *files* (``[(filename, bytes), ...]``) for parallel ingestion.

    Returns the new ``job_id`` immediately; processing continues in the pool.
    """
    if not files:
        raise ValueError("No files provided")

    job = _new_job_record([fn for fn, _ in files])
    with _jobs_lock:
        _jobs[job["job_id"]] = job

    pool = _get_pool()
    for idx, (filename, content) in enumerate(files):
        _set_file_status(job["job_id"], idx, status="processing", started_at=_now())
        fut = pool.submit(ingest_file, filename, content)
        fut.add_done_callback(_make_done_callback(job["job_id"], idx))

    logger.info("Submitted ingestion job %s with %d files", job["job_id"], len(files))
    return job["job_id"]


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a snapshot copy of the job state, or ``None`` if unknown."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        # Return a shallow copy so callers can't mutate internal state.
        return {
            **job,
            "files": [dict(f) for f in job["files"]],
        }


def list_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return [
            {
                "job_id": j["job_id"],
                "created_at": j["created_at"],
                "total": j["total"],
                "completed": j["completed"],
                "done": j["done"],
            }
            for j in _jobs.values()
        ]
