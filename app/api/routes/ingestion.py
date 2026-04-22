from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import (
    JobStatus,
    JobSubmitResponse,
    UploadResponse,
)
from app.services import job_manager
from app.services.ingestion import ingest_file

router = APIRouter(tags=["ingestion"])


@router.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Synchronous ingest (blocks until all files are processed)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for uf in files:
        content = await uf.read()
        result = ingest_file(uf.filename or "unknown", content)
        results.append(result)

    return UploadResponse(ingested=len(results), results=results)


@router.post("/upload/async", response_model=JobSubmitResponse)
async def upload_async(files: list[UploadFile] = File(...)) -> JobSubmitResponse:
    """
    Schedule ingestion across a worker process pool and return a ``job_id``
    immediately. Poll :http:get:`/jobs/{job_id}` for live per-file progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    payload: list[tuple[str, bytes]] = []
    for uf in files:
        content = await uf.read()
        payload.append((uf.filename or "unknown", content))

    job_id = job_manager.submit_job(payload)
    return JobSubmitResponse(job_id=job_id, total=len(payload))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JobStatus(**job)
