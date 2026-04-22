from fastapi import APIRouter

router = APIRouter(tags=["ops"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
