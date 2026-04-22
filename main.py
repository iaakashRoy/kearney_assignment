# Entry point — re-exports the FastAPI app for uvicorn: `uvicorn main:app`
from app.main import app  # noqa: F401
