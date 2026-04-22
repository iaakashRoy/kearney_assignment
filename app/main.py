"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import catalog, health, ingestion, query
from app.core.logging import configure_logging, get_logger
from app.db.sqlite import init_db
from app.services.job_manager import shutdown_pool

configure_logging()
logger = get_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    logger.info("Starting up — initialising database")
    init_db()
    yield
    logger.info("Shutting down — stopping ingestion worker pool")
    shutdown_pool()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Enterprise Document Intelligence Platform",
        description=(
            "Ingest mixed documents and component photos, "
            "answer structured SQL queries and analytical RAG questions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router)
    application.include_router(ingestion.router)
    application.include_router(query.router)
    application.include_router(catalog.router)

    return application


app = create_app()
