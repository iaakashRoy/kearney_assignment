from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncIterator, Callable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnalyticalQueryResponse,
    QueryRequest,
    StructuredQueryResponse,
)
from app.services import rag, sql_agent
from app.services.agents.graph import run_analytical_query
from app.services.agents.trace import use_emitter

router = APIRouter(tags=["query"])


@router.post("/query/structured", response_model=StructuredQueryResponse)
def query_structured(body: QueryRequest) -> StructuredQueryResponse:
    """Natural-language → SQL → result rows."""
    return sql_agent.structured_query(body.question)


@router.post("/query/analytical", response_model=AnalyticalQueryResponse)
def query_analytical(body: QueryRequest) -> AnalyticalQueryResponse:
    """Retrieve relevant chunks + component data, then reason with the LLM."""
    return rag.analytical_query(body.question)


# ─── SSE plumbing shared by both /stream endpoints ───
_SENTINEL = object()


def _sse_stream_response(runner: Callable[[], None]) -> StreamingResponse:
    """Run *runner* in a worker thread and stream emitted trace events as SSE."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit_threadsafe(event: str, payload: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, payload))

    def worker() -> None:
        try:
            with use_emitter(emit_threadsafe):
                runner()
        except Exception as exc:  # noqa: BLE001 — surface to client
            emit_threadsafe("run.error", {"error": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream() -> AsyncIterator[str]:
        yield ": stream open\n\n"
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                yield "event: end\ndata: {}\n\n"
                return
            event, payload = item
            data = json.dumps(payload, default=str, ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/query/analytical/stream")
async def query_analytical_stream(body: QueryRequest) -> StreamingResponse:
    """SSE variant of ``/query/analytical`` — streams agent-trace events."""
    return _sse_stream_response(lambda: run_analytical_query(body.question))


@router.post("/query/structured/stream")
async def query_structured_stream(body: QueryRequest) -> StreamingResponse:
    """SSE variant of ``/query/structured`` — streams SQL agent events
    (proposed SQL, validation, execution, optional self-correction retry)."""
    return _sse_stream_response(lambda: sql_agent.structured_query(body.question))
