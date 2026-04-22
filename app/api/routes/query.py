from fastapi import APIRouter

from app.models.schemas import (
    AnalyticalQueryResponse,
    QueryRequest,
    StructuredQueryResponse,
)
from app.services import rag, sql_agent

router = APIRouter(tags=["query"])


@router.post("/query/structured", response_model=StructuredQueryResponse)
def query_structured(body: QueryRequest) -> StructuredQueryResponse:
    """Natural-language → SQL → result rows."""
    return sql_agent.structured_query(body.question)


@router.post("/query/analytical", response_model=AnalyticalQueryResponse)
def query_analytical(body: QueryRequest) -> AnalyticalQueryResponse:
    """Retrieve relevant chunks + component data, then reason with the LLM."""
    return rag.analytical_query(body.question)
