"""
HTTP route handlers: POST /query and GET /health.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.generation import GenerationError
from app.services.rag_pipeline import run_rag_pipeline
from app.services.retrieval import RetrievalError, retrieval_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    history = [turn.model_dump() for turn in request.history] if request.history else None

    try:
        result = run_rag_pipeline(request.question, top_k=request.top_k, history=history)
    except RetrievalError as exc:
        logger.error("Retrieval error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc
    except GenerationError as exc:
        logger.error("Generation error: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Unexpected error handling /query")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        suggested_questions=result.get("suggested_questions", []),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vector_db_status = "ready" if retrieval_service.is_ready else "not_loaded"
    return HealthResponse(status="healthy", vector_db=vector_db_status, ollama="not_checked")