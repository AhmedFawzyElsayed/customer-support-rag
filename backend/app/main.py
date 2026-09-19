"""
FastAPI application entrypoint.

Loads the vector store and embedding model exactly once at startup
(via lifespan), never on a per-request basis.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.utils.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading embedding model and vector store...")
    try:
        retrieval_service.load()
    except Exception as exc:
        # We log but do not crash the app -- /health will report the store
        # as not ready, and /query will return a clear 503 until it's fixed.
        logger.error("Startup retrieval load failed: %s", exc)
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ShopEase Customer Support RAG API",
    description="Retrieval-Augmented Generation API answering ShopEase FAQ questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
