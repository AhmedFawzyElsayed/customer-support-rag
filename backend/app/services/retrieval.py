"""
Retrieval service.

Single responsibility: load the persisted Chroma vector store, embed the
incoming query, and return the top-k most relevant FAQ chunks with metadata.

This module deliberately knows nothing about prompting or the LLM -- that
lives in generation.py / rag_pipeline.py.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval cannot be performed (bad input or store unavailable)."""


class RetrievalService:
    """Wraps a persisted Chroma collection for FAQ chunk retrieval."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedder: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the embedding model and the persisted Chroma collection once."""
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        logger.info("Loading Chroma vector store from: %s", settings.VECTOR_DB_PATH)
        self._client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)

        try:
            self._collection = self._client.get_collection(name=settings.COLLECTION_NAME)
        except Exception as exc:  # collection missing / store not built yet
            logger.error("Could not load collection '%s': %s", settings.COLLECTION_NAME, exc)
            raise RetrievalError(
                f"Vector store collection '{settings.COLLECTION_NAME}' not found at "
                f"'{settings.VECTOR_DB_PATH}'. Run the notebook to build and persist it first."
            ) from exc

        logger.info("Retrieval service ready. Collection count: %s", self._collection.count())

    @property
    def is_ready(self) -> bool:
        return self._collection is not None and self._embedder is not None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top-k chunks (text + metadata + distance) for a query."""
        if not query or not query.strip():
            raise RetrievalError("Query must not be empty.")

        k = top_k or settings.TOP_K
        if k < 1:
            raise RetrievalError("top_k must be a positive integer.")

        if not self.is_ready:
            raise RetrievalError("Retrieval service is not initialized. Call load() at startup.")

        query_embedding = self._embedder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(documents)

        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                }
            )
        return chunks


# Module-level singleton, initialized during FastAPI lifespan startup.
retrieval_service = RetrievalService()
