"""
Structured logging setup for the RAG backend.

Logs request lifecycle events (received, retrieval done, generation done,
errors, latency) without ever logging raw customer-identifying content.
"""
import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported under uvicorn --reload)
        return

    root.setLevel(settings.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
