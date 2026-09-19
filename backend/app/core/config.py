"""
Centralized configuration for the RAG backend.

All environment-dependent values are read here, once, using pydantic-settings.
Never hard-code model names, paths, or ports elsewhere in the codebase --
import `settings` from this module instead.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- Vector store ---
    VECTOR_DB_PATH: str = "data/vector_store"
    COLLECTION_NAME: str = "shopease_faq"
    TOP_K: int = 4

    # --- Retrieval confidence ---
    # If the closest retrieved chunk's distance is above this, the system
    # refuses immediately instead of calling the LLM. Calibrated empirically:
    # in-scope questions measured <= 0.86, out-of-scope questions measured
    # >= 1.15 on this project's embedding model, so 1.0 sits safely between.
    DISTANCE_THRESHOLD: float = 1.0

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:8501"  # comma-separated list

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is only parsed once per process."""
    return Settings()


settings = get_settings()