"""
Thin wrapper around the backend REST API.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 300  # small safety margin above the two sequential LLM calls


class ApiClientError(Exception):
    """Raised when the backend request fails or returns an error."""


def ask_question(question: str, top_k: int | None = None, history: list[dict] | None = None) -> dict:
    payload = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if history:
        payload["history"] = history

    try:
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise ApiClientError(f"Could not reach the backend at {API_BASE_URL}: {exc}") from exc

    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise ApiClientError(f"Backend returned {response.status_code}: {detail}")

    return response.json()


def check_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise ApiClientError(f"Backend health check failed: {exc}") from exc