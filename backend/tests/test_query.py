"""
API tests for /query and /health.

Test 1: a valid question returns 200.
Test 2: an invalid (empty) question returns 422.

Retrieval and generation are monkeypatched so these tests do not require
a running Ollama instance or a pre-built vector store -- they verify the
API contract, not the model outputs.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import rag_pipeline


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    def fake_run_rag_pipeline(question: str, top_k: int | None = None, history: list | None = None) -> dict:
        return {
            "answer": "Standard shipping usually takes 3 to 5 business days after processing.",
            "sources": ["shipping_faq.md - How long does standard shipping take?"],
            "suggested_questions": ["How long does express shipping take?"],
        }

    monkeypatch.setattr(rag_pipeline, "run_rag_pipeline", fake_run_rag_pipeline)
    # The route module imported the function by reference, so patch there too.
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "run_rag_pipeline", fake_run_rag_pipeline)
    yield


client = TestClient(app)


def test_query_valid_question_returns_200():
    response = client.post("/query", json={"question": "How long does standard shipping take?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_empty_question_returns_422():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"