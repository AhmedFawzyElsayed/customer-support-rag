"""
Unit tests for the RAG orchestration logic (app/services/rag_pipeline.py).

These tests never touch a real embedding model, vector store, or Ollama --
they monkeypatch retrieval and generation so the pipeline's own
decision-making (greeting detection, distance-based refusal, source
de-duplication) can be verified quickly and deterministically.
"""
from app.services import rag_pipeline
from app.services.generation import NO_ANSWER_TEXT


def test_is_greeting_recognizes_common_greetings():
    assert rag_pipeline._is_greeting("hi")
    assert rag_pipeline._is_greeting("Hello!")
    assert rag_pipeline._is_greeting("  Good Morning ")


def test_is_greeting_rejects_real_questions():
    assert not rag_pipeline._is_greeting("How long does shipping take?")
    assert not rag_pipeline._is_greeting("hi, how long does shipping take?")


def test_build_search_query_blends_previous_question():
    history = [{"question": "How long does a refund take?", "answer": "5-7 days"}]
    result = rag_pipeline._build_search_query("does that include weekends?", history)
    assert "How long does a refund take?" in result
    assert "does that include weekends?" in result


def test_build_search_query_returns_question_unchanged_without_history():
    result = rag_pipeline._build_search_query("How long does shipping take?", None)
    assert result == "How long does shipping take?"


def test_best_distance_picks_the_minimum():
    chunks = [{"distance": 0.8}, {"distance": 0.3}, {"distance": 0.5}]
    assert rag_pipeline._best_distance(chunks) == 0.3


def test_best_distance_handles_empty_list():
    assert rag_pipeline._best_distance([]) is None


def test_greeting_short_circuits_before_retrieval(monkeypatch):
    """A greeting should never trigger retrieval or generation at all."""
    called = {"retrieve": False, "generate": False}

    def fake_retrieve(*args, **kwargs):
        called["retrieve"] = True
        return []

    def fake_generate(*args, **kwargs):
        called["generate"] = True
        return "should not happen"

    monkeypatch.setattr(rag_pipeline.retrieval_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate)

    result = rag_pipeline.run_rag_pipeline("hello")

    assert result["answer"].startswith("Hello!")
    assert result["sources"] == []
    assert called["retrieve"] is False
    assert called["generate"] is False


def test_low_confidence_retrieval_refuses_without_calling_llm(monkeypatch):
    """If nothing relevant is found, the pipeline must refuse without ever
    calling the LLM -- this is the core grounding guarantee of the system."""

    def fake_retrieve(query, top_k=None):
        return [{"text": "irrelevant", "metadata": {"source": "x.md"}, "distance": 5.0}]

    def fake_generate(*args, **kwargs):
        raise AssertionError("generate_answer should not be called for a low-confidence match")

    monkeypatch.setattr(rag_pipeline.retrieval_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate)

    result = rag_pipeline.run_rag_pipeline("some unrelated nonsense question")

    assert result["answer"] == NO_ANSWER_TEXT
    assert result["sources"] == []


def test_good_match_calls_llm_and_returns_deduplicated_sources(monkeypatch):
    chunks = [
        {"text": "chunk1", "metadata": {"source": "a.md", "question": "Q1"}, "distance": 0.2},
        {"text": "chunk2", "metadata": {"source": "a.md", "question": "Q1"}, "distance": 0.3},  # duplicate label
        {"text": "chunk3", "metadata": {"source": "b.md", "question": "Q2"}, "distance": 0.4},
    ]

    def fake_retrieve(query, top_k=None):
        return chunks

    def fake_generate(question, retrieved_chunks, history=None):
        return "A real grounded answer."

    def fake_suggestions(question, answer):
        return ["Follow-up 1?", "Follow-up 2?"]

    monkeypatch.setattr(rag_pipeline.retrieval_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate)
    monkeypatch.setattr(rag_pipeline, "generate_suggested_questions", fake_suggestions)

    result = rag_pipeline.run_rag_pipeline("How long does a refund take?")

    assert result["answer"] == "A real grounded answer."
    assert result["sources"] == ["a.md - Q1", "b.md - Q2"]  # de-duplicated
    assert result["suggested_questions"] == ["Follow-up 1?", "Follow-up 2?"]