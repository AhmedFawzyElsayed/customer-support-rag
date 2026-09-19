"""
Unit tests for prompt-building and response-parsing logic in
app/services/generation.py. No real Ollama calls are made.
"""
from app.services import generation


def test_build_context_formats_chunks_with_source_and_question():
    chunks = [
        {
            "text": "Standard shipping takes 3-5 days.",
            "metadata": {"source": "shipping_faq.md", "question": "How long does shipping take?"},
        },
    ]
    context = generation.build_context(chunks)
    assert "shipping_faq.md" in context
    assert "How long does shipping take?" in context
    assert "3-5 days" in context


def test_build_context_handles_no_chunks():
    assert generation.build_context([]) == "(no relevant context was retrieved)"


def test_build_history_block_empty_when_no_history():
    assert generation.build_history_block(None) == ""
    assert generation.build_history_block([]) == ""


def test_build_history_block_includes_recent_turns():
    history = [
        {"question": "How long does a refund take?", "answer": "5-7 business days."},
    ]
    block = generation.build_history_block(history)
    assert "How long does a refund take?" in block
    assert "5-7 business days." in block


def test_generate_suggested_questions_parses_numbered_list(monkeypatch):
    fake_response = (
        "1. Does it include weekends?\n"
        "2. What about international refunds?\n"
        "3. Can I get store credit instead?"
    )
    monkeypatch.setattr(generation, "_call_ollama", lambda prompt: fake_response)

    suggestions = generation.generate_suggested_questions(
        "How long does a refund take?", "5-7 business days."
    )

    assert suggestions == [
        "Does it include weekends?",
        "What about international refunds?",
        "Can I get store credit instead?",
    ]


def test_generate_suggested_questions_returns_empty_list_on_failure(monkeypatch):
    def raise_error(prompt):
        raise RuntimeError("Ollama unreachable")

    monkeypatch.setattr(generation, "_call_ollama", raise_error)

    assert generation.generate_suggested_questions("Q", "A") == []


def test_generate_answer_returns_trouble_message_when_empty(monkeypatch):
    monkeypatch.setattr(generation, "_call_ollama", lambda prompt: "")

    answer = generation.generate_answer("Q", [])

    assert answer == generation.GENERATION_TROUBLE_TEXT