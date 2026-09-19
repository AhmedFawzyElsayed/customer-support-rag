"""
RAG orchestration.

Coordinates: question -> retrieval -> context -> prompt -> Ollama ->
grounded answer + sources. Greetings and low-confidence retrievals skip
the main LLM call entirely; real answers get a best-effort set of
follow-up suggestions.

Retrieval is attempted twice when needed: first using the question exactly
as typed, and only if that doesn't find a good match, retried once more
blended with the previous question -- this rescues short follow-ups like
"does that include weekends?" without breaking plain, unrelated new
questions asked right after a different topic.
"""
from __future__ import annotations

from app.core.config import settings
from app.services.generation import NO_ANSWER_TEXT, generate_answer, generate_suggested_questions
from app.services.retrieval import retrieval_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

GREETING_PATTERNS = {
    "hi", "hello", "hey", "hiya", "yo", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "what's up", "whats up", "sup",
}

GREETING_RESPONSE = (
    "Hello! \U0001F44B I'm the ShopEase Customer Support Assistant. "
    "Ask me anything about your account, orders, shipping, or returns and refunds."
)


def _is_greeting(text: str) -> bool:
    normalized = text.strip().lower().strip("!?. ")
    return normalized in GREETING_PATTERNS


def _build_search_query(question: str, history: list[dict] | None) -> str:
    """Blend in the previous question so short follow-ups keep their topic."""
    if not history:
        return question
    last_question = history[-1].get("question", "").strip()
    if not last_question:
        return question
    return f"{last_question} {question}"


def _best_distance(chunks: list[dict]) -> float | None:
    distances = [c["distance"] for c in chunks if c.get("distance") is not None]
    return min(distances) if distances else None


def run_rag_pipeline(
    question: str, top_k: int | None = None, history: list[dict] | None = None
) -> dict:
    """Run the full RAG pipeline for a single question.

    Returns:
        dict with keys "answer" (str), "sources" (list[str]), and
        "suggested_questions" (list[str]).
    """
    logger.info("Received question (len=%d chars)", len(question))

    if _is_greeting(question):
        logger.info("Detected greeting; skipping retrieval/generation.")
        return {"answer": GREETING_RESPONSE, "sources": [], "suggested_questions": []}

    # Step 1: try retrieval using the question exactly as the user typed it.
    chunks = retrieval_service.retrieve(question, top_k=top_k)
    best_distance = _best_distance(chunks)
    logger.info("Direct retrieval best distance: %s", best_distance)

    # Step 2: only if that alone didn't find a good match, retry once more
    # blended with the previous question -- this rescues short follow-ups
    # without contaminating plain, unrelated new questions.
    if (best_distance is None or best_distance > settings.DISTANCE_THRESHOLD) and history:
        fallback_query = _build_search_query(question, history)
        fallback_chunks = retrieval_service.retrieve(fallback_query, top_k=top_k)
        fallback_distance = _best_distance(fallback_chunks)
        logger.info("History-assisted retry distance: %s", fallback_distance)
        if fallback_distance is not None and fallback_distance <= settings.DISTANCE_THRESHOLD:
            chunks = fallback_chunks
            best_distance = fallback_distance

    logger.info("Final best distance: %s (threshold: %s)", best_distance, settings.DISTANCE_THRESHOLD)

    if best_distance is None or best_distance > settings.DISTANCE_THRESHOLD:
        logger.info("No sufficiently relevant chunk found; refusing without calling the LLM.")
        return {"answer": NO_ANSWER_TEXT, "sources": [], "suggested_questions": []}

    answer = generate_answer(question, chunks, history=history)
    logger.info("Generation complete (answer length=%d chars)", len(answer))

    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        faq_question = meta.get("question", "")
        label = f"{source} - {faq_question}" if faq_question else source
        if label not in sources:
            sources.append(label)

    suggested_questions = generate_suggested_questions(question, answer)

    return {"answer": answer, "sources": sources, "suggested_questions": suggested_questions}