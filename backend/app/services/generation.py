"""
Generation service.

Single responsibility: take a question + retrieved context (+ a little bit of
recent conversation history), build a grounding prompt, call the local Ollama
LLM, and return the generated answer. Also offers a best-effort helper that
asks the LLM for a few natural follow-up questions after a real answer.

think=False is passed on every call: qwen3.5 (and similar "reasoning" models)
otherwise spend part of their response generating an internal, invisible
chain-of-thought before writing the visible answer. Disabling this via
Ollama's official "think" setting skips that step entirely, producing the
same final answer dramatically faster and without any risk of the model
running out of room mid-thought (the empty-response bug from earlier).
"""
from __future__ import annotations

import re

import ollama

from app.core.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

NO_ANSWER_TEXT = "I could not find this information in the available FAQ."
GENERATION_TROUBLE_TEXT = "Sorry, I had trouble generating an answer just now. Please try asking again."

MAX_HISTORY_TURNS = 3  # keep the prompt short on local hardware

PROMPT_TEMPLATE = """You are the ShopEase Customer Support Assistant.

Answer the user's question using ONLY the information provided in the
retrieved context below. You may use the recent conversation history only
to understand what the user is referring to (e.g. "it" or "that") -- never
as a source of factual information itself.

Rules:
1. Do not invent information.
2. Do not use external knowledge.
3. If the answer is not present in the context, respond exactly with:
   "{no_answer}"
4. Keep the answer clear and concise -- a few sentences at most.
5. Mention the relevant source when appropriate.
{history_block}
Retrieved Context:
{context}

User Question:
{question}

Answer:"""

SUGGESTION_PROMPT_TEMPLATE = """Based on this customer support exchange, suggest exactly 3 short, \
natural follow-up questions the customer might ask next. Only suggest questions that a ShopEase FAQ \
about accounts, orders, shipping, or returns/refunds could realistically answer. Do not repeat the \
original question.

Respond with ONLY a numbered list (1., 2., 3.), one short question per line, and nothing else -- no \
intro, no explanation.

Original question: {question}
Answer given: {answer}

Follow-up questions:"""


class GenerationError(Exception):
    """Raised when the LLM call fails or Ollama is unreachable."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    if not chunks:
        return "(no relevant context was retrieved)"

    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        question = meta.get("question", "")
        parts.append(f"[Source: {source} | FAQ: {question}]\n{chunk['text']}")
    return "\n\n".join(parts)


def build_history_block(history: list[dict] | None) -> str:
    """Format the last few conversation turns, or return an empty string if none."""
    if not history:
        return ""

    recent = history[-MAX_HISTORY_TURNS:]
    lines = ["Recent conversation (for context only, not a source of facts):"]
    for turn in recent:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines) + "\n"


def build_prompt(question: str, context: str, history: list[dict] | None = None) -> str:
    history_block = build_history_block(history)
    return PROMPT_TEMPLATE.format(
        no_answer=NO_ANSWER_TEXT, context=context, question=question, history_block=history_block
    )


def _call_ollama(prompt: str) -> str:
    client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    response = client.chat(
        model=settings.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
    )
    return response.get("message", {}).get("content", "").strip()


def generate_answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """Call Ollama with the grounded prompt and return the raw answer text."""
    context = build_context(chunks)
    prompt = build_prompt(question, context, history)

    try:
        answer = _call_ollama(prompt)
    except Exception as exc:
        logger.error("Ollama generation failed: %s", exc)
        raise GenerationError(f"Failed to generate an answer from Ollama: {exc}") from exc

    if not answer:
        logger.warning("Ollama returned an empty response.")
        return GENERATION_TROUBLE_TEXT
    return answer


def generate_suggested_questions(question: str, answer: str) -> list[str]:
    """Ask the LLM for a few natural follow-up questions.

    Best-effort only: returns an empty list on any failure rather than
    breaking the main answer the user actually asked for.
    """
    prompt = SUGGESTION_PROMPT_TEMPLATE.format(question=question, answer=answer)

    try:
        raw = _call_ollama(prompt)
    except Exception as exc:
        logger.warning("Suggested-question generation failed (non-fatal): %s", exc)
        return []

    suggestions = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\d]+[\.\)]\s*|^[-*]\s*", "", line).strip()
        if cleaned:
            suggestions.append(cleaned)

    return suggestions[:3]