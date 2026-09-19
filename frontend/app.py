"""
ShopEase AI Support Assistant - Streamlit frontend.

A chat-style UI that sends the user's question (plus a little recent
conversation history) to the FastAPI backend's /query endpoint and
displays the grounded answer, its sources, and a few clickable
follow-up question suggestions for the most recent turn.

The main question box is wrapped in a form so pressing Enter submits it,
in addition to the "Ask Question" button.
"""
import streamlit as st

from api_client import API_BASE_URL, ApiClientError, ask_question, check_health

st.set_page_config(page_title="ShopEase AI Support Assistant", page_icon="🛒", layout="centered")

st.title("🛒 ShopEase AI Support Assistant")
st.caption(f"Connected to backend: {API_BASE_URL}")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, sources, suggestions), newest first

MAX_HISTORY_TURNS_SENT = 3


def handle_ask(question_text: str) -> None:
    """Send a question to the backend (with recent history) and store the result."""
    prior_turns = list(reversed(st.session_state.history))[-MAX_HISTORY_TURNS_SENT:]
    history_payload = [{"question": q, "answer": a} for q, a, _, _ in prior_turns]

    with st.spinner("Retrieving relevant FAQs and generating an answer..."):
        try:
            result = ask_question(question_text.strip(), history=history_payload)
            st.session_state.history.insert(
                0,
                (
                    question_text.strip(),
                    result.get("answer", ""),
                    result.get("sources", []),
                    result.get("suggested_questions", []),
                ),
            )
        except ApiClientError as exc:
            st.error(f"⚠️ {exc}")


with st.sidebar:
    st.header("Backend status")
    if st.button("Check backend health"):
        try:
            health = check_health()
            st.success(f"API status: {health.get('status')}")
            st.write(f"Vector DB: {health.get('vector_db')}")
        except ApiClientError as exc:
            st.error(str(exc))

    st.divider()
    if st.button("🔄 New conversation"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.markdown(
        "Ask about **orders, shipping, returns/refunds, or your account**. "
        "Answers are grounded only in the ShopEase FAQ knowledge base. "
        "Follow-up questions like 'what about that?' are supported."
    )

with st.form(key="ask_form", clear_on_submit=True):
    question_input = st.text_input(
        "Ask your question:",
        placeholder="e.g. How long does standard shipping take?",
    )
    submitted = st.form_submit_button("Ask Question", type="primary")

if submitted:
    if not question_input or not question_input.strip():
        st.warning("Please enter a question before asking.")
    else:
        handle_ask(question_input)

if st.session_state.history:
    st.divider()
    for idx, (q, answer, sources, suggestions) in enumerate(st.session_state.history):
        st.markdown(f"**You asked:** {q}")
        st.markdown("**Answer**")
        st.write(answer)
        if sources:
            st.markdown("**Sources**")
            for src in sources:
                st.markdown(f"- {src}")

        if idx == 0 and suggestions:
            st.markdown("**You might also ask:**")
            cols = st.columns(len(suggestions))
            for col, suggestion in zip(cols, suggestions):
                if col.button(suggestion, key=f"suggestion_{idx}_{suggestion}"):
                    handle_ask(suggestion)
                    st.rerun()

        st.markdown("---")