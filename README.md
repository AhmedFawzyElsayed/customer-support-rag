# ShopEase AI Customer Support RAG Assistant

A production-minded **Retrieval-Augmented Generation (RAG)** application that answers e-commerce
customer-support questions by retrieving relevant FAQ content and generating a grounded answer with a
local Ollama LLM — never from the model's own general knowledge.

Built for the Level 2 Summer Training graduation project (Core Track — text-only RAG).

---

## 1. Overview

**Domain:** Customer support FAQs for a fictional e-commerce company, **ShopEase**.

**Problem statement:** Customers ask repetitive questions about accounts, orders, shipping, and
returns/refunds. This assistant retrieves the exact FAQ entry relevant to a question and asks a local LLM
to answer strictly from that retrieved content, citing its source — so it never hallucinates policy details
that aren't in the knowledge base.

---

## 2. Architecture

```text
                         ┌─────────────────┐
                         │     User        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Streamlit     │
                         │    Frontend     │
                         └────────┬────────┘
                                  │ HTTP POST /query
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI      │
                         │    Backend      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ RAG Orchestrator│
                         │ (rag_pipeline.py)│
                         └───────┬───┬─────┘
                                 │   │
                   ┌─────────────┘   └──────────────┐
                   ▼                                ▼
          ┌─────────────────┐              ┌─────────────────┐
          │ Sentence        │              │ Chroma Vector   │
          │ Transformer     │              │ Database        │
          │ (query embed)   │              │ (persisted)     │
          └─────────────────┘              └────────┬────────┘
                                                     │ top-k chunks
                                                     ▼
                                            ┌─────────────────┐
                                            │ Prompt Builder  │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ Ollama Local LLM│
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                          Grounded Answer + Sources
```

### RAG workflow

```text
OFFLINE (notebook)                       ONLINE (FastAPI, per request)
===================                      ===============================
FAQ Documents                            User Question
   ↓                                        ↓
Load                                     Embed query
   ↓                                        ↓
Clean                                    Retrieve top-k chunks from Chroma
   ↓                                        ↓
Chunk (Q+A aware)                        Build grounded prompt
   ↓                                        ↓
Embed                                    Call Ollama
   ↓                                        ↓
Store in Chroma                          Return answer + sources
   ↓
Persist to data/vector_store/
```

The vector store is built **once**, offline, in the notebook, and persisted to disk. The FastAPI backend
loads it once at startup (`lifespan`) and never rebuilds it per request.

---

## 3. Technology Stack

| Layer            | Technology                                  |
|------------------|----------------------------------------------|
| Embeddings       | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database  | `chromadb` (persistent client)               |
| LLM inference    | `ollama` (local, e.g. `llama3.2`)            |
| Backend          | `FastAPI`, `pydantic`, `pydantic-settings`   |
| Frontend         | `Streamlit`                                  |
| Testing          | `pytest`, `httpx` (FastAPI `TestClient`)     |
| Notebook         | `Jupyter`, `pandas`                          |

---

## 4. Project Structure

```text
customer-support-rag/
│
├── notebooks/
│   └── rag_pipeline.ipynb        # offline pipeline: load → clean → chunk → embed → store → evaluate
│
├── data/
│   ├── documents/                # 4 source FAQ markdown files
│   │   ├── account_faq.md
│   │   ├── orders_faq.md
│   │   ├── shipping_faq.md
│   │   └── returns_refunds_faq.md
│   └── vector_store/              # persisted Chroma DB (produced by the notebook)
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, CORS, lifespan startup
│   │   ├── api/routes/query.py    # POST /query, GET /health
│   │   ├── core/config.py         # pydantic-settings configuration
│   │   ├── schemas/query.py       # QueryRequest / QueryResponse
│   │   ├── services/
│   │   │   ├── retrieval.py       # load Chroma, embed query, retrieve top-k
│   │   │   ├── generation.py      # build grounded prompt, call Ollama
│   │   │   └── rag_pipeline.py    # orchestrates retrieval + generation
│   │   └── utils/logging_config.py
│   ├── tests/test_query.py        # 2+ TestClient tests (happy path + 422)
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/
│   ├── app.py                     # Streamlit chat UI
│   ├── api_client.py              # backend HTTP wrapper (reads API_BASE_URL)
│   ├── .env.example
│   └── requirements.txt
│
├── .gitignore
├── README.md
└── requirements.txt                # combined dev environment (notebook + backend + frontend)
```

---

## 5. Domain & Data

Four synthetic-but-realistic FAQ documents for a fictional company, **ShopEase**, covering:

- **Account** — signup, password reset, security, privacy, rewards
- **Orders** — placing/cancelling orders, payments, promo codes, order status
- **Shipping** — delivery times, costs, tracking, carriers, international shipping
- **Returns & Refunds** — return windows, refund timing, exchanges, damaged items

Each file is a plain Markdown document with one `## Question` header per FAQ entry, which the notebook's
chunking strategy is built around (see Section 2 of the notebook: each chunk keeps a question and its
answer together).

---

## 6. Setup & Installation

### Prerequisites

```bash
python --version   # 3.10+
ollama --version
git --version
```

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/customer-support-rag.git
cd customer-support-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull an Ollama model

```bash
ollama pull llama3.2
ollama serve   # if not already running
```

### 3. Run the notebook (builds the vector store)

```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

Run **Kernel → Restart & Run All**. This produces `data/vector_store/` containing the persisted Chroma
collection and a `pipeline_config.json` snapshot.

### 4. Run the backend

```bash
cd backend
cp .env.example .env        # edit if your Ollama model/paths differ
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` and try `POST /query` from Swagger UI.

### 5. Run the frontend

```bash
cd frontend
cp .env.example .env
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

### 6. Run tests

```bash
cd backend
pytest
```

---

## 7. Environment Variables

**Backend (`backend/.env`):**

| Variable          | Default                        | Description                                  |
|-------------------|---------------------------------|-----------------------------------------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434`       | Ollama server URL                             |
| `OLLAMA_MODEL`    | `llama3.2`                     | Local model name to use for generation        |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`             | Sentence-Transformers model for embeddings    |
| `VECTOR_DB_PATH`  | `data/vector_store`            | Path to the persisted Chroma store            |
| `COLLECTION_NAME` | `shopease_faq`                 | Chroma collection name                        |
| `TOP_K`           | `4`                             | Default number of chunks retrieved per query  |
| `API_HOST`        | `0.0.0.0`                       | Bind host for uvicorn                         |
| `API_PORT`        | `8000`                          | Bind port for uvicorn                         |
| `CORS_ORIGINS`    | `http://localhost:8501`        | Comma-separated allowed frontend origins      |
| `LOG_LEVEL`       | `INFO`                          | Logging verbosity                             |

**Frontend (`frontend/.env`):**

| Variable        | Default                   | Description                    |
|-----------------|---------------------------|---------------------------------|
| `API_BASE_URL`  | `http://localhost:8000`  | Backend base URL                |

---

## 8. API Reference

### `POST /query`

Request:

```json
{ "question": "How long does standard shipping take?" }
```

Response:

```json
{
  "answer": "Standard shipping usually takes 3 to 5 business days after processing.",
  "sources": ["shipping_faq.md - How long does standard shipping take?"]
}
```

`curl` example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I cancel my order after payment?"}'
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "healthy", "vector_db": "ready", "ollama": "not_checked" }
```

---

## 9. Evaluation Results

15 in-scope questions across all four FAQ categories, plus 2 intentionally out-of-scope questions, were run
through the retrieval pipeline in `notebooks/rag_pipeline.ipynb` (Section 6). The notebook records, for each
question: expected source, retrieved source, whether the answer was grounded, and correctness — saved to
`data/vector_store/evaluation_results.csv` after running the notebook.

Out-of-scope questions ("Who is the CEO of ShopEase?", "Can I pay using cryptocurrency?") are expected to
produce: `"I could not find this information in the available FAQ."` — verifying the assistant does not
hallucinate when the knowledge base has no relevant answer.

Observed failure cases and mitigations are documented in the notebook (Section 6): near-duplicate FAQ
phrasing across categories, short/ambiguous queries, and why grounding is enforced at the **prompt** level
rather than relying on retrieval distance alone.

---

## 10. Screenshots

*Add screenshots of the running Streamlit app and Swagger UI (`/docs`) here before submission.*

---

## 11. Known Limitations

- Answer quality depends on the local Ollama model chosen; smaller models may follow the grounding
  instructions less reliably.
- The knowledge base is synthetic; it does not reflect a real company's actual policies.
- Chroma's `query()` always returns `top_k` results even for out-of-scope questions — refusal to answer is
  enforced by the prompt, not by a relevance threshold.
- No authentication/rate-limiting is implemented; not intended for production traffic as-is.

## 12. Future Improvements

- Add a similarity-score threshold to skip generation entirely when no chunk is relevant enough.
- Add conversation memory for multi-turn follow-up questions.
- Add streaming responses from Ollama to the Streamlit UI.
- Add a `/feedback` endpoint to collect thumbs up/down on answers for future evaluation.

---

## 13. Reproducing From Scratch (stranger test)

```bash
git clone <repo-url>
cd customer-support-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
jupyter notebook notebooks/rag_pipeline.ipynb   # Kernel -> Restart & Run All
cd backend && cp .env.example .env && uvicorn app.main:app --reload &
cd ../frontend && cp .env.example .env && streamlit run app.py
```
