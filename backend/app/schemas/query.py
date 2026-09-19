"""
Request/response models for the /query endpoint.
"""
from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    """One previous question/answer pair, used for simple conversation memory."""
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The customer's question.")
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Optional override for number of chunks to retrieve."
    )
    history: list[ChatTurn] | None = Field(
        default=None,
        description="Optional recent conversation turns, oldest first, for simple follow-up context.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value.strip()


class SourceChunk(BaseModel):
    source: str
    category: str | None = None
    question: str | None = None
    chunk_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    suggested_questions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    vector_db: str
    ollama: str