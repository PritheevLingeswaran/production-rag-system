from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.common import DocumentFilter


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        examples=["How many projects are there in the resume?"],
    )
    top_k: int = Field(default=12, ge=1, le=50)
    filter: DocumentFilter | None = None
    rewrite_query: bool | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "rag-smart-qa-api"
    environment: str = "dev"
    checks: dict[str, str] = Field(default_factory=dict)
