from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional

from schemas.common import DocumentFilter


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=50)
    filter: Optional[DocumentFilter] = None
    rewrite_query: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str = "ok"
