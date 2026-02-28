from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float = Field(ge=-1.0, le=1.0)
    text: str


class Refusal(BaseModel):
    is_refusal: bool
    reason: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[SourceChunk]
    refusal: Refusal
    metrics: Dict[str, Any] = Field(default_factory=dict)
