from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class DocumentFilter(BaseModel):
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
