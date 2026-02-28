from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    total_tokens: int = 0
    cost_usd: float = 0.0


class EmbeddingsBackend(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbedResult:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> EmbedResult:
        raise NotImplementedError
