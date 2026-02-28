from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostBreakdown:
    embedding_usd: float = 0.0
    llm_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return float(self.embedding_usd + self.llm_usd)
