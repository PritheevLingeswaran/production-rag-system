from __future__ import annotations

from retrieval.rerank import LexicalReranker
from utils.settings import RerankConfig


def test_lexical_reranker_promotes_passages_with_better_query_coverage() -> None:
    reranker = LexicalReranker(
        RerankConfig(
            provider="lexical",
            query_weight=0.8,
            retrieval_weight=0.2,
            min_query_term_coverage=0.2,
        )
    )

    ranked = reranker.rerank(
        "docker compose healthchecks",
        [
            "This passage discusses docker compose healthchecks and readiness probes.",
            "This passage only mentions docker.",
            "This passage is unrelated to deployment.",
        ],
        base_scores=[0.3, 0.9, 0.2],
        top_k=3,
    )

    assert ranked[0].idx == 0
    assert ranked[0].score > ranked[1].score
