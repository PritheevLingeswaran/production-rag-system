from __future__ import annotations

from evaluation.resume_metrics import (
    answer_matches,
    build_resume_metrics,
    hallucination_summary,
    has_valid_citation,
    normalize_answer,
)


def test_normalize_answer_strips_citations_and_source_suffix() -> None:
    text = "B.Tech - Computer Science Engineering. [resume:p1:c6] (source: resume:p1:c6)"
    assert normalize_answer(text) == "b tech computer science engineering"


def test_answer_matches_uses_normalized_exact_match() -> None:
    assert answer_matches(
        "2024-2028 (Expected) [Pritheev_Resume.pdf:p1:c6]",
        ["2024 2028 expected"],
    )


def test_has_valid_citation_requires_returned_source_ids() -> None:
    assert has_valid_citation("Yes [a] [b]", ["a", "b", "c"]) is True
    assert has_valid_citation("Yes [a] [missing]", ["a", "b", "c"]) is False
    assert has_valid_citation("Yes", ["a"]) is False


def test_hallucination_summary_aggregates_rates() -> None:
    rows = [
        {
            "hallucinated": True,
            "valid_citation": False,
            "unsupported_claim": True,
            "refusal_correct": False,
            "false_refusal": False,
            "refusal": False,
            "expected_behavior": "answer",
            "cost_usd": 0.0,
            "embedding_tokens": 0,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
        },
        {
            "hallucinated": False,
            "valid_citation": True,
            "unsupported_claim": False,
            "refusal_correct": True,
            "false_refusal": False,
            "refusal": True,
            "expected_behavior": "refuse",
            "cost_usd": 0.0,
            "embedding_tokens": 0,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
        },
    ]
    summary = hallucination_summary(rows)
    assert summary["hallucination_rate"] == 0.5
    assert summary["refusal_correctness_rate"] == 1.0
    assert summary["citation_grounding_failure_rate"] == 1.0


def test_build_resume_metrics_only_includes_measured_values() -> None:
    metrics = build_resume_metrics(
        dataset_stats={
            "measured": True,
            "documents_indexed": 2,
            "chunks_indexed": 9,
            "index_sizes_bytes": {"vector_index": 10},
        },
        latency_dense={"measured": False},
        latency_hybrid={
            "measured": True,
            "summary": {
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 2.0,
                "p99_latency_ms": 3.0,
            },
        },
        retrieval={
            "measured": True,
            "summary_by_k": {
                "5": {
                    "precision_dense": 0.1,
                    "precision_hybrid": 0.2,
                    "recall_dense": 0.3,
                    "recall_hybrid": 0.4,
                    "hit_rate_dense": 0.5,
                    "hit_rate_hybrid": 0.6,
                }
            },
            "mrr_dense": 0.1,
            "mrr_hybrid": 0.2,
        },
        hallucination={
            "measured": True,
            "baseline_dense": {
                "hallucination_rate": 0.5,
                "citation_grounding_failure_rate": 0.2,
            },
            "strict_grounded_hybrid": {
                "hallucination_rate": 0.1,
                "citation_grounding_failure_rate": 0.0,
                "avg_cost_per_query_usd": 0.0,
                "avg_embedding_tokens_per_query": 0.0,
                "avg_llm_tokens_in_per_query": 0.0,
                "avg_llm_tokens_out_per_query": 0.0,
            },
        },
        load_test={"measured": False},
    )
    assert metrics["documents_indexed"] == 2
    assert metrics["precision_at_5_hybrid"] == 0.2
    assert "max_tested_concurrency_successful" not in metrics
