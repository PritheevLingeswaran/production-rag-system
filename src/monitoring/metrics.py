from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency for /query requests",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
REQUEST_COST_USD = Counter("rag_request_cost_usd_total", "Total USD cost across requests")
REQUEST_TOKENS = Counter("rag_request_tokens_total", "Total tokens used across requests", ["kind"])
REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests by path and status code",
    ["path", "status_code"],
)
REQUEST_ERRORS = Counter(
    "rag_request_errors_total",
    "Total query pipeline errors by stage",
    ["stage"],
)
REQUEST_REFUSALS = Counter(
    "rag_refusals_total",
    "Total query refusals",
    ["reason"],
)
REQUEST_GROUNDED = Counter(
    "rag_grounded_answers_total",
    "Total grounded answers and non-grounded answers",
    ["grounded"],
)
RETRIEVAL_TOP_SCORE = Histogram(
    "rag_retrieval_top_score",
    "Top retrieval score observed per query",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RETRIEVAL_TOP_GAP = Histogram(
    "rag_retrieval_top_gap",
    "Score gap between top-1 and top-2 retrieval hits",
    buckets=(0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.0),
)
