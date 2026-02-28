from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency for /query requests",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
REQUEST_COST_USD = Counter("rag_request_cost_usd_total", "Total USD cost across requests")
REQUEST_TOKENS = Counter("rag_request_tokens_total", "Total tokens used across requests", ["kind"])
