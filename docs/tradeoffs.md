# Tradeoffs

Not implemented in this baseline:
- multi-tenant auth + document ACL enforcement
- distributed ingestion/index build
- OpenTelemetry tracing + APM
- learned reranking/fusion beyond basic CrossEncoder hook

## Hybrid retrieval tradeoffs (BM25 + dense)

What we implemented:
- **True sparse retrieval over the full corpus** using a persistent BM25 index (`rank-bm25`).
- **Score fusion** with a tunable weight: `final = dense_weight*dense + (1-dense_weight)*bm25`.
- Candidate sets are retrieved independently (dense_k + bm25_k) and fused by union.

Why this approach:
- It’s the simplest “industry-real” hybrid baseline that runs locally, is config-driven, and is easy to debug.
- The union is where recall gains come from: BM25 can surface exact-term evidence that dense misses.

Known tradeoffs / what we did NOT do:
- **BM25 runtime is O(N) per query** with `rank-bm25`. This is fine for local corpora, but not for 10^6+ chunks.
  At scale, replace BM25 with Lucene/Elasticsearch/OpenSearch (true inverted index) or a hosted sparse retriever.
- **Score normalization** uses max-score scaling for BM25 (cheap, monotonic). For certain distributions, more robust
  normalization (e.g., rank-based fusion like RRF) can be more stable.
- **Fusion method** is linear weighted sum. In production, you often use:
  - Reciprocal Rank Fusion (RRF)
  - Learned-to-rank fusion / reranking
  - Domain-specific heuristics (e.g., boost titles, headers, recency)
- **Metadata filtering** is applied *after* scoring for BM25 (we still score the full corpus to meet requirements).
  For large corpora, filtering should happen during retrieval to avoid scoring irrelevant docs.

Rationale: these depend on deployment constraints (scale, infra, compliance) and would make the repo non-runnable by default.
