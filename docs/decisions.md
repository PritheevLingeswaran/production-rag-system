# Decisions

## ADR-001: Keep orchestration explicit

Decision:
- The repository uses direct Python services and retriever/answerer classes instead of a large orchestration framework.

Why:
- Easier to test, type-check, and reason about.
- Reviewers can see the actual retrieval, fusion, and refusal logic.
- Metrics and failure handling stay under repository control.

## ADR-002: Default to Chroma, keep FAISS optional

Decision:
- Chroma remains the default local vector store.
- FAISS stays available for leaner local experimentation.

Why Chroma:
- Persistent local state with minimal setup.
- Good reviewer ergonomics for a runnable repo.
- Fits the project goal of proving a deployable local baseline.

Why not FAISS only:
- FAISS is fast and simple, but Chroma gives easier persistence and metadata ergonomics for the document workflow.

Operational limit:
- Neither default is presented as the final answer for large-scale multi-tenant production. At that point, a managed vector platform or a stronger internal platform choice would be more appropriate.

## ADR-003: Use hybrid retrieval as the main default

Decision:
- The main runtime path prefers hybrid retrieval rather than dense-only retrieval.

Why:
- BM25 recovers exact lexical evidence that dense retrieval can miss.
- Dense retrieval covers semantic similarity when query and evidence wording differ.
- The combination better supports strict grounding and refusal decisions than dense-only retrieval on its own.

## ADR-004: Use RRF as the default hybrid fusion strategy

Decision:
- RRF is the default hybrid fusion mode, with weighted fusion still kept available.

Why:
- More stable when dense and sparse score scales drift.
- Easier to reason about than learned fusion in a portable repository.
- Good fit for a repo that must stay reproducible without external ranking infrastructure.

When to use weighted fusion instead:
- When the corpus is stable and the team has measured score calibration enough to justify a tuned dense weight.

## ADR-005: Enable reranking only when needed

Decision:
- Reranking is optional and not forced into the default path.

Why:
- Reranking improves quality only when candidate ambiguity is worth the added latency.
- For small corpora or offline reproducibility, pure hybrid retrieval is often enough.
- Keeping reranking opt-in makes latency and dependency costs more explicit.

When to use `RRF + rerank`:
- When reviewers or operators care more about final answer precision than raw latency.
- When first-pass hybrid retrieval returns multiple plausible chunks with close scores.

## ADR-006: Enforce strict grounding and refusal logic

Decision:
- The answerer refuses when evidence is weak or citations are missing/invalid.

Why:
- In a production-style RAG system, unsupported answers are usually more damaging than a refusal.
- This makes evaluation easier because unsupported behavior is visible and measurable.
- It aligns the API contract, metrics, and docs around a clear safety stance.

## ADR-007: Prioritize reproducibility over hidden hosted dependencies

Decision:
- The repo must still evaluate and run locally without external secrets.

Why:
- Reviewers need to verify engineering quality from a clean checkout.
- Offline fallback paths make CI stable.
- Saved artifacts in `experiments/metrics/` become inspectable evidence instead of claims.

## ADR-008: Accept latency vs quality tradeoffs explicitly

Decision:
- The repository documents and measures latency rather than pretending every quality improvement is free.

Tradeoff summary:
- Dense-only is usually simpler.
- Hybrid improves evidence recall.
- Rerank may improve final relevance.
- Strict refusal reduces hallucination but can increase false refusals.

This tradeoff is intentional and visible in code, metrics, and docs.
