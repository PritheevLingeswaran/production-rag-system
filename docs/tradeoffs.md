# Tradeoffs

## Operational limits

- Local file storage is simple and reproducible, but it is not a multi-region document platform.
- SQLite keeps the repo easy to run, but it is not the final answer for high-write multi-tenant production.
- Chroma and FAISS are good local baselines, but large-scale search would likely move to a stronger managed or distributed retrieval tier.

## Retrieval tradeoffs

### Dense only

Upside:
- Fast and simple.

Downside:
- Can miss exact-token evidence that matters for grounded answering.

### Hybrid retrieval

Upside:
- Better evidence coverage across semantic and lexical mismatch.

Downside:
- More moving parts.
- More tuning surface.
- Slightly higher operational complexity for debugging.

### RRF

Upside:
- Robust when score scales are not directly comparable.

Downside:
- Rank fusion loses some score magnitude information.

### Reranking

Upside:
- Can improve final evidence precision when candidate lists are ambiguous.

Downside:
- Adds latency and, depending on provider, heavier dependencies.

## Strict refusal tradeoffs

Upside:
- Safer than confidently answering from weak evidence.
- Reviewer-visible hallucination discipline.

Downside:
- Some borderline answerable questions will refuse.
- Product UX can feel conservative unless the UI explains refusal well.

## Monitoring tradeoffs

Upside:
- Request IDs, logs, and Prometheus metrics make operational behavior observable.

Downside:
- This repo stops at metrics and structured logging. It does not yet provide full distributed tracing or APM.

## CI and reproducibility tradeoffs

Upside:
- Stable CI without external secrets.
- Offline paths make review easier.

Downside:
- Secret-free CI means hosted LLM behavior is not exercised in every run.
- Some hosted-model token and cost fields are only populated when credentials are present.

## What this repo still does not claim

- Large-corpus benchmark leadership
- Multi-tenant security hardening
- Elastic horizontal ingest workers
- Full cloud-native autoscaling behavior
- Production throughput guarantees without environment-specific measurement
