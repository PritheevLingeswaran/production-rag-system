# rag-smart-qa

Config-driven FastAPI RAG system with strict grounding, hybrid retrieval (BM25 + dense), and reproducible evaluation.

## Core guarantees
- Answers must be grounded in retrieved evidence.
- Citation validation is enforced.
- If evidence quality is weak, system refuses: `Not available in the provided documents.`
- Metrics are exposed via Prometheus and evaluation scripts.

## API endpoints
- `GET /healthz`
- `POST /query`
- `GET /metrics`
- `GET /stats` (docs/chunks/vectors + active index paths)
- `POST /retrieve/bm25` (baseline sparse retrieval)
- `POST /retrieve/hybrid` (hybrid retrieval only)
- `POST /debug/retrieval` (stage-wise scores/counts; enable via config/env)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Build index and run API
```bash
source .venv/bin/activate
PYTHONPATH=src python -m scripts.ingest_data
PYTHONPATH=src python -m scripts.build_index --config configs/dev.yaml
PYTHONPATH=src python -m scripts.run_api
```

Open Swagger: `http://127.0.0.1:8000/docs`

## Quick verification
```bash
curl -sS http://127.0.0.1:8000/healthz | python3 -m json.tool
curl -sS -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How many projects are there in the resume?","top_k":8,"rewrite_query":false}' \
  | python3 -m json.tool
```

## Debug retrieval path
Enable one of:
- Config: `api.enable_debug_retrieval_endpoint: true`
- Env: `RAG_DEBUG_RETRIEVAL=1`

Then:
```bash
curl -sS -X POST "http://127.0.0.1:8000/debug/retrieval" \
  -H "Content-Type: application/json" \
  -d '{"query":"How many projects are there in the resume?","top_k":8,"rewrite_query":false}' \
  | python3 -m json.tool
```

Returned debug includes stage counts/scores:
- `dense_hits`, `bm25_hits`, `fusion_hits`, `rerank_hits`, `final_hits`
- `threshold_applied`
- `top_scores` per stage

## Reproducible resume metrics

Generate all machine-readable artifacts under `experiments/metrics/`:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m scripts.measure_resume_metrics
```

This script produces:
- `experiments/metrics/dataset_stats.json`
- `experiments/metrics/latency_dense.json`
- `experiments/metrics/latency_hybrid.json`
- `experiments/metrics/retrieval_comparison.json`
- `experiments/metrics/hallucination_report.json`
- `experiments/metrics/load_test_report.json`
- `experiments/metrics/resume_metrics.json`
- `experiments/metrics/resume_bullets.md`

What is measured:
- Dataset size from `data/processed/chunks/chunks.jsonl` and the persisted index directories.
- Dense-only vs hybrid retrieval latency on the fixed query set in `evaluation/datasets/resume_retrieval_eval.jsonl`.
- Precision@1/3/5/10, Recall@1/3/5/10, MRR, and hit rate@k for dense-only vs hybrid retrieval.
- Offline hallucination rate from the gold file `evaluation/datasets/resume_hallucination_eval.jsonl`.
- Real HTTP load-test results against a local `uvicorn` process.

What is offline vs online:
- Offline and reproducible without API keys: dataset stats, retrieval latency, retrieval quality, and the fallback-answer hallucination evaluation.
- Online only when valid model credentials exist: end-to-end remote LLM generation and any non-zero token or cost accounting for hosted models.

Hallucination rule used by the script:
- A response is counted as hallucinated if it answers a refusal-required question without refusing.
- A response is also counted as hallucinated if it answers an answer-required question with missing/invalid citations or an answer that does not match the gold acceptable answers.

Load-test claim boundary:
- Do not claim concurrency capacity unless `experiments/metrics/load_test_report.json` shows a successful concurrency level.
- Do not claim RPS unless it was explicitly measured and selected for reporting.

## Current measured snapshot

From the run on March 8, 2026:
- Documents indexed: `2`
- Chunks indexed: `9`
- Vector index size on disk: `725156` bytes
- Dense retrieval latency: avg `16.226 ms`, p95 `30.329 ms`
- Hybrid retrieval latency: avg `7.194 ms`, p95 `8.058 ms`
- Retrieval at `k=5`: dense precision `0.2`, hybrid precision `0.2`, dense recall `0.7708`, hybrid recall `0.7708`
- Hallucination rate: dense baseline `0.2727`, strict grounded hybrid `0.0`
- Load test: no successful concurrency level measured in this run; see `experiments/metrics/load_test_report.json`

## Make targets
```bash
make stats
make eval-retrieval
make eval-grounding
make stability-60m
```

## Notes on common confusion
- `embedding_tokens = 0` is expected for local `sentence_transformers` backend because token accounting is not provided.
- `num_hits = 0` is usually threshold/filtering or missing index/path mismatch; use `/debug/retrieval` and `/stats` to prove where it drops.

## License
MIT
