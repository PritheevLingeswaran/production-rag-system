# Project Metrics Sheet

Evidence-backed values from the current repository/run state.
If a value is not directly measured by artifacts or reproducible commands, it is marked as
`N/A (not measured)`.

## 1) Reliability

- Uptime (%): N/A (not measured)
- Error rate (%): N/A (not measured)
- Crash rate under load: N/A (not measured)

## 2) Performance

- Average latency (ms): N/A (not measured)
- p95 latency (ms): N/A (not measured)
- Throughput (RPS): N/A (not measured)
- Concurrent users tested: N/A (not measured)

## 3) Scale

- Number of source documents: 2 (from `data/raw/documents`)
- Dataset size (samples/docs): 5 chunks (from `data/processed/chunks/chunks.jsonl`)
- Events processed (streaming): N/A (not measured)
- Vector DB size: 5 vectors (indexed chunks)
- Embedding/model size: all-MiniLM-L6-v2 (384-dim embeddings)

## 4) ML Evaluation

- Baseline comparison: N/A (not measured)
- Improvement vs baseline (%): N/A (not measured)
- Cross-validation used (yes/no): N/A (not measured)
- Fold count: N/A (not measured)
- Precision@k improvement (%): N/A (not measured)
- Hallucination/error reduction (%): N/A (not measured)
- Drift detection accuracy (%): N/A (not measured)

## 5) Observability

- Prometheus metrics count: 3 (`rag_request_latency_seconds`, `rag_request_cost_usd_total`, `rag_request_tokens_total`)
- Structured logging implemented (yes/no): yes
- Monitoring dashboards implemented (yes/no): N/A (artifact not found)

## 6) Deployment

- Dockerized (yes/no): yes
- CI/CD enabled (yes/no): yes
- Test coverage (%): 50 (from `pytest --cov=src`)
- Config-driven architecture (yes/no): yes

## 7) Cost / Efficiency

- Cost reduction vs baseline (%): N/A (not measured)
- Memory reduction vs baseline (%): N/A (not measured)

## Evidence Links

- Load test report path: docs/load_test_results.json (not generated in this run)
- Evaluation report path: docs/evaluation_results.md (not generated in this run)
- Dashboard screenshot path: docs/assets/monitoring_dashboard.png (not found)
- CI pipeline link/path: .github/workflows/ci.yml
