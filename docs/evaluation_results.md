# Evaluation Results

This document summarizes the measured outputs written by `python -m scripts.measure_resume_metrics`.

Run date: March 8, 2026

## Artifacts

- `experiments/metrics/dataset_stats.json`
- `experiments/metrics/latency_dense.json`
- `experiments/metrics/latency_hybrid.json`
- `experiments/metrics/retrieval_comparison.json`
- `experiments/metrics/hallucination_report.json`
- `experiments/metrics/load_test_report.json`
- `experiments/metrics/resume_metrics.json`
- `experiments/metrics/resume_bullets.md`

## Dataset size

- Source documents indexed: `2`
- Chunks indexed: `9`
- Vectors indexed: `9`
- Vector index size on disk: `725156` bytes
- BM25 index size on disk: `9575` bytes

## Retrieval latency

Measured on the fixed query set in `evaluation/datasets/resume_retrieval_eval.jsonl`.

- Dense-only latency: avg `16.226 ms`, p50 `9.276 ms`, p95 `30.329 ms`, p99 `30.799 ms`
- Hybrid latency: avg `7.194 ms`, p50 `7.21 ms`, p95 `8.058 ms`, p99 `8.182 ms`

Notes:
- The benchmark performs an unmeasured warm-up call per mode before timing.
- These are local retrieval-path measurements, not hosted-LLM end-to-end latencies.

## Retrieval quality

Gold set: `evaluation/datasets/resume_retrieval_eval.jsonl`

At `k=1`:
- Dense precision `0.3333`, hybrid precision `0.1667`
- Dense recall `0.3333`, hybrid recall `0.1667`

At `k=3`:
- Dense precision `0.1667`, hybrid precision `0.1667`
- Dense recall `0.4583`, hybrid recall `0.4583`

At `k=5`:
- Dense precision `0.2`, hybrid precision `0.2`
- Dense recall `0.7708`, hybrid recall `0.7708`

At `k=10`:
- Dense precision `0.1667`, hybrid precision `0.1667`
- Dense recall `1.0`, hybrid recall `1.0`

Ranking metrics:
- Dense MRR: `0.4986`
- Hybrid MRR: `0.4111`

Where hybrid helped or hurt in this run:
- No query in this gold set improved hybrid recall over dense at `k=5`.
- No query in this gold set hurt hybrid precision relative to dense at `k=5`.
- Hybrid was worse at `k=1` on this tiny corpus, so a top-1 precision gain cannot be claimed.

Interpretation:
- This corpus has only `9` chunks, so the dense baseline already recovers most relevant chunks by `k=5`.
- On a larger or more lexically diverse corpus, hybrid may improve recall; this run does not demonstrate that improvement.

## Hallucination evaluation

Gold set: `evaluation/datasets/resume_hallucination_eval.jsonl`

Rule implemented in code:
- Count a response as hallucinated if it answers a refusal-required question without refusing.
- Also count a response as hallucinated if it answers an answer-required question with missing or invalid citations, or with an answer that does not match the gold acceptable answers.

Measured results:
- Baseline dense RAG hallucination rate: `0.2727`
- Strict grounded hybrid hallucination rate: `0.0`
- Baseline dense citation grounding failure rate: `1.0`
- Strict grounded hybrid citation grounding failure rate: `0.0`
- Refusal correctness rate: `1.0` for both configurations on the refusal-required subset

Important caveat:
- This run uses the repo's offline fallback answer path because no OpenAI credentials were available in the shell.
- These values are valid for the local fallback-answer evaluation, not for a hosted LLM answerer.

## Cost and tokens

For this run:
- Average cost per query: `0.0 USD`
- Average embedding tokens per query: `0.0`
- Average LLM input tokens per query: `0.0`
- Average LLM output tokens per query: `0.0`

Why all zeros:
- The active retrieval backend is local `sentence-transformers`.
- No hosted LLM generation was executed in this run.

## Load testing

Load test target:
- Local `uvicorn` process started by the measurement script
- Endpoint: `POST /query`

Tested concurrency levels:
- `10`, `25`, `50`, `100`, `200`

Measured outcome in this run:
- Every tested level failed all requests within the configured load-test run.
- `max_tested_concurrency_successful` is therefore not measured.

Claim boundary:
- Do not claim concurrent query handling capacity from this run.
- Use `experiments/metrics/load_test_report.json` as evidence that the test was attempted, and that no successful level was established.
