# Evaluation Results

This file is **generated** by `make eval`.

It reports:
- Corpus size (chunks, unique sources)
- Retrieval quality (dense vs hybrid): precision@k, recall@k, MRR
- Hallucination rate (heuristic, optional LLM judge)
- Latency p95 for the local eval run (retrieval, generation, end-to-end)
- Cost per query

To include load-handling capability (HTTP p95 + throughput):
1) Start the API (`make run`)
2) Run `make loadtest`
3) Run `make eval` again (it will embed `docs/load_test_results.json` if present)