from __future__ import annotations

import argparse
import asyncio
import json

from measure_production_metrics import evaluate_retrieval, load_eval_jsonl, parse_payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate retrieval quality: BM25 baseline vs hybrid (Precision@k, Recall@k)."
    )
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--hybrid_url", required=True)
    ap.add_argument("--bm25_url", required=True)
    ap.add_argument("--method", default="POST")
    ap.add_argument("--payload", default="{}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    eval_data = load_eval_jsonl(args.eval_jsonl)
    out = asyncio.run(
        evaluate_retrieval(
            eval_data=eval_data,
            hybrid_url=args.hybrid_url,
            bm25_url=args.bm25_url,
            method=args.method,
            payload_template=parse_payload(args.payload),
            k=args.k,
            timeout_s=args.timeout,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
