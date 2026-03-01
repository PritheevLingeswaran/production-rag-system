from __future__ import annotations

import argparse
import asyncio
import json

from measure_production_metrics import parse_payload, stability_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a duration-based API stability test.")
    ap.add_argument("--query_url", required=True)
    ap.add_argument("--method", default="POST")
    ap.add_argument("--payload", default="{}")
    ap.add_argument("--duration_minutes", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--stability_query", default="How many projects are there in the resume?")
    ap.add_argument("--out_csv", default="stability_60m.csv")
    args = ap.parse_args()

    out = asyncio.run(
        stability_test(
            url=args.query_url,
            method=args.method,
            payload_template=parse_payload(args.payload),
            query=args.stability_query,
            duration_minutes=args.duration_minutes,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
            out_csv=args.out_csv,
        )
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
