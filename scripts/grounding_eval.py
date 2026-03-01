from __future__ import annotations

import argparse
import asyncio
import json

from measure_production_metrics import (
    auto_label_grounding_csv,
    export_grounding_sheet,
    parse_payload,
    score_grounding_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounding evaluation export/label/score workflow.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export")
    exp.add_argument("--query_url", required=True)
    exp.add_argument("--method", default="POST")
    exp.add_argument("--payload", default="{}")
    exp.add_argument("--grounding_queries", required=True)
    exp.add_argument("--out_csv", default="grounding_eval.csv")
    exp.add_argument("--concurrency", type=int, default=3)
    exp.add_argument("--timeout", type=float, default=60.0)

    auto = sub.add_parser("autolabel")
    auto.add_argument("--out_csv", default="grounding_eval.csv")

    score = sub.add_parser("score")
    score.add_argument("--out_csv", default="grounding_eval.csv")

    args = ap.parse_args()

    if args.cmd == "export":
        asyncio.run(
            export_grounding_sheet(
                queries_path=args.grounding_queries,
                out_csv=args.out_csv,
                query_url=args.query_url,
                method=args.method,
                payload_template=parse_payload(args.payload),
                timeout_s=args.timeout,
                concurrency=args.concurrency,
            )
        )
        return

    if args.cmd == "autolabel":
        print(json.dumps(auto_label_grounding_csv(args.out_csv), indent=2))
        return

    print(json.dumps(score_grounding_csv(args.out_csv), indent=2))


if __name__ == "__main__":
    main()
