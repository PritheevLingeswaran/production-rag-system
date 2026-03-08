from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an annotation template from indexed chunks.")
    parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks/chunks.jsonl",
        help="Indexed chunks file to sample from.",
    )
    parser.add_argument(
        "--output-path",
        default="evaluation/datasets/gold_template.jsonl",
        help="Where to write the starter template.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of chunks to convert into template rows.",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    rows = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "id": f"annotate-{idx + 1}",
                    "query": "",
                    "gold_answer": "",
                    "relevant_chunk_ids": [obj["chunk_id"]],
                    "answerability": "",
                    "notes": f"Source: {obj['source']} page={obj['page']}",
                }
            )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
