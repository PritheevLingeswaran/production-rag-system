from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from evaluation.resume_metrics import (
    benchmark_retrieval_latency,
    build_resume_bullets,
    build_resume_metrics,
    default_run_environment,
    evaluate_hallucination,
    evaluate_retrieval,
    load_hallucination_examples,
    load_retrieval_examples,
    make_answerer,
    make_baseline_answerer_settings,
    make_eval_settings,
    make_strict_hybrid_answerer_settings,
    run_load_test,
    start_local_api,
    summarize_dataset_stats,
    write_json,
)
from retrieval.retriever import Retriever
from retrieval.vector_store import build_vector_store
from utils.config import ensure_dirs, load_settings
from utils.logging import configure_logging


def _blocked(reason: str) -> dict[str, Any]:
    return {"measured": False, "reason": reason}


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Measure reproducible resume metrics for the current RAG repository."
    )
    parser.add_argument(
        "--retrieval-dataset",
        default="evaluation/datasets/resume_retrieval_eval.jsonl",
    )
    parser.add_argument(
        "--hallucination-dataset",
        default="evaluation/datasets/resume_hallucination_eval.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/metrics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load-concurrency-levels",
        default="10,25,50,100,200",
        help="Comma-separated concurrency levels for HTTP load testing.",
    )
    parser.add_argument(
        "--load-requests-per-level",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--load-timeout-s",
        type=float,
        default=60.0,
    )
    args = parser.parse_args()

    settings = load_settings()
    ensure_dirs(settings)
    configure_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_examples = load_retrieval_examples(args.retrieval_dataset)
    hallucination_examples = load_hallucination_examples(args.hallucination_dataset)

    dataset_stats = summarize_dataset_stats(settings)
    write_json(output_dir / "dataset_stats.json", dataset_stats)

    try:
        eval_settings = make_eval_settings(settings)
        retriever = Retriever(eval_settings, build_vector_store(eval_settings))
    except Exception as exc:
        reason = f"retriever initialization failed: {exc}"
        latency_dense = _blocked(reason)
        latency_hybrid = _blocked(reason)
        retrieval_report = _blocked(reason)
        hallucination_report = _blocked(reason)
        write_json(output_dir / "latency_dense.json", latency_dense)
        write_json(output_dir / "latency_hybrid.json", latency_hybrid)
        write_json(output_dir / "retrieval_comparison.json", retrieval_report)
        write_json(output_dir / "hallucination_report.json", hallucination_report)
    else:
        latency_dense = benchmark_retrieval_latency(
            retriever,
            retrieval_examples,
            mode="dense",
            top_k=args.top_k,
        )
        latency_hybrid = benchmark_retrieval_latency(
            retriever,
            retrieval_examples,
            mode="hybrid",
            top_k=args.top_k,
        )
        retrieval_report = evaluate_retrieval(
            retriever,
            retrieval_examples,
            top_ks=[1, 3, 5, 10],
        )
        baseline_answerer = make_answerer(make_baseline_answerer_settings(settings))
        strict_hybrid_answerer = make_answerer(make_strict_hybrid_answerer_settings(settings))
        hallucination_report = evaluate_hallucination(
            retriever,
            baseline_answerer,
            strict_hybrid_answerer,
            hallucination_examples,
            top_k=args.top_k,
        )
        write_json(output_dir / "latency_dense.json", latency_dense)
        write_json(output_dir / "latency_hybrid.json", latency_hybrid)
        write_json(output_dir / "retrieval_comparison.json", retrieval_report)
        write_json(output_dir / "hallucination_report.json", hallucination_report)

    env = default_run_environment()
    load_report: dict[str, Any]
    server = None
    try:
        python_executable = str((Path(".venv") / "bin" / "python").resolve())
        server, base_url = start_local_api(
            python_executable=python_executable,
            env=env,
        )
        concurrency_levels = [
            int(value) for value in args.load_concurrency_levels.split(",") if value
        ]
        load_report = asyncio.run(
            run_load_test(
                base_url=base_url,
                endpoint="/query",
                queries=[example.query for example in retrieval_examples],
                top_k=args.top_k,
                concurrency_levels=concurrency_levels,
                requests_per_level=args.load_requests_per_level,
                timeout_s=args.load_timeout_s,
            )
        )
    except Exception as exc:
        load_report = _blocked(f"load test failed: {exc}")
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)

    write_json(output_dir / "load_test_report.json", load_report)

    resume_metrics = build_resume_metrics(
        dataset_stats=dataset_stats,
        latency_dense=latency_dense,
        latency_hybrid=latency_hybrid,
        retrieval=retrieval_report,
        hallucination=hallucination_report,
        load_test=load_report,
    )
    write_json(output_dir / "resume_metrics.json", resume_metrics)

    bullets = build_resume_bullets(resume_metrics)
    (output_dir / "resume_bullets.md").write_text(bullets + "\n", encoding="utf-8")

    print(json.dumps(resume_metrics, indent=2))


if __name__ == "__main__":
    main()
