from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.resume_metrics import (
    benchmark_retrieval_latency,
    build_resume_bullets,
    build_cost_report,
    build_refusal_report,
    build_resume_metrics,
    default_run_environment,
    evaluate_hallucination,
    evaluate_retrieval_variants,
    load_hallucination_examples,
    load_retrieval_examples,
    make_answerer,
    make_baseline_answerer_settings,
    make_dense_settings,
    make_eval_settings,
    make_hybrid_settings,
    make_strict_hybrid_answerer_settings,
    run_load_test,
    summarize_retrieval_diagnostics,
    start_local_api,
    summarize_dataset_stats,
    validate_hallucination_examples,
    validate_retrieval_examples,
    write_json,
    write_jsonl,
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
    retrieval_validation = validate_retrieval_examples(retrieval_examples)
    hallucination_validation = validate_hallucination_examples(hallucination_examples)

    dataset_stats = summarize_dataset_stats(settings)
    dataset_stats["evaluation_datasets"] = {
        "retrieval": retrieval_validation,
        "hallucination": hallucination_validation,
    }
    write_json(output_dir / "dataset_stats.json", dataset_stats)

    try:
        dense_settings = make_dense_settings(settings)
        weighted_settings = make_hybrid_settings(settings, fusion_method="weighted", rerank_enabled=False)
        rrf_settings = make_hybrid_settings(settings, fusion_method="rrf", rerank_enabled=False)
        rrf_rerank_settings = make_hybrid_settings(
            settings,
            fusion_method="rrf",
            rerank_enabled=True,
        )
        default_settings = make_eval_settings(settings)
        retrievers = {
            "dense": Retriever(dense_settings, build_vector_store(dense_settings)),
            "hybrid_weighted": Retriever(weighted_settings, build_vector_store(weighted_settings)),
            "hybrid_rrf": Retriever(rrf_settings, build_vector_store(rrf_settings)),
            "hybrid_rrf_reranked": Retriever(
                rrf_rerank_settings,
                build_vector_store(rrf_rerank_settings),
            ),
            "hybrid_default": Retriever(default_settings, build_vector_store(default_settings)),
        }
    except Exception as exc:
        reason = f"retriever initialization failed: {exc}"
        latency_dense = _blocked(reason)
        latency_hybrid_weighted = _blocked(reason)
        latency_hybrid_rrf = _blocked(reason)
        retrieval_report = _blocked(reason)
        hallucination_report = _blocked(reason)
        refusal_report = _blocked(reason)
        cost_report = _blocked(reason)
        write_json(output_dir / "latency_dense.json", latency_dense)
        write_json(output_dir / "latency_hybrid_weighted.json", latency_hybrid_weighted)
        write_json(output_dir / "latency_hybrid_rrf.json", latency_hybrid_rrf)
        write_json(output_dir / "retrieval_comparison.json", retrieval_report)
        write_jsonl(output_dir / "retrieval_diagnostics.jsonl", [])
        write_json(output_dir / "hallucination_report.json", hallucination_report)
        write_json(output_dir / "refusal_report.json", refusal_report)
        write_json(output_dir / "cost_report.json", cost_report)
    else:
        latency_dense = benchmark_retrieval_latency(
            retrievers["dense"],
            retrieval_examples,
            mode="dense",
            label="dense",
            top_k=args.top_k,
        )
        latency_hybrid_weighted = benchmark_retrieval_latency(
            retrievers["hybrid_weighted"],
            retrieval_examples,
            mode="hybrid",
            label="hybrid_weighted",
            top_k=args.top_k,
        )
        latency_hybrid_rrf = benchmark_retrieval_latency(
            retrievers["hybrid_rrf"],
            retrieval_examples,
            mode="hybrid",
            label="hybrid_rrf",
            top_k=args.top_k,
        )
        retrieval_report, diagnostics = evaluate_retrieval_variants(
            {
                "dense": retrievers["dense"],
                "hybrid_weighted": retrievers["hybrid_weighted"],
                "hybrid_rrf": retrievers["hybrid_rrf"],
                "hybrid_rrf_reranked": retrievers["hybrid_rrf_reranked"],
            },
            retrieval_examples,
            top_ks=[1, 3, 5, 10],
        )
        retrieval_report["diagnostic_summary"] = summarize_retrieval_diagnostics(diagnostics)
        retrieval_report["default_runtime_variant"] = retrieval_report["selected_default_hybrid"]
        selected_hybrid_retriever = retrievers.get(
            retrieval_report["selected_default_hybrid"],
            retrievers["hybrid_rrf"],
        )
        baseline_answerer = make_answerer(make_baseline_answerer_settings(settings))
        strict_hybrid_answerer = make_answerer(make_strict_hybrid_answerer_settings(settings))
        hallucination_report = evaluate_hallucination(
            retrievers["dense"],
            selected_hybrid_retriever,
            baseline_answerer,
            strict_hybrid_answerer,
            hallucination_examples,
            top_k=args.top_k,
        )
        refusal_report = build_refusal_report(hallucination_report)
        cost_report = build_cost_report(hallucination_report)
        write_json(output_dir / "latency_dense.json", latency_dense)
        write_json(output_dir / "latency_hybrid_weighted.json", latency_hybrid_weighted)
        write_json(output_dir / "latency_hybrid_rrf.json", latency_hybrid_rrf)
        write_json(output_dir / "retrieval_comparison.json", retrieval_report)
        write_jsonl(output_dir / "retrieval_diagnostics.jsonl", diagnostics)
        write_json(output_dir / "hallucination_report.json", hallucination_report)
        write_json(output_dir / "refusal_report.json", refusal_report)
        write_json(output_dir / "cost_report.json", cost_report)

    env = default_run_environment()
    load_report: dict[str, Any]
    server = None
    try:
        venv_python = Path(".venv/bin/python")
        python_executable = str(venv_python.absolute()) if venv_python.exists() else sys.executable
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
        latency_hybrid_weighted=latency_hybrid_weighted,
        latency_hybrid_rrf=latency_hybrid_rrf,
        retrieval=retrieval_report,
        hallucination=hallucination_report,
        refusal_report=refusal_report,
        cost_report=cost_report,
        load_test=load_report,
    )
    write_json(output_dir / "resume_metrics.json", resume_metrics)

    bullets = build_resume_bullets(resume_metrics)
    (output_dir / "resume_bullets.md").write_text(bullets + "\n", encoding="utf-8")

    print(json.dumps(resume_metrics, indent=2))


if __name__ == "__main__":
    main()
