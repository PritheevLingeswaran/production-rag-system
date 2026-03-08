from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from evaluation.performance import LatencyStats, summarize_latency
from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    compare_hybrid_vs_dense,
    hurt_precision_case,
    improved_recall_case,
    precision_at_k,
    recall_at_k,
)
from generation.answerer import Answerer
from retrieval.retriever import Retriever
from retrieval.vector_store import (
    ChromaVectorStore,
    FaissVectorStore,
    build_vector_store,
)
from utils.settings import Settings

RetrievalMode = Literal["dense", "hybrid"]


@dataclass(frozen=True)
class RetrievalExample:
    id: str
    query: str
    relevant_chunk_ids: set[str]


@dataclass(frozen=True)
class HallucinationExample:
    id: str
    question: str
    expected_behavior: Literal["answer", "refuse"]
    acceptable_answers: list[str]
    notes: str = ""


@dataclass(frozen=True)
class LoadTestRow:
    concurrency: int
    total_requests: int
    success_count: int
    failure_count: int
    failure_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    measured: bool = True
    error: str | None = None


def load_retrieval_examples(path: str) -> list[RetrievalExample]:
    items: list[RetrievalExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                RetrievalExample(
                    id=str(obj["id"]),
                    query=str(obj["query"]),
                    relevant_chunk_ids={str(v) for v in obj["relevant_chunk_ids"]},
                )
            )
    return items


def load_hallucination_examples(path: str) -> list[HallucinationExample]:
    items: list[HallucinationExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                HallucinationExample(
                    id=str(obj["id"]),
                    question=str(obj["question"]),
                    expected_behavior=str(obj["expected_behavior"]),
                    acceptable_answers=[str(v) for v in obj.get("acceptable_answers", [])],
                    notes=str(obj.get("notes", "")),
                )
            )
    return items


def normalize_answer(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\(source:[^)]+\)", "", text, flags=re.IGNORECASE)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(answer: str, acceptable_answers: list[str]) -> bool:
    if not acceptable_answers:
        return True
    norm_answer = normalize_answer(answer)
    return any(norm_answer == normalize_answer(candidate) for candidate in acceptable_answers)


def has_valid_citation(answer: str, source_ids: list[str]) -> bool:
    cited_ids = re.findall(r"\[([^\]]+)\]", answer)
    if not cited_ids:
        return False
    valid = set(source_ids)
    return all(cid in valid for cid in cited_ids)


def summarize_dataset_stats(settings: Settings) -> dict[str, Any]:
    chunks_path = Path(settings.paths.chunks_dir) / "chunks.jsonl"
    unique_sources: set[str] = set()
    chunk_count = 0
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                chunk_count += 1
                unique_sources.add(str(obj["source"]))

    raw_documents = sorted(Path(settings.paths.raw_dir).glob("*"))
    vector_store = build_vector_store(settings)
    vector_count = 0
    if isinstance(vector_store, ChromaVectorStore):
        vector_count = int(vector_store._collection.count())
    elif isinstance(vector_store, FaissVectorStore) and vector_store.index is not None:
        vector_count = int(vector_store.index.ntotal)

    vector_index_path = (
        Path(settings.vector_store.chroma.persist_dir)
        if settings.vector_store.provider == "chroma"
        else Path(settings.paths.indexes_dir) / "faiss"
    )
    bm25_path = Path(settings.paths.indexes_dir) / "bm25"

    return {
        "measured": True,
        "documents_indexed": len(unique_sources),
        "chunks_indexed": chunk_count,
        "vectors_indexed": vector_count,
        "raw_documents_present": len(raw_documents),
        "paths": {
            "chunks": str(chunks_path),
            "vector_index": str(vector_index_path),
            "bm25_index": str(bm25_path),
        },
        "index_sizes_bytes": {
            "vector_index": directory_size_bytes(vector_index_path),
            "bm25_index": directory_size_bytes(bm25_path),
            "chunks_file": file_size_bytes(chunks_path),
        },
    }


def file_size_bytes(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return int(path.stat().st_size)


def directory_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def make_eval_settings(base: Settings) -> Settings:
    settings = base.model_copy(deep=True)
    settings.retrieval.query_rewrite.enabled = False
    settings.retrieval.rerank.enabled = False
    return settings


def make_baseline_answerer_settings(base: Settings) -> Settings:
    settings = make_eval_settings(base)
    settings.generation.strict_refusal = False
    settings.retrieval.refuse_if_top_score_below = 0.0
    settings.retrieval.refuse_if_top_gap_below = -1.0
    return settings


def make_strict_hybrid_answerer_settings(base: Settings) -> Settings:
    settings = make_eval_settings(base)
    settings.generation.strict_refusal = True
    return settings


def make_answerer(settings: Settings) -> Answerer:
    answerer = Answerer(settings)
    if not settings.embeddings.openai.api_key:
        answerer._disable_remote_generation = True
    return answerer


def benchmark_retrieval_latency(
    retriever: Retriever,
    examples: list[RetrievalExample],
    *,
    mode: RetrievalMode,
    top_k: int,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    latencies_s: list[float] = []

    # Warm the embedding model and vector store so benchmark numbers reflect
    # steady-state query latency instead of one-time model/materialization cost.
    if examples:
        retriever.retrieve(
            examples[0].query,
            top_k=top_k,
            rewrite_override=False,
            mode_override=mode,
        )

    for example in examples:
        started = time.perf_counter()
        result = retriever.retrieve(
            example.query,
            top_k=top_k,
            rewrite_override=False,
            mode_override=mode,
        )
        elapsed_s = time.perf_counter() - started
        latencies_s.append(elapsed_s)
        per_query.append(
            {
                "id": example.id,
                "query": example.query,
                "latency_ms": round(elapsed_s * 1000.0, 3),
                "num_hits": len(result.hits),
                "query_used": result.query_used,
            }
        )

    stats = summarize_latency(latencies_s)
    return {
        "measured": True,
        "mode": mode,
        "query_count": len(examples),
        "top_k": top_k,
        "summary": latency_stats_dict(stats),
        "per_query": per_query,
    }


def latency_stats_dict(stats: LatencyStats) -> dict[str, float | int]:
    return {
        "n": stats.n,
        "avg_latency_ms": round(stats.avg_ms, 3),
        "p50_latency_ms": round(stats.p50_ms, 3),
        "p95_latency_ms": round(stats.p95_ms, 3),
        "p99_latency_ms": round(stats.p99_ms, 3),
    }


def evaluate_retrieval(
    retriever: Retriever,
    examples: list[RetrievalExample],
    *,
    top_ks: list[int],
) -> dict[str, Any]:
    summary_by_k: dict[str, dict[str, float]] = {}
    improved_by_k: dict[str, list[dict[str, Any]]] = {str(k): [] for k in top_ks}
    hurt_by_k: dict[str, list[dict[str, Any]]] = {str(k): [] for k in top_ks}
    per_example: list[dict[str, Any]] = []

    max_k = max(top_ks)
    dense_mrrs: list[float] = []
    hybrid_mrrs: list[float] = []

    for example in examples:
        dense_result = retriever.retrieve(
            example.query,
            top_k=max_k,
            rewrite_override=False,
            mode_override="dense",
        )
        hybrid_result = retriever.retrieve(
            example.query,
            top_k=max_k,
            rewrite_override=False,
            mode_override="hybrid",
        )

        dense_ids = [hit.chunk.chunk_id for hit in dense_result.hits]
        hybrid_ids = [hit.chunk.chunk_id for hit in hybrid_result.hits]
        example_metrics: dict[str, Any] = {
            "id": example.id,
            "query": example.query,
            "relevant_chunk_ids": sorted(example.relevant_chunk_ids),
            "dense_chunk_ids": dense_ids,
            "hybrid_chunk_ids": hybrid_ids,
            "metrics_by_k": {},
        }

        dense_mrr = 0.0
        hybrid_mrr = 0.0
        for idx, chunk_id in enumerate(dense_ids, start=1):
            if chunk_id in example.relevant_chunk_ids:
                dense_mrr = 1.0 / idx
                break
        for idx, chunk_id in enumerate(hybrid_ids, start=1):
            if chunk_id in example.relevant_chunk_ids:
                hybrid_mrr = 1.0 / idx
                break
        dense_mrrs.append(dense_mrr)
        hybrid_mrrs.append(hybrid_mrr)

        for k in top_ks:
            cmp = compare_hybrid_vs_dense(
                dense_retrieved=dense_ids,
                hybrid_retrieved=hybrid_ids,
                relevant=example.relevant_chunk_ids,
                k=k,
            )
            dense_hit = 1.0 if recall_at_k(dense_ids, example.relevant_chunk_ids, k) > 0 else 0.0
            hybrid_hit = (
                1.0 if recall_at_k(hybrid_ids, example.relevant_chunk_ids, k) > 0 else 0.0
            )
            example_metrics["metrics_by_k"][str(k)] = {
                "dense": retrieval_metrics_dict(cmp.dense),
                "hybrid": retrieval_metrics_dict(cmp.hybrid),
                "dense_hit_rate": dense_hit,
                "hybrid_hit_rate": hybrid_hit,
            }
            if improved_recall_case(cmp):
                improved_by_k[str(k)].append(
                    {
                        "id": example.id,
                        "query": example.query,
                        "delta_recall": round(cmp.delta_recall, 4),
                    }
                )
            if hurt_precision_case(cmp):
                hurt_by_k[str(k)].append(
                    {
                        "id": example.id,
                        "query": example.query,
                        "delta_precision": round(cmp.delta_precision, 4),
                    }
                )
        per_example.append(example_metrics)

    for k in top_ks:
        dense_precisions = [
            precision_at_k(item["dense_chunk_ids"], set(item["relevant_chunk_ids"]), k)
            for item in per_example
        ]
        hybrid_precisions = [
            precision_at_k(item["hybrid_chunk_ids"], set(item["relevant_chunk_ids"]), k)
            for item in per_example
        ]
        dense_recalls = [
            recall_at_k(item["dense_chunk_ids"], set(item["relevant_chunk_ids"]), k)
            for item in per_example
        ]
        hybrid_recalls = [
            recall_at_k(item["hybrid_chunk_ids"], set(item["relevant_chunk_ids"]), k)
            for item in per_example
        ]
        dense_hits = [1.0 if value > 0 else 0.0 for value in dense_recalls]
        hybrid_hits = [1.0 if value > 0 else 0.0 for value in hybrid_recalls]

        summary_by_k[str(k)] = {
            "precision_dense": round(mean(dense_precisions), 4),
            "precision_hybrid": round(mean(hybrid_precisions), 4),
            "recall_dense": round(mean(dense_recalls), 4),
            "recall_hybrid": round(mean(hybrid_recalls), 4),
            "hit_rate_dense": round(mean(dense_hits), 4),
            "hit_rate_hybrid": round(mean(hybrid_hits), 4),
        }

    return {
        "measured": True,
        "query_count": len(examples),
        "top_ks": top_ks,
        "summary_by_k": summary_by_k,
        "mrr_dense": round(mean(dense_mrrs), 4),
        "mrr_hybrid": round(mean(hybrid_mrrs), 4),
        "improved_recall_examples": improved_by_k,
        "hurt_precision_examples": hurt_by_k,
        "per_example": per_example,
    }


def retrieval_metrics_dict(metrics: RetrievalMetrics) -> dict[str, float]:
    return {
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "mrr": round(metrics.mrr, 4),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_hallucination(
    retriever: Retriever,
    baseline_answerer: Answerer,
    strict_hybrid_answerer: Answerer,
    examples: list[HallucinationExample],
    *,
    top_k: int,
) -> dict[str, Any]:
    baseline_rows = []
    hybrid_rows = []

    for example in examples:
        dense_result = retriever.retrieve(
            example.question,
            top_k=top_k,
            rewrite_override=False,
            mode_override="dense",
        )
        hybrid_result = retriever.retrieve(
            example.question,
            top_k=top_k,
            rewrite_override=False,
            mode_override="hybrid",
        )
        baseline_rows.append(
            score_generation(
                answerer=baseline_answerer,
                example=example,
                retrieval_mode="dense",
                retrieval_result=dense_result,
            )
        )
        hybrid_rows.append(
            score_generation(
                answerer=strict_hybrid_answerer,
                example=example,
                retrieval_mode="hybrid",
                retrieval_result=hybrid_result,
            )
        )

    baseline_summary = hallucination_summary(baseline_rows)
    hybrid_summary = hallucination_summary(hybrid_rows)
    return {
        "measured": True,
        "rule": {
            "hallucination": (
                "A response is counted as hallucinated when it answers a refusal-required "
                "question without refusing, or when it answers an answer-required question "
                "with unsupported content: invalid/missing citations or an answer that does not "
                "match the gold acceptable answers."
            ),
            "citation_grounding_failure": (
                "Non-refusal answer with no valid citation pointing to a returned "
                "source chunk ID."
            ),
            "refusal_correctness": (
                "Refusal-required examples are correct only when the system explicitly refuses."
            ),
        },
        "datasets": {
            "examples": len(examples),
            "answer_expected": sum(1 for item in examples if item.expected_behavior == "answer"),
            "refusal_expected": sum(1 for item in examples if item.expected_behavior == "refuse"),
        },
        "baseline_dense": baseline_summary,
        "strict_grounded_hybrid": hybrid_summary,
        "per_example": {
            "baseline_dense": baseline_rows,
            "strict_grounded_hybrid": hybrid_rows,
        },
    }


def score_generation(
    *,
    answerer: Answerer,
    example: HallucinationExample,
    retrieval_mode: RetrievalMode,
    retrieval_result: Any,
) -> dict[str, Any]:
    generation = answerer.generate(example.question, retrieval_result.hits)
    source_ids = [source.chunk_id for source in generation.sources]
    refusal = bool(generation.refusal.is_refusal)
    valid_citation = has_valid_citation(generation.answer, source_ids) if not refusal else True
    answer_ok = (
        answer_matches(generation.answer, example.acceptable_answers) if not refusal else False
    )
    refusal_correct = example.expected_behavior == "refuse" and refusal
    false_refusal = example.expected_behavior == "answer" and refusal

    hallucinated = False
    unsupported_claim = False
    if example.expected_behavior == "refuse":
        hallucinated = not refusal_correct
        unsupported_claim = hallucinated
    elif not refusal:
        unsupported_claim = (not valid_citation) or (not answer_ok)
        hallucinated = unsupported_claim

    return {
        "id": example.id,
        "question": example.question,
        "expected_behavior": example.expected_behavior,
        "retrieval_mode": retrieval_mode,
        "answer": generation.answer,
        "refusal": refusal,
        "refusal_reason": generation.refusal.reason,
        "source_ids": source_ids,
        "valid_citation": valid_citation,
        "answer_matches_gold": answer_ok,
        "refusal_correct": refusal_correct,
        "false_refusal": false_refusal,
        "unsupported_claim": unsupported_claim,
        "hallucinated": hallucinated,
        "cost_usd": round(float(retrieval_result.embedding_cost_usd + generation.llm_cost_usd), 8),
        "embedding_tokens": int(retrieval_result.embedding_tokens),
        "llm_tokens_in": int(generation.llm_tokens_in),
        "llm_tokens_out": int(generation.llm_tokens_out),
    }


def hallucination_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_refusal_answers = [row for row in rows if not row["refusal"]]
    refusal_examples = [row for row in rows if row["expected_behavior"] == "refuse"]
    answer_examples = [row for row in rows if row["expected_behavior"] == "answer"]
    return {
        "example_count": len(rows),
        "hallucination_rate": round(mean([1.0 if row["hallucinated"] else 0.0 for row in rows]), 4),
        "citation_grounding_failure_rate": round(
            mean([1.0 if not row["valid_citation"] else 0.0 for row in non_refusal_answers]),
            4,
        ),
        "unsupported_claim_rate": round(
            mean([1.0 if row["unsupported_claim"] else 0.0 for row in rows]),
            4,
        ),
        "refusal_correctness_rate": round(
            mean([1.0 if row["refusal_correct"] else 0.0 for row in refusal_examples]),
            4,
        ),
        "false_refusal_rate": round(
            mean([1.0 if row["false_refusal"] else 0.0 for row in answer_examples]),
            4,
        ),
        "answered_rate": round(mean([1.0 if not row["refusal"] else 0.0 for row in rows]), 4),
        "avg_cost_per_query_usd": round(mean([float(row["cost_usd"]) for row in rows]), 8),
        "avg_embedding_tokens_per_query": round(
            mean([float(row["embedding_tokens"]) for row in rows]),
            4,
        ),
        "avg_llm_tokens_in_per_query": round(
            mean([float(row["llm_tokens_in"]) for row in rows]),
            4,
        ),
        "avg_llm_tokens_out_per_query": round(
            mean([float(row["llm_tokens_out"]) for row in rows]),
            4,
        ),
    }


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def start_local_api(
    *,
    python_executable: str,
    env: dict[str, str],
    host: str = "127.0.0.1",
    port: int | None = None,
) -> tuple[subprocess.Popen[str], str]:
    chosen_port = port or reserve_port()
    cmd = [
        python_executable,
        "-m",
        "uvicorn",
        "api.app:app",
        "--host",
        host,
        "--port",
        str(chosen_port),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    base_url = f"http://{host}:{chosen_port}"

    deadline = time.time() + 60.0
    with httpx.Client() as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("API server exited before becoming ready.")
            try:
                response = client.get(f"{base_url}/healthz", timeout=1.0)
                if response.status_code == 200:
                    return proc, base_url
            except Exception:
                time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Timed out waiting for local API to become ready.")


async def run_load_test(
    *,
    base_url: str,
    endpoint: str,
    queries: list[str],
    top_k: int,
    concurrency_levels: list[int],
    requests_per_level: int,
    timeout_s: float,
) -> dict[str, Any]:
    rows: list[LoadTestRow] = []
    url = base_url.rstrip("/") + endpoint

    for concurrency in concurrency_levels:
        total_requests = max(requests_per_level, concurrency)
        sem = asyncio.Semaphore(concurrency)
        latencies_s: list[float] = []
        failure_count = 0

        try:
            limits = httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            )
            async with httpx.AsyncClient(limits=limits) as client:
                async def one_request(
                    index: int,
                    *,
                    client: httpx.AsyncClient = client,
                    latencies_s: list[float] = latencies_s,
                    sem: asyncio.Semaphore = sem,
                ) -> None:
                    nonlocal failure_count
                    payload = {
                        "query": queries[index % len(queries)],
                        "top_k": top_k,
                        "rewrite_query": False,
                    }
                    started = time.perf_counter()
                    async with sem:
                        try:
                            response = await client.post(url, json=payload, timeout=timeout_s)
                            if response.status_code < 200 or response.status_code >= 300:
                                failure_count += 1
                            else:
                                latencies_s.append(time.perf_counter() - started)
                        except Exception:
                            failure_count += 1

                await asyncio.gather(
                    *[asyncio.create_task(one_request(index)) for index in range(total_requests)]
                )

            stats = summarize_latency(latencies_s)
            success_count = len(latencies_s)
            rows.append(
                LoadTestRow(
                    concurrency=concurrency,
                    total_requests=total_requests,
                    success_count=success_count,
                    failure_count=failure_count,
                    failure_rate=round(failure_count / total_requests, 4),
                    avg_latency_ms=round(stats.avg_ms, 3),
                    p50_latency_ms=round(stats.p50_ms, 3),
                    p95_latency_ms=round(stats.p95_ms, 3),
                    p99_latency_ms=round(stats.p99_ms, 3),
                )
            )
        except Exception as exc:
            rows.append(
                LoadTestRow(
                    concurrency=concurrency,
                    total_requests=total_requests,
                    success_count=0,
                    failure_count=total_requests,
                    failure_rate=1.0,
                    avg_latency_ms=0.0,
                    p50_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    measured=False,
                    error=str(exc),
                )
            )

    successful_levels = [
        row.concurrency
        for row in rows
        if row.measured and row.failure_count == 0 and row.success_count > 0
    ]
    return {
        "measured": any(row.measured for row in rows),
        "base_url": base_url,
        "endpoint": endpoint,
        "results": [asdict(row) for row in rows],
        "max_tested_concurrency_successful": max(successful_levels) if successful_levels else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_resume_metrics(
    *,
    dataset_stats: dict[str, Any],
    latency_dense: dict[str, Any],
    latency_hybrid: dict[str, Any],
    retrieval: dict[str, Any],
    hallucination: dict[str, Any],
    load_test: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if dataset_stats.get("measured"):
        metrics["documents_indexed"] = dataset_stats["documents_indexed"]
        metrics["chunks_indexed"] = dataset_stats["chunks_indexed"]
        vector_size = dataset_stats["index_sizes_bytes"].get("vector_index")
        if vector_size is not None:
            metrics["vector_index_size_bytes"] = vector_size

    if latency_hybrid.get("measured"):
        summary = latency_hybrid["summary"]
        metrics["avg_latency_ms"] = summary["avg_latency_ms"]
        metrics["p95_latency_ms"] = summary["p95_latency_ms"]
        metrics["p99_latency_ms"] = summary["p99_latency_ms"]
        metrics["avg_latency_ms_hybrid"] = summary["avg_latency_ms"]
        metrics["p95_latency_ms_hybrid"] = summary["p95_latency_ms"]

    if latency_dense.get("measured"):
        summary = latency_dense["summary"]
        metrics["avg_latency_ms_dense"] = summary["avg_latency_ms"]
        metrics["p95_latency_ms_dense"] = summary["p95_latency_ms"]

    if retrieval.get("measured"):
        for k in [1, 3, 5, 10]:
            key = str(k)
            if key in retrieval["summary_by_k"]:
                block = retrieval["summary_by_k"][key]
                metrics[f"precision_at_{k}_dense"] = block["precision_dense"]
                metrics[f"precision_at_{k}_hybrid"] = block["precision_hybrid"]
                metrics[f"recall_at_{k}_dense"] = block["recall_dense"]
                metrics[f"recall_at_{k}_hybrid"] = block["recall_hybrid"]
                metrics[f"hit_rate_at_{k}_dense"] = block["hit_rate_dense"]
                metrics[f"hit_rate_at_{k}_hybrid"] = block["hit_rate_hybrid"]
        metrics["mrr_dense"] = retrieval["mrr_dense"]
        metrics["mrr_hybrid"] = retrieval["mrr_hybrid"]

    if hallucination.get("measured"):
        baseline = hallucination["baseline_dense"]
        hybrid = hallucination["strict_grounded_hybrid"]
        metrics["hallucination_rate_dense"] = baseline["hallucination_rate"]
        metrics["hallucination_rate_hybrid"] = hybrid["hallucination_rate"]
        metrics["citation_grounding_failure_rate_dense"] = baseline[
            "citation_grounding_failure_rate"
        ]
        metrics["citation_grounding_failure_rate_hybrid"] = hybrid[
            "citation_grounding_failure_rate"
        ]
        metrics["avg_cost_per_query_usd"] = hybrid["avg_cost_per_query_usd"]
        metrics["avg_embedding_tokens_per_query"] = hybrid["avg_embedding_tokens_per_query"]
        metrics["avg_llm_tokens_in_per_query"] = hybrid["avg_llm_tokens_in_per_query"]
        metrics["avg_llm_tokens_out_per_query"] = hybrid["avg_llm_tokens_out_per_query"]

    if load_test.get("measured"):
        results = [
            row
            for row in load_test["results"]
            if row["measured"] and int(row["success_count"]) > 0
        ]
        if results:
            best = results[-1]
            metrics["max_tested_concurrency_successful"] = load_test.get(
                "max_tested_concurrency_successful"
            )
            metrics["load_test_avg_latency_ms"] = best["avg_latency_ms"]
            metrics["load_test_p95_latency_ms"] = best["p95_latency_ms"]
            metrics["load_test_failure_rate"] = best["failure_rate"]

    return metrics


def _metric_detail(metrics: dict[str, Any], key: str, label: str) -> str | None:
    value = metrics.get(key)
    if value is None:
        return None
    return f"{value} {label}"


def _metric_delta(
    metrics: dict[str, Any],
    *,
    newer_key: str,
    older_key: str,
    label: str,
) -> str | None:
    newer = metrics.get(newer_key)
    older = metrics.get(older_key)
    if newer is None or older is None:
        return None
    delta = round(float(newer) - float(older), 4)
    if abs(delta) < 1e-9:
        return None
    return f"{label} by {delta:+.4f}"


def build_resume_bullets(metrics: dict[str, Any]) -> str:
    p95 = metrics.get("p95_latency_ms")
    concurrency = metrics.get("max_tested_concurrency_successful")
    docs = metrics.get("documents_indexed")
    chunks = metrics.get("chunks_indexed")

    measured_parts = [
        part
        for part in [
            (
                f"{docs} documents / {chunks} chunks"
                if docs is not None and chunks is not None
                else None
            ),
            _metric_detail(metrics, "p95_latency_ms", "ms p95 hybrid retrieval latency"),
            (
                f"load-tested to {concurrency} concurrent queries with zero failures"
                if concurrency is not None
                else None
            ),
        ]
        if part is not None
    ]

    retrieval_delta = _metric_delta(
        metrics,
        newer_key="recall_at_5_hybrid",
        older_key="recall_at_5_dense",
        label="Recall@5 improved",
    )
    precision_delta = _metric_delta(
        metrics,
        newer_key="precision_at_5_hybrid",
        older_key="precision_at_5_dense",
        label="Precision@5 changed",
    )
    halluc_delta = _metric_delta(
        metrics,
        newer_key="hallucination_rate_hybrid",
        older_key="hallucination_rate_dense",
        label="hallucination rate changed",
    )

    short = "Built a hybrid RAG system with reproducible evaluation"
    if measured_parts:
        short += f" ({'; '.join(measured_parts)})."
    else:
        short += "."

    strong = "Implemented a reproducible dense-vs-hybrid RAG evaluation pipeline"
    details = []
    if retrieval_delta is not None:
        details.append(retrieval_delta)
    if precision_delta is not None:
        details.append(precision_delta)
    if halluc_delta is not None:
        details.append(halluc_delta)
    if p95 is not None:
        details.append(f"hybrid p95 latency {p95} ms")
    if concurrency is not None:
        details.append(f"verified at {concurrency} concurrent queries")
    if details:
        strong += ": " + ", ".join(details) + "."
    else:
        strong += "."

    senior = (
        "Productionized measurement for a hybrid RAG stack with code-generated resume metrics, "
        "covering corpus size, dense-vs-hybrid retrieval quality, and offline hallucination checks"
    )
    if measured_parts:
        senior += f" ({'; '.join(measured_parts)})."
    else:
        senior += "."

    return "\n".join(
        [
            "# Resume Bullets",
            "",
            "## Short Bullet",
            f"- {short}",
            "",
            "## Strong Bullet",
            f"- {strong}",
            "",
            "## Senior/Staff-Style Bullet",
            f"- {senior}",
            "",
            "## Claim Boundaries",
            (
                "- These bullets include only values written into "
                "`experiments/metrics/resume_metrics.json` by code execution."
            ),
            (
                "- No throughput or RPS claim is included because resume metrics "
                "omit it unless explicitly measured and selected."
            ),
        ]
    )


def default_run_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env.setdefault("RAG_ENV", "dev")
    env["OPENAI_API_KEY"] = ""
    env["OPENAI_BASE_URL"] = ""
    env["OPENAI_ORG"] = ""
    return env
