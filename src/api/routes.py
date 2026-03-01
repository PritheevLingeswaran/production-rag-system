from __future__ import annotations

import time
from typing import Any, cast

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.deps import get_answerer, get_retriever
from monitoring.metrics import REQUEST_COST_USD, REQUEST_LATENCY, REQUEST_TOKENS
from schemas.query import HealthResponse, QueryRequest
from schemas.response import QueryResponse, Refusal, SourceChunk
from utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


def _record_usage_metrics(
    *,
    latency_s: float,
    embedding_tokens: int,
    llm_in: int,
    llm_out: int,
    total_cost: float,
) -> None:
    REQUEST_LATENCY.observe(latency_s)
    REQUEST_COST_USD.inc(total_cost)
    REQUEST_TOKENS.labels(kind="embedding").inc(float(embedding_tokens))
    REQUEST_TOKENS.labels(kind="llm_in").inc(float(llm_in))
    REQUEST_TOKENS.labels(kind="llm_out").inc(float(llm_out))


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    retriever: object = Depends(get_retriever),
    answerer: object = Depends(get_answerer),
) -> QueryResponse:
    start = time.perf_counter()
    filter_sub = req.filter.source if req.filter and req.filter.source else None
    retriever_impl = cast(Any, retriever)
    answerer_impl = cast(Any, answerer)
    try:
        r = retriever_impl.retrieve(
            question=req.query,
            top_k=req.top_k,
            filter_source_substr=filter_sub,
            rewrite_override=req.rewrite_query,
        )
    except Exception as e:
        # Common failure mode in dev: rewrite model/network issues.
        log.exception("query.retrieve_failed", error=str(e))
        if req.rewrite_query is not False:
            try:
                r = retriever_impl.retrieve(
                    question=req.query,
                    top_k=req.top_k,
                    filter_source_substr=filter_sub,
                    rewrite_override=False,
                )
            except Exception as e2:
                log.exception("query.retrieve_retry_failed", error=str(e2))
                latency_s = time.perf_counter() - start
                _record_usage_metrics(
                    latency_s=latency_s,
                    embedding_tokens=0,
                    llm_in=0,
                    llm_out=0,
                    total_cost=0.0,
                )
                return QueryResponse(
                    answer=(
                        "I cannot answer right now because retrieval is "
                        "temporarily unavailable."
                    ),
                    confidence=0.0,
                    sources=[],
                    refusal=Refusal(
                        is_refusal=True,
                        reason=(
                            "Retrieval backend error. Check model/API "
                            "configuration and connectivity."
                        ),
                    ),
                    metrics={
                        "error": "retrieval_failed",
                        "latency_ms": round(latency_s * 1000.0, 2),
                    },
                )
        else:
            latency_s = time.perf_counter() - start
            _record_usage_metrics(
                latency_s=latency_s,
                embedding_tokens=0,
                llm_in=0,
                llm_out=0,
                total_cost=0.0,
            )
            return QueryResponse(
                answer=(
                    "I cannot answer right now because retrieval is temporarily unavailable."
                ),
                confidence=0.0,
                sources=[],
                refusal=Refusal(
                    is_refusal=True,
                    reason=(
                        "Retrieval backend error. Check model/API configuration and "
                        "connectivity."
                    ),
                ),
                metrics={
                    "error": "retrieval_failed",
                    "latency_ms": round(latency_s * 1000.0, 2),
                },
            )

    try:
        g = answerer_impl.generate(req.query, r.hits)
    except Exception as e:
        log.exception("query.generate_failed", error=str(e))
        latency_s = time.perf_counter() - start
        total_cost = float(r.embedding_cost_usd)
        _record_usage_metrics(
            latency_s=latency_s,
            embedding_tokens=int(r.embedding_tokens),
            llm_in=0,
            llm_out=0,
            total_cost=total_cost,
        )
        fallback_sources = [
            SourceChunk(
                chunk_id=h.chunk.chunk_id,
                source=h.chunk.source,
                page=h.chunk.page,
                score=float(h.score),
                text=h.chunk.text,
            )
            for h in r.hits
        ]
        return QueryResponse(
            answer="I found relevant sources, but answer generation is temporarily unavailable.",
            confidence=0.0,
            sources=fallback_sources,
            refusal=Refusal(
                is_refusal=True,
                reason="Generation backend error. Check model/API configuration and connectivity.",
            ),
            metrics={
                "query_used": r.query_used,
                "embedding_tokens": r.embedding_tokens,
                "embedding_cost_usd": r.embedding_cost_usd,
                "llm_tokens_in": 0,
                "llm_tokens_out": 0,
                "llm_cost_usd": 0.0,
                "total_cost_usd": total_cost,
                "num_hits": len(r.hits),
                "error": "generation_failed",
                "latency_ms": round(latency_s * 1000.0, 2),
            },
        )

    latency_s = time.perf_counter() - start
    total_cost = float(r.embedding_cost_usd + g.llm_cost_usd)
    _record_usage_metrics(
        latency_s=latency_s,
        embedding_tokens=int(r.embedding_tokens),
        llm_in=int(g.llm_tokens_in),
        llm_out=int(g.llm_tokens_out),
        total_cost=total_cost,
    )

    return QueryResponse(
        answer=g.answer,
        confidence=g.confidence,
        sources=g.sources,
        refusal=g.refusal,
        metrics={
            "query_used": r.query_used,
            "embedding_tokens": r.embedding_tokens,
            "embedding_cost_usd": r.embedding_cost_usd,
            "llm_tokens_in": g.llm_tokens_in,
            "llm_tokens_out": g.llm_tokens_out,
            "llm_cost_usd": g.llm_cost_usd,
            "total_cost_usd": total_cost,
            "num_hits": len(r.hits),
            "latency_ms": round(latency_s * 1000.0, 2),
        },
    )


@router.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
