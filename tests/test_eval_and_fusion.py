from __future__ import annotations

from measure_production_metrics import extract_doc_ids
from retrieval.bm25 import BM25DocHit
from retrieval.retriever import Retriever
from retrieval.vector_store import IndexedChunk, SearchHit
from utils.settings import Settings


def _chunk(cid: str, source: str = "doc.txt") -> IndexedChunk:
    return IndexedChunk(chunk_id=cid, source=source, page=1, text=f"text-{cid}", metadata={})


def test_extract_doc_ids_from_sources_and_ids() -> None:
    assert extract_doc_ids({"ids": ["a", "b"]}) == ["a", "b"]
    assert extract_doc_ids({"sources": [{"doc_id": "d1"}, {"id": "d2"}, {"source_id": "d3"}]}) == [
        "d1",
        "d2",
        "d3",
    ]
    assert extract_doc_ids({"retrieved": [{"doc_id": "x"}]}) == ["x"]


def test_apply_min_score_cutoff_falls_back_to_unfiltered_when_all_removed() -> None:
    hits = [
        SearchHit(chunk=_chunk("c1"), score=0.1),
        SearchHit(chunk=_chunk("c2"), score=0.08),
    ]
    out, threshold_applied = Retriever._apply_min_score_cutoff(hits, min_score=0.2)
    assert threshold_applied is False
    assert [h.chunk.chunk_id for h in out] == ["c1", "c2"]


def test_fusion_ranking_deterministic_order() -> None:
    settings = Settings()
    settings.retrieval.hybrid.dense_weight = 0.6
    retriever = Retriever.__new__(Retriever)
    retriever.settings = settings

    dense_hits = [
        SearchHit(chunk=_chunk("a"), score=0.9),
        SearchHit(chunk=_chunk("b"), score=0.2),
    ]
    sparse_hits = [
        BM25DocHit(idx=0, chunk_id="b", score=8.0),
        BM25DocHit(idx=1, chunk_id="c", score=4.0),
    ]
    chunk_by_id = {"a": _chunk("a"), "b": _chunk("b"), "c": _chunk("c")}

    fused = retriever._fuse_dense_and_sparse(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        chunk_by_id=chunk_by_id,
        top_k=3,
    )
    # Expected fused scores:
    # a = 0.6*0.9 + 0.4*0   = 0.54
    # b = 0.6*0.2 + 0.4*1.0 = 0.52
    # c = 0.6*0.0 + 0.4*0.5 = 0.20
    assert [h.chunk.chunk_id for h in fused] == ["a", "b", "c"]
