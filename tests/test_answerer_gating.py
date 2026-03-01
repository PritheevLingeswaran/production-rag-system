from __future__ import annotations

from dataclasses import dataclass

from generation.answerer import Answerer
from retrieval.vector_store import IndexedChunk, SearchHit
from utils.settings import Settings


def _hit(chunk_id: str, score: float, text: str) -> SearchHit:
    return SearchHit(
        chunk=IndexedChunk(
            chunk_id=chunk_id,
            source="data/raw/documents/Pritheev_Resume.pdf",
            page=1,
            text=text,
            metadata={},
        ),
        score=score,
    )


def test_refusal_gate_triggers_on_low_top_score() -> None:
    settings = Settings()
    answerer = Answerer(settings)
    answerer._disable_remote_generation = True
    out = answerer.generate("How many projects are there?", [_hit("a", 0.10, "text")])
    assert out.refusal.is_refusal is True
    assert out.answer == "Not available in the provided documents."


def test_refusal_gate_triggers_on_ambiguous_top_gap() -> None:
    settings = Settings()
    settings.retrieval.refuse_if_top_score_below = 0.0
    settings.retrieval.refuse_if_top_gap_below = 0.05
    answerer = Answerer(settings)
    answerer._disable_remote_generation = True
    out = answerer.generate(
        "How many projects are there?",
        [
            _hit("a", 0.70, "Project one (rag-smart-qa)"),
            _hit("b", 0.68, "Project two (realtime-ml-drift)"),
        ],
    )
    assert out.refusal.is_refusal is True
    assert "ambiguous" in out.refusal.reason.lower()


def test_fallback_answer_gets_citation_when_strict_refusal_enabled() -> None:
    settings = Settings()
    settings.retrieval.refuse_if_top_score_below = 0.0
    settings.retrieval.refuse_if_top_gap_below = 0.0
    answerer = Answerer(settings)
    answerer._disable_remote_generation = True
    out = answerer.generate(
        "How many projects are there in the resume?",
        [
            _hit(
                "resume:p1:c1",
                0.9,
                "Production-Grade Hybrid RAG System (rag-smart-qa)\n"
                "Production-Grade Real-Time ML Drift Detection System (realtime-ml-drift)\n"
                "Production-Grade ML Decision & Evaluation Platform (ml-failure-analysis-framework)",
            )
        ],
    )
    assert out.refusal.is_refusal is False
    assert "[resume:p1:c1]" in out.answer


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 20


class _StubClientNoCitations:
    def chat(self, **_: object) -> tuple[str, _Usage]:
        return (
            '{"answer":"This is an unsupported uncited answer.","cited_chunk_ids":[],"refusal":{"is_refusal":false,"reason":""}}',
            _Usage(),
        )


def test_uncited_llm_answer_is_refused_under_strict_policy() -> None:
    settings = Settings()
    settings.retrieval.refuse_if_top_score_below = 0.0
    settings.retrieval.refuse_if_top_gap_below = 0.0
    answerer = Answerer(settings)
    answerer.client = _StubClientNoCitations()
    out = answerer.generate("What is the answer?", [_hit("x:p1:c0", 0.9, "Supported text")])
    assert out.refusal.is_refusal is True
    assert "citations" in out.refusal.reason.lower()
