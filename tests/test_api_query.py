from fastapi.testclient import TestClient

from api import deps
from api.app import create_app
from retrieval.vector_store import IndexedChunk, SearchHit
from schemas.response import Refusal, SourceChunk


class DummyRetriever:
    def retrieve(
        self,
        question: str,
        top_k: int,
        filter_source_substr: str | None = None,
        rewrite_override: bool | None = None,
    ) -> object:
        chunk = IndexedChunk(
            chunk_id="x:p1:c0", source="doc.txt", page=1, text="Warranty is 12 months.", metadata={}
        )
        return type(
            "R",
            (),
            {
                "query_used": question,
                "hits": [SearchHit(chunk=chunk, score=0.9)],
                "embedding_tokens": 0,
                "embedding_cost_usd": 0.0,
            },
        )


class DummyAnswerer:
    def generate(self, question: str, hits: list[SearchHit]) -> object:
        return type(
            "G",
            (),
            {
                "answer": "The warranty period is 12 months. [x:p1:c0]",
                "confidence": 0.9,
                "sources": [
                    SourceChunk(
                        chunk_id="x:p1:c0",
                        source="doc.txt",
                        page=1,
                        score=0.9,
                        text="Warranty is 12 months.",
                    )
                ],
                "refusal": Refusal(is_refusal=False, reason=""),
                "llm_tokens_in": 0,
                "llm_tokens_out": 0,
                "llm_cost_usd": 0.0,
            },
        )


class FailingAnswerer:
    def generate(self, question: str, hits: list[SearchHit]) -> None:
        raise RuntimeError("upstream generation error")


def _dummy_retriever() -> DummyRetriever:
    return DummyRetriever()


def _dummy_answerer() -> DummyAnswerer:
    return DummyAnswerer()


def _failing_answerer() -> FailingAnswerer:
    return FailingAnswerer()


def test_query_endpoint() -> None:
    app = create_app()
    app.dependency_overrides[deps.get_retriever] = _dummy_retriever
    app.dependency_overrides[deps.get_answerer] = _dummy_answerer

    client = TestClient(app)
    r = client.post("/query", json={"query": "warranty?", "top_k": 3})
    assert r.status_code == 200
    j = r.json()
    assert j["refusal"]["is_refusal"] is False
    assert j["sources"][0]["chunk_id"] == "x:p1:c0"


def test_query_endpoint_generation_failure_returns_refusal() -> None:
    app = create_app()
    app.dependency_overrides[deps.get_retriever] = _dummy_retriever
    app.dependency_overrides[deps.get_answerer] = _failing_answerer

    client = TestClient(app)
    r = client.post("/query", json={"query": "warranty?", "top_k": 3})
    assert r.status_code == 200
    j = r.json()
    assert j["refusal"]["is_refusal"] is True
    assert j["metrics"]["error"] == "generation_failed"
    assert len(j["sources"]) == 1
