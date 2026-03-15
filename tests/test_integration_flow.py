from __future__ import annotations

import io
from pathlib import Path
from typing import Any, ClassVar

import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from embeddings.base import EmbedResult
from retrieval.vector_store import IndexedChunk, SearchHit, VectorStore


class FakeEmbeddingsBackend:
    def embed_texts(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(
            vectors=[self._vector(text) for text in texts], total_tokens=0, cost_usd=0.0
        )

    def embed_query(self, text: str) -> EmbedResult:
        return EmbedResult(vectors=[self._vector(text)], total_tokens=0, cost_usd=0.0)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("project")),
            float(lowered.count("resume")),
            float(lowered.count("rag")),
        ]


class FakePersistentVectorStore(VectorStore):
    _registry: ClassVar[dict[str, tuple[list[IndexedChunk], list[list[float]]]]] = {}

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        key = settings.paths.indexes_dir
        stored = self._registry.get(key, ([], []))
        self.chunks = list(stored[0])
        self.vectors = list(stored[1])

    def add(self, chunks: list[IndexedChunk], vectors: Any) -> None:
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(
        self,
        query_vector: Any,
        top_k: int,
        filter_source_substr: str | None = None,
    ) -> list[SearchHit]:
        query = (
            query_vector[0]
            if isinstance(query_vector, list) and query_vector and isinstance(query_vector[0], list)
            else query_vector
        )
        hits: list[tuple[float, IndexedChunk]] = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=False):
            if filter_source_substr and filter_source_substr not in chunk.source:
                continue
            score = sum(float(a) * float(b) for a, b in zip(query, vector, strict=False))
            hits.append((score, chunk))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [SearchHit(chunk=chunk, score=score) for score, chunk in hits[:top_k]]

    def save(self) -> None:
        self._registry[self.settings.paths.indexes_dir] = (list(self.chunks), list(self.vectors))

    @classmethod
    def load(cls, settings: Any) -> FakePersistentVectorStore:
        return cls(settings)


def test_upload_index_and_query_flow(tmp_path: Path, monkeypatch: Any) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw_dir = tmp_path / "data" / "raw" / "documents"
    processed_dir = tmp_path / "data" / "processed"
    base_config = {
        "app": {"name": "rag-smart-qa", "environment": "test"},
        "paths": {
            "raw_dir": str(raw_dir),
            "uploads_dir": str(raw_dir / "uploads"),
            "processed_dir": str(processed_dir),
            "chunks_dir": str(processed_dir / "chunks"),
            "metadata_dir": str(processed_dir / "metadata"),
            "indexes_dir": str(processed_dir / "indexes"),
            "app_db_path": str(processed_dir / "metadata" / "app.db"),
        },
        "embeddings": {
            "provider": "sentence_transformers",
            "model": "fake-local",
            "sentence_transformers": {"model_name": "fake-local", "local_files_only": True},
        },
        "vector_store": {"provider": "pinecone", "top_k": 8},
        "retrieval": {
            "query_rewrite": {"enabled": False, "model": "gpt-4o-mini"},
            "hybrid": {
                "enabled": True,
                "fusion_method": "rrf",
                "bm25_k": 10,
                "dense_k": 10,
                "rrf_k": 30,
            },
            "cache": {"enabled": True, "max_entries": 32},
            "rerank": {"enabled": False, "provider": "lexical"},
            "min_score": 0.0,
            "refuse_if_top_score_below": 0.0,
            "refuse_if_top_gap_below": 0.0,
        },
        "summaries": {"enabled": True, "max_context_chars": 2000, "max_points": 3},
        "auth": {"enabled": False, "provider": "none", "demo_user_id": "local-user"},
    }
    (config_dir / "base.yaml").write_text(yaml.safe_dump(base_config), encoding="utf-8")
    (config_dir / "dev.yaml").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("RAG_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("RAG_ENV", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("RAG_SKIP_STARTUP_VALIDATION", "1")
    monkeypatch.setattr(
        "retrieval.retriever.build_embeddings_backend", lambda settings: FakeEmbeddingsBackend()
    )
    monkeypatch.setattr(
        "services.document_service.build_embeddings_backend",
        lambda settings: FakeEmbeddingsBackend(),
    )

    def fake_build_vector_store(settings: Any) -> FakePersistentVectorStore:
        return FakePersistentVectorStore(settings)

    monkeypatch.setattr("services.document_service.build_vector_store", fake_build_vector_store)

    app = create_app()
    client = TestClient(app)

    resume_text = (
        "Production-Grade Hybrid RAG System (rag-smart-qa)\n"
        "Production-Grade Real-Time ML Drift Detection System (realtime-ml-drift)\n"
        "Production-Grade ML Decision & Evaluation Platform (ml-failure-analysis-framework)\n"
    )

    upload = client.post(
        "/api/v1/documents/upload",
        files=[("files", ("resume.txt", io.BytesIO(resume_text.encode("utf-8")), "text/plain"))],
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    detail = client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["status"] == "ready"

    chat = client.post(
        "/api/v1/chat/query",
        json={
            "question": "How many projects are there in the resume?",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 5,
        },
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["refusal"]["is_refusal"] is False
    assert payload["answer"].startswith("3")
    assert payload["citations"]
