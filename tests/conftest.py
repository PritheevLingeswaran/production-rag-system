import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_ENV", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("OPENAI_ORG", "")
    yield


@pytest.fixture(autouse=True)
def _clear_dependency_caches() -> Iterator[None]:
    from api import deps

    deps.get_settings.cache_clear()
    deps.get_store.cache_clear()
    deps.get_retriever.cache_clear()
    deps.get_answerer.cache_clear()
    deps.get_metadata_service.cache_clear()
    deps.get_storage_service.cache_clear()
    deps.get_summary_service.cache_clear()
    deps.get_document_service.cache_clear()
    deps.get_chat_service.cache_clear()
    deps.get_auth_service.cache_clear()
    yield
    deps.get_settings.cache_clear()
    deps.get_store.cache_clear()
    deps.get_retriever.cache_clear()
    deps.get_answerer.cache_clear()
    deps.get_metadata_service.cache_clear()
    deps.get_storage_service.cache_clear()
    deps.get_summary_service.cache_clear()
    deps.get_document_service.cache_clear()
    deps.get_chat_service.cache_clear()
    deps.get_auth_service.cache_clear()
