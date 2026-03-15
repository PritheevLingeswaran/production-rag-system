from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import ensure_dirs, load_settings


def test_load_settings_merges_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        """
app:
  environment: dev
paths:
  raw_dir: ${RAW_DIR}
  uploads_dir: ${RAW_DIR}/uploads
  processed_dir: ${PROCESSED_DIR}
  chunks_dir: ${PROCESSED_DIR}/chunks
  metadata_dir: ${PROCESSED_DIR}/metadata
  indexes_dir: ${PROCESSED_DIR}/indexes
  app_db_path: ${PROCESSED_DIR}/metadata/app.db
retrieval:
  hybrid:
    enabled: false
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "dev.yaml").write_text(
        """
retrieval:
  hybrid:
    enabled: true
    fusion_method: rrf
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("RAG_ENV", "dev")
    monkeypatch.setenv("RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))

    settings = load_settings()
    ensure_dirs(settings)

    assert settings.retrieval.hybrid.enabled is True
    assert settings.retrieval.hybrid.fusion_method == "rrf"
    assert settings.paths.raw_dir == str(tmp_path / "raw")
    assert (tmp_path / "processed" / "chunks").exists()
    assert (tmp_path / "processed" / "metadata").exists()


def test_load_settings_requires_base_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ENV", "dev")

    with pytest.raises(FileNotFoundError):
        load_settings()
