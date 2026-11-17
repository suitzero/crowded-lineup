"""Application settings loaded via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CROWDED_", extra="ignore")

    app_name: str = "Crowded Lineup Arena & Intelligence API"
    environment: str = "dev"
    database_url: str = ""
    blob_root: Path = Path("/tmp/crowded_lineup_blobs")
    corpus_work_dir: Path = Path("/tmp/crowded_lineup_corpus")
    audit_log_path: Path | None = None
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    bootstrap_corpus: bool = True
