"""
backend/app/config.py
=====================
Application settings loaded from environment variables / .env file.

Never commit .env to git. Copy .env.example and fill in real values.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    db_host:     str = "localhost"
    db_port:     int = 5432
    db_name:     str = "sih26146"
    db_user:     str = "sih_user"
    db_password: str = "changeme"

    # ── GCP ───────────────────────────────────────────────────────────────────
    gcp_project: str = "visata-ai-505904"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host:    str = "0.0.0.0"
    api_port:    int = 8000
    debug:       bool = False

    # ── Offline mode ─────────────────────────────────────────────────────────
    # When True, skip BigQuery and use local parquet snapshots.
    # Set to True during the hackathon to avoid venue network dependency.
    offline_mode: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # silently ignore unknown .env keys
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    lru_cache means this is only constructed once per process — efficient.
    """
    return Settings()
