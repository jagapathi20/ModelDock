"""Configuration loaded from environment variables.

Every setting has an env var named ``MODELDOCK_<FIELD_NAME>`` (case-insensitive),
e.g. ``MODELDOCK_REDIS_URL``. A ``.env`` file in the working directory is read too,
but real environment variables win over it.

Both process types (API and worker) use this one class. Fields that only one
side needs are still declared here so there is a single place to look up what
can be configured.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODELDOCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown MODELDOCK_* vars are ignored, not an error
    )

    # --- General -----------------------------------------------------------
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- API process -------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # --- Redis (request queues, reply lists, cache, rate limits) -----------
    redis_url: str = "redis://localhost:6379/0"

    # --- Postgres (model registry, clients, API keys) ----------------------
    # The +asyncpg driver suffix is what SQLAlchemy's async engine needs (Stage 5).
    postgres_dsn: SecretStr = SecretStr(
        "postgresql+asyncpg://modeldock:modeldock@localhost:5432/modeldock"
    )

    # --- Object storage (model weights; MinIO locally, R2/S3 in deployment)
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: SecretStr = SecretStr("minioadmin")
    s3_bucket: str = "models"
    s3_region: str = "us-east-1"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once.

    Cached so every caller sees the same object and env parsing happens once.
    Tests that change env vars call ``get_settings.cache_clear()`` first.
    """
    return Settings()
