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

from pydantic import Field
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

    # --- Redis (request queues, reply lists) ------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # Postgres (postgres_dsn) and object storage (s3_*) settings were removed in Stage 1.5:
    # the first release runs neither (C1.23). They come back, as SecretStr where they hold
    # credentials (C0.2), in the stage that uses them: Postgres with S1, S3 with the registry.


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once.

    Cached so every caller sees the same object and env parsing happens once.
    Tests that change env vars call ``get_settings.cache_clear()`` first.
    """
    return Settings()
