"""Configuration loaded from environment variables.

Every setting has an env var named ``MODELDOCK_<FIELD_NAME>`` (case-insensitive),
e.g. ``MODELDOCK_REDIS_URL``. A ``.env`` file in the working directory is read too,
but real environment variables win over it.

Both process types (API and worker) use this one class. Fields that only one
side needs are still declared here so there is a single place to look up what
can be configured.
"""

from functools import lru_cache
from pathlib import Path
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

    # Largest request body the API reads, in bytes (C2.2). Larger bodies get 413 before
    # they are parsed. Separate from a model's max_chars (422), which limits one field.
    max_body_bytes: int = Field(default=16_384, ge=1)

    # --- Redis (request queues, reply lists) ------------------------------
    # SecretStr (C2.4): a managed Redis URL carries a password
    # (rediss://default:<password>@host:port). Unwrap it only where the client is created.
    # The response cache's separate instance (stretch S2) gets a cache_redis_url, also SecretStr.
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # --- Requests ------------------------------------------------------------
    # One number sets both the request's deadline_ms (arrival + this) and the API's 504
    # timer (C1.16).
    request_timeout_ms: int = Field(default=500, ge=1)

    # --- Worker process --------------------------------------------------------
    # Which catalog entry (common/models.py) this worker serves. One worker serves one
    # model version (D9), so these are per process, set per worker.
    worker_model: str = "sentiment"
    worker_version: int = Field(default=1, ge=1)
    # PyTorch intra-op threads per worker. Workers x threads = performance cores (C1.5).
    worker_threads: int = Field(default=4, ge=1)
    # Where model directories live; the catalog names a directory inside it.
    artifacts_dir: Path = Path("artifacts")

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
