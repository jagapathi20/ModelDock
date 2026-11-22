import pytest
from pydantic import ValidationError

from common.config import Settings, get_settings


def test_defaults_point_at_local_compose_services():
    s = Settings()
    assert s.env == "dev"
    assert s.redis_url.get_secret_value() == "redis://localhost:6379/0"
    assert s.request_timeout_ms == 500  # C1.16
    assert s.max_body_bytes == 16_384  # C2.2


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("MODELDOCK_REDIS_URL", "redis://redis:6379/1")
    monkeypatch.setenv("MODELDOCK_API_PORT", "9100")
    s = Settings()
    assert s.redis_url.get_secret_value() == "redis://redis:6379/1"
    assert s.api_port == 9100  # parsed from string to int


def test_env_var_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("modeldock_log_level", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_dotenv_file_is_read_but_real_env_wins(tmp_path, monkeypatch):
    # conftest already chdir'd into tmp_path, so .env here is the one loaded.
    (tmp_path / ".env").write_text("MODELDOCK_LOG_LEVEL=DEBUG\nMODELDOCK_ENV=test\n")
    monkeypatch.setenv("MODELDOCK_ENV", "prod")
    s = Settings()
    assert s.log_level == "DEBUG"  # only in .env, so .env is read
    assert s.env == "prod"  # in both, so the real env var wins


def test_invalid_values_fail_at_startup(monkeypatch):
    monkeypatch.setenv("MODELDOCK_API_PORT", "70000")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("MODELDOCK_API_PORT", "8000")
    monkeypatch.setenv("MODELDOCK_ENV", "staging")  # not one of dev/test/prod
    with pytest.raises(ValidationError):
        Settings()


def test_removed_settings_in_an_old_env_file_are_ignored(tmp_path):
    # A .env written for Stage 0/1 still sets the Postgres and S3 variables removed in
    # Stage 1.5. extra="ignore" means they are skipped, so an old .env doesn't break startup.
    (tmp_path / ".env").write_text(
        "MODELDOCK_S3_BUCKET=models\nMODELDOCK_POSTGRES_DSN=postgresql://x\n"
    )
    s = Settings()
    assert not hasattr(s, "s3_bucket")
    assert not hasattr(s, "postgres_dsn")


def test_redis_url_is_masked_when_printed(monkeypatch):
    # A managed Redis URL carries a password (C2.4). Logging the settings must not leak it.
    monkeypatch.setenv("MODELDOCK_REDIS_URL", "rediss://default:hunter2@cache.example.com:6379")
    s = Settings()
    assert "hunter2" not in repr(s)
    assert "hunter2" not in str(s.redis_url)
    assert s.redis_url.get_secret_value().endswith("@cache.example.com:6379")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
