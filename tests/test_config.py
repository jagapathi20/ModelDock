import pytest
from pydantic import ValidationError

from common.config import Settings, get_settings


def test_defaults_point_at_local_compose_services():
    s = Settings()
    assert s.env == "dev"
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.s3_endpoint_url == "http://localhost:9000"
    assert s.s3_bucket == "models"


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("MODELDOCK_REDIS_URL", "redis://redis:6379/1")
    monkeypatch.setenv("MODELDOCK_API_PORT", "9100")
    s = Settings()
    assert s.redis_url == "redis://redis:6379/1"
    assert s.api_port == 9100  # parsed from string to int


def test_env_var_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("modeldock_log_level", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_dotenv_file_is_read_but_real_env_wins(tmp_path, monkeypatch):
    # conftest already chdir'd into tmp_path, so .env here is the one loaded.
    (tmp_path / ".env").write_text("MODELDOCK_S3_BUCKET=from-dotenv\nMODELDOCK_ENV=test\n")
    monkeypatch.setenv("MODELDOCK_ENV", "prod")
    s = Settings()
    assert s.s3_bucket == "from-dotenv"
    assert s.env == "prod"


def test_invalid_values_fail_at_startup(monkeypatch):
    monkeypatch.setenv("MODELDOCK_API_PORT", "70000")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("MODELDOCK_API_PORT", "8000")
    monkeypatch.setenv("MODELDOCK_ENV", "staging")  # not one of dev/test/prod
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_masked_when_printed():
    s = Settings()
    assert "minioadmin" not in repr(s.s3_secret_key)
    assert s.s3_secret_key.get_secret_value() == "minioadmin"
    assert "modeldock:modeldock" not in repr(s)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
