import pytest

from common.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Make every test start from a clean configuration.

    - Clears the get_settings() cache so env changes in one test can't leak
      into another.
    - Removes any MODELDOCK_* vars from the developer's shell.
    - Runs from an empty temp dir so a local .env file is not picked up.
    """
    import os

    for key in list(os.environ):
        if key.startswith("MODELDOCK_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- Redis fixtures (Stage 2) ---------------------------------------------------------------
#
# Unit tests use fakeredis: an in-memory implementation of the Redis commands, so they need no
# server. Tests marked `redis` (tests/test_roundtrip_redis.py) run against a real server at
# TEST_REDIS_URL (default: database 15 of the Compose Redis, so they never touch your dev data
# in database 0), and are skipped if none is reachable. The name has no MODELDOCK_ prefix on
# purpose: the autouse fixture above deletes MODELDOCK_* variables.

import os  # noqa: E402

import fakeredis  # noqa: E402
import httpx  # noqa: E402

from api.main import create_app  # noqa: E402
from common.config import Settings  # noqa: E402

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def fake_server():
    """One in-memory Redis "server" shared by every client a test creates."""
    return fakeredis.FakeServer()


@pytest.fixture
def make_api(fake_server):
    """Build an API app wired to the fake server; returns an async context manager.

        async with make_api(request_timeout_ms=200) as (client, app):
            await client.post(...)

    It runs the app's lifespan (so the reply listener starts) and gives an httpx client that
    calls the app in-process.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _make(**overrides):
        settings = Settings(**overrides)
        app = create_app(
            settings, redis_factory=lambda url, **kw: fakeredis.FakeAsyncRedis(server=fake_server)
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, app

    return _make
