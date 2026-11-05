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
