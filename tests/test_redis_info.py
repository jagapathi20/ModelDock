"""Benchmarks record which Redis they ran against (scripts/redis_info.py)."""

from scripts.redis_info import describe, redis_server_info


def test_unreachable_redis_is_recorded_as_none():
    assert redis_server_info("redis://127.0.0.1:1/0") is None
    assert "not reachable" in describe(None)


def test_native_redis_gets_a_warning():
    native = {
        "redis_version": "8.10.2",
        "os": "Darwin 25.1.0 arm64",
        "redis_mode": "standalone",
        "hz": "10",
    }
    assert "not running in a Docker container" in describe(native)


def test_container_redis_has_no_warning():
    container = {
        "redis_version": "7.4.7",
        "os": "Linux 6.10.14-linuxkit aarch64",
        "redis_mode": "standalone",
        "hz": "10",
    }
    assert describe(container) == "redis: 7.4.7 on Linux 6.10.14-linuxkit aarch64, hz=10"
