"""Record which Redis server a benchmark ran against (C1.15: Redis runs in Docker).

``localhost:6379`` can be answered by more than one server: the Compose container, or a Redis
installed directly on the machine (e.g. with Homebrew) that keeps running when Docker is stopped.
Results must say which one they measured. Redis's ``INFO server`` reports the server's ``os``:
``Linux ...`` for the container (Docker Desktop runs containers in a Linux VM), ``Darwin ...``
for a Redis running natively on macOS.
"""

from typing import Any

import redis


def redis_server_info(url: str) -> dict[str, Any] | None:
    """Version, OS, mode and timer frequency of the server at ``url``; None if unreachable."""
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=2)
        info = client.info("server")
        hz = client.config_get("hz").get("hz")
    except redis.RedisError:
        return None
    return {
        "redis_version": info.get("redis_version"),
        "os": info.get("os"),
        "redis_mode": info.get("redis_mode"),
        "hz": hz,
    }


def describe(info: dict[str, Any] | None) -> str:
    """One line for the console, with a warning if Redis isn't the containerised one."""
    if info is None:
        return "redis: not reachable (server info not recorded)"
    line = f"redis: {info['redis_version']} on {info['os']}, hz={info['hz']}"
    if not str(info.get("os", "")).startswith("Linux"):
        line += (
            "\nwarning: this Redis is not running in a Docker container; benchmarks should use "
            "the Compose Redis (C1.15). Stop the native one (e.g. `brew services stop redis`)."
        )
    return line
