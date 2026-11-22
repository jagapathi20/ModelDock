"""Run one worker process: ``uv run python -m worker``.

It serves the catalog entry named by ``MODELDOCK_WORKER_MODEL`` / ``MODELDOCK_WORKER_VERSION``
(default: sentiment v1), loaded from ``MODELDOCK_ARTIFACTS_DIR``. Command-line flags override them:

    uv run python -m worker --model iris
"""

import argparse
import logging

import redis

from common.config import get_settings
from common.models import get_spec
from worker.loop import Worker, check_spec_matches
from worker.runners import build_runner


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default=settings.worker_model)
    p.add_argument("--version", type=int, default=settings.worker_version)
    args = p.parse_args()

    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    spec = get_spec(args.model, args.version)

    if spec.kind == "transformer_classifier":
        # Process-wide, and before the first inference (see configure_cpu_threads).
        from worker.runners.transformer import configure_cpu_threads

        configure_cpu_threads(settings.worker_threads)

    runner = build_runner(spec.kind)
    runner.load(settings.artifacts_dir / spec.artifact)
    check_spec_matches(spec, runner)

    # socket_timeout must exceed the BLPOP timeout (1 s), or every idle wait would surface as a
    # timeout error. health_check_interval makes redis-py ping a connection that sat idle, so a
    # connection Redis dropped is noticed before a command is sent on it.
    client = redis.Redis.from_url(
        settings.redis_url.get_secret_value(),
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        health_check_interval=30,
    )
    worker = Worker(spec, runner, client)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass  # Ctrl-C while developing. Graceful SIGTERM shutdown is Stage 5.
    finally:
        logging.getLogger("modeldock.worker").info("stats: %s", worker.stats)


if __name__ == "__main__":
    main()
