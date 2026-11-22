"""Model worker process: loads one model version and runs batched inference.

Runners (one per kind of model) are in ``worker.runners``; the queue loop is in ``worker.loop``.
Run a worker with ``uv run python -m worker`` (see ``worker/__main__.py``).
"""
