"""FastAPI application for the API process.

Run locally with:
    uv run uvicorn api.main:app --port 8000

Request path for ``POST /v1/models/{name}/predict`` (Stage 2 part of DESIGN.md section 3):

1. Body size limit, in middleware: over 16 KB -> 413 before parsing (C2.2).
2. Resolve the model to its serving version (one per model for now) -> 404 if unknown.
3. Parse JSON and validate it against the version's input schema -> 422 (C2.1).
4. Enqueue and wait for the reply (api/rpc.py) -> 504 on timeout, 503 if Redis fails.
5. Map the reply: ok -> 200, invalid_input -> 422, inference_failed -> 500.

Auth, rate limiting, caching and backpressure slot in between 2 and 4 in later stages.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from redis.asyncio import Redis

from api.limits import BodySizeLimit
from api.rpc import EnqueueFailed, PendingReplies, ReplyListener, RequestTimeout, RpcClient
from common.config import Settings, get_settings
from common.models import UnknownModelError, serving_version
from common.protocol import INVALID_INPUT, ReplyMessage, reply_key

log = logging.getLogger("modeldock.api")

RedisFactory = Callable[..., Redis]


def create_app(
    settings: Settings | None = None, redis_factory: RedisFactory | None = None
) -> FastAPI:
    """Build the FastAPI app.

    ``redis_factory(url, **options)`` creates a Redis client; tests pass one that returns an
    in-memory fake. Nothing connects until the lifespan starts, and even then connections open
    lazily: an API process started while Redis is down comes up, reports not ready, and
    recovers by itself when Redis returns.
    """
    settings = settings or get_settings()
    factory = redis_factory or Redis.from_url

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        url = settings.redis_url.get_secret_value()  # the only place the secret is unwrapped
        # socket_timeout (2 s) exceeds the listener's BLPOP timeout (1 s), or every idle wait
        # would be reported as a timeout error (C1.19).
        options = {"socket_timeout": 2.0, "socket_connect_timeout": 2.0}
        commands = factory(url, **options)
        listener_client = factory(url, **options)  # dedicated: BLPOP keeps its connection busy

        # Unique per process, so each process reads only its own replies (D4).
        api_instance_id = uuid.uuid4().hex
        pending = PendingReplies()
        listener = ReplyListener(listener_client, reply_key(api_instance_id), pending)
        app.state.pending = pending
        app.state.listener = listener
        app.state.rpc = RpcClient(commands, api_instance_id, pending, settings.request_timeout_ms)
        listener.start()
        log.info("API instance %s listening on %s", api_instance_id, listener.key)
        try:
            yield
        finally:
            await listener.stop()
            await commands.aclose()
            await listener_client.aclose()

    app = FastAPI(title="ModelDock API", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        # Liveness answers one question: is this process running and able to handle an HTTP
        # request? It deliberately checks no dependencies. If it checked Redis, a Redis outage
        # would make the orchestrator restart every API process, which fixes nothing.
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        # Readiness: can this process serve right now? Only if its reply listener ran within
        # the last 3 s (C1.19). That covers a crashed listener, a stuck one, and Redis being
        # unreachable (the listener's BLPOP keeps failing, so its heartbeat stops).
        listener: ReplyListener | None = getattr(request.app.state, "listener", None)
        if listener is not None and listener.healthy():
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready", "reason": "reply listener"}, status_code=503)

    @app.post("/v1/models/{name}/predict")
    async def predict(name: str, request: Request) -> JSONResponse:
        started = time.perf_counter()
        try:
            spec = serving_version(name)
        except UnknownModelError as e:
            return _error(404, "unknown_model", str(e))

        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(422, INVALID_INPUT, "request body is not valid JSON")
        try:
            model_input = spec.validate_input(payload)
        except ValidationError as e:
            return _error(
                422, INVALID_INPUT, "input does not match the model's schema", _details(e)
            )

        try:
            reply = await request.app.state.rpc.call(spec, model_input)
        except RequestTimeout as e:
            return _error(
                504, "timeout", "no reply from the model worker in time", request_id=str(e)
            )
        except EnqueueFailed:
            log.exception("enqueue failed")
            return _error(
                503,
                "backend_unavailable",
                "request queue unavailable",
                headers={"Retry-After": "1"},
            )
        return _reply_response(spec.name, reply, app_ms=(time.perf_counter() - started) * 1000)

    return app


def _reply_response(model: str, reply: ReplyMessage, app_ms: float) -> JSONResponse:
    headers = {"X-Request-Id": reply.request_id, "X-Model-Version": str(reply.version)}
    if reply.status == "ok":
        if reply.timings is not None:
            # Server-Timing is a standard header for server-side timings (browsers' dev tools
            # show it). The overhead benchmark reads it: app = time inside this handler.
            t = reply.timings
            headers["Server-Timing"] = (
                f"app;dur={app_ms:.3f}, queue;dur={t.queue_wait_ms:.3f}, "
                f"batch;dur={t.batch_wait_ms:.3f}, inference;dur={t.inference_ms:.3f}"
            )
            headers["X-Batch-Size"] = str(t.batch_size)
        body = {
            "request_id": reply.request_id,
            "model": model,
            "version": reply.version,
            "output": reply.output,
        }
        return JSONResponse(body, headers=headers)
    assert reply.error is not None
    status = 422 if reply.error.code == INVALID_INPUT else 500
    return _error(
        status, reply.error.code, reply.error.message, request_id=reply.request_id, headers=headers
    )


def _error(
    status: int,
    code: str,
    message: str,
    details: Any = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Every error has the same shape: ``{"error": {"code", "message", ...}}`` (C2.5)."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    if request_id is not None:
        error["request_id"] = request_id
    return JSONResponse({"error": error}, status_code=status, headers=headers)


def _details(e: ValidationError) -> list[dict[str, Any]]:
    # Location and message only: pydantic's "input" field would echo the client's data back
    # (up to the whole body), and "ctx" can hold objects that aren't JSON-serializable.
    return [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in e.errors()]


app = create_app()
