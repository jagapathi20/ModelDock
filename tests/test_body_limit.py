"""Body size limit middleware (C2.2): 413 before the body is parsed."""

import httpx
import pytest
from fastapi import FastAPI, Request

from api.limits import BodySizeLimit

LIMIT = 100


@pytest.fixture
def app():
    app = FastAPI()
    app.state.handler_ran = False
    app.add_middleware(BodySizeLimit, max_bytes=LIMIT)

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        request.app.state.handler_ran = True
        return {"size": len(await request.body())}

    return app


async def _post(app, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/echo", **kwargs)


async def test_body_at_the_limit_passes(app):
    resp = await _post(app, content=b"x" * LIMIT)
    assert resp.status_code == 200
    assert resp.json() == {"size": LIMIT}


async def test_declared_length_over_limit_is_rejected_before_the_handler(app):
    resp = await _post(app, content=b"x" * (LIMIT + 1))
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "body_too_large"
    assert app.state.handler_ran is False  # rejected from the header alone


async def test_chunked_body_over_limit_is_rejected(app):
    # A generator body has no Content-Length, so httpx sends it chunked: the middleware has to
    # count bytes as they arrive.
    async def chunks():
        for _ in range(5):
            yield b"x" * 40

    resp = await _post(app, content=chunks())
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "body_too_large"


async def test_chunked_body_under_limit_passes(app):
    async def chunks():
        yield b"x" * 30
        yield b"x" * 30

    resp = await _post(app, content=chunks())
    assert resp.status_code == 200
    assert resp.json() == {"size": 60}


async def test_invalid_content_length_is_400(app):
    # httpx always sets a correct Content-Length, so call the middleware directly.
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [(b"content-length", b"abc")],
    }
    await BodySizeLimit(app, max_bytes=LIMIT)(scope, receive, send)
    assert sent[0]["status"] == 400
