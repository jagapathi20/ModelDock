import httpx

from api.main import create_app


async def test_live_returns_200():
    app = create_app()
    # ASGITransport calls the app in-process: no server, no network port.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")

    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


async def test_unknown_route_is_404():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/nope")

    assert resp.status_code == 404
