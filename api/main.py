"""FastAPI application for the API process.

Run locally with:
    uv run uvicorn api.main:app --reload
"""

from fastapi import FastAPI

from common.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    A factory (rather than a module-level app built inline) lets tests create a
    fresh app with their own Settings, and later stages will attach startup and
    shutdown work here (Redis pool, reply-list listener, routing table).
    """
    settings = settings or get_settings()
    app = FastAPI(title="ModelDock API", version="0.1.0")
    app.state.settings = settings

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        # Liveness answers one question: is this process running and able to
        # handle an HTTP request? It deliberately checks no dependencies.
        # If it checked Redis, a Redis outage would make the orchestrator
        # restart every API process, which fixes nothing and adds churn.
        # Dependency checks belong in /health/ready (added in later stages).
        return {"status": "alive"}

    return app


app = create_app()
