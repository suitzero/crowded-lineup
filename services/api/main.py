"""FastAPI application entrypoint with CORS, Prometheus metrics, and health probe."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from services.api.config import Settings
from services.api.routers import documents, intelligence, leaderboard, telemetry
from services.api.state import ServiceContainer

HTTP_REQUESTS_TOTAL = Counter(
    "crowded_lineup_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_LATENCY_SECONDS = Histogram(
    "crowded_lineup_http_request_duration_seconds",
    "HTTP request latency seconds",
    ["method", "path"],
)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    app = FastAPI(title=cfg.app_name, version="2.0.0")
    app.state.container = ServiceContainer(cfg)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        t0 = perf_counter()
        response: Response = await call_next(request)
        elapsed = perf_counter() - t0
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        HTTP_LATENCY_SECONDS.labels(method=request.method, path=request.url.path).observe(elapsed)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        c: ServiceContainer = app.state.container
        return {
            "status": "ok",
            "indexed_chunks": c.index.count(),
            "documents": len(c.repo.documents),
            "heats": len(c.repo.heats),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(telemetry.router)
    app.include_router(leaderboard.router)
    app.include_router(documents.router)
    app.include_router(intelligence.router)
    return app


app = create_app()
