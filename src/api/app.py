from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_retriever, get_settings, validate_runtime_readiness
from api.middleware import ObservabilityMiddleware
from api.routes import router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app.name)
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.on_event("startup")
    def _startup_checks() -> None:
        if os.environ.get("RAG_SKIP_STARTUP_VALIDATION", "0") == "1":
            return
        validate_runtime_readiness()
        # Warm up retriever dependencies so index/model issues fail fast at startup.
        _ = get_retriever()

    return app


app = create_app()
