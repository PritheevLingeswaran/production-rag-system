from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_document_service, get_settings, validate_runtime_readiness
from schemas.api_common import ApiErrorResponse
from schemas.query import HealthResponse
from services.document_service import DocumentService
from utils.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:  # noqa: B008
    return HealthResponse(status="ok", environment=settings.app.environment, checks={"api": "ok"})


@router.get("/readyz", response_model=HealthResponse)
@router.get("/readiness", response_model=HealthResponse)
@router.get(
    "/api/v1/readiness",
    response_model=HealthResponse,
    responses={503: {"model": ApiErrorResponse}},
    include_in_schema=False,
)
def readiness(
    document_service: DocumentService = Depends(get_document_service),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> HealthResponse:
    try:
        validate_runtime_readiness()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "runtime_not_ready", "message": str(exc)},
        ) from exc
    _ = document_service
    return HealthResponse(
        status="ok",
        environment=settings.app.environment,
        checks={"runtime": "ok", "documents": "ok"},
    )
