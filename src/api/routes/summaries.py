from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user_id, get_document_service, get_metadata_service
from schemas.api_common import ApiErrorResponse
from schemas.summaries import SummaryResponse
from services.document_service import DocumentService
from services.metadata_service import MetadataService

router = APIRouter(tags=["summaries"])


@router.get(
    "/documents/{document_id}/summary",
    response_model=SummaryResponse,
    responses={404: {"model": ApiErrorResponse}},
)
def get_document_summary(
    document_id: str,
    owner_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service),  # noqa: B008
    metadata: MetadataService = Depends(get_metadata_service),  # noqa: B008
) -> SummaryResponse:
    _ = document_service.get_document_detail(document_id, owner_id)
    summary = metadata.get_summary(document_id)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "summary_not_found", "message": "Summary not found."},
        )
    return SummaryResponse.model_validate(summary)
