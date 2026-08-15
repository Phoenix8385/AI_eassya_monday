"""``POST /analyze``.

Deliberately thin: validate (Pydantic already did), check the models are up,
delegate to the service layer, return. No feature logic, no model access.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from app.services import classifier, model_loader

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse an essay for AI-generated writing",
    responses={
        422: {"model": ErrorResponse, "description": "Essay failed validation"},
        503: {"model": ErrorResponse, "description": "Models are not loaded"},
    },
)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if not (model_loader.is_loaded() and classifier.is_loaded()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded; the service is not ready.",
        )

    # Scoring is synchronous, CPU-bound torch work. Off the event loop it goes,
    # so one long essay cannot stall every other connection.
    return await run_in_threadpool(classifier.score_essay, payload.essay)
