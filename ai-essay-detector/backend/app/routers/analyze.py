"""``POST /analyze``.

Thin by design: validate, rate-limit, delegate, return. No feature logic and no
model access -- ``services.classifier`` owns scoring, ``services.model_loader``
owns the model's lifetime.

Validation happens in two layers on purpose. Pydantic enforces a character
bound in ``models/schemas.py``, which is cheap and rejects obvious junk before
any work. The real limit is a word count, and that lives here: it needs to
report the actual number back to the caller, and counting words is handler
logic rather than a schema constraint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from app.services import classifier, model_loader
from app.services.rate_limit import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])

_settings = get_settings()

# Process-wide, created once at import. See services/rate_limit.py for what
# this does and does not protect against.
_limiter = SlidingWindowRateLimiter(
    limit=_settings.rate_limit_requests,
    window_seconds=_settings.rate_limit_window_seconds,
    max_tracked_clients=_settings.rate_limit_max_tracked_clients,
)

GENERIC_FAILURE = "Analysis failed — please try again."


def client_key(request: Request, settings: Settings) -> str:
    """Identify the caller for rate-limiting purposes.

    ``X-Forwarded-For`` is only read when TRUST_FORWARDED_FOR is on. Trusting it
    unconditionally would make the limit trivially bypassable -- any caller can
    set that header, and each spoofed value looks like a fresh client.
    """
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Reject callers over their quota with a 429 and a Retry-After."""
    if not settings.rate_limit_enabled:
        return

    key = client_key(request, settings)
    decision = _limiter.check(key)
    if decision.allowed:
        return

    retry_after = max(int(decision.retry_after + 0.999), 1)
    logger.warning("Rate limit hit by %s; retry in %ss", key, retry_after)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded — at most {settings.rate_limit_requests} "
            f"analyses every {int(settings.rate_limit_window_seconds)} seconds. "
            f"Try again in {retry_after} second(s)."
        ),
        headers={"Retry-After": str(retry_after)},
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse an essay for AI-generated writing",
    dependencies=[Depends(enforce_rate_limit)],
    responses={
        400: {"model": ErrorResponse, "description": "Essay too short or too long"},
        422: {"model": ErrorResponse, "description": "Body failed validation"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Analysis failed"},
        503: {"model": ErrorResponse, "description": "Models are not loaded"},
    },
)
async def analyze(
    payload: AnalyzeRequest, settings: Settings = Depends(get_settings)
) -> AnalyzeResponse:
    if not (model_loader.is_loaded() and classifier.is_loaded()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded; the service is not ready.",
        )

    essay = payload.essay

    # The real length check. Pydantic's character bound is a cheap proxy; this
    # reports the actual count so the caller knows how much more to write.
    word_count = len(essay.split())

    if word_count < settings.min_essay_words:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Essay is {word_count} words — at least "
                f"{settings.min_essay_words} words are needed for reliable "
                "sentence-level analysis."
            ),
        )

    if word_count > settings.max_essay_words:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Essay is {word_count} words — the maximum is "
                f"{settings.max_essay_words}. Scoring time grows with length, "
                "and this is far longer than any admissions essay. Submit a "
                "single essay rather than a combined document."
            ),
        )

    # Scoring is synchronous, CPU-bound torch work. Off the event loop it goes,
    # so one long essay cannot stall every other connection.
    try:
        return await run_in_threadpool(classifier.score_essay, essay)

    except classifier.EssayTooShortError as exc:
        # Cleared both length gates but yielded no analysable prose.
        # Bad input, not a server fault.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    except HTTPException:
        raise  # already shaped for the client; don't rewrite it as a 500

    except (RuntimeError, MemoryError) as exc:
        # Torch's failure channel: CUDA OOM, allocation failures, a model in a
        # bad state. Real cause to the log, generic message to the caller.
        logger.exception("Scoring failed on a %d-word essay", word_count)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=GENERIC_FAILURE
        ) from exc

    except (ValueError, KeyError, IndexError, OSError) as exc:
        # Tokenizer and feature-extraction failures: unencodable input, a
        # missing feature the artifact expects, a cache file that vanished.
        logger.exception("Tokenizer/feature failure on a %d-word essay", word_count)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=GENERIC_FAILURE
        ) from exc

    except Exception as exc:  # noqa: BLE001 - backstop, still logged and generic
        # Not a bare except standing in for thought: the specific handlers
        # above cover the known failure modes, and this exists so an unforeseen
        # one is still logged in full and still answered generically rather
        # than escaping as a stack trace.
        logger.exception("Unexpected scoring failure on a %d-word essay", word_count)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=GENERIC_FAILURE
        ) from exc
